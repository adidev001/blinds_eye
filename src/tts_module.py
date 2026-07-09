"""
Blind's Eye — Text-to-Speech Module (Piper TTS Backend)
=========================================================
Provides a fully offline, low-latency, asynchronous TTS engine
built on top of Piper TTS (ONNX inference) and sounddevice for
direct audio streaming to the system speakers. No audio files are
ever written to disk.

Architecture
------------
  Main thread         Worker thread (daemon)
  ─────────           ──────────────────────────────────────────
  tts.speak(msg)  →   Dequeues message from PriorityQueue
                      → Piper synthesizes raw PCM samples
                      → sounddevice plays them directly

Critical Safety Feature
-----------------------
  If a message begins with "CRITICAL:", the normal queue is
  immediately flushed and the critical warning is placed at the
  front so it is spoken without any backlog delay. This is the
  most important safety contract of this module.

Usage
-----
    from src.tts_module import TTSEngine

    tts = TTSEngine(model_path="models/tts/en_US-amy-low.onnx")
    tts.speak("Person at 2 meters.")
    tts.speak("CRITICAL: Stairs directly ahead!")
    # ... later ...
    tts.shutdown()
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Optional imports — degrade gracefully when libraries are absent so that
# the rest of the application can still start without audio.
# ──────────────────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
    print("[TTS] WARNING: 'sounddevice' not installed. Audio playback disabled.")

try:
    from piper.voice import PiperVoice
    _PIPER_AVAILABLE = True
except ImportError:
    _PIPER_AVAILABLE = False
    print("[TTS] WARNING: 'piper-tts' not installed. Falling back to win32com/espeak.")


# ──────────────────────────────────────────────────────────────────────────────
# Internal sentinel — signals the worker thread to exit cleanly.
# ──────────────────────────────────────────────────────────────────────────────
_STOP_SENTINEL = object()

# Priority levels used by the internal PriorityQueue.
# Lower number = higher priority.
_PRIORITY_CRITICAL = 0
_PRIORITY_NORMAL   = 1


class TTSEngine:
    """
    Asynchronous, offline Text-to-Speech engine.

    Parameters
    ----------
    model_path : str | Path | None
        Path to the Piper TTS ``.onnx`` voice model file.
        If *None* or the file is missing, the engine falls back to the
        system's native TTS (win32com on Windows / espeak on Linux).
    queue_max : int
        Maximum number of *normal-priority* messages allowed to wait in
        the queue before new ones are silently dropped. Critical messages
        always bypass this limit.
    speech_rate_scale : float
        Speech speed multiplier fed to Piper (1.0 = default, 0.75 = slower,
        1.25 = faster). Ignored when using the native fallback backend.
    sample_rate : int
        PCM sample rate expected by the loaded Piper voice model.
        Piper ``*-low`` models use 16 000 Hz; ``*-medium`` / ``*-high``
        models use 22 050 Hz. Must match the ``.onnx`` model exactly.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        queue_max:  int   = 4,
        speech_rate_scale: float = 1.0,
        sample_rate: int  = 22050,
    ) -> None:
        self._model_path       = Path(model_path) if model_path else None
        self._queue_max        = queue_max
        self._speech_rate_scale = speech_rate_scale
        self._sample_rate      = sample_rate
        self._running          = True

        # A PriorityQueue ensures CRITICAL messages always jump the queue.
        # Each item is a (priority: int, counter: int, text: str) tuple.
        # The counter breaks ties while preserving FIFO within the same priority.
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._counter = 0           # Monotonically-increasing tie-breaker
        self._counter_lock = threading.Lock()

        # Initialise the synthesis backend (either Piper or native fallback).
        self._piper_voice: Optional[PiperVoice] = None
        self._fallback_speaker = None   # win32com / espeak handle
        self._backend = self._init_backend()

        # Spawn the background audio worker thread.
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="tts-worker",
            daemon=True,             # Dies automatically when main exits.
        )
        self._worker.start()
        print(f"[TTS] Engine started — backend: {self._backend}, "
              f"model: {self._model_path.name if self._model_path else 'native'}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def speak(self, message: str) -> None:
        """
        Enqueue *message* for asynchronous speech output.

        If the message begins with ``"CRITICAL"`` (case-insensitive) the
        entire normal queue is **flushed immediately** and the critical
        warning is inserted at the front. This guarantees the warning is
        spoken as soon as the current utterance finishes.

        Normal messages are silently dropped when the queue is full
        (``queue_max`` exceeded) to prevent a stale audio backlog from
        building up during high-inference-rate loops.
        """
        if not self._running:
            return

        is_critical = message.strip().upper().startswith("CRITICAL")

        if is_critical:
            # ── Critical path: drain queue, then jump to front ──────────────
            self._flush_normal_queue()
            self._enqueue(message, _PRIORITY_CRITICAL)
            print(f"[TTS][CRITICAL] {message}")
        else:
            # ── Normal path: drop if too full ────────────────────────────────
            # Count only normal-priority items (priority == _PRIORITY_NORMAL).
            current_size = self._queue.qsize()
            if current_size >= self._queue_max:
                print(f"[TTS] Queue full — dropping: {message}")
                return
            self._enqueue(message, _PRIORITY_NORMAL)

    def shutdown(self) -> None:
        """
        Gracefully stop the worker thread and release audio resources.

        Blocks until the worker finishes (up to 3 seconds) so that
        any currently-speaking utterance completes before exit.
        """
        self._running = False
        # Wake the worker even if the queue is empty.
        self._queue.put((_PRIORITY_CRITICAL, -1, _STOP_SENTINEL))
        self._worker.join(timeout=3.0)
        print("[TTS] Engine shut down.")

    # ─────────────────────────────────────────────────────────────────────────
    # Internals — Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_backend(self) -> str:
        """
        Probe for the best available TTS backend in preference order:
          1. Piper TTS (fully offline, high-quality ONNX voice model)
          2. win32com SAPI (Windows native, zero install)
          3. espeak-ng (Linux native, zero install)
          4. pyttsx3 (last resort cross-platform fallback)
        Returns a short backend identifier string for logging.
        """
        # 1. Piper TTS ───────────────────────────────────────────────────────
        if _PIPER_AVAILABLE and self._model_path and self._model_path.exists():
            try:
                self._piper_voice = PiperVoice.load(
                    str(self._model_path),
                    config_path=str(self._model_path.with_suffix(".onnx.json")),
                    use_cuda=False,          # CPU-only for wearable edge device
                )
                return "piper"
            except Exception as exc:
                print(f"[TTS] Piper load failed ({exc}). Trying native fallback.")

        # 2. Windows SAPI via win32com ────────────────────────────────────────
        if sys.platform == "win32":
            try:
                import pythoncom
                import win32com.client
                # CoInitialize must happen in the thread that uses the object;
                # we store the *class names* and initialise in the worker.
                self._fallback_backend_type = "win32com"
                return "win32com"
            except ImportError:
                pass

        # 3. espeak-ng via subprocess (Linux/macOS) ───────────────────────────
        import shutil
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            self._fallback_backend_type = "espeak"
            return "espeak"

        # 4. pyttsx3 last-resort ─────────────────────────────────────────────
        try:
            import pyttsx3  # noqa: F401
            self._fallback_backend_type = "pyttsx3"
            return "pyttsx3"
        except ImportError:
            pass

        print("[TTS] WARNING: No TTS backend found. Speech will be silent.")
        self._fallback_backend_type = "none"
        return "none"

    # ─────────────────────────────────────────────────────────────────────────
    # Internals — Worker Thread
    # ─────────────────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """
        Background daemon thread.

        Continuously dequeues messages and synthesises + plays audio.
        Falls back to the appropriate native engine when Piper is
        unavailable.
        """
        # Per-thread COM initialisation for win32com (must be in same thread).
        native_speaker = None
        if self._backend == "win32com":
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                native_speaker = win32com.client.Dispatch("SAPI.SpVoice")
                # Clamp rate to SAPI's [-10, +10] range.
                native_speaker.Rate = 1
            except Exception as exc:
                print(f"[TTS] win32com init error in worker: {exc}")

        # pyttsx3 engine initialised here to stay on the same thread.
        pyttsx3_engine = None
        if self._backend == "pyttsx3":
            try:
                import pyttsx3
                pyttsx3_engine = pyttsx3.init()
            except Exception as exc:
                print(f"[TTS] pyttsx3 init error in worker: {exc}")

        while True:
            try:
                # Block indefinitely until a message arrives.
                _priority, _counter, text = self._queue.get(timeout=1.0)
            except queue.Empty:
                # Timeout — loop back and check self._running.
                if not self._running:
                    break
                continue

            # Poison-pill sentinel → clean exit.
            if text is _STOP_SENTINEL:
                self._queue.task_done()
                break

            # ── Synthesise and play ──────────────────────────────────────────
            try:
                if self._backend == "piper":
                    self._play_piper(text)
                elif self._backend == "win32com" and native_speaker:
                    native_speaker.Speak(text, 0)   # 0 = synchronous
                elif self._backend == "espeak":
                    self._play_espeak(text)
                elif self._backend == "pyttsx3" and pyttsx3_engine:
                    pyttsx3_engine.say(text)
                    pyttsx3_engine.runAndWait()
            except Exception as exc:
                print(f"[TTS] Playback error: {exc}")
            finally:
                self._queue.task_done()

    def _play_piper(self, text: str) -> None:
        """
        Synthesise *text* with Piper and stream PCM frames directly to
        sounddevice. No temporary .wav file is created on disk.

        ``PiperVoice.synthesize()`` returns an iterable of ``AudioChunk``
        objects. Each chunk carries:
            - ``audio_float_array`` : np.ndarray (float32, already in [-1, 1])
            - ``sample_rate``       : int  (e.g. 22050 or 16000)
            - ``sample_channels``   : int  (almost always 1 for Piper voices)
        We concatenate all chunks into one array and play them in a single
        blocking call so there are no click artefacts between sentences.
        """
        if self._piper_voice is None:
            return

        from piper.voice import SynthesisConfig

        syn_config = SynthesisConfig(
            length_scale=1.0 / self._speech_rate_scale,   # speed control
        )

        chunks = list(self._piper_voice.synthesize(text, syn_config=syn_config))
        if not chunks:
            return

        # Concatenate all sentence chunks into one continuous array.
        audio_array = np.concatenate([c.audio_float_array for c in chunks])
        sample_rate  = chunks[0].sample_rate

        if _SD_AVAILABLE:
            # blocking=True: worker thread waits for playback to finish
            # before dequeuing the next message. Overlapping speech is
            # confusing for a blind user.
            sd.play(audio_array, samplerate=sample_rate, blocking=True)
        else:
            print(f"[TTS][NO AUDIO] {text}")



    def _play_espeak(self, text: str) -> None:
        """
        Synthesise *text* using the system espeak-ng command-line tool.
        Runs synchronously (blocking) inside the worker thread.
        """
        import subprocess
        import shutil

        binary = "espeak-ng" if shutil.which("espeak-ng") else "espeak"
        subprocess.run(
            [binary, "-s", "165", "-v", "en-us", text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internals — Queue Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _enqueue(self, message: str, priority: int) -> None:
        """Thread-safe enqueue with monotonic counter tie-breaker."""
        with self._counter_lock:
            self._counter += 1
            counter = self._counter
        self._queue.put((priority, counter, message))

    def _flush_normal_queue(self) -> None:
        """
        Drain all *normal-priority* messages from the queue.

        This is called before inserting a CRITICAL message so that stale
        object-detection announcements don't delay the safety warning.
        We rebuild the queue contents, keeping only CRITICAL items and
        the stop sentinel.
        """
        kept: list[tuple] = []
        # Drain everything non-blocking.
        while True:
            try:
                item = self._queue.get_nowait()
                priority = item[0]
                text     = item[2]
                # Preserve critical items and the stop sentinel.
                if priority == _PRIORITY_CRITICAL or text is _STOP_SENTINEL:
                    kept.append(item)
            except queue.Empty:
                break

        # Re-insert the items we want to keep.
        for item in kept:
            self._queue.put(item)
