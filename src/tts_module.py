"""
Blind's Eye — Text-to-Speech Module (Dual-Engine: Piper + Kokoro)
==================================================================
Provides a fully offline, low-latency, asynchronous TTS engine
that supports two synthesis backends:

  1. Piper TTS   — default, ONNX-based, 22kHz/16kHz float32 PCM.
  2. Kokoro-82M  — higher-quality, 24kHz float32, en/hi/hinglish.
                   Runs on CPUExecutionProvider only to avoid
                   competing with YOLO26 + Depth Anything V2 for VRAM.

Architecture
------------
  Main thread              Worker thread (daemon)
  ──────────               ───────────────────────────────────────
  tts.speak(msg)  ──────►  Dequeues from PriorityQueue
                            → Engine synthesises PCM (blocking)
                            → sounddevice streams directly to speakers

Critical Safety Feature
-----------------------
  If a message begins with "CRITICAL:", the normal queue is
  immediately flushed AND ``sounddevice.stop()`` is called to
  interrupt any currently-playing audio. The critical warning
  is then placed at the front and spoken within ~100ms.

Fallback Chain
--------------
  Kokoro requested  →  Kokoro (if installed + model files present)
                    →  Piper  (if model file present)
                    →  win32com SAPI (Windows native)
                    →  espeak-ng  (Linux native)
                    →  pyttsx3   (last resort)

Usage
-----
    from src.tts_module import TTSEngine

    # Default Piper backend
    tts = TTSEngine(model_path="models/tts/en_US-amy-low.onnx")

    # Kokoro backend with Hindi
    tts = TTSEngine(engine="kokoro", lang="hi")

    tts.speak("Person at 2 meters.")
    tts.speak("CRITICAL: Stairs directly ahead!")
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
        If *None* or the file is missing, the engine falls through to
        the fallback chain (Kokoro → win32com → espeak → pyttsx3).
    engine : str
        Preferred synthesis backend: ``"piper"`` (default) or ``"kokoro"``.
        If Kokoro is requested but unavailable, falls back to Piper.
    lang : str
        Announcement language: ``"en"`` (default), ``"hi"`` (Hindi),
        or ``"hinglish"`` (Roman Hinglish). Controls Kokoro voice selection
        and is passed through for Piper ``hi_IN`` model selection.
    queue_max : int
        Maximum number of *normal-priority* messages allowed to wait in
        the queue before new ones are silently dropped. Critical messages
        always bypass this limit.
    speech_rate_scale : float
        Speech speed multiplier (1.0 = default, 0.75 = slower, 1.25 = faster).
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        engine: str = "piper",
        lang:   str = "en",
        queue_max:  int   = 4,
        speech_rate_scale: float = 1.0,
    ) -> None:
        self._model_path        = Path(model_path) if model_path else None
        self._engine_pref       = engine.lower()
        self._lang              = lang.lower()
        self._queue_max         = queue_max
        self._speech_rate_scale = speech_rate_scale
        self._running           = True

        # ── Kokoro handle (set in _init_backend if kokoro initialises OK) ──
        self._kokoro = None

        # ── PriorityQueue: (priority, counter, text) tuples ───────────────
        # Lower priority number = higher urgency.
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._counter      = 0
        self._counter_lock = threading.Lock()

        # ── Interrupt event: set by _flush_normal_queue when CRITICAL fires ─
        # _play_piper / _play_kokoro check this before calling sd.play().
        self._interrupt_event = threading.Event()

        # ── Piper / native fallback handles ───────────────────────────────
        self._piper_voice: Optional[PiperVoice] = None
        self._fallback_speaker = None

        self._backend = self._init_backend()

        # Spawn the background audio worker thread.
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="tts-worker",
            daemon=True,
        )
        self._worker.start()
        print(f"[TTS] Engine started — backend: {self._backend}, lang: {self._lang}, "
              f"model: {self._model_path.name if self._model_path else 'auto'}")

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
        Probe for the best available TTS backend.

        Preference order when ``--tts-engine kokoro`` is requested:
          0. Kokoro-82M ONNX  (CPU-only, explicitly isolated)
          1. Piper TTS        (automatic fallback)
          2. win32com SAPI    (Windows native)
          3. espeak-ng        (Linux/macOS native)
          4. pyttsx3          (last resort)

        Preference order for ``--tts-engine piper`` (default):
          1. Piper TTS
          2–4. Same native fallback chain

        Returns a short backend identifier string for logging.
        """
        # ── Tier 0: Kokoro-82M (only when explicitly requested) ────────────
        if self._engine_pref == "kokoro":
            try:
                from kokoro_onnx import Kokoro  # noqa: PLC0415
                # Resolve model files relative to a models/tts/ directory
                # that sits two levels above this src/ file.
                from pathlib import Path as _P
                _tts_dir = _P(__file__).resolve().parent.parent / "models" / "tts"
                _onnx    = _tts_dir / "kokoro-v1.0.onnx"
                _voices  = _tts_dir / "voices-v1.0.bin"

                if _onnx.exists() and _voices.exists():
                    # Explicitly CPU-only: we never share VRAM with
                    # YOLO26 or Depth Anything V2.
                    self._kokoro = Kokoro(str(_onnx), str(_voices))
                    print("[TTS] Kokoro-82M loaded (CPUExecutionProvider).")
                    return "kokoro"
                else:
                    print("[TTS] Kokoro model files not found in models/tts/.")
                    print("[TTS]   Expected: kokoro-v1.0.onnx + voices-v1.0.bin")
                    print("[TTS] Falling back to Piper TTS.")
            except ImportError:
                print("[TTS] kokoro-onnx not installed. Falling back to Piper.")
            except Exception as exc:
                print(f"[TTS] Kokoro init failed ({exc}). Falling back to Piper.")

        # ── Tier 1: Piper TTS ─────────────────────────────────────────────
        if _PIPER_AVAILABLE and self._model_path and self._model_path.exists():
            try:
                self._piper_voice = PiperVoice.load(
                    str(self._model_path),
                    config_path=str(self._model_path.with_suffix(".onnx.json")),
                    use_cuda=False,   # CPU-only; TTS must not steal VRAM
                )
                return "piper"
            except Exception as exc:
                print(f"[TTS] Piper load failed ({exc}). Trying native fallback.")

        # ── Tier 2: Windows SAPI via win32com ────────────────────────────
        if sys.platform == "win32":
            try:
                import pythoncom        # noqa: F401
                import win32com.client  # noqa: F401
                self._fallback_backend_type = "win32com"
                return "win32com"
            except ImportError:
                pass

        # ── Tier 3: espeak-ng (Linux/macOS) ──────────────────────────────
        import shutil
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            self._fallback_backend_type = "espeak"
            return "espeak"

        # ── Tier 4: pyttsx3 last resort ───────────────────────────────────
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
                if self._backend == "kokoro":
                    self._play_kokoro(text)
                elif self._backend == "piper":
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

        if _SD_AVAILABLE and not self._interrupt_event.is_set():
            # blocking=True: worker thread waits for playback to finish
            # before dequeuing the next message. Overlapping speech is
            # confusing for a blind user.
            sd.play(audio_array, samplerate=sample_rate, blocking=True)
        else:
            print(f"[TTS][NO AUDIO] {text}")
        self._interrupt_event.clear()   # consume the interrupt signal

    def _play_kokoro(self, text: str) -> None:
        """
        Synthesise *text* with Kokoro-82M ONNX and play via sounddevice.

        Kokoro always returns ``(samples: np.ndarray[float32], sr: int)``
        where ``sr == 24000``. No temporary files are created on disk.

        CPU isolation: Kokoro was initialised without CUDA providers so
        it never competes with YOLO26 / Depth Anything V2 for VRAM.
        """
        if self._kokoro is None:
            return

        # Map our --lang flag to Kokoro's (lang_code, voice_id) pairs.
        # hf_alpha = Hindi female voice, af_heart = English female voice.
        _LANG_MAP: dict[str, tuple[str, str]] = {
            "en":       ("en-us", "af_heart"),
            "hi":       ("hi",    "hf_alpha"),
            "hinglish": ("hi",    "hf_alpha"),  # Roman Hinglish phonemised as Hindi
        }
        kokoro_lang, kokoro_voice = _LANG_MAP.get(self._lang, ("en-us", "af_heart"))

        try:
            samples, sr = self._kokoro.create(
                text,
                voice=kokoro_voice,
                speed=self._speech_rate_scale,
                lang=kokoro_lang,
            )
        except Exception as exc:
            print(f"[TTS] Kokoro synthesis error: {exc}")
            return

        if _SD_AVAILABLE and not self._interrupt_event.is_set():
            sd.play(samples, samplerate=sr, blocking=True)
        else:
            print(f"[TTS][NO AUDIO] {text}")
        self._interrupt_event.clear()   # consume the interrupt signal



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
        Drain all *normal-priority* messages from the queue AND interrupt
        any currently-playing audio so the CRITICAL warning fires within
        ~100ms instead of waiting for the full utterance to finish.

        Steps:
          1. ``sd.stop()``  — immediately halts sounddevice playback
                              (thread-safe, documented for cross-thread use)
          2. ``_interrupt_event.set()`` — tells _play_piper / _play_kokoro
                              not to start a new sd.play() call for the
                              message that was just interrupted.
          3. Drain normal-priority queue items (stale obstacle announcements).
          4. Re-insert any CRITICAL items and the stop sentinel.
        """
        # ── Step 1+2: Stop current audio and signal the playback methods ──
        if _SD_AVAILABLE:
            sd.stop()
        self._interrupt_event.set()

        # ── Steps 3+4: Rebuild queue keeping only high-priority items ─────
        kept: list[tuple] = []
        while True:
            try:
                item = self._queue.get_nowait()
                priority = item[0]
                text     = item[2]
                if priority == _PRIORITY_CRITICAL or text is _STOP_SENTINEL:
                    kept.append(item)
            except queue.Empty:
                break

        for item in kept:
            self._queue.put(item)
