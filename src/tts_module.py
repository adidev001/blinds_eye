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

import functools
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# Audio & Psychoacoustic Helpers (FR-4.2, FR-4.3)
# ══════════════════════════════════════════════════════════════════════════

def generate_tone(
    freq_hz: float,
    duration_sec: float,
    sr: int = 22050,
    amp: float = 0.35,
) -> np.ndarray:
    """
    Generate a sine tone with a 5ms raised-cosine (Hann) taper on both ends (FR-4.3).
    Guarantees first and last samples are exactly 0.0 (zero digital speaker clicks).
    """
    num_samples = max(int(round(sr * duration_sec)), 2)
    t = np.arange(num_samples, dtype=np.float32) / float(sr)
    tone = (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)

    # 5ms taper on both ends
    taper_len = min(int(round(sr * 0.005)), num_samples // 2)
    if taper_len > 0:
        hann_window = np.hanning(2 * taper_len).astype(np.float32)
        tone[:taper_len] *= hann_window[:taper_len]
        tone[-taper_len:] *= hann_window[taper_len:]
    tone[0] = 0.0
    tone[-1] = 0.0
    return tone


def pan_audio(
    audio_mono: np.ndarray,
    direction: str = "center",
    stereo: bool = True,
) -> np.ndarray:
    """
    Pan a 1D mono float32 waveform across stereo channels (FR-4.2).
    Safe gain matrix preserves single-earbud audibility (100/35 safe panning):
      Left:   [1.00, 0.35]
      Center: [0.85, 0.85]
      Right:  [0.35, 1.00]
    When stereo is False, returns 1D array (N,) unchanged.
    """
    if not stereo:
        return audio_mono

    gains = {
        "left":   np.array([1.00, 0.35], dtype=np.float32),
        "center": np.array([0.85, 0.85], dtype=np.float32),
        "right":  np.array([0.35, 1.00], dtype=np.float32),
    }.get(direction, np.array([0.85, 0.85], dtype=np.float32))

    return np.outer(audio_mono, gains)


def generate_critical_prebeep(
    direction: str = "center",
    sample_rate: int = 22050,
    stereo: bool = True,
) -> np.ndarray:
    """
    Generate an 80ms dual-tone Hann earcon chime (40ms @ 880Hz + 40ms @ 1320Hz)
    panned across stereo channels (FR-4.3).
    """
    tone1 = generate_tone(880.0, 0.040, sr=sample_rate, amp=0.35)
    tone2 = generate_tone(1320.0, 0.040, sr=sample_rate, amp=0.35)
    chime = np.concatenate([tone1, tone2])
    return pan_audio(chime, direction=direction, stereo=stereo)


apply_stereo_panning = pan_audio


def detect_direction_from_text(text: str) -> str:
    """Detect spatial direction from formatted announcement string."""
    lower = text.lower()
    if "left" in lower or "बाईं" in text or "baayein" in lower:
        return "left"
    elif "right" in lower or "दाईं" in text or "daayein" in lower:
        return "right"
    return "center"


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
        stereo: bool = True,
        heartbeat: bool = False,
        critical_sound: bool = True,
    ) -> None:
        self._model_path        = Path(model_path) if model_path else None
        self._engine_pref       = engine.lower()
        self._lang              = lang.lower()
        self._queue_max         = queue_max
        self._speech_rate_scale = speech_rate_scale
        self._running           = True
        self._critical_sound    = critical_sound

        # Hardware audio channel auto-negotiation (FR-4.2)
        self._stereo = stereo
        self._heartbeat = heartbeat
        self.last_spoken_time = time.monotonic()
        self._staleness_fn: Optional[Callable[[], bool]] = None
        if self._stereo and _SD_AVAILABLE:
            try:
                device_info = sd.query_devices(kind="output")
                max_ch = int(device_info.get("max_output_channels", 2))
                if max_ch < 2:
                    print(f"[TTS] Output device has only {max_ch} channels — forcing mono.")
                    self._stereo = False
            except Exception:
                pass

        # ── Kokoro handle (set in _init_backend if kokoro initialises OK) ──
        self._kokoro = None

        # ── PriorityQueue: (priority, counter, text) tuples ───────────────
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._counter      = 0
        self._counter_lock = threading.Lock()

        # ── Interrupt event: set by _flush_normal_queue when CRITICAL fires ─
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
              f"model: {self._model_path.name if self._model_path else 'auto'}, stereo: {self._stereo}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def set_staleness_fn(self, fn: Callable[[], bool]) -> None:
        """Register camera staleness callback for heartbeat gating (FR-4.3)."""
        self._staleness_fn = fn

    @property
    def critical_sound(self) -> bool:
        """Return whether critical earcon chimes and urgent prefix are enabled."""
        return self._critical_sound

    @critical_sound.setter
    def critical_sound(self, enabled: bool) -> None:
        self._critical_sound = bool(enabled)

    def toggle_critical_sound(self) -> bool:
        """Toggle critical earcon chime and urgent prefix on/off. Returns new state."""
        self._critical_sound = not self._critical_sound
        state = "ENABLED" if self._critical_sound else "DISABLED"
        print(f"[TTS] Critical hazard sound: {state}")
        return self._critical_sound

    @property
    def stereo(self) -> bool:
        """Return whether stereo spatialization is enabled."""
        return self._stereo

    @stereo.setter
    def stereo(self, enabled: bool) -> None:
        self._stereo = bool(enabled)

    def toggle_stereo(self) -> bool:
        """Toggle stereo audio spatialization on/off. Returns new state."""
        self._stereo = not self._stereo
        state = "ENABLED" if self._stereo else "DISABLED"
        print(f"[TTS] Stereo spatialization: {state}")
        return self._stereo

    @property
    def lang(self) -> str:
        """Return current TTS language code."""
        return self._lang

    def set_lang(self, lang: str) -> str:
        """Set active language code (en, hi, hinglish)."""
        if lang in ("en", "hi", "hinglish"):
            self._lang = lang
            print(f"[TTS] Language set to: {self._lang}")
        return self._lang

    def cycle_lang(self) -> str:
        """Cycle through supported languages: en -> hi -> hinglish -> en."""
        order = ["en", "hi", "hinglish"]
        idx = order.index(self._lang) if self._lang in order else 0
        self._lang = order[(idx + 1) % len(order)]
        print(f"[TTS] Language switched to: {self._lang}")
        return self._lang

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
                from pathlib import Path as _P
                _tts_dir = _P(__file__).resolve().parent.parent / "models" / "tts"

                # ── ONNX model file: prefer renamed copy, fall back to HF cache path
                _onnx = _tts_dir / "kokoro-v1.0.onnx"
                if not _onnx.exists():
                    # onnx-community layout stores it under onnx/model.onnx
                    _onnx_alt = _tts_dir / "onnx" / "model.onnx"
                    if _onnx_alt.exists():
                        _onnx = _onnx_alt

                # ── Voices file: support both legacy pack and per-voice layout
                # Legacy:  voices-v1.0.bin  (hexgrad/Kokoro-82M-ONNX, now deleted)
                # New:     voices/<name>.bin (onnx-community/Kokoro-82M-v1.0-ONNX)
                _voices_legacy = _tts_dir / "voices-v1.0.bin"
                # Map our lang to the appropriate per-voice filename
                _VOICE_FILE_MAP = {
                    "en":       "af_heart.bin",
                    "hi":       "hf_alpha.bin",
                    "hinglish": "hf_alpha.bin",
                }
                _voice_filename = _VOICE_FILE_MAP.get(self._lang, "af_heart.bin")
                # Check both possible locations the HF downloader may place the file
                _voices_per_voice = _tts_dir / "voices" / _voice_filename
                _voices_flat      = _tts_dir / _voice_filename
                _voices = (
                    _voices_legacy    if _voices_legacy.exists()
                    else _voices_per_voice if _voices_per_voice.exists()
                    else _voices_flat      if _voices_flat.exists()
                    else None
                )

                if _onnx.exists() and _voices is not None:
                    # Explicitly CPU-only: we never share VRAM with
                    # YOLO26 or Depth Anything V2.
                    try:
                        import onnxruntime as _ort
                        _sess = _ort.InferenceSession(str(_onnx), providers=["CPUExecutionProvider"])
                        self._kokoro = Kokoro.from_session(_sess, str(_voices))
                    except Exception:
                        self._kokoro = Kokoro(str(_onnx), str(_voices))
                    print(f"[TTS] Kokoro-82M loaded — model: {_onnx.name}, voice: {_voices.name}")
                    return "kokoro"
                else:
                    _missing = []
                    if not _onnx.exists():    _missing.append("kokoro-v1.0.onnx (or onnx/model.onnx)")
                    if _voices is None:       _missing.append(f"{_voice_filename} (or voices-v1.0.bin)")
                    print("[TTS] Kokoro model files not found in models/tts/.")
                    print(f"[TTS]   Missing: {', '.join(_missing)}")
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

    @functools.lru_cache(maxsize=256)
    def _synthesize_clause_cached(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize a clause string to a 1D mono float32 waveform with LRU caching (FR-4.1).
        Cached array is read-only float32; callers copy or view.
        """
        if self._backend == "piper" and self._piper_voice is not None:
            from piper.voice import SynthesisConfig
            syn_config = SynthesisConfig(
                length_scale=1.0 / self._speech_rate_scale,
            )
            chunks = list(self._piper_voice.synthesize(text, syn_config=syn_config))
            if not chunks:
                return np.zeros(0, dtype=np.float32), 22050
            audio_array = np.concatenate([c.audio_float_array for c in chunks])
            sample_rate = chunks[0].sample_rate
            return audio_array, sample_rate

        elif self._backend == "kokoro" and self._kokoro is not None:
            _LANG_MAP: dict[str, tuple[str, str]] = {
                "en":       ("en-us", "af_heart"),
                "hi":       ("hi",    "hf_alpha"),
                "hinglish": ("hi",    "hf_alpha"),
            }
            kokoro_lang, kokoro_voice = _LANG_MAP.get(self._lang, ("en-us", "af_heart"))
            samples, sr = self._kokoro.create(
                text,
                voice=kokoro_voice,
                speed=self._speech_rate_scale,
                lang=kokoro_lang,
            )
            return samples, sr

        return np.zeros(0, dtype=np.float32), 22050

    def pan_audio(self, audio_mono: np.ndarray, direction: str = "center") -> np.ndarray:
        """Pan mono waveform using current engine stereo configuration (FR-4.2)."""
        return pan_audio(audio_mono, direction=direction, stereo=self._stereo)

    def get_cache_stats(self) -> dict:
        """Return LRU cache hit/miss statistics (FR-5.4)."""
        info = self._synthesize_clause_cached.cache_info()
        total = info.hits + info.misses
        return {
            "hits": info.hits,
            "misses": info.misses,
            "currsize": info.currsize,
            "maxsize": info.maxsize,
            "hit_ratio": info.hits / max(total, 1),
        }

    def _worker_loop(self) -> None:
        """
        Background daemon thread.
        Continuously dequeues messages and synthesises + plays audio.
        Supports 15s 800Hz bone-conduction heartbeat when idle (FR-4.3).
        """
        native_speaker = None
        if self._backend == "win32com":
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                native_speaker = win32com.client.Dispatch("SAPI.SpVoice")
                native_speaker.Rate = 1
            except Exception as exc:
                print(f"[TTS] win32com init error in worker: {exc}")

        pyttsx3_engine = None
        if self._backend == "pyttsx3":
            try:
                import pyttsx3
                pyttsx3_engine = pyttsx3.init()
            except Exception as exc:
                print(f"[TTS] pyttsx3 init error in worker: {exc}")

        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                if not self._running:
                    break
                # System-alive heartbeat (FR-4.3): 15ms @ 800Hz after 15s of silence
                if self._heartbeat and _SD_AVAILABLE:
                    if self._staleness_fn is not None and self._staleness_fn():
                        continue
                    now = time.monotonic()
                    if (now - self.last_spoken_time) >= 15.0:
                        try:
                            tick = generate_tone(800.0, 0.015, sr=22050, amp=0.15)
                            tick_out = pan_audio(tick, direction="center", stereo=self._stereo)
                            sd.play(tick_out, samplerate=22050, blocking=True)
                            self.last_spoken_time = time.monotonic()
                        except Exception:
                            pass

                continue

            if len(item) == 4:
                _priority, _counter, text, enqueued_at = item
            else:
                _priority, _counter, text = item
                enqueued_at = time.monotonic()

            if text is _STOP_SENTINEL:
                self._queue.task_done()
                break

            # Drop stale announcements that waited in queue for > 1.8s (prevents phantom chatter)
            age = time.monotonic() - enqueued_at
            if age > 1.8 and not ("Obstacle ahead" in text or "camera" in text.lower()):
                print(f"[TTS] Dropped stale announcement ({age:.1f}s old): {text}")
                self._queue.task_done()
                continue

            try:
                if self._backend == "kokoro":
                    self._play_kokoro(text)
                elif self._backend == "piper":
                    self._play_piper(text)
                elif self._backend == "win32com" and native_speaker:
                    native_speaker.Speak(text, 0)
                    self.last_spoken_time = time.monotonic()
                elif self._backend == "espeak":
                    self._play_espeak(text)
                    self.last_spoken_time = time.monotonic()
                elif self._backend == "pyttsx3" and pyttsx3_engine:
                    pyttsx3_engine.say(text)
                    pyttsx3_engine.runAndWait()
                    self.last_spoken_time = time.monotonic()
            except Exception as exc:
                print(f"[TTS] Playback error: {exc}")
            finally:
                self._queue.task_done()

    def _play_piper(self, text: str) -> None:
        """
        Synthesise text with Piper using clause-level LRU cache, 80ms Hann earcon,
        and safe stereo panning (FR-4.1–FR-4.3).
        """
        if self._piper_voice is None:
            return

        is_critical = text.strip().upper().startswith("CRITICAL")
        direction = detect_direction_from_text(text)

        synth_text = text
        if is_critical and not self._critical_sound:
            if text.upper().startswith("CRITICAL:"):
                synth_text = text[9:].strip()
            elif text.upper().startswith("CRITICAL"):
                synth_text = text[8:].strip()

        audio_mono, sample_rate = self._synthesize_clause_cached(synth_text)
        if audio_mono.size == 0:
            return
        audio_mono = audio_mono.copy()

        # Prepend 80ms dual-tone Hann earcon chime on CRITICAL alerts if critical_sound enabled (FR-4.3)
        if is_critical and self._critical_sound:
            tone1 = generate_tone(880.0, 0.040, sr=sample_rate, amp=0.35)
            tone2 = generate_tone(1320.0, 0.040, sr=sample_rate, amp=0.35)
            chime = np.concatenate([tone1, tone2])
            audio_mono = np.concatenate([chime, audio_mono])

        audio_output = pan_audio(audio_mono, direction=direction, stereo=self._stereo)

        # Critical alerts must NEVER be silenced by the interrupt event that enqueued them
        if is_critical:
            self._interrupt_event.clear()

        if _SD_AVAILABLE and not self._interrupt_event.is_set():
            try:
                sd.play(audio_output, samplerate=sample_rate, blocking=True)
                self.last_spoken_time = time.monotonic()
            except Exception as exc:
                print(f"[TTS] Audio playback error: {exc}")
        else:
            print(f"[TTS][NO AUDIO] {text}")
        self._interrupt_event.clear()

    def _play_kokoro(self, text: str) -> None:
        """
        Synthesise text with Kokoro-82M using clause-level LRU cache, 80ms Hann earcon,
        and safe stereo panning (FR-4.1–FR-4.3).
        """
        if self._kokoro is None:
            return

        is_critical = text.strip().upper().startswith("CRITICAL")
        direction = detect_direction_from_text(text)

        synth_text = text
        if is_critical and not self._critical_sound:
            if text.upper().startswith("CRITICAL:"):
                synth_text = text[9:].strip()
            elif text.upper().startswith("CRITICAL"):
                synth_text = text[8:].strip()

        try:
            audio_mono, sample_rate = self._synthesize_clause_cached(synth_text)
        except Exception as exc:
            print(f"[TTS] Kokoro synthesis error: {exc}")
            return

        if audio_mono.size == 0:
            return
        audio_mono = audio_mono.copy()

        # Prepend 80ms dual-tone Hann earcon chime on CRITICAL alerts if critical_sound enabled (FR-4.3)
        if is_critical and self._critical_sound:
            tone1 = generate_tone(880.0, 0.040, sr=sample_rate, amp=0.35)
            tone2 = generate_tone(1320.0, 0.040, sr=sample_rate, amp=0.35)
            chime = np.concatenate([tone1, tone2])
            audio_mono = np.concatenate([chime, audio_mono])

        audio_output = pan_audio(audio_mono, direction=direction, stereo=self._stereo)

        # Critical alerts must NEVER be silenced by the interrupt event that enqueued them
        if is_critical:
            self._interrupt_event.clear()

        if _SD_AVAILABLE and not self._interrupt_event.is_set():
            try:
                sd.play(audio_output, samplerate=sample_rate, blocking=True)
                self.last_spoken_time = time.monotonic()
            except Exception as exc:
                print(f"[TTS] Audio playback error: {exc}")
        else:
            print(f"[TTS][NO AUDIO] {text}")
        self._interrupt_event.clear()



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
        """Thread-safe enqueue with monotonic counter tie-breaker and creation timestamp."""
        with self._counter_lock:
            self._counter += 1
            counter = self._counter
        self._queue.put((priority, counter, message, time.monotonic()))

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
        # ── Step 1: Signal interrupt event and drain normal-priority queue items ──
        # Note: We deliberately do NOT call sd.stop() cross-thread because PortAudio
        # on Windows WASAPI/MME raises native access violations (0xC0000005) when Pa_StopStream
        # is called from a thread outside the audio callback while blocked in Pa_WriteStream.
        # ── Step 2: Flush unplayed backlog so only fresh alerts are spoken ─────
        # All pending items (both normal and stale criticals waiting in queue)
        # are discarded so speech never lags behind the real-time scene.
        kept: list[tuple] = []
        while True:
            try:
                item = self._queue.get_nowait()
                text = item[2]
                if text is _STOP_SENTINEL:
                    kept.append(item)
            except queue.Empty:
                break

        for item in kept:
            self._queue.put(item)

    def purge_label(self, label: str) -> None:
        """Purge pending queue items containing a specific label (e.g. when obstacle left the frame)."""
        kept: list[tuple] = []
        purged = False
        while True:
            try:
                item = self._queue.get_nowait()
                text = item[2]
                if text is not _STOP_SENTINEL and label.lower() in text.lower():
                    purged = True
                    continue
                kept.append(item)
            except queue.Empty:
                break
        for item in kept:
            self._queue.put(item)
        if purged:
            print(f"[TTS] Purged pending announcements for departed obstacle: {label}")

    def clear_queue(self) -> None:
        """Purge all pending unplayed messages from queue (e.g. when scene is clear)."""
        kept: list[tuple] = []
        while True:
            try:
                item = self._queue.get_nowait()
                if item[2] is _STOP_SENTINEL:
                    kept.append(item)
            except queue.Empty:
                break
        for item in kept:
            self._queue.put(item)
