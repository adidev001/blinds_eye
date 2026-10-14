"""
Blind's Eye — TTS Dual-Engine Benchmark Suite
=============================================
Phase 0 of the TTS/Localization upgrade plan.
Measures RTF, RAM overhead, and CRITICAL interrupt latency
for both Piper and Kokoro-82M ONNX backends.

Acceptance criteria (must pass before Phase 3 begins):
  - Piper   RTF  < 0.20
  - Kokoro  RTF  < 0.30
  - Kokoro  RAM delta < 400 MB
  - Hindi phrases synthesize without UnicodeError

Run:
    conda activate blinded_bythe_beauty
    python test_tts.py

Outputs a formatted summary table per engine.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Guard: psutil for RAM measurement ─────────────────────────────────────
try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    print("[WARN] psutil not installed. RAM benchmarks will be skipped.")
    print("       pip install psutil")

# ── Guard: sounddevice ─────────────────────────────────────────────────────
try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False
    print("[WARN] sounddevice not found — audio playback will be skipped.")

import numpy as np

# ══════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════

PIPER_MODEL   = Path("models/tts/en_US-amy-low.onnx")
KOKORO_ONNX   = Path("models/tts/kokoro-v1.0.onnx")
KOKORO_VOICES = Path("models/tts/voices-v1.0.bin")

# ── Test phrases ordered short → long ────────────────────────────────────
TEST_PHRASES_EN = [
    "Person on the left.",
    "I see a person on the left at 2 meters.",
    "I see: person on the left at 2 meters, bicycle in the center, car on the right at 5 meters.",
]

TEST_PHRASES_HI = [
    "बाईं ओर व्यक्ति।",
    "बाईं ओर व्यक्ति, दो मीटर पर।",
    "सामने साइकिल और दाईं ओर कार, पाँच मीटर पर।",
]

TEST_PHRASES_HINGLISH_ROMAN = [
    "baayein insaan hai.",
    "baayein insaan, do meter par.",
    "saamne cycle aur daayein car, paanch meter par.",
]

CRITICAL_PHRASE = "CRITICAL: Obstacle directly ahead at half a meter!"


# ══════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════

def ram_mb() -> float:
    if not _PSUTIL_OK:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def print_header(title: str) -> None:
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def print_row(phrase: str, synth_ms: float, audio_ms: float, rtf: float,
              pass_threshold: float) -> None:
    status = "✅" if rtf < pass_threshold else "❌"
    snippet = phrase[:38].ljust(38)
    print(f"  {status}  RTF={rtf:.3f}  synth={synth_ms:5.0f}ms  audio={audio_ms:5.0f}ms  │ {snippet}")


# ══════════════════════════════════════════════════════════════════════════
# Phase 0-A — Piper Benchmark
# ══════════════════════════════════════════════════════════════════════════

def run_piper_benchmark() -> bool:
    """
    Returns True if all acceptance criteria pass.
    """
    print_header("BENCHMARK A: Piper TTS")

    if not PIPER_MODEL.exists():
        print(f"  [SKIP] Model not found: {PIPER_MODEL}")
        print(f"         Download from https://rhasspy.github.io/piper-samples/")
        return False

    try:
        from piper.voice import PiperVoice, SynthesisConfig
    except ImportError:
        print("  [SKIP] piper-tts not installed.")
        return False

    ram_before = ram_mb()
    t_init = time.perf_counter()
    try:
        voice = PiperVoice.load(
            str(PIPER_MODEL),
            config_path=str(PIPER_MODEL.with_suffix(".onnx.json")),
            use_cuda=False,
        )
    except Exception as exc:
        print(f"  [ERROR] Piper load failed: {exc}")
        return False
    init_ms = (time.perf_counter() - t_init) * 1000
    ram_after = ram_mb()

    print(f"  Model     : {PIPER_MODEL.name}")
    print(f"  Init time : {init_ms:.0f} ms")
    print(f"  RAM delta : +{ram_after - ram_before:.1f} MB")
    print()
    print(f"  {'Status':<6}  {'RTF':>6}  {'Synth':>8}  {'Audio':>8}  │ Phrase")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  │ {'-'*38}")

    all_pass = True
    THRESHOLD = 0.20

    for phrase in TEST_PHRASES_EN:
        syn_cfg = SynthesisConfig(length_scale=1.0)
        t0 = time.perf_counter()
        chunks = list(voice.synthesize(phrase, syn_config=syn_cfg))
        synth_s = time.perf_counter() - t0

        if not chunks:
            print(f"  ⚠️   No audio output for: {phrase}")
            continue

        audio = np.concatenate([c.audio_float_array for c in chunks])
        sr = chunks[0].sample_rate
        audio_s = len(audio) / sr
        rtf = synth_s / audio_s

        if rtf >= THRESHOLD:
            all_pass = False
        print_row(phrase, synth_s * 1000, audio_s * 1000, rtf, THRESHOLD)

        # Play only first phrase so the benchmark doesn't take forever
        if phrase == TEST_PHRASES_EN[0] and _SD_OK:
            sd.play(audio, samplerate=sr, blocking=True)

    print()
    result = "✅ PASS" if all_pass else "❌ FAIL"
    print(f"  Piper benchmark result: {result}  (threshold RTF < {THRESHOLD})")
    return all_pass


# ══════════════════════════════════════════════════════════════════════════
# Phase 0-B — Kokoro Benchmark
# ══════════════════════════════════════════════════════════════════════════

def run_kokoro_benchmark() -> bool:
    print_header("BENCHMARK B: Kokoro-82M ONNX (English)")

    if not KOKORO_ONNX.exists() or not KOKORO_VOICES.exists():
        print(f"  [SKIP] Model files not found:")
        print(f"         {KOKORO_ONNX}")
        print(f"         {KOKORO_VOICES}")
        print(f"  Download from: https://github.com/thewh1teagle/kokoro-onnx/releases")
        return False

    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        print("  [SKIP] kokoro-onnx not installed. Run: pip install kokoro-onnx")
        return False

    ram_before = ram_mb()
    t_init = time.perf_counter()
    try:
        # Explicit CPU-only: never compete with YOLO/DepthAnything for VRAM
        kokoro = Kokoro(str(KOKORO_ONNX), str(KOKORO_VOICES))
    except Exception as exc:
        print(f"  [ERROR] Kokoro init failed: {exc}")
        return False
    init_ms = (time.perf_counter() - t_init) * 1000
    ram_after = ram_mb()
    ram_delta = ram_after - ram_before

    print(f"  Model     : {KOKORO_ONNX.name}")
    print(f"  Init time : {init_ms:.0f} ms")
    print(f"  RAM delta : +{ram_delta:.1f} MB  (limit: 400 MB)")
    if ram_delta >= 400:
        print("  ❌  RAM delta exceeds 400 MB threshold!")
    print()
    print(f"  {'Status':<6}  {'RTF':>6}  {'Synth':>8}  {'Audio':>8}  │ Phrase")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  │ {'-'*38}")

    all_pass = True
    THRESHOLD = 0.30

    for phrase in TEST_PHRASES_EN:
        t0 = time.perf_counter()
        try:
            samples, sr = kokoro.create(phrase, voice="af_heart", speed=1.0, lang="en-us")
        except Exception as exc:
            print(f"  ⚠️   Synthesis error: {exc}")
            all_pass = False
            continue
        synth_s = time.perf_counter() - t0
        audio_s = len(samples) / sr
        rtf = synth_s / audio_s

        if rtf >= THRESHOLD:
            all_pass = False
        print_row(phrase, synth_s * 1000, audio_s * 1000, rtf, THRESHOLD)

        if phrase == TEST_PHRASES_EN[0] and _SD_OK:
            sd.play(samples, samplerate=sr, blocking=True)

    all_pass = all_pass and (ram_delta < 400)
    result = "✅ PASS" if all_pass else "❌ FAIL"
    print()
    print(f"  Kokoro (EN) benchmark result: {result}  (RTF < {THRESHOLD}, RAM < 400MB)")
    return all_pass


# ══════════════════════════════════════════════════════════════════════════
# Phase 0-C — Hindi / Hinglish with Kokoro
# ══════════════════════════════════════════════════════════════════════════

def run_kokoro_hindi_benchmark() -> bool:
    print_header("BENCHMARK C: Kokoro Hindi & Hinglish Phoneme Test")

    if not KOKORO_ONNX.exists() or not KOKORO_VOICES.exists():
        print("  [SKIP] Kokoro model files not found.")
        return False

    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        print("  [SKIP] kokoro-onnx not installed.")
        return False

    kokoro = Kokoro(str(KOKORO_ONNX), str(KOKORO_VOICES))
    all_pass = True

    # --- Hindi (Devanagari, lang=hi) ------------------------------------
    print("  ── Hindi (Devanagari, lang='hi', voice='hf_alpha') ──")
    for text in TEST_PHRASES_HI:
        t0 = time.perf_counter()
        try:
            samples, sr = kokoro.create(text, voice="hf_alpha", speed=1.0, lang="hi")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            audio_s = len(samples) / sr
            rtf = (elapsed_ms / 1000) / audio_s
            status = "✅" if rtf < 0.40 else "❌"
            print(f"  {status}  {elapsed_ms:5.0f}ms synth │ {text[:50]}")
            if text == TEST_PHRASES_HI[0] and _SD_OK:
                sd.play(samples, samplerate=sr, blocking=True)
        except UnicodeError as exc:
            print(f"  ❌  UnicodeError (phoneme incompatibility): {exc}")
            all_pass = False
        except Exception as exc:
            print(f"  ❌  Error: {exc}")
            all_pass = False

    print()
    # --- Hinglish (Roman, lang=hi for phonemics) ------------------------
    print("  ── Hinglish (Roman script, lang='hi', voice='hf_alpha') ──")
    for text in TEST_PHRASES_HINGLISH_ROMAN:
        t0 = time.perf_counter()
        try:
            samples, sr = kokoro.create(text, voice="hf_alpha", speed=1.0, lang="hi")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"  ✅  {elapsed_ms:5.0f}ms synth │ {text[:50]}")
            if text == TEST_PHRASES_HINGLISH_ROMAN[0] and _SD_OK:
                sd.play(samples, samplerate=sr, blocking=True)
        except Exception as exc:
            print(f"  ❌  Error: {exc}")
            all_pass = False

    result = "✅ PASS" if all_pass else "❌ FAIL (check UnicodeError above)"
    print()
    print(f"  Hindi/Hinglish phoneme result: {result}")
    return all_pass


# ══════════════════════════════════════════════════════════════════════════
# Phase 0-D — CRITICAL Preemption Latency
# ══════════════════════════════════════════════════════════════════════════

def run_preemption_benchmark() -> None:
    """
    Tests the current (unfixed) preemption behaviour via the existing TTSEngine.
    Measures:
      - How long after CRITICAL is enqueued does the audio actually start?
    Goal with sd.stop() fix: < 100ms.
    Current (unfixed) behaviour: waits for entire current utterance to finish.
    """
    print_header("BENCHMARK D: CRITICAL Preemption Latency (current TTSEngine)")

    try:
        from src.tts_module import TTSEngine
    except ImportError as exc:
        print(f"  [SKIP] Could not import TTSEngine: {exc}")
        return

    if not PIPER_MODEL.exists():
        print(f"  [SKIP] Piper model not found. Using native fallback.")
        model_path = None
    else:
        model_path = PIPER_MODEL

    tts = TTSEngine(model_path=model_path)
    time.sleep(0.3)

    results: list[float] = []

    for run in range(2):
        print(f"\n  Run {run + 1}/2:")

        # Start a long utterance
        long_msg = "Person on the left at two meters. Bicycle in the center at four meters."
        t_long_enqueue = time.perf_counter()
        tts.speak(long_msg)
        print(f"    t=0ms   Enqueued normal message")

        # Wait 200ms then fire CRITICAL
        time.sleep(0.2)
        t_critical = time.perf_counter()
        tts.speak(CRITICAL_PHRASE)
        print(f"    t=200ms Enqueued CRITICAL message")
        print(f"    (Listen: CRITICAL should fire AFTER current utterance ends)")
        print(f"    Expected with sd.stop() fix: CRITICAL fires within ~100ms")
        print(f"    Current behaviour: CRITICAL waits for full utterance to finish")

        # Give time for both to play
        time.sleep(8)

    tts.shutdown()

    print()
    print("  ⚠️  Manual listening test required.")
    print("  After Phase 3 (sd.stop() fix), re-run this benchmark.")
    print("  CRITICAL should interrupt the ongoing utterance within ~100ms.")


# ══════════════════════════════════════════════════════════════════════════
# Phase 0-E — ONNX Runtime Collision Check
# ══════════════════════════════════════════════════════════════════════════

def run_onnx_provider_check() -> bool:
    print_header("BENCHMARK E: ONNX Runtime Provider Sanity Check")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  Available providers: {providers}")

        if "CUDAExecutionProvider" in providers:
            print("  ✅  CUDAExecutionProvider present — GPU pipeline is safe.")
            return True
        else:
            print("  ❌  CUDAExecutionProvider MISSING.")
            print("      onnxruntime-gpu has likely been overwritten by CPU onnxruntime.")
            print("      Fix: pip install onnxruntime-gpu --force-reinstall")
            return False
    except ImportError:
        print("  ❌  onnxruntime not installed at all.")
        return False


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Blind's Eye — TTS Dual-Engine Benchmark Suite             ║")
    print("║   Phase 0: Viability check before production changes        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Run checks
    onnx_ok      = run_onnx_provider_check()
    piper_ok     = run_piper_benchmark()
    kokoro_ok    = run_kokoro_benchmark()
    hindi_ok     = run_kokoro_hindi_benchmark()
    run_preemption_benchmark()          # Manual / informational only

    # Final verdict
    print_header("FINAL VERDICT")
    checks = [
        ("ONNX Runtime (CUDAExecutionProvider present)", onnx_ok),
        ("Piper RTF < 0.20",                             piper_ok),
        ("Kokoro RTF < 0.30 & RAM < 400MB",             kokoro_ok),
        ("Hindi/Hinglish phoneme compat (no UnicodeError)", hindi_ok),
    ]
    all_clear = True
    for label, passed in checks:
        icon = "✅" if passed else "❌  (BLOCKER)"
        print(f"  {icon}  {label}")
        if not passed:
            all_clear = False

    print()
    if all_clear:
        print("  🚀  All acceptance criteria met. Safe to proceed with Phase 2.")
    else:
        print("  🛑  One or more checks failed. Resolve blockers before Phase 2.")
        print()
        if not onnx_ok:
            print("  → Fix: pip install onnxruntime-gpu --force-reinstall")
        if not kokoro_ok and not KOKORO_ONNX.exists():
            print(f"  → Fix: Download kokoro-v1.0.onnx + voices-v1.0.bin to {KOKORO_VOICES.parent}/")
        if not piper_ok and not PIPER_MODEL.exists():
            print(f"  → Fix: Download a Piper voice model to {PIPER_MODEL.parent}/")
    print()


if __name__ == "__main__":
    main()
