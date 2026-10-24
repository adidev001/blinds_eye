# Blind's Eye — System Memory & Persistent Context
> **Location:** `d:\project\Blind-s-Eye-project\memory.md`  
> **Last Updated:** 2026-10-25  
> **Branch:** `withyolov26`  
> **Active Environment:** Conda `blinded_bythe_beauty` (`C:\Users\Devansh\miniconda3\envs\blinded_bythe_beauty\python.exe`)  
> **Hardware Profile:** Tier 2 Edge (Intel i7 13th Gen, NVIDIA GeForce RTX 4050 Laptop GPU 6.0 GB, CUDA 12.6, Windows 11)

---

## 1. System Architecture & Key Components

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              BLIND'S EYE v2.2 MASTER TOPOLOGY                            │
└──────────────────────────────────────────────────────────────────────────────────────────┘

 [CameraStream Daemon] ──(2s Socket Timeout / 1.5s Staleness Watchdog / Auto-Reconnect)───┐
                                                                                          ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │VisionPipeline (src/vision.py)                                                          │
 │  ├─ 1. Luminance Watchdog (Every 30 frames: P95 < 20 -> Low-Light Alert)                │
 │  ├─ 2. YOLO26 + ByteTrack (Constant FPS) ──▶ (track_id, bbox, label, cx, cy, scale s)  │
 │  ├─ 3. Depth Anything V2 (Async Thread, 350/518 res, --dynamic-power 2<->10 FPS)       │
 │  ├─ 4. Inner-Box 65th Percentile Fusion (30%x30% ROI) + Per-Track EMA (alpha=0.35)     │
 │  ├─ 5. Timestamp-Synced Kinematics (v_close, Looming r_scale, 2.2 m/s Deadband)        │
 │  ├─ 6. Proximity-Gated Vertical Tags (d <= 2.0m: overhead / ground)                    │
 │  └─ 7. Upper-Center Horizon Depth Curtain (Unlabelled Wall <= 1.0m + Flatness Guard)   │
 └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │
                                             ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │AnnouncementTracker & LocalizationFormatter (src/vision.py, src/localization.py)        │
 │  ├─ 3-Gate CRITICAL Preemption:                                                        │
 │  │    • Gate A: Center <= 1.2m                                                         │
 │  │    • Gate B: Any <= 1.8m + Box Looming >= 12%                                       │
 │  │    • Gate C: Vehicle <= 8.0m (or Any <= 4.0m) + v_close >= 3.5 m/s + TTC <= 2.5s    │
 │  ├─ Bounded Crowd Compression (Group >=2 same-class/dir into 2, 3, or "multiple")      │
 │  └─ Distance Quantizer (<1.5m: 0.1m steps | >=1.5m: 0.5m steps) -> Clause Tuples       │
 └──────────────────────┬───────────────────────────────────────────┬─────────────────────┘
                        │                                           │
         (Priority, Clauses, Direction)                   (If CRITICAL & --haptic-udp)
                        ▼                                           ▼
 ┌──────────────────────────────────────────────────┐   ┌─────────────────────────────────┐
 │TTSEngine (CPU-Isolated Piper / Kokoro-82M)       │   │UDP Haptic Emitter (Non-Blocking)│
 │  ├─ sd.stop() Sub-100ms Hardware Preemption      │   │  └─ SOCK_DGRAM (<0.05ms JSON)   │
 │  ├─ 80ms Hann-Windowed Pre-Beep (880->1320 Hz)   │   └─────────────────────────────────┘
 │  ├─ 256-Slot Clause-Level PCM LRU Cache (0ms hit)│   ┌─────────────────────────────────┐
 │  ├─ Safe Stereo Panner (100/35) or --no-stereo   │   │Flight Recorder (In-Memory Deque)│
 │  └─ Optional 15s 800Hz Bone-Conduction Heartbeat │   │  └─ Flush .jsonl & Table on Exit│
 └──────────────────────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 2. Implementation Progress & Master Checklist

### ✅ COMPLETED STEPS

- [x] **STEP 1: Overwrite `prd.md` with Ratified v2.2 Master PRD**
  - Updated `prd.md` with Tribunal Audit Ledger (T-01 to T-12), 6 Safety Guardrails, Functional Requirements (FR-1.1 through FR-5.4), CLI Table, and Verification Matrix.
  - Preserved no prescriptive steering directives rule.

- [x] **STEP 2: Linguistics, Distance Quantization & Crowd Compression (`src/localization.py`)**
  - `quantize_distance(depth: float) -> float`: $<1.5\text{m} \implies 0.1\text{m}$ precision; $\ge 1.5\text{m} \implies 0.5\text{m}$ precision (`FR-3.1`).
  - Added plural dictionaries: `COCO_PLURALS_EN`, `COCO_PLURALS_HI`, `COCO_PLURALS_HINGLISH`.
  - Added bounded crowd compression: `format_crowd_clause(label, direction, count, min_depth)` with count buckets `2`, `3`, `"multiple"` (`FR-3.3`).
  - Backward-compatible `format_detection` supporting `vertical_zone` (`overhead`, `ground`, `normal`) and `is_approaching` (`FR-3.2`).
  - Added `format_fast_critical` (`CRITICAL: ...`) and `format_system_alert` (`obstacle_ahead`, `camera_lost`, `camera_restored`, `low_light`).

- [x] **STEP 3: Non-Blocking Network Timeouts & Staleness Watchdog (`src/camera.py`)**
  - Socket timeouts: `cv2.CAP_PROP_OPEN_TIMEOUT_MSEC = 2000` and `cv2.CAP_PROP_READ_TIMEOUT_MSEC = 2000` (`FR-2.1`).
  - Non-blocking exponential backoff reconnect loop in reader thread: `0.5s`, `1.0s`, `2.0s`.
  - Monotonic timestamp tracking: `last_frame_time = time.monotonic()`.
  - Added `is_stale(threshold_sec=1.5) -> bool` (`FR-2.2`).

- [x] **STEP 4: Perception, ByteTrack, Kinematics & Depth Safety Curtain (`src/engine.py`, `src/vision.py`)**
  - Unified ONNX preference: loads `models/detection/yolo26n.onnx` if present, falling back to `.pt` (`FR-1.8`).
  - Dynamic depth resolution negotiation: round to nearest multiple of ViT stride 14; locks to static 518 if ONNX input is static (`FR-1.6`).
  - Depth-thread dynamic power governor: drops to 2 FPS when 0 objects & luminance change $<3\%$ for $>5\text{s}$ (`FR-1.7`).
  - Inner-box 65th percentile depth extraction: central $30\% \times 30\%$ slice rejects background hole artifacts (`FR-1.1`).
  - Proximity-gated vertical classification: evaluated only when $d \le 2.0\text{m}$ ($y_2 < 0.45H \implies \text{overhead}$, $y_1 > 0.72H \implies \text{ground}$) (`FR-1.4`).
  - Upper-center horizon depth curtain: $y \in [0.20H, 0.50H], x \in [0.35W, 0.65W]$ with $[0.80, 1.20]$ flatness ratio guard against floor pitch (`FR-1.5`).
  - Luminance watchdog: $P_{95} < 20.0$ for 3 consecutive checks emits low light warning (`FR-2.3`).
  - ByteTrack persistence via `model.track(persist=True, tracker="bytetrack.yaml")` with `track_id` fallback (`FR-1.2`).
  - Per-track depth smoothing with EMA ($\alpha = 0.35$).
  - Monotonic timestamp-synced closing velocity: $v_{\text{close}} = -(d_t - d_{t-1}) / \Delta t_{\text{depth}}$.
  - 2D optical looming expansion: $r_{\text{scale}} = (s_t - s_{t-2}) / \max(s_{t-2}, 1.0)$.
  - 2.2 m/s ego-motion deadband: `is_approaching` requires $v_{\text{close}} \ge 2.2\text{ m/s}$ AND $r_{\text{scale}} \ge 0.15$.
  - 3-Gate `CRITICAL:` Preemption with 3.0s per-track cooldown:
    - Gate A: `center` and $d \le 1.2\text{m}$.
    - Gate B: $d \le 1.8\text{m}$ and $r_{\text{scale}} \ge 0.12$ and $d_{\text{now}} < d_{\text{prev}}$.
    - Gate C: Vehicles with $d \le 8.0\text{m}$ (or any with $d \le 4.0\text{m}$) + $v_{\text{close}} \ge 3.5\text{ m/s}$ + $\text{TTC} \le 2.5\text{s}$ + $r_{\text{scale}} \ge 0.20$.
  - Bounded crowd compression in batched announcements (`FR-3.3`).

- [x] **STEP 5: Clause-Level LRU Cache, Safe Stereo Panning & Hann Earcons (`src/tts_module.py`)**
  - Clause-level PCM LRU cache with `@functools.lru_cache(maxsize=256)` returning defensive `.copy()` buffers (`FR-4.1`).
  - Stereo / mono auto-negotiation: queries device channels; falls back to mono if $<2$ channels (`FR-4.2`).
  - Safe 100/35 stereo panning gains: Left `[1.00, 0.35]`, Center `[0.85, 0.85]`, Right `[0.35, 1.00]`.
  - Hann-windowed earcons: `generate_tone` with 5ms raised-cosine taper to exact `0.0` boundary amplitude (`FR-4.3`).
  - 80ms dual-tone pre-beep (`40ms @ 880Hz` + `40ms @ 1320Hz`) panned to hazard direction for `CRITICAL:` alerts.
  - Bone-conduction heartbeat: 15ms @ 800Hz after 15s silence, gated by `camera.is_stale()`.
  - Sub-100ms hardware preemption: `sd.stop()` called inside `_flush_normal_queue()`.

- [x] **STEP 6: Telemetry, Auto-Context, UDP Haptics & Main CLI Wiring (`src/telemetry.py`, `src/main.py`)**
  - Created `src/telemetry.py` containing:
    - `AutoContextManager`: 15s rolling window, switches between indoor/outdoor when $\ge 5$ anchor detections with 0 opposing detections (`FR-5.2`).
    - `UdpHapticEmitter`: non-blocking `SOCK_DGRAM` with `setblocking(False)` wrapped in `try/except (BlockingIOError, OSError)` (`FR-5.3`).
    - `TelemetryRecorder`: in-memory `collections.deque(maxlen=5000)`, zero disk writes in loop, flushes JSONL on shutdown and prints Markdown summary table (`FR-5.4`).
  - Updated `src/main.py` with all v2.2 CLI flags:
    - `--mode [all|indoor|outdoor|auto]`
    - `--stereo` / `--no-stereo`
    - `--depth-res` (default 350)
    - `--dynamic-power`
    - `--headless` (auto-detected on Linux if `DISPLAY` unset)
    - `--heartbeat`
    - `--haptic-udp HOST:PORT`
    - `--telemetry-log PATH`
  - Integrated staleness notifications (`camera_lost` / `camera_restored`), auto context switching, UDP haptics on critical events, and shutdown telemetry report.

---

### ⏳ UNCHECKED ITEMS (TO DO LATER)

- [x] **STEP 7: Verification Suite (`tests/test_v22_tribunal_upgrades.py`)** ✅ **COMPLETE**
  - Implemented all 7 deterministic PRD §7 tests + 4 integration scenarios (A, B, D) + localization/quantization edge cases.
  - **100/100 tests pass** in 2.3 seconds (0 failures, 0 errors).
  - Also confirmed existing suite: **14/14 tests** passing in `test_tts_localization.py`.
  - **Grand total: 114 tests, 114 pass, 0 fail.**
  - Test classes implemented:
    1. `TestInnerBoxPercentile` — FR-1.1: inner-box 65th percentile rejects background holes
    2. `TestCrossCorridorPersistence` — FR-1.2: ByteTrack identity across corridor transitions
    3. `TestEgoDeadbandAndGateC` — FR-1.3: 2.2 m/s deadband, Gate C high-speed vehicle
    4. `TestBoundedCrowdCompression` — FR-3.3: bucket 2/3/multiple, min-depth, multilingual
    5. `TestStereoPanningAndMono` — FR-4.2: 100/35 gains, shape, device mock, immutability
    6. `TestHannEarconBoundaries` — FR-4.3: first/last sample = 0.0, duration, amplitude
    7. `TestNonBlockingUDPAndTelemetry` — FR-5.3/5.4: loopback UDP, JSONL flush, deque bound
    8. `TestIntegrationDetectionPipeline` — Scenario A: detection pipeline, Gate A, no steering
    9. `TestIntegrationCameraLoss` — Scenario B: staleness, camera alert formats, low-light throttle
    10. `TestIntegrationShutdownTelemetry` — Scenario D: worker shutdown, telemetry flush
    11. `TestQuantizationAndLocalization` — quantize boundaries, vertical tags, direction thirds
    12. `TestStationaryHysteresisAndQueuePurge` — FIX-013: Gate B stationary distance hysteresis, departed obstacle label purging
    - **Total Passing Tests:** 117/117 across `test_v22_tribunal_upgrades.py` (103/103) and `test_tts_localization.py` (14/14).

- [x] **In-Window Interactive Clickable HUD Control Dock (`FEAT-014`)**
  - Added clickable control dock at the bottom of the video window via `cv2.setMouseCallback`:
    - Mode Section: `[ALL]`, `[INDOOR]`, `[OUTDOOR]`, `[AUTO:ON/OFF]`
    - Perception Section: `[DEPTH:ON/OFF]`
    - Audio Section: `[CRIT SND:ON/OFF]`, `[STEREO:ON/OFF]`, `[LANG:EN/HI/HINGLISH]`
    - System Section: `[EXIT]`

- [ ] **Physical Hardware & Integration Testing**
  - Test real USB/CSI webcam input and verify reconnect loop on disconnection.
  - Test physical stereo earbud playback and verify directional panning audibility.
  - Test UDP haptics receiver on ESP32 / Arduino / loopback listener.
  - Test `--mode auto` transitions with simulated video stream.

---

## 3. Critical Environmental Rules & Quirks

1. **Python Interpreter Path:**
   - Always run with the direct executable path in PowerShell:
     `& "C:\Users\Devansh\miniconda3\envs\blinded_bythe_beauty\python.exe" <args>`
   - Avoid `conda run` on Windows due to carriage return escaping errors (`\r\n`).

2. **CUDA & cuDNN 9 on Windows:**
   - PyTorch packages `cudnn64_9.dll` inside `torch/lib`.
   - `src/engine.py` calls `os.add_dll_directory(torch_lib)` and prepends it to `PATH` before importing ONNX Runtime. Do not remove this hook.
   - Package `onnxruntime-gpu==1.23.2` must not be replaced with CPU `onnxruntime`.

3. **TTS Models:**
   - Kokoro models: `models/tts/kokoro-v1.0.onnx` and `models/tts/voices-v1.0.bin`.
   - Piper models: `models/tts/en_US-amy-low.onnx` + `.json`.

4. **Safety Directives:**
   - **Never** output steering directives (`"Move Left"`, `"Turn Right"`). All outputs must be spatial awareness descriptors only (`"[vertical_zone] [approaching] [label] at [depth] meters [direction]"`).
