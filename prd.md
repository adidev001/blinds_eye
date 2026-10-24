# Product Requirements Document (PRD): Blind's Eye v2.2

| Document Metadata | Details |
| --- | --- |
| **Product Name** | Blind's Eye — Edge-Native Spatial Awareness & Kinematic Collision Avoidance |
| **Document Version** | `2.2.0-MASTER-PRD` (Ratified by Tribunal Sessions I & II) |
| **Target Branch** | `withyolov26` |
| **Core AI Stack** | YOLO26 (NMS-Free + ByteTrack) + Depth Anything V2 (ONNX) + Piper / Kokoro-82M TTS |
| **Supported Languages** | English (`en`), Hindi (`hi`), Hinglish (`hinglish`) |
| **Target File Path** | `prd.md` (and exportable to `Blind_Eye_PRD_v2.2.docx`) |

---

## 1. Executive Summary & Product Vision

**Blind's Eye v2.2** is a 100% offline, safety-critical spatial navigation and collision-avoidance system engineered for blind and visually impaired users. Operating on wearable cameras, smart glasses, or smartphone IP streams, it fuses **NMS-free YOLO26 object detection and ByteTrack multi-object tracking** with **Depth Anything V2 monocular depth estimation**, filters cognitive clutter via a **kinematic state machine**, and outputs directional acoustic warnings through a **CPU-isolated Dual-Engine TTS pipeline (Piper + Kokoro-82M)** and **non-blocking UDP haptics**.

This **v2.2 Master Specification** consolidates all architectural upgrades ratified across **Tribunal Session I** (configurable stereo/mono hardware negotiation, ego-motion deadband filtering, high-speed vehicle Time-to-Collision preemption, and unified ONNX runtime execution) and **Tribunal Session II** (bounded crowd compression, Hann-windowed psychoacoustic earcons, depth-thread dynamic power scaling, fire-and-forget UDP haptics, and lockless in-memory telemetry recording).

---

## 2. Tribunal Audit Ledger (v2.0 $\rightarrow$ v2.2 Evolution)

| ID | Legacy v2.0 Limitation | Red-Team Failure Mode Identified | Ratified v2.2 Engineering Solution |
| --- | --- | --- | --- |
| **T-01** | Single-pixel centroid depth (`D[cy, cx]`) | Center pixel on hollow objects (chairs, cycles, legs apart) reads background wall 10m away. | **Inner-Box 65th Percentile Sampling** over a tight $30\% \times 30\%$ core ROI (`FR-1.1`). |
| **T-02** | Corridor-keyed tracking (`f"{label}_{dir}"`) | Crossing from `left` to `center` resets the **0.5s** debounce timer, silencing moving hazards. | **ByteTrack (`track_id`) Persistence** + per-track Depth EMA smoothing (`FR-1.2`). |
| **T-03** | Asynchronous depth differentiation & walking ego-motion | Differentiating stale depth frames creates velocity spikes; walking at **1.4 m/s** makes parked cars look moving. | **Timestamp-Synced Depth Velocity** + **2.2 m/s Ego-Motion Deadband** + **2D Optical Scale Looming** (`FR-1.3`). |
| **T-04** | Late vehicle warnings ($d \le 1.8\text{m}$) | A vehicle at **10 m/s** gives only **0.18s** reaction time at **1.8m**. | **Gate C High-Speed Vehicle Alert** up to **8.0m** when $v_{\text{close}} \ge 3.5\text{ m/s}$ and $\text{TTC} \le 2.5\text{s}$ (`FR-1.3`). |
| **T-05** | Unlabelled obstacle blind spot | COCO-80 lacks walls, closed doors, and pillars; user walks into blank walls in silence. | **Upper-Center Horizon Depth Curtain** ($\le 1.0\text{m}$) with vertical flatness verification (`FR-1.5`). |
| **T-06** | Main-loop FPS throttling breaks Kalman filters | Dropping YOLO FPS to **3 FPS** distorts ByteTrack Kalman covariances and creates an **833ms** blind spot. | **Depth-Only Dynamic Power Governor**: YOLO stays at constant FPS; only background DA-V2 scales down to **2 FPS** (`FR-1.7`). |
| **T-07** | Dual CUDA allocator overhead | PyTorch (`.pt`) + ONNX Runtime (`.onnx`) allocate two separate GPU memory pools. | **Unified ONNX Runtime Preference**: Auto-load `yolo26n.onnx` inside Ultralytics `.track()` when present (`FR-1.8`). |
| **T-08** | Silent IP camera hangs & covered lenses | Wi-Fi drop blocks `cap.read()` for 30s; dark/covered lens outputs 0 detections (implying clear path). | **2s Socket Timeouts, 1.5s Staleness Watchdog, & 95th-Percentile Darkness Check** (`FR-2.1–2.3`). |
| **T-09** | Crowd audio spam & LRU cache thrashing | 6 pedestrians trigger 6 repetitive clauses; raw count strings explode LRU cache permutations. | **Bounded Crowd Compression** bucketing counts strictly into `2`, `3`, or `multiple` ($\ge 4$) (`FR-3.3`). |
| **T-10** | PortAudio mono crash & single-earbud blindness | 2-channel stereo crashes 1-channel speakers; 100/0 panning mutes alerts for single-earbud users. | **`--stereo` / `--no-stereo` Flags**, hardware channel auto-query, and **100/35 Safe Stereo Panning** (`FR-4.2`). |
| **T-11** | Speaker pops & bone-conduction roll-off | Unwindowed sine waves click loudly; **220 Hz** tones vanish on bone-conduction headsets. | **Hann-Windowed Earcons** (`880->1320 Hz` pre-beep + `800 Hz` bone-conduction heartbeat tick) (`FR-4.3`). |
| **T-12** | Blocking HTTP webhooks & disk I/O jitter | TCP retries stall critical alerts; runtime file flushing causes **15–40ms** frame drops. | **Non-Blocking UDP Datagram Haptics** (`FR-5.3`) & **In-Memory `deque` Telemetry Flush-on-Exit** (`FR-5.4`). |

---

## 3. Binding Safety Guardrails (Strictly Prohibited Anti-Patterns)

1. **NO Prescriptive Movement Directives (`"Move Left"` / `"Move Right"`):** Monocular vision without ground-plane LiDAR cannot verify open manholes, stairs, or curbs in adjacent corridors. Report obstacle positions only.
2. **NO Verbal `"Fixed"` or `"Stationary"` Labels:** Announcing *"stationary chair"* adds cognitive noise. Silence on motion status implies stationary; only explicitly prepend `"approaching"` when kinematic thresholds are exceeded.
3. **NO Depth-Only Velocity Triggers:** Every kinematic `approaching` or `CRITICAL:` state must be co-verified by 2D pixel bounding-box scale expansion ($r_{\text{scale}}$) to reject global monocular depth normalization shifts.
4. **NO HSV Traffic Light Color Inference:** Sensor saturation and turn-lane ambiguity make 640p HSV color thresholding unsafe for street crossing. Announce `"traffic light"` as a landmark only.
5. **NO Laplacian Blur Occlusion Checks:** Blank painted walls have near-zero Laplacian variance and trigger false occlusion warnings. Use 95th-percentile luminance checks only.
6. **NO Runtime Disk Writes or Blocking Network Calls:** Neither TTS synthesis, telemetry logging, nor haptic broadcasting may touch the filesystem or open blocking TCP sockets during the active perception loop.

---

## 4. Functional Requirements (FR)

### 4.1 Module 1: Perception, Kinematics & Spatial Fusion (`src/vision.py`, `src/engine.py`)

#### FR-1.1: Inner-Box 65th Percentile Depth Extraction

For any detected bounding box $(x_1, y_1, x_2, y_2)$ with width $w = x_2 - x_1$ and height $h = y_2 - y_1$, extract the inner $30\% \times 30\%$ core region of the disparity map $D$:


$$rx_1 = \lfloor x_1 + 0.35w \rfloor, \quad rx_2 = \lceil x_2 - 0.35w \rceil, \quad ry_1 = \lfloor y_1 + 0.35h \rfloor, \quad ry_2 = \lceil y_2 - 0.35h \rceil$$


Compute raw disparity $d_{\text{raw}}$ as the **65th percentile** of $D[ry_1:ry_2, rx_1:rx_2]$ (falling back to $D[c_y, c_x]$ if empty), and metric distance:


$$\text{Distance}_{\text{raw}} = \frac{\text{depth\_scale}}{\max(d_{\text{raw}}, 10^{-6})}$$

#### FR-1.2: ByteTrack Persistence & Per-Track Depth EMA

* Run YOLO26 via `model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)` (falling back to `model.predict()` if tracking is unavailable).
* Key all `AnnouncementTracker` state records by `track_id`.
* Smooth metric distance per track using an Exponential Moving Average ($\alpha = 0.35$):

$$d_{\text{smooth}, t} = 0.35 \cdot \text{Distance}_{\text{raw}, t} + 0.65 \cdot d_{\text{smooth}, t-1}$$


* **Cross-Corridor Transition:** If a tracked object moves across corridors (e.g., `left` $\rightarrow$ `center`), preserve its `first_seen` timestamp. If an already-spoken object enters the `center` corridor at $d_{\text{smooth}, t} \le 2.5\text{m}$, reset its `spoken` flag once so the user is alerted that the obstacle is now directly ahead.

#### FR-1.3: Timestamp-Synced Kinematics, Ego-Motion Deadband & 3-Gate `CRITICAL:` Preemption

* **Depth Timestamping:** Each disparity map computed by the background depth thread must carry a monotonic `depth_timestamp` and `depth_frame_id`.
* **Closing Velocity ($v_{\text{close}}$):** Updated **only** when `depth_frame_id` increments ($\Delta t_{\text{depth}} = t_{\text{depth}, k} - t_{\text{depth}, k-1} > 0$):

$$v_{\text{close}} = -\frac{d_{\text{smooth}, k} - d_{\text{smooth}, k-1}}{\Delta t_{\text{depth}}}$$


* **2D Optical Scale Looming ($r_{\text{scale}}$):** For bounding box scale $s_t = \sqrt{w_t \cdot h_t}$, compute expansion over two depth updates:

$$r_{\text{scale}} = \frac{s_k - s_{k-2}}{\max(s_{k-2}, 1.0)}$$


* **Motion State Classification (Ego-Motion Deadband = 2.2 m/s):**
* **Stationary (`is_approaching = False`):** $v_{\text{close}} < 2.2\text{ m/s}$ OR $r_{\text{scale}} < 0.15$. Announced without any motion adjective.
* **Moderately Approaching (`is_approaching = True`):** $v_{\text{close}} \ge 2.2\text{ m/s}$ AND $r_{\text{scale}} \ge 0.15$. Prepends localized approaching modifier.


* **3-Gate Automatic `CRITICAL:` Preemption (bypasses `--speak-interval`, requires $\ge 0.2\text{s}$ track age, 3.0s per-track cooldown):**
* **Gate A (Static Immediate Hazard):** Corridor is `center` AND $d_{\text{smooth}} \le 1.2\text{m}$.
* **Gate B (Close Expanding Hazard):** Any corridor, $d_{\text{smooth}} \le 1.8\text{m}$ AND $r_{\text{scale}} \ge 0.12$ AND $d_{\text{smooth}, k} < d_{\text{smooth}, k-1}$.
* **Gate C (High-Speed Approaching Vehicle):** For vehicle classes (`car`, `motorcycle`, `bus`, `truck`, `bicycle`, `train`) at $d_{\text{smooth}} \le 8.0\text{m}$ (or any class at $d_{\text{smooth}} \le 4.0\text{m}$):

$$v_{\text{close}} \ge 3.5\text{ m/s} \quad \text{AND} \quad \text{TTC} = \frac{d_{\text{smooth}}}{v_{\text{close}}} \le 2.5\text{s} \quad \text{AND} \quad r_{\text{scale}} \ge 0.20$$





#### FR-1.4: Proximity-Gated Vertical Hazard Classification

Evaluated **strictly when $d_{\text{smooth}} \le 2.0\text{m}$** (to prevent horizon perspective distortion):

* **`overhead`:** $y_2 < 0.45H$ (entire bounding box is in the upper 45% of frame height).
* **`ground`:** $y_1 > 0.72H$ (entire bounding box is in the lower 28% of frame height).
* **`normal`:** All other cases or whenever $d_{\text{smooth}} > 2.0\text{m}$.

#### FR-1.5: Upper-Center Horizon Depth Safety Curtain

When YOLO detects zero objects in the `center` corridor:

1. Slice the upper-center horizon band of $D$: $y \in [0.20H, 0.50H]$, $x \in [0.35W, 0.65W]$.
2. Compute $d_{\text{curtain}} = \frac{\text{depth\_scale}}{\max(P_{80}(\text{ROI}_{\text{curtain}}), 10^{-6})}$.
3. **Vertical Flatness Guard:** Compare median disparity of the upper half $[0.20H, 0.35H]$ and lower half $[0.35H, 0.50H]$. If their ratio lies within $[0.80, 1.20]$ (confirming a vertical wall/door rather than a pitched floor) AND $d_{\text{curtain}} \le 1.0\text{m}$ continuously for **0.6s**, emit a localized unlabelled obstacle alert (throttled to once every **4.0s**).

#### FR-1.6: Dynamic ONNX Depth Resolution Negotiation

Inspect `depth_session.get_inputs()[0].shape` at startup. If spatial dimensions are dynamic (`str` or `None`), default to **350×350** (`--depth-res 350`, enforced as a multiple of ViT patch size **14**). If static (`518`), automatically lock to **518×518**.

#### FR-1.7: Depth-Thread Dynamic Power Governor (`--dynamic-power`)

* The main YOLO26 + ByteTrack loop **must never throttle its FPS** so Kalman filter covariances remain accurate.
* When `--dynamic-power` is passed, the background Depth Anything V2 worker scales its target sleep interval:
* **Idle (`2.0 FPS`):** When 0 objects are tracked AND mean absolute frame luminance diff $< 3.0\%$ for **5.0s**.
* **Active (`8.0–10.0 FPS`):** Wakes immediately via `threading.Event` as soon as YOLO detects any object or frame motion exceeds $3.0\%$.



#### FR-1.8: Unified ONNX Runtime Model Loading

In `src/engine.py`, if `models/detection/yolo26n.onnx` exists alongside `yolo26n.pt`, prefer loading `yolo26n.onnx` via Ultralytics so YOLO26 and Depth Anything V2 share `onnxruntime-gpu` providers, falling back to `yolo26n.pt` if absent.

---

### 4.2 Module 2: Camera Resilience & Fault Diagnostics (`src/camera.py`, `src/vision.py`)

#### FR-2.1: Non-Blocking Network Timeouts & Auto-Reconnect

Configure `cv2.VideoCapture` with `CAP_PROP_OPEN_TIMEOUT_MSEC = 2000` and `CAP_PROP_READ_TIMEOUT_MSEC = 2000`. If frame reading fails in the daemon thread, execute an exponential backoff reconnect loop (`0.5s`, `1.0s`, `2.0s`) without blocking the main thread.

#### FR-2.2: Frame Staleness Watchdog

Record `self.last_frame_time = time.monotonic()` on every successful frame decode and expose `camera.is_stale(threshold_sec: float = 1.5) -> bool`. If stale, pause inference and emit `CRITICAL: Camera signal lost` (throttled every **5.0s**), announcing `"Camera restored"` upon recovery.

#### FR-2.3: 95th-Percentile Luminance Darkness / Occlusion Check

Every 30 frames in `VisionPipeline`, compute `np.percentile(frame, 95) < 20.0`. If `True` for 3 consecutive checks, trigger a localized `"Camera blocked or low light"` warning (throttled every **10.0s**).

---

### 4.3 Module 3: Linguistics, Localization & Crowd Compression (`src/localization.py`)

#### FR-3.1: Dual-Regime Distance Quantization

$$\text{quantize}(d) = \begin{cases} \text{round}(d, 1) & \text{if } d < 1.5\text{m} \\ \frac{\text{round}(2d)}{2} & \text{if } d \ge 1.5\text{m} \end{cases}$$

#### FR-3.2: Multi-Language Template Matrix (`en`, `hi`, `hinglish`)

| Template Key | English (`en`) | Hindi (`hi`) | Hinglish (`hinglish`) |
| --- | --- | --- | --- |
| **Overhead Prefix** | `"overhead {label}"` | `"सिर के पास {label}"` | `"sir ke paas {label}"` |
| **Ground Prefix** | `"ground-level {label}"` | `"नीचे जमीन पर {label}"` | `"neeche zameen par {label}"` |
| **Approaching Prefix** | `"approaching {label}"` | `"आता हुआ {label}"` | `"aata hua {label}"` |
| **Fast Critical (`Gate C`)** | `"CRITICAL: Fast approaching {label} {dir} at {d} meters"` | `"CRITICAL: तेज़ आता हुआ {label} {dir}, {d} मीटर पर"` | `"CRITICAL: Tez aata hua {label} {dir}, {d} meter par"` |
| **Crowd Clause (`count=2,3`)** | `"{count} {plural_label} {dir}, closest at {d} meters"` | `"{dir} {count} {plural_label}, सबसे पास {d} मीटर पर"` | `"{dir} {count} {plural_label}, sabse paas {d} meter par"` |
| **Crowd Clause (`count>=4`)** | `"multiple {plural_label} {dir}, closest at {d} meters"` | `"{dir} कई {plural_label}, सबसे पास {d} मीटर पर"` | `"{dir} kai {plural_label}, sabse paas {d} meter par"` |
| **Depth Curtain Alert** | `"obstacle ahead at {d} meters"` | `"सामने रुकावट, {d} मीटर पर"` | `"saamne rukawat, {d} meter par"` |
| **Camera Lost (`CRITICAL`)** | `"CRITICAL: Camera signal lost"` | `"CRITICAL: कैमरा सिग्नल बंद है"` | `"CRITICAL: Camera signal band hai"` |
| **Low Light / Covered** | `"Camera blocked or low light"` | `"कैमरा ढका है या अंधेरा है"` | `"Camera dhaka hai ya andhera hai"` |

#### FR-3.3: Bounded Crowd Compression & Curated Pluralization

When $\ge 2$ unspoken objects of the same `label` occupy the same `direction` corridor in a single announcement batch:

* Collapse them into a single crowd clause using $d = \min(d_{\text{smooth}})$ of the group.
* Bucket counts strictly into **`2`**, **`3`**, or **`multiple`** ($\ge 4$) so clause-level LRU cache permutations stay bounded.
* Use a curated plural dictionary (`person` $\rightarrow$ `people` / `लोग` / `log`, `car` $\rightarrow$ `cars` / `कारें` / `gaadiyan`, `motorcycle` $\rightarrow$ `motorcycles` / `बाइक` / `bikes`, `chair` $\rightarrow$ `chairs` / `कुर्सियाँ` / `kursiyan`, `dog` $\rightarrow$ `dogs` / `कुत्ते` / `kutte`, etc.) with safe fallback.

---

### 4.4 Module 4: Psychoacoustics, Stereo Panning & Cached TTS (`src/tts_module.py`)

#### FR-4.1: Clause-Level In-Memory PCM LRU Cache

Wrap raw mono waveform synthesis (`_synthesize_clause(text: str) -> np.ndarray`) in `@functools.lru_cache(maxsize=256)`. Each cached array is a read-only `float32` 1D NumPy buffer ($\le 36\text{ MB}$ worst-case RAM for 256 entries), giving **0.0ms** synthesis latency on repeat phrases.

#### FR-4.2: `--stereo` / `--no-stereo` & PortAudio Channel Auto-Negotiation

* Add CLI flags `--stereo` (default: `True`) and `--no-stereo` (`False`).
* At `TTSEngine` startup, inspect `sd.query_devices(kind="output")["max_output_channels"]`. If `< 2` or `--no-stereo` is set, force 1-channel mono output `(N,)` to prevent `PortAudioError`.
* When stereo is active, transform each clause's `(N,)` mono array into a `(N, 2)` stereo array using safe panning gains that preserve single-earbud audibility:

$$\text{Gain}_{\text{Left}} = \begin{bmatrix} 1.00 & 0.35 \end{bmatrix}, \quad \text{Gain}_{\text{Center}} = \begin{bmatrix} 0.85 & 0.85 \end{bmatrix}, \quad \text{Gain}_{\text{Right}} = \begin{bmatrix} 0.35 & 1.00 \end{bmatrix}$$



#### FR-4.3: Hann-Windowed Earcons & Bone-Conduction Heartbeat (`--heartbeat`)

* **Hann Envelope Helper:** Every synthesized sine tone of length $M$ samples must apply a raised-cosine taper (`np.hanning(2 * taper_samples)`) to the first and last **5ms** of samples so waveform boundaries start and end at exact `0.0` amplitude (zero digital speaker clicks).
* **Critical Pre-Beep (80ms):** Automatically prepend an 80ms dual-tone chime (`40ms @ 880 Hz` followed by `40ms @ 1320 Hz` at `0.35` amplitude, stereo-panned to the hazard direction) ahead of any `CRITICAL:` utterance.
* **System-Alive Heartbeat (`--heartbeat`):** When enabled, if no audio has been played for **15.0s** and `!camera.is_stale()`, play a **15ms**, **800 Hz** Hann-windowed micro-tick at `0.15` amplitude (engineered above the **350 Hz** bone-conduction roll-off threshold).

---

### 4.5 Module 5: Runtime Orchestration, UDP Haptics & Telemetry (`src/main.py`)

#### FR-5.1: Headless Execution (`--headless`)

Add `--headless` CLI flag (auto-enabled on Linux/Docker when `DISPLAY` is unset). Skips all OpenCV bounding-box drawing, depth colormap rendering, and `cv2.imshow()` calls.

#### FR-5.2: Opt-In Auto Context Hysteresis (`--mode [all|indoor|outdoor|auto]`)

Default remains `all`. When `--mode auto` is selected, track high-confidence ($\ge 0.65$) detections over a 15-second rolling window and switch between `indoor` and `outdoor` only when $\ge 5$ target-mode anchor detections occur with **0** opposing-mode detections.

#### FR-5.3: Fire-and-Forget Non-Blocking UDP Haptic Emitter (`--haptic-udp HOST:PORT`)

* Initialize a single persistent `socket.socket(socket.AF_INET, socket.SOCK_DGRAM)` with `sock.setblocking(False)`.
* Whenever a `CRITICAL:` event fires, dispatch a compact UTF-8 JSON packet:
```json
{"event": "CRITICAL", "label": "car", "dir": "center", "dist": 4.5, "ttc": 1.2, "ts": 1728940000.1}

```


* Wrap `sock.sendto()` in `try ... except (BlockingIOError, OSError): pass` to guarantee $< 0.05\text{ ms}$ execution with zero network stall risk.

#### FR-5.4: Lockless In-Memory Flight Recorder (`--telemetry-log PATH`)

* During active runtime, append telemetry dictionaries (timestamp, YOLO ms, Depth ms, TTS ms, cache hit boolean, active track count, announcement text, critical gate triggered) to an in-memory `collections.deque(maxlen=5000)`.
* **Zero disk writes occur inside the `while` loop.**
* Inside the `finally:` shutdown block, flush the `deque` to `PATH` (`.jsonl`) and print a formatted Markdown **Session Telemetry Summary** (p50/p95 frame latency, TTS cache hit ratio, total announcements, critical preemptions).

---

## 5. System Architecture & Concurrency Topology (v2.2)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              BLIND'S EYE v2.2 MASTER TOPOLOGY                            │
└──────────────────────────────────────────────────────────────────────────────────────────┘

 [CameraStream Daemon] ──(2s Socket Timeout / 1.5s Staleness Watchdog / Auto-Reconnect)───┐
                                                                                          ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │VisionPipeline (src/vision.py)                                                          │
 │  ├─ 1. Luminance Watchdog (Every 30 frames: P95 < 20 -> Low-Light / Covered Warning)   │
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

## 6. Complete CLI Reference Specification (v2.2)

| CLI Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--mode` | `all|indoor|outdoor|auto` | `all` | Context filter (`auto` uses 15s 5-anchor hysteresis) |
| `--camera` | `str` | `"0"` | Webcam index, RTSP/HTTP URL, or `.mp4` file path |
| `--tts-engine` | `piper|kokoro` | `piper` | Primary offline TTS engine (auto-fallbacks enabled) |
| `--lang` | `en|hi|hinglish` | `en` | Announcement localization language |
| `--stereo` / `--no-stereo` | `flag` | `--stereo` | Enable safe 100/35 stereo panning or force 1-channel mono |
| `--depth-res` | `int` | `350` | Dynamic DA-V2 input resolution (multiple of 14; auto-locks if static) |
| `--dynamic-power` | `flag` | `False` | Scale background depth thread to 2 FPS during static scenes |
| `--headless` | `flag` | `auto` | Disable OpenCV GUI rendering (`cv2.imshow`) |
| `--heartbeat` | `flag` | `False` | Emit 15ms 800Hz bone-conduction tick after 15s of silence |
| `--haptic-udp` | `str` | `None` | Fire-and-forget UDP target `HOST:PORT` for critical haptics |
| `--telemetry-log` | `str` | `None` | Path to write session `.jsonl` metrics on exit |

---

## 7. Verification Matrix & Unit Test Suite Specification

Add `tests/test_v22_tribunal_upgrades.py` to verify all v2.2 requirements deterministically without hardware webcams:

1. **`test_inner_box_percentile_rejects_background_hole`:** Synthetic disparity map with `0.0` at `(cx, cy)` and `0.5` across surrounding core pixels returns `0.5` via 65th percentile.
2. **`test_cross_corridor_track_persistence`:** Object with `track_id=42` moving from `left` (`0.3s`) to `center` (`0.3s`) triggers announcement at `0.6s` without debounce reset.
3. **`test_ego_motion_deadband_and_gate_c_vehicle`:**
* Stationary car approached at walking speed ($v_{\text{close}} = 1.4\text{ m/s}$) is **not** marked `approaching`.
* Fast car at $d = 6.5\text{m}$ with $v_{\text{close}} = 5.0\text{ m/s}$ ($\text{TTC} = 1.3\text{s}$) and $r_{\text{scale}} = 0.25$ immediately triggers `Gate C` `CRITICAL:` alert.


4. **`test_bounded_crowd_compression`:** Passing 5 unspoken `person` detections in `center` at `[2.1m, 2.4m, 3.0m, 3.2m, 4.0m]` produces a single `"multiple people ahead, closest at 2.0 meters"` clause.
5. **`test_stereo_panning_and_mono_fallback`:** Verifies `(N, 2)` output with `[1.0, 0.35]` left gains when `--stereo` is active, and `(N,)` 1D output when `--no-stereo` is set.
6. **`test_hann_earcon_zero_boundary_crossings`:** Asserts first and last samples of the 80ms critical pre-beep and 15ms 800Hz heartbeat are `0.0` (`< 1e-6`) to guarantee zero speaker click.
7. **`test_nonblocking_udp_and_deque_telemetry`:** Verifies UDP dispatch to an unreachable port completes in `< 1.0ms` without raising an exception, and telemetry deque flushes valid JSONL on shutdown.

---

## 8. One-Command Script to Generate `prd.md` and `Blind_Eye_PRD_v2.2.docx`

Run the following Python script in your repo root to automatically save the Markdown PRD to `prdv2.md` and generate a formatted Microsoft Word document (`Blind_Eye_PRD_v2.2.docx`):

```python
"""
Run:
    pip install python-docx
    python generate_prd_doc.py
"""
from pathlib import Path

def build_word_doc(md_path: Path, docx_path: Path) -> None:
    try:
        from docx import Document
    except ImportError:
        print("[INFO] Install python-docx (`pip install python-docx`) to export .docx")
        return

    doc = Document()
    doc.add_heading("Blind's Eye v2.2 — Master Product Requirements Document", 0)

    for line in md_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        if stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=1)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=2)
        elif stripped.startswith("#### "):
            doc.add_heading(stripped[5:], level=3)
        elif stripped.startswith(("- ", "* ")):
            doc.add_paragraph(stripped[2:], style="List Bullet")
        else:
            doc.add_paragraph(stripped)

    doc.save(str(docx_path))
    print(f"[SUCCESS] Generated Word Document: {docx_path.resolve()}")

if __name__ == "__main__":
    md_file = Path("prd.md")
    if md_file.exists():
        build_word_doc(md_file, Path("Blind_Eye_PRD_v2.2.docx"))
    else:
        print("Save the PRD Markdown above to prd.md first, then run this script.")

```
