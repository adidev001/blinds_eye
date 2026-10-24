# Blind's Eye — System Modes & Operational Profiles

> **Comprehensive Reference Guide**  
> This document categorizes and defines all operational modes in the Blind's Eye assistive spatial awareness system. Modes are organized group-wise across context filtering, perception pipelines, hardware execution tiers, audio synthesis, camera ingestion, temporal tracking, and power management.

---

## Table of Contents
1. [Group 1: Context-Aware Environmental Filtering Modes](#group-1-context-aware-environmental-filtering-modes)
2. [Group 2: Perception & Vision Pipeline Operating Modes](#group-2-perception--vision-pipeline-operating-modes)
3. [Group 3: Hardware Execution & Compute Tier Modes](#group-3-hardware-execution--compute-tier-modes)
4. [Group 4: Speech Synthesis (TTS) & Localization Modes](#group-4-speech-synthesis-tts--localization-modes)
5. [Group 5: Camera & Video Ingestion Modes](#group-5-camera--video-ingestion-modes)
6. [Group 6: Temporal Debouncing & Announcement Tracking Modes](#group-6-temporal-debouncing--announcement-tracking-modes)
7. [Group 7: Power, Thermal & Wearable Battery Profiles](#group-7-power-thermal--wearable-battery-profiles)
8. [Summary CLI Flag Reference](#summary-cli-flag-reference)

---

## Group 1: Context-Aware Environmental Filtering Modes

Context filtering controls which object classes are permitted into the spatial tracking and announcement pipeline. In real-time assistive systems for visually impaired users, acoustic clutter is dangerous. These modes purge irrelevant objects and hallucinated detections *before* they enter the spatial debouncer.

### 1.1 Indoor Mode (`--mode indoor`)
* **Target Environment:** Homes, offices, classrooms, stores, indoor corridors.
* **Filter Strategy:** Whitelists 48 household and indoor-specific classes from COCO-80 while rejecting outdoor vehicle and transit classes.
* **Permitted Classes:**
  * **Furniture & Fixtures:** `chair`, `couch`, `bed`, `dining table`, `toilet`, `sink`, `potted plant`, `clock`, `vase`
  * **Electronics & Appliances:** `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `refrigerator`, `hair drier`
  * **Daily Items & Tableware:** `book`, `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `scissors`, `toothbrush`, `backpack`, `umbrella`, `handbag`, `tie`, `teddy bear`
  * **Food Items:** `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`
  * **Living Entities:** `person`, `cat`, `dog`
* **Ignored Hazards / Noise:** Filters out transient false positives such as a chair misidentified as a `truck` or `train`.
* **Example Launch:**
  ```bash
  python -m src.main --mode indoor --confidence 0.5
  ```

### 1.2 Outdoor Mode (`--mode outdoor`)
* **Target Environment:** Streets, sidewalks, public squares, parks, transport stops.
* **Filter Strategy:** Whitelists 28 transit, pedestrian, and urban obstacle classes; suppresses small domestic clutter (e.g., cutlery, cups, appliances).
* **Permitted Classes:**
  * **Vehicles & Transit:** `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`
  * **Street Infrastructure:** `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`
  * **Living Entities:** `person`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`
  * **Pedestrian Accessories:** `backpack`, `umbrella`, `handbag`
* **Example Launch:**
  ```bash
  python -m src.main --mode outdoor --confidence 0.6
  ```

### 1.3 All-Class Pass-Through Mode (`--mode all`) — Default
* **Target Environment:** Unconstrained exploration, laboratory testing, model evaluation.
* **Filter Strategy:** No class filter applied (`_allowed_classes = None`). Every detected object among all 80 COCO classes is evaluated.
* **Example Launch:**
  ```bash
  python -m src.main --mode all
  ```

### 1.4 Auto Context-Switching Mode (`--mode auto`)
* **Target Environment:** Seamless transitions between indoor structures and outdoor sidewalks.
* **Mechanism (`AutoContextManager`):** Maintains a 15-second sliding window of anchor detections.
  * Transitions `indoor -> outdoor` upon detecting $\ge 5$ outdoor anchor instances (e.g. `car`, `bus`, `traffic light`) with 0 indoor anchor detections.
  * Transitions `outdoor -> indoor` upon detecting $\ge 5$ indoor anchor instances (e.g. `chair`, `couch`, `bed`, `refrigerator`) with 0 outdoor anchor detections.
  * Hysteresis prevents ping-pong switching in doorways and lobbies.
* **Example Launch:**
  ```bash
  python -m src.main --mode auto
  ```

### 1.5 Planned & Experimental Context Modes (Roadmap)
* **Crosswalk / Street Crossing Mode:** Focuses strictly on pedestrian lights, zebra markings, curb drops, and oncoming vehicles with dynamic velocity thresholds.
* **Transit & Commute Mode:** Prioritizes platform edges, train/bus doors, empty seats, and ticketing gates.
* **Grocery / Shelf Scanning Mode:** Near-field hand-held detection for products, price tags, and shelf items.
* **Social / Interaction Mode:** Isolates human faces and upper bodies to announce person presence and proximity for conversation.

---

## Group 2: Perception & Vision Pipeline Operating Modes

These modes determine which neural networks execute during visual inference, balancing depth fidelity against execution framerate.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PERCEPTION PIPELINE MODES                            │
├───────────────────────────────────┬────────────────────────────────────┤
│ Full Multimodal Mode (Default)    │ No-Depth / Direction-Only Mode     │
│ [YOLO26] + [Depth Anything V2]    │ [YOLO26 Only]                      │
│ Output: Label + Azimuth + Distance│ Output: Label + Azimuth (L/C/R)    │
│ Speed: 10–20 FPS (Edge GPU)       │ Speed: 40–60 FPS (Ultra-fast)      │
└───────────────────────────────────┴────────────────────────────────────┘
```

### 2.1 Full Multimodal Perception Mode (Default)
* **Components:** End-to-end YOLO26 (detection + bounding box) + Depth Anything V2 Small (monocular depth estimation).
* **Output:** Object identification, horizontal direction (`left`, `center`, `right`), and metric distance (e.g., *"कुर्सी सामने 1.5 मीटर"* or *"Chair center 1.5 meters"*).
* **Inference Execution:**
  * YOLO runs on RGB input frames.
  * Depth Anything V2 computes an inverse relative disparity map.
  * The depth map is sampled over a 5×5 patch centered at the object's centroid (`cx`, `cy`).
  * Relative depth is transformed into rough metric distance using the calibration parameter `--depth-scale`.
* **Example Launch:**
  ```bash
  python -m src.main --depth-scale 3.0
  ```

### 2.2 Direction-Only / No-Depth Mode (`--no-depth`)
* **Components:** YOLO26 only. Depth Anything V2 is completely bypassed.
* **Output:** Object identification and azimuth sector only (e.g., *"Person on the left"*).
* **Benefits:**
  * Frees ~95 MB of GPU VRAM / system memory.
  * Increases framerate to 40–60+ FPS on edge devices.
  * Eliminates thermal load on fanless wearable hardware.
* **Example Launch:**
  ```bash
  python -m src.main --no-depth
  ```

### 2.3 Ground-Calibrated Metric Depth Mode (`--depth-scale <float>`)
* **Mechanism:** Adjusts the hyperbolic conversion coefficient:
  $$\text{depth\_meters} = \frac{\text{depth\_scale}}{\text{relative\_disp} + \epsilon}$$
* **Default:** `3.0` (calibrated for standard camera height and waist/chest mounting).
* **High Calibration (Outdoors / Distant):** `--depth-scale 4.5`
* **Close Range (Indoors / Desktops):** `--depth-scale 2.0`

---

## Group 3: Hardware Execution & Compute Tier Modes

Managed by `src/engine.py:HardwareProbe` and `ModelLoader`, the system dynamically inspects the host processor, VRAM, and ONNX Runtime providers to select optimal execution tiers.

| Tier Profile | Target Architecture | YOLO Model | Depth Model | Execution Provider |
|---|---|---|---|---|
| **Tier 1: Server** | Datacenter GPU (A100, V100, RTX 3090/4090, VRAM $\ge$ 20 GB) | `yolo26m.pt` (Medium) | `depth_anything_v2_large.engine` (TensorRT) | `TensorrtExecutionProvider`, `CUDAExecutionProvider` |
| **Tier 2: Edge GPU** | Mobile / Laptop GPU (RTX 4050, Jetson Orin Nano, 4–8 GB VRAM) | `yolo26n.pt` (Nano) | `depth_anything_v2_small.onnx` | `CUDAExecutionProvider` |
| **Tier 3: CPU Only** | Raspberry Pi 4/5, Intel NUC, Non-GPU PCs | `yolo26n.pt` (Nano) | `depth_anything_v2_small.onnx` | `OpenVINOExecutionProvider`, `CPUExecutionProvider` |

### Key Architectural Rule: GPU Isolation
Vision models (YOLO and Depth Anything V2) claim GPU acceleration. The TTS subsystem is strictly isolated to CPU threads to prevent CUDA Out-Of-Memory (OOM) faults and avoid context-switching latency.

---

## Group 4: Speech Synthesis (TTS) & Localization Modes

Controlled via `src/tts_module.py` and `src/localization.py`. Runs asynchronously on a dedicated worker thread with a thread-safe `PriorityQueue`.

### 4.1 Synthesis Engine Modes (`--tts-engine`)

#### Kokoro-82M High-Fidelity Mode (`--tts-engine kokoro`)
* **Engine:** 82-million parameter neural TTS running via ONNX Runtime on CPU.
* **Model Files:** `models/tts/kokoro-v1.0.onnx` + `models/tts/voices-v1.0.bin` (or `voices/hf_alpha.bin`).
* **Characteristics:** Studio-quality natural human intonation; zero robotic artifacts; 24 kHz sample rate output directly streamed via `sounddevice`.
* **Isolation:** Explicitly loads with `CPUExecutionProvider` so GPU VRAM remains 100% dedicated to vision models.
* **Example Launch:**
  ```bash
  python -m src.main --tts-engine kokoro --lang hi
  ```

#### Piper Fast Neural Mode (`--tts-engine piper`) — Default
* **Engine:** Lightweight VITS-based neural TTS.
* **Model Files:** `models/tts/en_US-amy-low.onnx` (English) or `hi_IN-rohan-medium.onnx` (Hindi).
* **Characteristics:** Sub-50ms synthesis latency, low CPU utilization, streaming PCM playback.
* **Example Launch:**
  ```bash
  python -m src.main --tts-engine piper --lang en
  ```

#### Native OS Fallback Mode (Automatic)
* **Engines:** Windows SAPI (`win32com`), Linux `espeak-ng`, or cross-platform `pyttsx3`.
* **Trigger:** Activates automatically if neural models or dependencies are missing. Guarantees the user is never left without audio feedback.

---

### 4.2 Language & Localization Modes (`--lang`)

| Language Flag | Mode | Localization Logic | Target Voice |
|---|---|---|---|
| `--lang en` | **English (Default)** | Standard spatial alerts (e.g., *"Chair center 1.5 meters"*). | Piper: `amy-low` / Kokoro: `af_heart` |
| `--lang hi` | **Devanagari Hindi** | Native Devanagari spatial phrasing with grammatical number (e.g., *"कुर्सी सामने 1.5 मीटर"*). | Piper: `rohan-medium` / Kokoro: `hf_alpha` |
| `--lang hinglish`| **Roman Hinglish** | Phonetic Romanized Hindi phrasing (e.g., *"kursi center 1.5 meter"*). | Kokoro: `hf_alpha` (Hindi phonemizer) |

### 4.3 Speech Rate Modulation (`--speech-rate`)
* **Standard:** `1.0` (Default normal conversational pacing).
* **Fast Assistive Mode:** `1.2` – `1.5` (Preferred by experienced screen-reader users).
* **Slow / Instructional Mode:** `0.8` (Clear pronunciation for noisy outdoor environments).

---

## Group 5: Camera & Video Ingestion Modes

Configured via `src/camera.py:CameraStream` with a dedicated capture thread that continuously discards stale frames to guarantee zero buffer lag.

### 5.1 Local USB / Webcam Mode (`--camera 0`) — Default
* **Source:** Hardware device index (`0`, `1`, etc.).
* **Protocol:** Local OS capture (DirectShow on Windows, V4L2 on Linux).
* **Characteristics:** Lowest possible latency, direct hardware bus bandwidth.
* **Example Launch:**
  ```bash
  python -m src.main --camera 0
  ```

### 5.2 IP Camera / Chest-Mounted Wireless Mode (`--camera <URL>`)
* **Source:** RTSP or HTTP MJPEG stream (e.g., phone app running IP Webcam strapped to chest).
* **Protocol:** TCP/UDP network socket stream.
* **Characteristics:** Decouples the wearable camera sensor from the heavy compute unit.
* **Example Launch:**
  ```bash
  python -m src.main --camera "http://192.168.1.10:8080/video"
  ```

### 5.3 Offline Video Benchmark Mode (`--camera path/to/file.mp4`)
* **Source:** Local pre-recorded video file.
* **Usage:** Algorithmic verification, repeatability testing, and benchmark regression runs.
* **Example Launch:**
  ```bash
  python -m src.main --camera "test_samples/hallway_walk.mp4"
  ```

---

## Group 6: Temporal Debouncing & Announcement Tracking Modes

Managed by `src/vision.py:AnnouncementTracker`. Prevents the system from chattering constantly or announcing flickering false positives.

```
                    TEMPORAL DEBOUNCING LIFECYCLE
                    
   Object First Appears
          │
          ▼
   ┌──────────────────────────────────────────────┐
   │ Streak Accumulation Phase                    │
   │ Must remain visible for >= --min-duration    │
   │ (Default: 0.5s)                              │
   └──────┬───────────────────────────────────────┘
          │ (If continuous)
          ▼
   ┌──────────────────────────────────────────────┐
   │ ANNOUNCEMENT DISPATCH                        │
   │ Speech is synthesized and queued             │
   └──────┬───────────────────────────────────────┘
          │
          ▼
   ┌──────────────────────────────────────────────┐
   │ Cooldown Throttle Phase                      │
   │ Suppressed from re-announcement for          │
   │ >= --speak-interval (Default: 2.0s)          │
   └──────┬───────────────────────────────────────┘
          │ (If object disappears)
          ▼
   ┌──────────────────────────────────────────────┐
   │ Absence Reset Phase                          │
   │ Memory cleared after >= --absence-reset      │
   │ (Default: 1.5s)                              │
   └──────────────────────────────────────────────┘
```

### 6.1 Fast Reaction Profile (Crowded / Rapid Navigation)
* Configures minimal latency before an object is verified and announced:
  ```bash
  python -m src.main --min-duration 0.2 --speak-interval 1.0 --absence-reset 1.0
  ```

### 6.2 Calm / Low-Chatter Profile (Stationary / Indoor Reading)
* High hysteresis to keep speech calm and infrequent:
  ```bash
  python -m src.main --min-duration 0.8 --speak-interval 4.0 --absence-reset 3.0
  ```

### 6.3 Standard Balanced Profile (Default)
* `--min-duration 0.5` · `--speak-interval 2.0` · `--absence-reset 1.5`

---

## Group 7: Power, Thermal & Wearable Battery Profiles

Blind's Eye supports battery-aware throttling for edge and wearable deployments (e.g., NVIDIA Jetson Orin Nano, laptop on battery).

### 7.1 High-Performance Tethered Profile
* **Goal:** Maximum spatial awareness and responsiveness when AC power is available.
* **Configuration:**
  ```bash
  python -m src.main --max-fps 15.0 --confidence 0.45 --frame-width 640
  ```

### 7.2 Thermally Capped Edge Profile (Default)
* **Goal:** Sustained execution without thermal throttling on laptops or Jetson enclosures.
* **Configuration:**
  ```bash
  python -m src.main --max-fps 10.0 --confidence 0.50 --frame-width 640
  ```

### 7.3 Ultra-Low Power / Battery Conservation Profile
* **Goal:** Extends battery life on wearable power banks; disables heavy depth tensor computation and reduces resolution.
* **Configuration:**
  ```bash
  python -m src.main --no-depth --max-fps 6.0 --frame-width 480 --confidence 0.60
  ```

---

## Summary CLI Flag Reference

| Flag | Parameter / Type | Default | Group | Description |
|---|---|---|---|---|
| `--mode` | `indoor \| outdoor \| all \| auto` | `all` | Group 1 | Context-aware environmental class filter (or auto-switching) |
| `--no-depth` | flag | `False` | Group 2 | Disables monocular depth; runs in azimuth-only mode |
| `--depth-scale` | `float` | `3.0` | Group 2 | Metric depth scaling coefficient |
| `--depth-res` | `int` | `350` | Group 2 | Depth map inference resolution (350 / 518) |
| `--dynamic-power` | flag | `False` | Group 7 | Throttles depth thread to 2 FPS when idle/static |
| `--tts-engine` | `piper \| kokoro` | `piper` | Group 4 | Neural speech synthesis backend |
| `--lang` | `en \| hi \| hinglish` | `en` | Group 4 | Language and localization formatting |
| `--tts-model` | `path` | `None` (auto) | Group 4 | Manual path override for Piper voice model |
| `--speech-rate` | `float` | `1.0` | Group 4 | Speech speed multiplier (e.g. 1.2 for faster) |
| `--stereo` / `--no-stereo` | flag | Auto-query | Group 4 | Enforces stereo 100/35 spatialization or mono downmix |
| `--critical-sound` / `--no-critical-sound` | flag | `True` | Group 4 | Toggles 80ms Hann earcon chimes and urgent prefix (press 'c' live) |
| `--heartbeat` | flag | `False` | Group 4 | Emits 15ms @ 800Hz earcon after 15s camera silence |
| `--camera` | `string` | `"0"` | Group 5 | Camera index (`0`) or IP stream URL |
| `--frame-width` | `int` | `640` | Group 5 | Video frame downsampling width in pixels |
| `--min-duration` | `float` (seconds) | `0.5` | Group 6 | Continuous visibility time before first announcement |
| `--speak-interval`| `float` (seconds) | `2.0` | Group 6 | Minimum cooldown time between repeated announcements |
| `--absence-reset` | `float` (seconds) | `1.5` | Group 6 | Disappearance time before spatial memory clears |
| `--max-fps` | `float` | `10.0` | Group 7 | Framerate cap for thermal and pipeline regulation |
| `--confidence` | `float` | `0.5` | Group 2 | Minimum YOLO object detection confidence threshold |
| `--headless` | flag | Auto (Linux) | Group 3 | Suppresses OpenCV GUI display windows |
| `--haptic-udp` | `HOST:PORT` | `None` | Group 4 | Dispatches non-blocking UDP JSON datagrams on critical hazard |
| `--telemetry-log`| `path` | `None` | Group 7 | Flushes in-memory telemetry deque to JSONL file on exit |

---

## Group 8: Interactive GUI & In-Window Control Dock Modes

For sighted developers, testing assistants, and caretakers observing the user's video feed, Blind's Eye provides an interactive, in-window clickable control dock rendered directly onto the OpenCV display frame (`cv2.setMouseCallback`).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        INTERACTIVE HUD CONTROL DOCK                    │
├────────────────────────────────────────────────────────────────────────┤
│ Mode Section:       [ALL]       [INDOOR]       [OUTDOOR]       [AUTO]  │
│ Perception Section: [DEPTH: ON/OFF]                                    │
│ Audio Section:      [CRIT SND: ON/OFF]   [STEREO: ON/OFF]   [LANG: EN] │
│ System Section:     [EXIT]                                             │
└────────────────────────────────────────────────────────────────────────┘
```

### 8.1 Mode Selection Dock
* **`[ALL]` Button:** Clears class whitelisting and evaluates all 80 COCO objects.
* **`[INDOOR]` Button:** Restricts spatial announcements to 48 indoor classes.
* **`[OUTDOOR]` Button:** Restricts spatial announcements to 28 transit and street classes.
* **`[AUTO]` Button:** Toggles dynamic hysteresis context switching (`AutoContextManager`). Active state is highlighted in emerald green.

### 8.2 Perception & Depth Toggle
* **`[DEPTH: ON/OFF]` Button:** Toggles Depth Anything V2 monocular depth inference on and off dynamically in real time without stopping the process.

### 8.3 Audio & Localization Dock
* **`[CRIT SND: ON/OFF]` Button:** Toggles the dual-tone 80ms Hann earcon chimes and urgent prefix on and off (also accessible via keyboard shortcut `'c'`).
* **`[STEREO: ON/OFF]` Button:** Toggles 100/35 safe stereo spatial panning vs. center mono downmixing.
* **`[LANG: EN / HI / HINGLISH]` Button:** Cycles announcement translation and speech formatting dynamically among English, Hindi, and Hinglish.

### 8.4 System Exit
* **`[EXIT]` Button:** Safely halts the camera stream, releases models, and terminates execution.

---

*Document created and maintained by Antigravity AI Agent · Blind's Eye Project*
