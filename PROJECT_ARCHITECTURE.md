# Blind's Eye: System Architecture & Feature Overview

This document provides a comprehensive end-to-end breakdown of the "Blind's Eye" project. It is designed to explain the system's logic, pipeline architecture, and current optimizations to facilitate brainstorming for new features, modes, or algorithmic improvements.

---

## 1. Project Mission
Blind's Eye is an assistive wearable/portable device software stack designed for visually impaired users. It processes live video (via webcam or IP camera), detects objects, estimates their real-world distances, and announces them via an offline, ultra-low-latency Text-to-Speech (TTS) engine.

## 2. Core Technologies
- **Object Detection:** YOLOv11 (via `ultralytics`)
- **Monocular Depth Estimation:** Depth Anything V2 Small (via `onnxruntime-gpu`)
- **Speech Synthesis:** Piper TTS (Offline, ONNX-based, native PCM audio streaming)
- **Vision/Matrix Math:** OpenCV (`cv2`) and NumPy
- **Parallelization:** Python `threading` and `queue.PriorityQueue`

---

## 3. End-to-End Pipeline Architecture

The system is designed around **asynchronous decoupling**. AI models, camera hardware, and audio hardware operate at vastly different speeds. To prevent audio lag or "ghost frames", the architecture isolates these components into non-blocking threads.

### A. The Camera Stream (`src/camera.py`)
- **Problem:** `cv2.VideoCapture.read()` buffers frames. If inference runs at 10 FPS but the camera produces 30 FPS, the buffer fills up, causing the AI to analyze frames from 3 seconds ago (a critical safety hazard).
- **Solution:** A background daemon thread continuously reads and discards frames as fast as possible, storing only the absolute newest frame in a variable. The main AI loop grabs this variable, guaranteeing zero-latency real-time video.

### B. The Vision Engine (`src/vision.py` & `src/engine.py`)
- **Hardware Routing:** `src/engine.py` dynamically detects NVIDIA GPUs. It loads YOLO via PyTorch/CUDA and forces `onnxruntime-gpu` to use `CUDAExecutionProvider` or `TensorrtExecutionProvider` for the Depth model.
- **Inference Pipeline (`process_frame`):**
  1. **Parallel Execution:** YOLO is incredibly fast (~15ms) but Depth Anything V2 is heavier (~150-200ms). To maximize FPS, Depth runs in a background thread while YOLO runs concurrently on the main thread.
  2. **Centroid Extraction:** YOLO outputs bounding boxes. The center `(cx, cy)` of the box is calculated.
  3. **Depth Fusion:** The `(cx, cy)` pixel coordinates are mapped onto the 2D matrix produced by Depth Anything V2. The relative disparity at that exact pixel is extracted.
  4. **Metric Conversion:** The disparity is converted to real-world meters using a user-calibrated formula: `Distance = Scale / Disparity`.

### C. The Tracker & Debouncer (`AnnouncementTracker`)
- **Problem:** YOLO occasionally hallucinates objects for a single frame. Previously, this flooded the TTS queue with false positives.
- **Temporal Debouncing (`--min-duration`):** An object must be continuously detected across multiple frames for `0.5s` (configurable) before it is considered "real".
- **Rate Limiting (`--speak-interval`):** To prevent spam, the system groups simultaneous detections (e.g., "I see a person, a chair, and a desk") and only speaks once every `2.0s`.
- **Absence Reset (`--absence-reset`):** If an object disappears for more than `1.5s`, its streak is broken. If it reappears, it is treated as a new object and re-announced.

### D. The TTS Engine (`src/tts_module.py`)
- **Problem:** Legacy TTS engines like `pyttsx3` block the main Python thread while speaking, freezing the video feed.
- **Solution:** Piper TTS runs in an isolated daemon thread, listening to a `PriorityQueue`.
- **Preemption & Safety:** If a "CRITICAL" message is pushed to the queue (e.g., a fast-approaching obstacle), the TTS thread instantly flushes all pending normal messages and announces the critical warning first.

---

## 4. Current Features & CLI Controls

| Feature | Implementation Logic |
|---|---|
| **Context Modes** | `--mode {indoor, outdoor, all}` filters YOLO classes *before* they hit the Tracker. This prevents outdoor objects (like a car hallucination) from triggering inside, keeping audio clean. |
| **FPS Capping** | `--max-fps 8` ensures the GPU doesn't overheat. It uses a smart `time.sleep()` at the end of the main loop, calculating exactly how many milliseconds are left to hit the target framerate. |
| **No-Depth Mode** | `--no-depth` completely disables the ONNX depth model, allowing YOLO to run at 40-60 FPS for users who only care about object direction (Left/Center/Right). |
| **IP Camera Support** | Accepts HTTP/RTSP URLs to run inference on a powerful laptop while the user wears a lightweight smartphone camera strapped to their chest. |

---

## 5. Potential Areas for Improvement (Brainstorming Fuel)

When chatting with Claude, here are some structural limitations and potential expansion areas to consider:

1. **Depth Accuracy Limits:** Monocular depth (Depth Anything V2) is relative. The `--depth-scale` calibration works for objects at ground-level, but fails for objects high up or very close to the lens. *Could we integrate temporal depth smoothing or a lightweight stereo-vision approach?*
2. **New Context Modes:** We currently have "indoor" and "outdoor". *Could we add a "Crosswalk Mode", "Grocery Store Mode", or "Face/Person Tracking Mode"?*
3. **Critical Danger Alerts:** The TTS queue supports "CRITICAL" priority, but the Vision logic currently doesn't flag anything as critical. *How do we mathematically determine if an object is dangerous (e.g., fast approaching bounding box expansion)?*
4. **Spatial Audio:** Currently, direction is announced as text ("person on the left"). *Could we use `sounddevice` stereo panning to actually play the audio in the user's left or right ear?*
5. **Tracking ID Assignment:** The current debouncer tracks objects by combining `Label_Direction` (e.g., `person_left`). If a person walks from left to right, the streak breaks. *Should we implement a lightweight tracker like DeepSORT or ByteTrack?*
