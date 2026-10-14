# 🦯 Blind's Eye

> **AI-powered real-time spatial awareness for the visually impaired**

Blind's Eye is a wearable assistive device that uses **YOLO26** for object detection and **Depth Anything V2** for monocular depth estimation to help blind users navigate safely. It announces detected obstacles, their direction (left / center / right), and distance in meters through a fully **offline, low-latency Text-to-Speech** engine (Piper TTS + Kokoro-82M ONNX) supporting English, Hindi, and Hinglish.

> 📖 **Comprehensive Engineering Dossier & Production Blueprint:** For an exhaustive architectural deep dive, hardware execution profiles, sub-100ms mid-utterance preemption mechanics, and edge production roadmap, see [**PROJECT_DOSSIER.md**](PROJECT_DOSSIER.md).

---

## ✨ Features

| Feature | Description |
|---|---|
| **Real-Time Detection** | YOLO26 identifies 80+ object classes — NMS-Free, 43% faster CPU inference |
| **3D Depth Estimation** | Depth Anything V2 (ONNX) calculates real-world distance in meters |
| **Offline TTS** | Piper TTS with priority queue — critical warnings preempt normal speech |
| **Context Modes** | Indoor / Outdoor / All filtering to reduce irrelevant announcements |
| **Temporal Debouncing** | Objects must be visible for 0.5s before being announced (no ghost audio) |
| **IP Camera Support** | Stream from network cameras via HTTP/RTSP URLs |
| **GPU Accelerated** | CUDA + TensorRT support for NVIDIA GPUs |
| **Docker Ready** | Containerized deployment with NVIDIA Container Toolkit |

---

## 📁 Project Structure

```
Blind-s-Eye-project/
├── src/
│   ├── main.py          # CLI entry point & main loop
│   ├── vision.py        # YOLO + Depth pipeline & AnnouncementTracker
│   ├── engine.py        # Hardware detection & model loading
│   ├── camera.py        # Threaded camera stream (anti-lag buffer)
│   └── tts_module.py    # Piper TTS engine (async, priority queue)
├── models/
│   ├── detection/       # YOLO weights (auto-downloaded)
│   ├── depth/           # Depth Anything V2 ONNX weights
│   └── tts/             # Piper voice model (.onnx + .json)
├── deployment/
│   └── Dockerfile       # NVIDIA CUDA Docker image
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/adidev001/blinds_eye.git
cd blinds_eye
```

### 2. Create Conda Environment

```bash
conda create -n blinds_eye python=3.10 -y
conda activate blinds_eye
```

### 3. Install Dependencies

**For GPU (NVIDIA — Recommended):**
```bash
pip install -r requirements.txt
pip uninstall -y onnxruntime
pip install onnxruntime-gpu --force-reinstall
```

**For CPU only:**
```bash
pip install -r requirements.txt
```

> [!IMPORTANT]
> If you have an NVIDIA GPU, you **must** uninstall the CPU `onnxruntime` first, then install `onnxruntime-gpu`. Having both installed causes the CPU version to silently override the GPU version, dropping Depth AI performance from 60 FPS to 5 FPS.

### 4. Download Models

- **YOLO weights** — Auto-downloaded on first run by Ultralytics
- **Depth Anything V2** — Auto-downloaded on first run from Hugging Face (~95 MB)
- **Piper TTS voice** — Place a `.onnx` voice model in `models/tts/`. Download voices from [Piper Samples](https://rhasspy.github.io/piper-samples/)

### 5. Run

```bash
python -m src.main
```

Press **`q`** in the video window to quit.

---

## 🎮 Usage Modes

### Context-Aware Filtering

Blind's Eye supports three detection modes to reduce audio noise based on your environment:

```bash
# 🏠 Indoor Mode — furniture, appliances, household items
python -m src.main --mode indoor

# 🌳 Outdoor Mode — vehicles, traffic lights, animals, street objects
python -m src.main --mode outdoor

# 🌍 All Mode (default) — announce every detected object
python -m src.main --mode all
```

<details>
<summary><b>📋 Indoor Classes (47 objects)</b></summary>

`person`, `cat`, `dog`, `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`, `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush`, `backpack`, `umbrella`, `handbag`, `tie`, `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`

</details>

<details>
<summary><b>📋 Outdoor Classes (27 objects)</b></summary>

`person`, `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`, `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`, `backpack`, `umbrella`, `handbag`

</details>

---

## ⚙️ CLI Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--mode` | `indoor\|outdoor\|all` | `all` | Context-aware detection filter |
| `--camera` | `str` | `0` | Camera index (e.g. `0`) or IP camera URL |
| `--confidence` | `float` | `0.5` | Minimum YOLO detection confidence |
| `--depth-scale` | `float` | `3.0` | Calibration factor: `distance = scale / disparity` |
| `--no-depth` | flag | — | Disable depth estimation (direction-only mode) |
| `--max-fps` | `float` | `10.0` | Cap inference FPS to save GPU power |
| `--min-duration` | `float` | `0.5` | Seconds an object must be visible before announcing |
| `--speak-interval` | `float` | `2.0` | Minimum seconds between speech announcements |
| `--absence-reset` | `float` | `1.5` | Seconds before a disappeared object can be re-announced |
| `--speech-rate` | `float` | `1.0` | TTS speed multiplier (1.0 = normal, 1.3 = faster) |
| `--tts-model` | `str` | auto | Path to Piper `.onnx` voice model |
| `--frame-width` | `int` | `640` | Resize camera frames to this width |

### Example Commands

```bash
# Basic laptop webcam (uses all defaults)
python -m src.main

# Indoor mode with IP camera from phone
python -m src.main --mode indoor --camera "http://192.168.1.10:8080/video"

# Outdoor mode, higher confidence, capped at 8 FPS
python -m src.main --mode outdoor --confidence 0.7 --max-fps 8

# No depth (blazing fast, direction-only announcements)
python -m src.main --no-depth

# Custom depth calibration
python -m src.main --depth-scale 2.5

# Quick response mode (announce after 0.2s instead of 0.5s)
python -m src.main --min-duration 0.2
```

---

## 🛠 Depth Calibration

Depth Anything V2 predicts *relative disparity* (inverse depth). To convert this into real-world meters, we apply: **`Distance = Scale / Disparity`**.

Because every camera lens is different, you may need to calibrate:

1. Place an object exactly **1.0 meter** from your camera
2. Run `python -m src.main` and listen to the announced distance
3. Adjust `--depth-scale`:
   - Says **2.0m** → lower the scale (e.g., `--depth-scale 1.5`)
   - Says **0.5m** → raise the scale (e.g., `--depth-scale 6.0`)
4. Repeat until the announced distance matches reality

---

## 🐳 Docker

### Build

```bash
docker build -t blinds-eye -f deployment/Dockerfile .
```

### Run (CPU)

```bash
docker run -it --rm \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    blinds-eye
```

### Run (NVIDIA GPU)

*Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)*

```bash
docker run -it --rm \
    --gpus all \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    blinds-eye
```

> [!TIP]
> On Windows with Docker Desktop (WSL2), mapping physical webcams is tricky. Use an IP camera app on your phone (e.g., **IP Webcam** for Android) and pass the URL:
> ```bash
> docker run -it --rm --gpus all blinds-eye python -m src.main --camera "http://192.168.1.10:8080/video"
> ```

---

## ⚠️ Known Issues & Troubleshooting

### Low FPS (5 FPS) despite having an NVIDIA GPU

**Cause:** Both `onnxruntime` and `onnxruntime-gpu` are installed simultaneously. The CPU version silently overrides the GPU version.

**Fix:**
```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu --force-reinstall
```

**Verify:** When the app starts, check the ONNX providers line:
```
# ❌ Bad (CPU only — will be very slow)
[INIT] ONNX providers: ['AzureExecutionProvider', 'CPUExecutionProvider']

# ✅ Good (GPU accelerated)
[INIT] ONNX providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Ghost audio (TTS announces objects that already disappeared)

**Cause:** In earlier versions, even a single-frame detection would trigger TTS.

**Fix:** Already resolved. The `--min-duration` flag (default: 0.5s) ensures objects must be continuously visible before being announced.

### No audio output

- Ensure `sounddevice` is installed: `pip install sounddevice`
- Place a Piper voice model (`.onnx` + `.onnx.json`) in `models/tts/`
- Download voices from: https://rhasspy.github.io/piper-samples/

### Camera not opening

- Laptop webcam: Ensure no other app (Zoom, Teams) is using the camera
- IP camera: Check the URL is reachable — try opening it in a browser first

---

## 🏗 Architecture

```
┌──────────────┐     ┌───────────────────────────────────────┐
│  Camera      │     │         VisionPipeline                │
│  (threaded)  │────▶│  ┌─────────┐    ┌──────────────────┐  │
│              │     │  │ YOLOv11 │    │ Depth Anything   │  │
└──────────────┘     │  │ (main   │    │ V2 (background   │  │
                     │  │ thread) │    │ thread, ONNX)    │  │
                     │  └────┬────┘    └────────┬─────────┘  │
                     │       │    Fuse           │            │
                     │       ▼    centroids      ▼            │
                     │  ┌─────────────────────────────────┐  │
                     │  │ Detection + Depth per object    │  │
                     │  └───────────────┬─────────────────┘  │
                     └──────────────────┼────────────────────┘
                                        │
                                        ▼
                     ┌──────────────────────────────────────┐
                     │  AnnouncementTracker                 │
                     │  • Temporal debouncing (min_duration) │
                     │  • Absence reset tracking            │
                     │  • Speak interval throttling         │
                     └───────────────┬──────────────────────┘
                                     │
                                     ▼
                     ┌──────────────────────────────────────┐
                     │  Piper TTS Engine (daemon thread)    │
                     │  • PriorityQueue (critical preempts) │
                     │  • Direct PCM → sounddevice stream   │
                     │  • Fully offline, no disk I/O        │
                     └──────────────────────────────────────┘
```

---

## 📄 License

This project is for educational and assistive technology purposes.

---

<p align="center">
  Built with ❤️ for accessibility
</p>
