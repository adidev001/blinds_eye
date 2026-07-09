Here is the updated, comprehensive Product Requirements Document (PRD) for project "Blind's Eye." This version integrates strict local environment management using either `conda` or `venv` alongside the Docker deployment strategy to ensure isolated, reproducible development across all team devices.

---

# Product Requirements Document (PRD): Project "Blind's Eye"

**Phase:** Architecture Refactor, Environment Isolation & Cross-Hardware Deployment

**Core Tech Stack:** Python, Ultralytics YOLOv11, Depth Anything V2, PyTorch, ONNX Runtime, Docker, Conda / Venv

## 1. Executive Summary

Project "Blind's Eye" is an AI-powered assistive device pipeline providing real-time spatial awareness and obstacle detection. This phase transitions the system from a computationally heavy RAFT-Stereo setup to an optimized monocular (single-camera) architecture. By running a lightweight YOLO object detector in parallel with Depth Anything V2 (DA-V2), the system delivers real-time, metric-depth-aware obstacle announcements.

To prevent dependency drift and ensure system stability across diverse host architectures, all local development must happen within isolated virtual environments (`conda` or `venv`), while production delivery is standardized via Docker.

## 2. Target Hardware & Execution Strategy

The inference engine must dynamically probe host hardware at runtime and allocate resources accordingly without crashing.

* **Tier 1: Server Compute (Professor's Device)**
* **Hardware:** NVIDIA A100 GPU (40GB/80GB VRAM).
* **Execution:** `CUDAExecutionProvider` inside Docker.
* **Model Tier:** DA-V2 Large + YOLOv11 Medium/Large.


* **Tier 2: Edge Compute (Devansh's Device)**
* **Hardware:** Intel i7-14700HX + NVIDIA RTX 4050 (6GB VRAM).
* **Execution:** `CUDAExecutionProvider` via Local Conda/Venv or Docker.
* **Model Tier:** DA-V2 Small (<1GB VRAM) + YOLOv11 Nano.


* **Tier 3: CPU-Only Compute (Friend's Device)**
* **Hardware:** Intel i5-13500H (No dedicated GPU).
* **Execution:** `OpenVINOExecutionProvider` or `CPUExecutionProvider` via Local Conda/Venv or Docker.
* **Model Tier:** DA-V2 Small (ONNX format) + YOLOv11 Nano (ONNX/OpenVINO format).



## 3. Environment Isolation Strategy

Developers may choose between standard Python `venv` or Anaconda/Miniconda based on their local system preferences.

### Option A: Standard Python `venv` (Recommended for lightweight setups)

Used to keep dependencies entirely contained within the project folder without global system pollution.

```bash
# Creation
python3 -m venv venv

# Activation (Windows)
.\venv\Scripts\activate

# Activation (Linux/Mac)
source venv/bin/activate

```

### Option B: Conda/Miniconda (Recommended for complex CUDA handling)

Used if native C++ bindings, CUDA toolkits, or ONNX Runtime dependencies require isolated binary packages outside of pip.

```bash
# Creation
conda create --name blinds_eye python=3.10 -y

# Activation
conda activate blinds_eye

```

---

## 4. Folder Structure (The Target Architecture)

The repository must maintain this explicit structure. Local environment folders (`venv` or `.conda`) reside inside the root directory but are strictly locked out of source control.

```text
Blind-s-Eye-Project/
│
├── .conda/                     # Local Conda environment binaries (Excluded from Git)
├── venv/                       # Local Venv environment binaries (Excluded from Git)
│
├── data/                       
│   ├── images/                 # Test images (e.g., 1.png, 2.png)
│   └── videos/                 # Test videos (e.g., car.gif)
│
├── models/                     # Weights directory (Excluded from Git)
│   ├── detection/
│   │   ├── yolo11n.pt          
│   │   └── coco.names          
│   └── depth/                  
│       ├── depth_anything_v2_small.onnx
│       └── depth_anything_v2_large.engine
│
├── src/                        # Core Application Logic
│   ├── __init__.py
│   ├── engine.py               # Hardware probing & model loading logic
│   ├── tts_module.py           # pyttsx3 Queue and background worker
│   ├── vision.py               # YOLO and DA-V2 parallel execution loops
│   └── main.py                 # Application entry point and orchestrator
│
├── deployment/                 
│   └── Dockerfile              # Cross-hardware container definition
│
├── .gitignore                  # Repository rules
├── environment.yml             # Conda environment definition file
├── requirements.txt            # Python pip dependencies
└── README.md                   # Setup and execution instructions

```

---

## 5. Configuration Files

### 5.1. `.gitignore`

This file guarantees that neither local environments nor massive AI model weights are ever accidentally pushed to GitHub.

```text
# Ignore Heavy Model Weights
models/**/*.pt
models/**/*.weights
models/**/*.engine
models/**/*.onnx

# Ignore legacy configs if not actively used
yolo-coco/
*.cfg

# Ignore Local Virtual Environments (CRITICAL)
venv/
.venv/
env/
.conda/
miniconda/

# Ignore Python Caches and IDE files
__pycache__/
*.pyc
.ipynb_checkpoints/
.vscode/
.idea/

# Ignore OS metadata
.DS_Store
Thumbs.db

```

### 5.2. `requirements.txt`

Dependencies optimized to facilitate smooth processing across both NVIDIA CUDA and Intel execution providers.

```text
# Core PyTorch (Provides base structures)
torch>=2.0.0
torchvision>=0.15.0

# Inference Engines (Cross-Hardware Support)
onnxruntime>=1.17.0
onnxruntime-gpu>=1.17.0
onnxruntime-openvino>=1.17.0

# Vision & Analytics
ultralytics>=8.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0

# Audio/TTS
pyttsx3>=2.90

```

### 5.3. `environment.yml`

For team members utilizing Conda, this file manages the environment creation comprehensively.

```yaml
name: blinds_eye
channels:
  - defaults
  - pytorch
  - nvidia
dependencies:
  - python=3.10
  - pip
  - pip:
    - -r requirements.txt

```

---

## 6. Implementation Roadmap (Start-to-Finish)

### Phase 1: Environment Setup & Tree Cleanup

1. **Initialize Environment:** Create and activate either the `venv` or `conda` environment locally before touching any code.
2. **Restructure the Tree:** Move test images and videos into `data/`. Move the `real_time_object_detection.py` script into `src/main.py`.
3. **Purge Legacy Configurations:** Delete `yolov3.weights` and `yolov3.cfg` to free repository space.
4. **Lock Git Rules:** Commit the `.gitignore` to the repository immediately before adding dependency changes.

### Phase 2: Modular Engine Development

1. **Extract the TTS Module:** Move the `pyttsx3` background worker queues out of the main script and wrap them into an object-oriented class in `src/tts_module.py`.
2. **Hardware Probing:** In `src/engine.py`, build the infrastructure to probe `torch.cuda.is_available()`.
3. **Model Instantiation:** Load the appropriate high/mid/low resource models dynamically depending on the hardware signature found during the probe step.

### Phase 3: Data Fusion Pipeline

1. **Threaded Vision Loop:** Implement non-blocking, parallel execution in `src/vision.py` for both the object detection and depth mapping threads.
2. **Centroid Extraction:** Extract spatial coordinates from the YOLO bounding box and overlay them onto the DA-V2 depth matrix to pull the exact metric distance.
3. **Announce Queue:** Format the text telemetry and stream it directly to the active TTS queue.

### Phase 4: Containerization & Cross-Platform Validation

1. **Dockerization:** Build the `Dockerfile` inside `deployment/` using an NVIDIA CUDA base runtime layer.
2. **Edge Test (RTX 4050):** Run the container locally using `docker run --gpus all` and verify VRAM overhead remains within safe bounds (<6GB).
3. **CPU Fallback Test (i5-13500H Simulation):** Launch the container without passing the GPU flags. Verify that the system handles the exception natively, successfully loads the OpenVINO ONNX engine, and executes without freezing.
4. **Final Deployment:** Push the pristine, fully-documented modular pipeline to GitHub for direct deployment onto the professor's A100 environment.