# Blind's Eye

Blind's Eye is an AI-powered real-time object detection and spatial awareness tool. It uses a webcam or IP camera feed to detect objects using **YOLO11** and estimates their real-world distances using **Depth Anything V2**. The detections are then spoken aloud using native text-to-speech (TTS) to provide environmental awareness.

## Features
- **Real-Time Detection:** Uses YOLO to identify objects and track their left/center/right direction.
- **Metric Depth Estimation:** Fuses YOLO centroids with the DA-V2 depth map to calculate distance.
- **Ultra-Low Latency TTS:** Asynchronous background thread using native Windows `win32com` SAPI for instant audio feedback without pipeline lag.
- **IP Camera Support:** Stream directly from network cameras via HTTP/RTSP URLs.
- **Anti-Lag Engine:** Background buffer-draining ensures you always process the absolute freshest frame.

---

## 🛠 Depth Calibration

Depth Anything V2 predicts *relative disparity* (inverse depth). To convert this into real-world meters, we apply a mathematical formula: `Distance = Scale / Disparity`. 

Because every camera lens and resolution is slightly different, you may need to calibrate the `--depth-scale` parameter to match your specific setup.

### How to Calibrate
1. Place a recognizable object (like a `laptop` or `cell phone`) exactly **1.0 meter** away from your camera.
2. Run the program with the default scale:
   ```bash
   python -m src.main
   ```
3. Listen to the distance it announces. 
   - If it says **2.0 meters**, your scale is *twice* as high as it should be. 
   - If it says **0.5 meters**, your scale is *half* what it should be.
4. Stop the program, and adjust the `--depth-scale` accordingly. (For example, if the default scale of 3.0 gives you 2.0 meters, lower the scale to 1.5).

```bash
python -m src.main --depth-scale 1.5
```
5. Repeat until the announced distance closely matches the real physical distance.

---

## 🚀 Usage

Run the main pipeline directly from your terminal:

```bash
# Default webcam (0)
python -m src.main

# Custom Depth Scale
python -m src.main --depth-scale 2.0

# Using an IP Camera (e.g. from an Android app like IP Webcam)
python -m src.main --camera "http://192.168.1.10:8080/video"

# Disable Depth Estimation (Only announce object direction)
python -m src.main --no-depth
```

### CLI Arguments
- `--camera`: Camera device index (integer) or IP camera stream URL (default: `0`).
- `--confidence`: Minimum confidence threshold for YOLO detections (default: `0.5`).
- `--speech-rate`: Speed of the TTS engine (default: `170`).
- `--absence-reset`: Seconds an object must disappear before it is re-announced (default: `1.5`).
- `--depth-scale`: Calibration factor to convert disparity to meters (default: `3.0`).
- `--no-depth`: Flag to bypass Depth Anything V2.

---

## 🐳 Dockerization

You can run Blind's Eye inside a Docker container. The included `Dockerfile` uses an NVIDIA CUDA base image to support hardware acceleration (if available) and falls back to the CPU automatically.

### 1. Build the Image
```bash
docker build -t blinds-eye -f deployment/Dockerfile .
```

### 2. Run the Container

**For CPU Only:**
```bash
docker run -it --rm \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    blinds-eye
```

**For NVIDIA GPU Acceleration:**
*(Requires NVIDIA Container Toolkit installed on the host)*
```bash
docker run -it --rm \
    --gpus all \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    blinds-eye
```

*(Note: If you're on Windows and using Docker Desktop with WSL2, mapping physical webcams into Docker can be complicated. The easiest workaround is to use an IP Camera app on your phone and pass the URL to the `--camera` flag!)*
