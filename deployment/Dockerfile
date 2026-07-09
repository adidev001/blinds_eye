# ==============================================================
# Blind's Eye — Production Dockerfile
# ==============================================================
# Cross-hardware container using NVIDIA CUDA base runtime.
# Supports GPU passthrough via: docker run --gpus all
# Falls back to CPU execution providers automatically.
#
# Build:
#   docker build -t blinds-eye -f deployment/Dockerfile .
#
# Run (GPU):
#   docker run --gpus all --device /dev/video0 blinds-eye
#
# Run (CPU fallback):
#   docker run --device /dev/video0 blinds-eye
# ==============================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libespeak1 \
        espeak \
        ffmpeg \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ---- Working directory ----
WORKDIR /app

# ---- Install Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Copy application code ----
COPY src/ ./src/

# ---- Copy model weights (if present) ----
# Models are typically volume-mounted at runtime, but copy any
# pre-bundled weights if available.
COPY models/ ./models/

# ---- Entrypoint ----
ENTRYPOINT ["python", "-m", "src.main"]
