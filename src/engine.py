"""
Blind's Eye — Hardware Probing & Model Loading Engine
======================================================
Dynamically detects host hardware (GPU model, VRAM, CPU) and selects
the appropriate model tier and ONNX Runtime execution provider.

Tier mapping (from PRD §2):
    Tier 1  — A100 (40/80 GB)  → DA-V2 Large  + YOLO Medium/Large
    Tier 2  — RTX 4050 (6 GB)  → DA-V2 Small  + YOLO Nano
    Tier 3  — CPU only          → DA-V2 Small (ONNX) + YOLO Nano (ONNX/OpenVINO)

Per PRD §6 Phase 2, Steps 2-3.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]


# ======================================================================
# Data Classes
# ======================================================================

class Tier(IntEnum):
    """Hardware tier classification."""
    TIER_1_SERVER = 1    # A100 / high-end datacenter GPU
    TIER_2_EDGE = 2      # Consumer NVIDIA GPU (≤ 8 GB VRAM)
    TIER_3_CPU = 3       # No dedicated GPU


@dataclass
class HardwareProfile:
    """Snapshot of the detected host hardware."""
    tier: Tier
    platform: str
    cpu_name: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    onnx_providers: list[str] = field(default_factory=list)


# ======================================================================
# Hardware Probe
# ======================================================================

class HardwareProbe:
    """
    Probes the local system and returns a `HardwareProfile`.

    The probe runs once on instantiation; results are cached in `.profile`.
    """

    # VRAM thresholds (GB) for tier classification
    _TIER1_VRAM_MIN = 20.0   # ≥ 20 GB → server class (A100 / A6000 etc.)

    def __init__(self) -> None:
        self.profile: HardwareProfile = self._probe()

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------

    def detect(self) -> dict:
        """Return the profile as a plain dict (handy for logging)."""
        return {
            "tier": self.profile.tier.name,
            "platform": self.profile.platform,
            "cpu": self.profile.cpu_name,
            "cuda": self.profile.cuda_available,
            "gpu": self.profile.gpu_name,
            "vram_gb": self.profile.gpu_vram_gb,
            "onnx_providers": self.profile.onnx_providers,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _probe(self) -> HardwareProfile:
        plat = platform.platform()
        cpu = platform.processor() or "unknown"
        cuda = False
        gpu_name: Optional[str] = None
        vram_gb: Optional[float] = None
        tier = Tier.TIER_3_CPU

        # --- CUDA / GPU detection via PyTorch --------------------------
        if torch is not None and torch.cuda.is_available():
            cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(vram_bytes / (1024 ** 3), 2)

            if vram_gb >= self._TIER1_VRAM_MIN:
                tier = Tier.TIER_1_SERVER
            else:
                tier = Tier.TIER_2_EDGE
        else:
            tier = Tier.TIER_3_CPU

        # --- ONNX Runtime providers ------------------------------------
        onnx_providers: list[str] = []
        if ort is not None:
            onnx_providers = ort.get_available_providers()

        return HardwareProfile(
            tier=tier,
            platform=plat,
            cpu_name=cpu,
            cuda_available=cuda,
            gpu_name=gpu_name,
            gpu_vram_gb=vram_gb,
            onnx_providers=onnx_providers,
        )


# ======================================================================
# Model Loader
# ======================================================================

# Default model directory relative to the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"

# Tier → YOLO weight filename mapping
_YOLO_TIER_MAP: dict[Tier, str] = {
    Tier.TIER_1_SERVER: "yolo11m.pt",    # Medium / Large for server
    Tier.TIER_2_EDGE:   "yolo11n.pt",    # Nano for edge
    Tier.TIER_3_CPU:    "yolo11n.pt",    # Nano for CPU
}

# Tier → Depth Anything V2 weight filename mapping
_DEPTH_TIER_MAP: dict[Tier, str] = {
    Tier.TIER_1_SERVER: "depth_anything_v2_large.engine",
    Tier.TIER_2_EDGE:   "depth_anything_v2_small.onnx",
    Tier.TIER_3_CPU:    "depth_anything_v2_small.onnx",
}


class ModelLoader:
    """
    Loads the YOLO and Depth Anything V2 models appropriate for the
    detected hardware tier.

    Parameters
    ----------
    profile : HardwareProfile
        The hardware profile returned by `HardwareProbe`.
    models_dir : Path | str | None
        Override for the models directory (defaults to ``<project>/models``).
    """

    def __init__(
        self,
        profile: HardwareProfile,
        models_dir: Optional[Path | str] = None,
    ) -> None:
        self._profile = profile
        self._models_dir = Path(models_dir) if models_dir else _MODELS_DIR
        self._yolo_model = None
        self._depth_session = None

    # ------------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------------

    def load_yolo(self):
        """
        Load the appropriate Ultralytics YOLO model.

        The model file is auto-downloaded by ultralytics if not found
        locally under ``models/detection/``.
        """
        if YOLO is None:
            raise ImportError("ultralytics is not installed")

        weight_name = _YOLO_TIER_MAP[self._profile.tier]
        weight_path = self._models_dir / "detection" / weight_name

        # Ultralytics auto-downloads if the path doesn't exist — just
        # pass the filename and let it resolve.
        source = str(weight_path) if weight_path.exists() else weight_name
        print(f"[ENGINE] Loading YOLO model: {source} (Tier {self._profile.tier.name})")

        self._yolo_model = YOLO(source)
        return self._yolo_model

    # ------------------------------------------------------------------
    # Depth Anything V2 (ONNX)
    # ------------------------------------------------------------------

    def load_depth(self):
        """
        Load the Depth Anything V2 ONNX model via ONNX Runtime.

        Selects execution providers based on the hardware profile:
            Tier 1/2 → CUDAExecutionProvider first
            Tier 3   → OpenVINOExecutionProvider → CPUExecutionProvider
        """
        if ort is None:
            raise ImportError("onnxruntime is not installed")

        weight_name = _DEPTH_TIER_MAP[self._profile.tier]
        weight_path = self._models_dir / "depth" / weight_name

        if not weight_path.exists():
            print(
                f"[ENGINE] Depth model not found at {weight_path}. "
                "Skipping depth loading — run without depth until weights are supplied."
            )
            return None

        # Build execution provider preference list
        providers = self._select_onnx_providers()
        print(f"[ENGINE] Loading depth model: {weight_path} with providers {providers}")

        self._depth_session = ort.InferenceSession(
            str(weight_path), providers=providers,
        )
        return self._depth_session

    def _select_onnx_providers(self) -> list[str]:
        """Choose ONNX Runtime execution providers by tier."""
        available = set(self._profile.onnx_providers)

        if self._profile.tier in (Tier.TIER_1_SERVER, Tier.TIER_2_EDGE):
            preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            preferred = [
                "OpenVINOExecutionProvider",
                "CPUExecutionProvider",
            ]

        return [p for p in preferred if p in available] or ["CPUExecutionProvider"]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def yolo(self):
        return self._yolo_model

    @property
    def depth(self):
        return self._depth_session
