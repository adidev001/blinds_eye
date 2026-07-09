"""
Blind's Eye — Threaded Vision Pipeline
========================================
Runs YOLO object detection and Depth Anything V2 monocular depth
estimation in parallel threads, fusing their outputs to produce
spatially-aware obstacle announcements.

Per PRD §6 Phase 3:
    1. Threaded Vision Loop — non-blocking, parallel execution
    2. Centroid Extraction — YOLO bbox centroid → DA-V2 depth matrix
    3. Announce Queue — formatted telemetry → TTS queue
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# Local imports (lazy — may be None if deps missing)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]


# ======================================================================
# Data Structures
# ======================================================================

@dataclass
class Detection:
    """A single detected object with spatial metadata."""
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int               # centroid x
    cy: int               # centroid y
    direction: str        # "left" | "center" | "right"
    depth_meters: Optional[float] = None  # metric depth at centroid


@dataclass
class FrameResult:
    """Result bundle for one processed frame."""
    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None
    timestamp: float = 0.0


# ======================================================================
# Direction Helper
# ======================================================================

def get_direction(cx: int, frame_width: int) -> str:
    """Return left / center / right depending on horizontal position."""
    if cx < frame_width / 3:
        return "left"
    elif cx < 2 * frame_width / 3:
        return "center"
    else:
        return "right"


# ======================================================================
# Vision Pipeline
# ======================================================================

class VisionPipeline:
    """
    Threaded vision pipeline that runs YOLO and DA-V2 in parallel.

    Parameters
    ----------
    yolo_model
        A loaded Ultralytics YOLO model instance.
    depth_session
        An ONNX Runtime InferenceSession for Depth Anything V2,
        or None to skip depth estimation.
    confidence : float
        Minimum detection confidence threshold.
    frame_width : int
        Target frame width for resizing.
    """

    def __init__(
        self,
        yolo_model,
        depth_session=None,
        confidence: float = 0.5,
        frame_width: int = 640,
        depth_scale: float = 3.0,
    ) -> None:
        self._yolo = yolo_model
        self._depth = depth_session
        self._confidence = confidence
        self._frame_width = frame_width
        self._depth_scale = depth_scale

        # Depth input shape cache (set on first frame if depth is loaded)
        self._depth_input_name: Optional[str] = None
        self._depth_input_shape: Optional[tuple] = None
        if self._depth is not None:
            inp = self._depth.get_inputs()[0]
            self._depth_input_name = inp.name
            # Expected shape: [1, 3, H, W]
            shape = inp.shape[2:]
            if not isinstance(shape[0], int) or not isinstance(shape[1], int):
                self._depth_input_shape = (518, 518) # DA-V2 standard size
            else:
                self._depth_input_shape = tuple(shape)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Run detection and depth estimation in parallel on *frame*.

        Returns a `FrameResult` containing detections (with optional
        per-object depth) and the raw depth map.
        """
        h, w = frame.shape[:2]
        if w != self._frame_width:
            scale = self._frame_width / float(w)
            frame = cv2.resize(frame, (self._frame_width, int(h * scale)))
            h, w = frame.shape[:2]

        # --- Launch parallel threads ----------------------------------
        det_result: list[Detection] = []
        depth_map: Optional[np.ndarray] = None
        det_error: Optional[Exception] = None
        depth_error: Optional[Exception] = None

        def _run_detection():
            nonlocal det_result, det_error
            try:
                det_result = self._detect(frame, w)
            except Exception as e:
                det_error = e

        def _run_depth():
            nonlocal depth_map, depth_error
            try:
                depth_map = self._estimate_depth(frame)
            except Exception as e:
                depth_error = e

        t_det = threading.Thread(target=_run_detection, name="yolo-det")
        t_det.start()

        if self._depth is not None:
            t_dep = threading.Thread(target=_run_depth, name="da-v2-depth")
            t_dep.start()
        else:
            t_dep = None

        t_det.join()
        if t_dep is not None:
            t_dep.join()

        if det_error:
            print(f"[VISION] Detection error: {det_error}")
        if depth_error:
            print(f"[VISION] Depth error: {depth_error}")

        # --- Fuse: overlay detection centroids onto depth map ----------
        if depth_map is not None and det_result:
            dh, dw = depth_map.shape[:2]
            scale_x = dw / w
            scale_y = dh / h
            for det in det_result:
                dx = int(det.cx * scale_x)
                dy = int(det.cy * scale_y)
                dx = min(max(dx, 0), dw - 1)
                dy = min(max(dy, 0), dh - 1)
                # Depth map values are in relative scale; convert later
                # if a metric scale factor is known.
                det.depth_meters = round(float(depth_map[dy, dx]), 2)

        return FrameResult(
            frame=frame,
            detections=det_result,
            depth_map=depth_map,
            timestamp=time.time(),
        )

    def draw_overlays(self, result: FrameResult) -> np.ndarray:
        """
        Draw bounding boxes, labels, direction lines, and depth info
        onto the frame.  Returns the annotated frame.
        """
        frame = result.frame.copy()
        h, w = frame.shape[:2]

        for det in result.detections:
            color = (0, 255, 0)
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            # Build label string
            parts = [det.label, det.direction, f"{int(det.confidence * 100)}%"]
            if det.depth_meters is not None:
                parts.append(f"{det.depth_meters:.1f}m")
            label_text = " ".join(parts)

            cv2.putText(
                frame, label_text,
                (det.x1, max(20, det.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

        # Direction guide lines (thirds)
        cv2.line(frame, (w // 3, 0), (w // 3, h), (255, 255, 255), 1)
        cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (255, 255, 255), 1)

        return frame

    # ------------------------------------------------------------------
    # Internal — Detection
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray, frame_w: int) -> list[Detection]:
        """Run YOLO inference and return a list of Detection objects."""
        results = self._yolo(frame, conf=self._confidence, verbose=False)[0]
        detections: list[Detection] = []

        for box in results.boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()
            conf = float(box.conf.cpu().numpy())
            cls = int(box.cls.cpu().numpy())
            label = self._yolo.names[cls]

            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            direction = get_direction(cx, frame_w)

            detections.append(Detection(
                label=label,
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
                cx=cx, cy=cy,
                direction=direction,
            ))

        return detections

    # ------------------------------------------------------------------
    # Internal — Depth Estimation
    # ------------------------------------------------------------------

    def _estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run Depth Anything V2 ONNX inference.

        Returns a 2-D depth map (float32, H×W) with relative depth
        values (higher = further).
        """
        if self._depth is None or self._depth_input_name is None:
            return None

        # Preprocess: resize, normalize, CHW, batch
        target_h, target_w = self._depth_input_shape  # type: ignore[misc]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h))
        img = img.astype(np.float32) / 255.0
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC → CHW → NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = self._depth.run(None, {self._depth_input_name: img})
        depth_map = outputs[0].squeeze()  # (H, W)

        # Convert disparity (relative inverse depth) to a rough metric distance
        # Z = Scale / Disparity
        depth_map = self._depth_scale / (depth_map + 1e-6)

        return depth_map


# ======================================================================
# Announcement Formatter
# ======================================================================

class AnnouncementTracker:
    """
    Tracks detected objects and decides when they should be announced.

    Re-announces an object only if it disappears for longer than
    `absence_reset` seconds.
    """

    def __init__(
        self,
        absence_reset: float = 1.5,
        speak_interval: float = 2.0,
    ) -> None:
        self._absence_reset = absence_reset
        self._speak_interval = speak_interval
        self._last_seen: dict[str, float] = {}
        self._pending: set[str] = set()
        self._last_speak_time: float = time.time()

    def update(self, detections: list[Detection]) -> Optional[str]:
        """
        Feed the latest detections.  Returns an announcement string
        if it's time to speak, otherwise None.
        """
        now = time.time()
        current_keys: set[str] = set()

        for det in detections:
            # Build a key that includes depth when available
            if det.depth_meters is not None:
                key = f"{det.label} on the {det.direction} at {det.depth_meters:.1f} meters"
            else:
                key = f"{det.label} on the {det.direction}"

            # Simplified key for tracking (without depth)
            track_key = f"{det.label}_{det.direction}"
            current_keys.add(track_key)

            if track_key not in self._last_seen or \
               (now - self._last_seen[track_key]) > self._absence_reset:
                self._pending.add(key)

            self._last_seen[track_key] = now

        # Purge disappeared objects
        self._last_seen = {
            k: v for k, v in self._last_seen.items()
            if now - v <= self._absence_reset + 0.5
        }

        # Emit announcement at interval
        if (now - self._last_speak_time) >= self._speak_interval and self._pending:
            items = sorted(self._pending)
            if len(items) > 1:
                msg = "I see: " + ", ".join(items)
            else:
                msg = f"I see {items[0]}"
            self._pending.clear()
            self._last_speak_time = now
            return msg

        return None
