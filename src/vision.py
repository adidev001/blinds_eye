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

# ─────────────────────────────────────────────────────────────────────────
# Context-Aware Filtering Modes
# ─────────────────────────────────────────────────────────────────────────
INDOOR_CLASSES = {
    "person", "cat", "dog", "chair", "couch", "potted plant", "bed", 
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush", "backpack", "umbrella", "handbag", "tie", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", 
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
    "donut", "cake"
}

OUTDOOR_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag"
}

# Local imports (lazy — may be None if deps missing)
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

from src.localization import LocalizationFormatter


# ======================================================================
# Data Structures
# ======================================================================

@dataclass
class Detection:
    """A single detected object with spatial and kinematic metadata (FR-1.1–FR-1.4)."""
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int               # centroid x
    cy: int               # centroid y
    direction: str        # "left" | "center" | "right"
    depth_meters: Optional[float] = None  # metric depth
    track_id: Optional[int] = None        # ByteTrack ID
    raw_disparity: Optional[float] = None # 65th percentile disparity
    vertical_zone: str = "normal"         # "normal" | "overhead" | "ground"
    is_approaching: bool = False          # kinematic approaching flag
    v_close: float = 0.0                  # closing velocity (m/s)
    r_scale: float = 0.0                  # 2D optical looming expansion rate
    ttc: Optional[float] = None           # time to collision (seconds)


@dataclass
class FrameResult:
    """Result bundle for one processed frame."""
    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None
    timestamp: float = 0.0
    depth_frame_id: int = 0
    depth_timestamp: float = 0.0
    curtain_meters: Optional[float] = None
    low_light_detected: bool = False
    yolo_ms: float = 0.0
    depth_ms: float = 0.0



# ======================================================================
# Direction & Depth ROI Helpers
# ======================================================================

def get_direction(cx: int, frame_width: int) -> str:
    """Return left / center / right depending on horizontal position."""
    if cx < frame_width / 3:
        return "left"
    elif cx < 2 * frame_width / 3:
        return "center"
    else:
        return "right"


def extract_roi_disparity(
    disparity_map: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> float:
    """
    Inner-Box 65th percentile depth extraction (FR-1.1).
    Extracts the central 30% x 30% core region of the disparity map:
    rx1 = int(x1 + 0.35 * w), rx2 = int(x2 - 0.35 * w)
    ry1 = int(y1 + 0.35 * h), ry2 = int(y2 - 0.35 * h)
    Returns the 65th percentile disparity, falling back to centroid disparity if empty.
    """
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)

    rx1 = int(x1 + 0.35 * w)
    rx2 = int(x2 - 0.35 * w)
    ry1 = int(y1 + 0.35 * h)
    ry2 = int(y2 - 0.35 * h)

    dh, dw = disparity_map.shape[:2]
    scale_x = dw / float(frame_w)
    scale_y = dh / float(frame_h)

    dx1 = max(0, min(int(rx1 * scale_x), dw - 1))
    dx2 = max(dx1 + 1, min(int(rx2 * scale_x) + 1, dw))
    dy1 = max(0, min(int(ry1 * scale_y), dh - 1))
    dy2 = max(dy1 + 1, min(int(ry2 * scale_y) + 1, dh))

    roi = disparity_map[dy1:dy2, dx1:dx2]
    if roi.size > 0:
        return float(np.percentile(roi, 65))

    cx = int(((x1 + x2) // 2) * scale_x)
    cy = int(((y1 + y2) // 2) * scale_y)
    cx = min(max(cx, 0), dw - 1)
    cy = min(max(cy, 0), dh - 1)
    return float(disparity_map[cy, cx])


# ======================================================================
# Vision Pipeline
# ======================================================================

class VisionPipeline:
    """
    Decoupled vision pipeline running YOLO26 with ByteTrack and
    Depth Anything V2 with dynamic resolution and power governor.
    """

    def __init__(
        self,
        yolo_model,
        depth_session=None,
        confidence: float = 0.5,
        frame_width: int = 640,
        depth_scale: float = 3.0,
        mode: str = "all",
        depth_res: int = 350,
        dynamic_power: bool = False,
    ) -> None:
        self._yolo = yolo_model
        self._depth = depth_session
        self._confidence = confidence
        self._frame_width = frame_width
        self._depth_scale = depth_scale
        self._dynamic_power = dynamic_power
        self._depth_res = depth_res

        self._mode = mode.lower()
        if self._mode == "indoor":
            self._allowed_classes = INDOOR_CLASSES
        elif self._mode == "outdoor":
            self._allowed_classes = OUTDOOR_CLASSES
        else:
            self._allowed_classes = None

        # Depth input shape negotiation (FR-1.6)
        self._depth_input_name: Optional[str] = None
        self._depth_input_shape: Optional[tuple] = None
        if self._depth is not None:
            inp = self._depth.get_inputs()[0]
            self._depth_input_name = inp.name
            shape = inp.shape[2:]
            if not isinstance(shape[0], int) or not isinstance(shape[1], int):
                # Dynamic shape: round depth_res to nearest multiple of ViT stride 14
                stride = 14
                res = max(int(round(depth_res / stride) * stride), 14)
                self._depth_input_shape = (res, res)
            else:
                self._depth_input_shape = tuple(shape)

        # Dynamic power & staleness state (FR-1.7, FR-2.3)
        self._frame_count = 0
        self._low_light_streak = 0
        self._depth_frame_id = 0
        self._depth_timestamp = 0.0
        self._prev_gray: Optional[np.ndarray] = None
        self._idle_since: Optional[float] = None
        self._last_depth_run_time: float = 0.0
        self._cached_disparity_map: Optional[np.ndarray] = None
        self._depth_enabled: bool = True

    @property
    def mode(self) -> str:
        """Return current context filtering mode."""
        return self._mode

    @property
    def depth_enabled(self) -> bool:
        """Return whether depth estimation is active."""
        return self._depth_enabled and (self._depth is not None)

    def toggle_depth(self) -> bool:
        """Toggle depth inference on/off. Returns new enabled state."""
        self._depth_enabled = not self._depth_enabled
        state = "ENABLED" if self._depth_enabled else "DISABLED"
        print(f"[VISION] Depth estimation: {state}")
        return self._depth_enabled

    def set_mode(self, mode: str) -> None:
        """Dynamically update context filtering mode (FR-5.2)."""
        self._mode = mode.lower()
        if self._mode == "indoor":
            self._allowed_classes = INDOOR_CLASSES
        elif self._mode == "outdoor":
            self._allowed_classes = OUTDOOR_CLASSES
        else:
            self._allowed_classes = None

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process frame through YOLO26, ByteTrack, and Depth Anything V2.
        """
        h, w = frame.shape[:2]
        if w != self._frame_width:
            scale = self._frame_width / float(w)
            frame = cv2.resize(frame, (self._frame_width, int(h * scale)))
            h, w = frame.shape[:2]

        self._frame_count += 1
        now = time.monotonic()

        # ── 1. Luminance Darkness / Occlusion Check (FR-2.3) ──────────────
        low_light_detected = False
        if self._frame_count % 30 == 0:
            p95 = float(np.percentile(frame, 95))
            if p95 < 20.0:
                self._low_light_streak += 1
                if self._low_light_streak >= 3:
                    low_light_detected = True
            else:
                self._low_light_streak = 0

        # ── 2. Run YOLO26 + ByteTrack Detection ───────────────────────────
        det_result: list[Detection] = []
        t_yolo0 = time.perf_counter()
        try:
            det_result = self._detect(frame, w, h)
        except Exception as e:
            print(f"[VISION] Detection error: {e}")
        yolo_ms = (time.perf_counter() - t_yolo0) * 1000.0

        # ── 3. Depth-Thread Dynamic Power Governor (FR-1.7) ───────────────
        should_run_depth = (self._depth is not None) and self._depth_enabled
        if should_run_depth and self._dynamic_power:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luminance_diff = 1.0
            if self._prev_gray is not None:
                diff_arr = cv2.absdiff(gray, self._prev_gray)
                luminance_diff = float(np.mean(diff_arr)) / 255.0
            self._prev_gray = gray

            if len(det_result) == 0 and luminance_diff < 0.03:
                if self._idle_since is None:
                    self._idle_since = now
                elif (now - self._idle_since) >= 5.0:
                    # Idle mode: throttle depth to 2 FPS (interval 0.5s)
                    if (now - self._last_depth_run_time) < 0.5:
                        should_run_depth = False
            else:
                self._idle_since = None

        # ── 4. Depth Estimation ───────────────────────────────────────────
        disp_map = self._cached_disparity_map
        depth_ms = 0.0
        if should_run_depth:
            t_d0 = time.perf_counter()
            try:
                disp_map = self._estimate_disparity(frame)
                self._cached_disparity_map = disp_map
                self._depth_frame_id += 1
                self._depth_timestamp = now
                self._last_depth_run_time = now
            except Exception as e:
                print(f"[VISION] Depth error: {e}")
            depth_ms = (time.perf_counter() - t_d0) * 1000.0

        # ── 5. Fusion & Proximity-Gated Vertical Tagging (FR-1.1, FR-1.4) ─
        if disp_map is not None and det_result:
            for det in det_result:
                d_raw = extract_roi_disparity(disp_map, (det.x1, det.y1, det.x2, det.y2), w, h)
                det.raw_disparity = d_raw
                metric_d = round(self._depth_scale / max(d_raw, 1e-6), 2)
                det.depth_meters = metric_d

                # Proximity-gated vertical tags (FR-1.4): evaluated only when d <= 2.0m
                if metric_d <= 2.0:
                    if det.y2 < int(0.45 * h):
                        det.vertical_zone = "overhead"
                    elif det.y1 > int(0.72 * h):
                        det.vertical_zone = "ground"
                    else:
                        det.vertical_zone = "normal"
                else:
                    det.vertical_zone = "normal"

        # ── 6. Upper-Center Horizon Depth Curtain (FR-1.5) ────────────────
        curtain_d = None
        has_center_object = any(d.direction == "center" for d in det_result)
        if not has_center_object and disp_map is not None:
            curtain_d = self.check_depth_curtain(disp_map)

        return FrameResult(
            frame=frame,
            detections=det_result,
            depth_map=disp_map,
            timestamp=now,
            depth_frame_id=self._depth_frame_id,
            depth_timestamp=self._depth_timestamp,
            curtain_meters=curtain_d,
            low_light_detected=low_light_detected,
            yolo_ms=yolo_ms,
            depth_ms=depth_ms,
        )


    def check_depth_curtain(self, disparity_map: np.ndarray) -> Optional[float]:
        """
        Upper-center horizon depth safety curtain (FR-1.5).
        Inspects y in [0.20H, 0.50H], x in [0.35W, 0.65W] with vertical flatness guard.
        """
        dh, dw = disparity_map.shape[:2]
        y1, y2 = int(0.20 * dh), int(0.50 * dh)
        x1, x2 = int(0.35 * dw), int(0.65 * dw)

        roi_curtain = disparity_map[y1:y2, x1:x2]
        if roi_curtain.size == 0:
            return None

        # Vertical flatness guard: ratio of upper half to lower half medians
        y_mid = int(0.35 * dh)
        upper_half = disparity_map[y1:y_mid, x1:x2]
        lower_half = disparity_map[y_mid:y2, x1:x2]

        if upper_half.size > 0 and lower_half.size > 0:
            med_upper = float(np.median(upper_half))
            med_lower = float(np.median(lower_half))
            if med_lower > 1e-6:
                ratio = med_upper / med_lower
                if not (0.80 <= ratio <= 1.20):
                    return None  # Pitch floor rejection

        p80_disp = float(np.percentile(roi_curtain, 80))
        d_curtain = self._depth_scale / max(p80_disp, 1e-6)
        if d_curtain <= 1.0:
            return round(d_curtain, 2)
        return None

    def _detect(self, frame: np.ndarray, frame_w: int, frame_h: int) -> list[Detection]:
        """Run YOLO inference with ByteTrack multi-object tracking (FR-1.2)."""
        if hasattr(self._yolo, "track"):
            try:
                yolo_res = self._yolo.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self._confidence,
                    verbose=False,
                )[0]
            except Exception:
                yolo_res = self._yolo(frame, conf=self._confidence, verbose=False)[0]
        else:
            yolo_res = self._yolo(frame, conf=self._confidence, verbose=False)[0]

        detections: list[Detection] = []
        names = self._yolo.names

        for box in yolo_res.boxes:
            conf = float(box.conf[0])
            if conf < self._confidence:
                continue

            cls_id = int(box.cls[0])
            label = names[cls_id]

            if self._allowed_classes and label not in self._allowed_classes:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            direction = get_direction(cx, frame_w)

            track_id = None
            if hasattr(box, "id") and box.id is not None:
                track_id = int(box.id[0])

            detections.append(Detection(
                label=label,
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
                cx=cx, cy=cy,
                direction=direction,
                track_id=track_id,
            ))

        return detections

    def _estimate_disparity(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Run Depth Anything V2 ONNX inference and return disparity map."""
        if self._depth is None or self._depth_input_name is None:
            return None

        target_h, target_w = self._depth_input_shape  # type: ignore[misc]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h))
        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = self._depth.run(None, {self._depth_input_name: img})
        disparity_map = outputs[0].squeeze()
        return disparity_map

    def draw_overlays(self, result: FrameResult) -> np.ndarray:
        """Draw bounding boxes, labels, direction lines, and kinematics onto frame."""
        frame = result.frame.copy()
        h, w = frame.shape[:2]

        for det in result.detections:
            color = (0, 0, 255) if det.is_approaching else (0, 255, 0)
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            parts = [det.label]
            if det.track_id is not None:
                parts.append(f"#{det.track_id}")
            parts.append(det.direction)
            if det.depth_meters is not None:
                parts.append(f"{det.depth_meters:.1f}m")
            if det.is_approaching:
                parts.append(f"+{det.v_close:.1f}m/s")
            if det.vertical_zone != "normal":
                parts.append(det.vertical_zone)

            label_text = " ".join(parts)
            cv2.putText(
                frame, label_text,
                (det.x1, max(20, det.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

        cv2.line(frame, (w // 3, 0), (w // 3, h), (255, 255, 255), 1)
        cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (255, 255, 255), 1)
        return frame


# ======================================================================
# Announcement Tracker & Kinematic State Machine
# ======================================================================

@dataclass
class _TrackRecord:
    first_seen: float
    last_seen: float
    d_smooth: Optional[float] = None
    d_smooth_prev: Optional[float] = None
    last_depth_frame_id: int = -1
    last_depth_timestamp: float = 0.0
    scale_history: list[float] = field(default_factory=list)
    v_close: float = 0.0
    r_scale: float = 0.0
    is_approaching: bool = False
    vertical_zone: str = "normal"
    announced: bool = False
    last_critical_time: float = 0.0
    last_corridor: str = "center"
    spoken_in_center: bool = False
    last_announced_dist: Optional[float] = None


class AnnouncementTracker:
    """
    Kinematic spatial tracker and announcement debouncer (FR-1.2, FR-1.3, FR-3.3).
    Features:
    - ByteTrack track_id persistence across corridors
    - Per-track Depth EMA smoothing (alpha=0.35)
    - Timestamp-synced closing velocity and 2D optical looming
    - 2.2 m/s ego-motion deadband
    - 3-Gate CRITICAL preemption with 3.0s per-track cooldown
    - Bounded crowd compression for multi-obstacle batches
    - Horizon depth curtain and low-light warning monitors
    """

    def __init__(
        self,
        absence_reset: float = 1.5,
        speak_interval: float = 2.0,
        min_duration: float = 0.5,
        lang: str = "en",
    ) -> None:
        self._absence_reset = absence_reset
        self._speak_interval = speak_interval
        self._min_duration = min_duration
        self._lang = lang
        self._fmt = LocalizationFormatter(lang)

        self._tracks: dict[int | str, _TrackRecord] = {}
        self._pending_items: dict[tuple[str, str], list[tuple[float, _TrackRecord]]] = {}
        self._last_speak_time: float = time.time()
        self._internal_frame_counter: int = 0

        # Safety curtain and low light timers
        self._curtain_start_time: Optional[float] = None
        self._last_curtain_alert: float = 0.0
        self._last_low_light_alert: float = 0.0

        # Critical event telemetry / haptic hooks (FR-5.3, FR-5.4)
        self.last_critical_event: Optional[dict] = None
        self.last_critical_gate: Optional[str] = None
        self._last_global_critical_time: float = 0.0

    @property
    def active_track_count(self) -> int:
        """Return count of active tracked objects."""
        return len(self._tracks)

    @property
    def lang(self) -> str:
        """Return current announcement language."""
        return self._lang

    def set_lang(self, lang: str) -> None:
        """Update tracker language formatter (en, hi, hinglish)."""
        if lang in ("en", "hi", "hinglish"):
            self._lang = lang
            self._fmt = LocalizationFormatter(lang)

    def update(
        self,
        detections: list[Detection],
        depth_frame_id: Optional[int] = None,
        depth_timestamp: Optional[float] = None,
        depth_curtain_meters: Optional[float] = None,
        low_light: bool = False,
    ) -> Optional[str]:
        """
        Process detections through tracking, kinematics, and critical preemption gates.
        Returns immediate CRITICAL alert string or periodic batched announcement string.
        """
        now = time.time()
        self.last_critical_event = None
        self.last_critical_gate = None
        self._internal_frame_counter += 1

        active_f_id = depth_frame_id if depth_frame_id is not None else self._internal_frame_counter
        active_f_ts = depth_timestamp if depth_timestamp is not None else now

        # ── System Diagnostics Alerts ─────────────────────────────────────
        if low_light and (now - self._last_low_light_alert) >= 10.0:
            self._last_low_light_alert = now
            return self._fmt.format_system_alert("low_light")

        if depth_curtain_meters is not None and depth_curtain_meters <= 1.0:
            if self._curtain_start_time is None:
                self._curtain_start_time = now
            elif (now - self._curtain_start_time) >= 0.6:
                if (now - self._last_curtain_alert) >= 4.0:
                    self._last_curtain_alert = now
                    return self._fmt.format_system_alert("obstacle_ahead", depth_curtain_meters)
        else:
            self._curtain_start_time = None

        # ── Process Detections & Kinematics ───────────────────────────────
        current_keys: set[int | str] = set()

        for det in detections:
            track_key: int | str = det.track_id if det.track_id is not None else f"{det.label}_{det.direction}"
            current_keys.add(track_key)

            # Streak tracking & absence reset
            if track_key not in self._tracks or (now - self._tracks[track_key].last_seen) > self._absence_reset:
                self._tracks[track_key] = _TrackRecord(
                    first_seen=now,
                    last_seen=now,
                    last_corridor=det.direction,
                )

            rec = self._tracks[track_key]
            rec.last_seen = now
            track_age = now - rec.first_seen

            # Cross-corridor transition (FR-1.2)
            if rec.last_corridor != det.direction:
                if det.direction == "center" and not rec.spoken_in_center:
                    if rec.d_smooth is not None and rec.d_smooth <= 2.5:
                        rec.announced = False
                        rec.spoken_in_center = True
                rec.last_corridor = det.direction

            # Depth EMA smoothing (alpha=0.35)
            if det.depth_meters is not None:
                raw_d = float(det.depth_meters)
                if rec.d_smooth is None:
                    rec.d_smooth = raw_d
                    rec.d_smooth_prev = raw_d
                else:
                    rec.d_smooth_prev = rec.d_smooth
                    rec.d_smooth = round(0.35 * raw_d + 0.65 * rec.d_smooth, 3)
            det.depth_meters = rec.d_smooth

            # Timestamp-synced kinematics (FR-1.3)
            w = max(det.x2 - det.x1, 1)
            h = max(det.y2 - det.y1, 1)
            s = float(np.sqrt(w * h))

            if active_f_id > rec.last_depth_frame_id:
                dt_depth = active_f_ts - rec.last_depth_timestamp if rec.last_depth_timestamp > 0 else 0.1
                if dt_depth > 0 and rec.d_smooth is not None and rec.d_smooth_prev is not None:
                    rec.v_close = - (rec.d_smooth - rec.d_smooth_prev) / dt_depth

                rec.scale_history.append(s)
                if len(rec.scale_history) > 10:
                    rec.scale_history = rec.scale_history[-10:]

                if len(rec.scale_history) >= 3:
                    s_prev2 = rec.scale_history[-3]
                    rec.r_scale = (s - s_prev2) / max(s_prev2, 1.0)
                elif len(rec.scale_history) == 2:
                    s_prev2 = rec.scale_history[0]
                    rec.r_scale = (s - s_prev2) / max(s_prev2, 1.0)
                else:
                    rec.r_scale = 0.0

                rec.last_depth_frame_id = active_f_id
                rec.last_depth_timestamp = active_f_ts

            det.v_close = rec.v_close
            det.r_scale = rec.r_scale

            # Ego-motion deadband: 2.2 m/s closing speed AND 15% scale expansion
            rec.is_approaching = (rec.v_close >= 2.2) and (rec.r_scale >= 0.15)
            det.is_approaching = rec.is_approaching
            rec.vertical_zone = getattr(det, "vertical_zone", "normal")

            # ── 3-Gate CRITICAL Preemption (FR-1.3) ───────────────────────
            # Must satisfy per-track cooldown (3.0s) AND global critical cooldown (2.5s)
            # to match natural TTS speech duration and prevent queue pileup.
            if (now - rec.last_critical_time) >= 3.0 and (now - self._last_global_critical_time) >= 2.5 and track_age >= 0.2:
                gate_a = (det.direction == "center" and rec.d_smooth is not None and rec.d_smooth <= 1.2)
                if gate_a and not rec.is_approaching:
                    # Suppress repeating stationary proximity reminders if distance hasn't shifted >= 0.25m
                    if rec.last_announced_dist is not None:
                        dist_delta = abs(rec.d_smooth - rec.last_announced_dist)
                        if dist_delta < 0.25 and (now - rec.last_critical_time) < 8.0:
                            gate_a = False

                gate_b = (
                    rec.d_smooth is not None
                    and rec.d_smooth <= 1.8
                    and rec.r_scale >= 0.12
                    and rec.d_smooth_prev is not None
                    and (rec.d_smooth_prev - rec.d_smooth) >= 0.04
                )
                if gate_b and not rec.is_approaching:
                    # Suppress stationary objects from re-triggering Gate B on optical jitter unless distance moves >= 0.25m
                    if rec.last_announced_dist is not None:
                        dist_delta = abs(rec.d_smooth - rec.last_announced_dist)
                        if dist_delta < 0.25 and (now - rec.last_critical_time) < 8.0:
                            gate_b = False

                gate_c = False
                if rec.v_close >= 3.5 and rec.d_smooth is not None:
                    ttc = rec.d_smooth / rec.v_close
                    det.ttc = ttc
                    is_vehicle = det.label in {"car", "motorcycle", "bus", "truck", "bicycle", "train"}
                    dist_max = 8.0 if is_vehicle else 4.0
                    if rec.d_smooth <= dist_max and ttc <= 2.5 and rec.r_scale >= 0.20:
                        gate_c = True

                if gate_c:
                    rec.last_critical_time = now
                    self._last_global_critical_time = now
                    rec.announced = True
                    rec.last_announced_dist = rec.d_smooth
                    self.last_critical_gate = "C"
                    self.last_critical_event = {
                        "event": "CRITICAL",
                        "label": det.label,
                        "dir": det.direction,
                        "dist": round(rec.d_smooth, 1),
                        "ttc": round(ttc, 2) if (rec.v_close > 0) else None,
                        "ts": now,
                    }
                    return self._fmt.format_fast_critical(det.label, det.direction, rec.d_smooth)
                elif gate_a or gate_b:
                    rec.last_critical_time = now
                    self._last_global_critical_time = now
                    rec.announced = True
                    rec.last_announced_dist = rec.d_smooth
                    self.last_critical_gate = "A" if gate_a else "B"
                    ttc_val = round(rec.d_smooth / rec.v_close, 2) if (rec.v_close > 0 and rec.d_smooth is not None) else None
                    self.last_critical_event = {
                        "event": "CRITICAL",
                        "label": det.label,
                        "dir": det.direction,
                        "dist": round(rec.d_smooth, 1) if rec.d_smooth is not None else None,
                        "ttc": ttc_val,
                        "ts": now,
                    }
                    is_app = rec.is_approaching
                    text = self._fmt.format_detection(
                        det.label,
                        det.direction,
                        rec.d_smooth,
                        rec.vertical_zone,
                        is_approaching=is_app,
                    )
                    return f"CRITICAL: {text}"


            # ── Normal Batch Accumulation ─────────────────────────────────
            if not rec.announced and track_age >= self._min_duration:
                group_key = (det.label, det.direction)
                if group_key not in self._pending_items:
                    self._pending_items[group_key] = []
                self._pending_items[group_key].append((rec.d_smooth or 999.0, rec))

        # Purge stale tracks
        expired = [k for k, r in self._tracks.items() if (now - r.last_seen) > self._absence_reset]
        for k in expired:
            del self._tracks[k]

        # ── Periodic Announcement Dispatch with Crowd Compression (FR-3.3)
        if (now - self._last_speak_time) >= self._speak_interval and self._pending_items:
            clauses: list[str] = []

            for (label, direction), items in sorted(self._pending_items.items()):
                count = len(items)
                min_d = min(d for d, _ in items)
                if min_d >= 900.0:
                    min_d = None

                if count >= 2:
                    # Bounded crowd compression
                    clause = self._fmt.format_crowd_clause(label, direction, count, min_d or 2.0)
                    clauses.append(clause)
                else:
                    _, rec = items[0]
                    clause = self._fmt.format_detection(
                        label,
                        direction,
                        min_d,
                        rec.vertical_zone,
                        rec.is_approaching,
                    )
                    clauses.append(clause)

                # Mark tracks as announced
                for _, rec in items:
                    rec.announced = True

            self._pending_items.clear()
            self._last_speak_time = now

            if len(clauses) == 1:
                return self._fmt.format_announcement([clauses[0]])
            elif clauses:
                return self._fmt.format_announcement(clauses)

        return None

