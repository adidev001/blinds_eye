"""
Blind's Eye — Telemetry, Haptics & Context Orchestration
=========================================================
Implements:
- Fire-and-forget non-blocking UDP haptic emitter (FR-5.3)
- Lockless in-memory flight recorder and session summary (FR-5.4)
- 15-second rolling window auto-context hysteresis manager (FR-5.2)
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
import socket
import time
from typing import Any, Optional

import numpy as np

from src.vision import Detection, INDOOR_CLASSES, OUTDOOR_CLASSES


# ======================================================================
# FR-5.2: Auto Context Hysteresis Manager
# ======================================================================

class AutoContextManager:
    """
    Tracks high-confidence (>= 0.65) detections over a 15-second rolling window
    and switches between 'indoor' and 'outdoor' only when >= 5 target-mode anchor
    detections occur with 0 opposing-mode detections (FR-5.2).
    """

    def __init__(
        self,
        window_sec: float = 15.0,
        min_anchors: int = 5,
        min_conf: float = 0.65,
    ) -> None:
        self.window_sec = window_sec
        self.min_anchors = min_anchors
        self.min_conf = min_conf
        self._history: collections.deque[tuple[float, str]] = collections.deque()
        self.indoor_anchors: set[str] = INDOOR_CLASSES - OUTDOOR_CLASSES
        self.outdoor_anchors: set[str] = OUTDOOR_CLASSES - INDOOR_CLASSES
        self.current_mode: str = "all"

    def update(self, detections: list[Detection], now: Optional[float] = None) -> Optional[str]:
        """
        Feed current detections. Returns new mode string if a context switch
        occurred, otherwise None.
        """
        if now is None:
            now = time.monotonic()

        # Append high-confidence detections
        for det in detections:
            if det.confidence >= self.min_conf:
                self._history.append((now, det.label))

        # Evict samples outside the 15-second rolling window
        cutoff = now - self.window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # Count anchor detections
        indoor_count = sum(1 for _, label in self._history if label in self.indoor_anchors)
        outdoor_count = sum(1 for _, label in self._history if label in self.outdoor_anchors)

        # Hysteresis transition logic
        if indoor_count >= self.min_anchors and outdoor_count == 0:
            if self.current_mode != "indoor":
                self.current_mode = "indoor"
                return "indoor"
        elif outdoor_count >= self.min_anchors and indoor_count == 0:
            if self.current_mode != "outdoor":
                self.current_mode = "outdoor"
                return "outdoor"

        return None


# ======================================================================
# FR-5.3: Fire-and-Forget Non-Blocking UDP Haptic Emitter
# ======================================================================

class UdpHapticEmitter:
    """
    Fire-and-forget non-blocking UDP socket emitter (FR-5.3).
    Transmits JSON alerts for CRITICAL hazards with zero blocking risk.
    """

    def __init__(self, target: Optional[str] = None) -> None:
        self.target = target
        self.host: Optional[str] = None
        self.port: Optional[int] = None
        self._sock: Optional[socket.socket] = None

        if target:
            try:
                parts = target.split(":")
                self.host = parts[0]
                self.port = int(parts[1])
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.setblocking(False)
            except Exception as e:
                print(f"[HAPTIC] Could not initialize UDP socket on {target}: {e}")
                self._sock = None

    def send(self, payload: dict[str, Any]) -> bool:
        """
        Dispatch JSON payload non-blockingly (<0.05ms execution).
        Returns True if sent without exception, False otherwise.
        """
        if self._sock is None or not self.host or not self.port:
            return False

        try:
            data = json.dumps(payload).encode("utf-8")
            self._sock.sendto(data, (self.host, self.port))
            return True
        except (BlockingIOError, OSError):
            return False
        except Exception:
            return False

    def close(self) -> None:
        """Close the underlying UDP socket."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


# ======================================================================
# FR-5.4: Lockless In-Memory Flight Recorder
# ======================================================================

class TelemetryRecorder:
    """
    Lockless in-memory flight recorder (FR-5.4).
    Appends telemetry metrics to collections.deque(maxlen=5000) inside
    the frame loop with zero disk I/O. Flushes to JSONL and generates
    Markdown session summary on shutdown.
    """

    def __init__(self, maxlen: int = 5000) -> None:
        self.records: collections.deque[dict[str, Any]] = collections.deque(maxlen=maxlen)

    def record(
        self,
        yolo_ms: float,
        depth_ms: float,
        tts_ms: float = 0.0,
        cache_hit: bool = False,
        active_tracks: int = 0,
        announcement: Optional[str] = None,
        critical_gate: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Append a frame telemetry record to the in-memory deque."""
        if timestamp is None:
            timestamp = time.time()

        self.records.append({
            "timestamp": timestamp,
            "yolo_ms": round(yolo_ms, 2),
            "depth_ms": round(depth_ms, 2),
            "tts_ms": round(tts_ms, 2),
            "cache_hit": cache_hit,
            "active_tracks": active_tracks,
            "announcement": announcement,
            "critical_gate": critical_gate,
        })

    def flush_jsonl(self, path: str | Path) -> int:
        """
        Write all accumulated records in the deque to a .jsonl file.
        Returns the number of records written.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(p, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec) + "\n")
                count += 1
        return count

    def generate_summary_markdown(self, tts_stats: Optional[dict] = None) -> str:
        """
        Generate a GitHub-flavored Markdown telemetry table summarizing
        p50/p95 latencies, announcement counts, and cache performance.
        """
        total_frames = len(self.records)
        if total_frames == 0:
            return (
                "### Blind's Eye Session Telemetry Summary\n\n"
                "*No frame telemetry records captured during this session.*\n"
            )

        yolo_times = [r["yolo_ms"] for r in self.records]
        depth_times = [r["depth_ms"] for r in self.records if r["depth_ms"] > 0]
        frame_times = [r["yolo_ms"] + r["depth_ms"] for r in self.records]

        yolo_p50 = float(np.percentile(yolo_times, 50))
        yolo_p95 = float(np.percentile(yolo_times, 95))

        depth_p50 = float(np.percentile(depth_times, 50)) if depth_times else 0.0
        depth_p95 = float(np.percentile(depth_times, 95)) if depth_times else 0.0

        frame_p50 = float(np.percentile(frame_times, 50))
        frame_p95 = float(np.percentile(frame_times, 95))

        total_announcements = sum(1 for r in self.records if r["announcement"] is not None)
        critical_preemptions = sum(
            1 for r in self.records 
            if r["critical_gate"] is not None or (r["announcement"] and r["announcement"].startswith("CRITICAL:"))
        )

        if tts_stats and "hit_ratio" in tts_stats:
            hit_ratio_str = f"{tts_stats['hit_ratio'] * 100.0:.1f}% ({tts_stats['hits']}/{tts_stats['hits'] + tts_stats['misses']})"
        else:
            cache_hits = sum(1 for r in self.records if r["cache_hit"])
            hit_ratio_str = f"{cache_hits / max(total_announcements, 1) * 100.0:.1f}%"

        md = [
            "### Blind's Eye Session Telemetry Summary",
            "| Metric | Value |",
            "| --- | --- |",
            f"| **Total Processed Frames** | {total_frames} |",
            f"| **YOLO Latency (p50 / p95)** | {yolo_p50:.1f} ms / {yolo_p95:.1f} ms |",
            f"| **Depth Latency (p50 / p95)** | {depth_p50:.1f} ms / {depth_p95:.1f} ms |",
            f"| **Frame Processing Latency (p50 / p95)** | {frame_p50:.1f} ms / {frame_p95:.1f} ms |",
            f"| **TTS Clause Cache Hit Ratio** | {hit_ratio_str} |",
            f"| **Total Dispatched Announcements** | {total_announcements} |",
            f"| **Critical Hazard Preemptions** | {critical_preemptions} |",
        ]
        return "\n".join(md) + "\n"

    def print_summary(self, tts_stats: Optional[dict] = None) -> None:
        """Print the Markdown summary table to stdout."""
        print()
        print(self.generate_summary_markdown(tts_stats=tts_stats))
