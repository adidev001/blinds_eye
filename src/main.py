"""
Blind's Eye — Application Entry Point & Orchestrator (v2.2)
============================================================
Initializes the hardware probe, loads tier-appropriate models,
starts the camera with non-blocking timeouts & staleness watchdog,
and runs the real-time vision + TTS + haptics + telemetry loop.

Usage
-----
    conda activate blinded_bythe_beauty
    python -m src.main
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path for direct script execution
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Callable, Optional
import cv2
import numpy as np

from src.engine import HardwareProbe, ModelLoader
from src.tts_module import TTSEngine
from src.vision import VisionPipeline, AnnouncementTracker
from src.camera import CameraStream
from src.telemetry import AutoContextManager, UdpHapticEmitter, TelemetryRecorder


# ======================================================================
# Interactive In-Window Clickable HUD Control Dock
# ======================================================================

class HudControlDock:
    """
    Interactive in-window clickable control dock for Blind's Eye.
    Renders styled section buttons at the bottom of the video window
    and registers mouse clicks to toggle modes, audio, depth, and exit.
    """

    def __init__(
        self,
        pipeline: VisionPipeline,
        tts: TTSEngine,
        tracker: AnnouncementTracker,
        auto_context: Optional[AutoContextManager] = None,
    ) -> None:
        self.pipeline = pipeline
        self.tts = tts
        self.tracker = tracker
        self.auto_context = auto_context
        self.auto_context_enabled = auto_context is not None
        self.exit_requested = False

        self._buttons = [
            # Row 0: Mode Section & Perception
            {
                "id": "mode_all", "row": 0, "label": lambda: "ALL",
                "active": lambda: (not self.auto_context_enabled and self.pipeline.mode == "all"),
                "on_click": self._set_all,
            },
            {
                "id": "mode_in", "row": 0, "label": lambda: "INDOOR",
                "active": lambda: (not self.auto_context_enabled and self.pipeline.mode == "indoor"),
                "on_click": self._set_indoor,
            },
            {
                "id": "mode_out", "row": 0, "label": lambda: "OUTDOOR",
                "active": lambda: (not self.auto_context_enabled and self.pipeline.mode == "outdoor"),
                "on_click": self._set_outdoor,
            },
            {
                "id": "mode_auto", "row": 0, "label": lambda: f"AUTO:{'ON' if self.auto_context_enabled else 'OFF'}",
                "active": lambda: self.auto_context_enabled,
                "on_click": self._toggle_auto,
            },
            {
                "id": "depth", "row": 0, "label": lambda: f"DEPTH:{'ON' if self.pipeline.depth_enabled else 'OFF'}",
                "active": lambda: self.pipeline.depth_enabled,
                "on_click": self.pipeline.toggle_depth,
            },
            # Row 1: Audio Section & System
            {
                "id": "crit_snd", "row": 1, "label": lambda: f"CRIT SND:{'ON' if self.tts.critical_sound else 'OFF'}",
                "active": lambda: self.tts.critical_sound,
                "on_click": self.tts.toggle_critical_sound,
            },
            {
                "id": "stereo", "row": 1, "label": lambda: f"STEREO:{'ON' if self.tts.stereo else 'OFF'}",
                "active": lambda: self.tts.stereo,
                "on_click": self.tts.toggle_stereo,
            },
            {
                "id": "lang", "row": 1, "label": lambda: f"LANG:{self.tts.lang.upper()}",
                "active": lambda: True,
                "on_click": self._cycle_lang,
            },
            {
                "id": "exit", "row": 1, "label": lambda: "EXIT",
                "active": lambda: False,
                "on_click": self._req_exit,
                "bg": (40, 40, 180),
            },
        ]
        self._layout_rects: dict[str, tuple[int, int, int, int]] = {}

    def _set_all(self) -> None:
        self.auto_context_enabled = False
        self.pipeline.set_mode("all")
        print("[USER DOCK] Mode switched to: ALL")

    def _set_indoor(self) -> None:
        self.auto_context_enabled = False
        self.pipeline.set_mode("indoor")
        print("[USER DOCK] Mode switched to: INDOOR")

    def _set_outdoor(self) -> None:
        self.auto_context_enabled = False
        self.pipeline.set_mode("outdoor")
        print("[USER DOCK] Mode switched to: OUTDOOR")

    def _toggle_auto(self) -> None:
        self.auto_context_enabled = not self.auto_context_enabled
        if not self.auto_context_enabled:
            self.pipeline.set_mode("all")
        print(f"[USER DOCK] Auto context: {'ENABLED' if self.auto_context_enabled else 'DISABLED'}")

    def _cycle_lang(self) -> None:
        new_lang = self.tts.cycle_lang()
        self.tracker.set_lang(new_lang)

    def _req_exit(self) -> None:
        print("[USER DOCK] Exit clicked.")
        self.exit_requested = True

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for b_id, rect in self._layout_rects.items():
                x1, y1, x2, y2 = rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    for b in self._buttons:
                        if b["id"] == b_id:
                            b["on_click"]()
                            return

    def render(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        dock_h = 74
        y_top = max(0, h - dock_h)

        # Semi-transparent dock backdrop
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_top), (w, h), (18, 18, 22), -1)
        cv2.line(overlay, (0, y_top), (w, y_top), (60, 60, 70), 1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        row0_btns = [b for b in self._buttons if b["row"] == 0]
        row1_btns = [b for b in self._buttons if b["row"] == 1]

        for row_idx, row_btns in enumerate([row0_btns, row1_btns]):
            n = len(row_btns)
            if n == 0:
                continue
            pad_x = 6
            row_y1 = y_top + 5 + row_idx * 33
            row_y2 = row_y1 + 28
            btn_w = (w - pad_x * (n + 1)) // n

            for i, btn in enumerate(row_btns):
                bx1 = pad_x + i * (btn_w + pad_x)
                bx2 = bx1 + btn_w
                self._layout_rects[btn["id"]] = (bx1, row_y1, bx2, row_y2)

                is_active = btn["active"]()
                label = btn["label"]()

                if "bg" in btn:
                    bg_color = btn["bg"]
                    txt_color = (255, 255, 255)
                    border_color = (80, 80, 220)
                elif is_active:
                    bg_color = (35, 120, 35)       # Rich vibrant green
                    txt_color = (255, 255, 255)
                    border_color = (60, 220, 60)
                else:
                    bg_color = (38, 38, 44)        # Dark slate gray
                    txt_color = (190, 190, 190)
                    border_color = (65, 65, 75)

                cv2.rectangle(frame, (bx1, row_y1), (bx2, row_y2), bg_color, -1)
                cv2.rectangle(frame, (bx1, row_y1), (bx2, row_y2), border_color, 1)

                font_scale = 0.38 if btn_w < 110 else 0.42
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                tx = bx1 + max(2, (btn_w - tw) // 2)
                ty = row_y1 + ((row_y2 - row_y1) + th) // 2
                cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, 1, cv2.LINE_AA)

        return frame


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="blinds-eye",
        description="Blind's Eye — AI-powered real-time spatial awareness assistant (v2.2)",
    )
    ap.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum detection confidence (default: 0.5)",
    )
    ap.add_argument(
        "--frame-width", type=int, default=640,
        help="Resize frames to this width (default: 640)",
    )
    ap.add_argument(
        "--speech-rate", type=float, default=1.0,
        help="TTS speech speed multiplier: 1.0=normal, 1.3=faster, 0.8=slower (default: 1.0)",
    )
    ap.add_argument(
        "--tts-engine", type=str, choices=["piper", "kokoro"], default="piper",
        help="TTS synthesis engine: 'piper' (default) or 'kokoro' (Kokoro-82M ONNX)",
    )
    ap.add_argument(
        "--lang", type=str, choices=["en", "hi", "hinglish"], default="en",
        help="Announcement language: 'en' (default), 'hi' (Hindi), 'hinglish' (Roman Hinglish)",
    )
    ap.add_argument(
        "--tts-model", type=str, default=None,
        help="Path to a Piper TTS .onnx voice model (auto-detected from models/tts/ if omitted)",
    )
    ap.add_argument(
        "--mode", type=str, choices=["all", "indoor", "outdoor", "auto"], default="all",
        help="Context-aware filtering mode: all, indoor, outdoor, or auto (default: all)",
    )
    ap.add_argument(
        "--stereo", action=argparse.BooleanOptionalAction, default=True,
        help="Enable safe 100/35 stereo panning or force 1-channel mono (default: --stereo)",
    )
    ap.add_argument(
        "--depth-res", type=int, default=350,
        help="Dynamic DA-V2 input resolution (multiple of 14; auto-locks if static) (default: 350)",
    )
    ap.add_argument(
        "--dynamic-power", action="store_true", default=False,
        help="Scale background depth thread to 2 FPS during static scenes",
    )
    ap.add_argument(
        "--headless", action="store_true", default=False,
        help="Disable OpenCV GUI rendering (cv2.imshow)",
    )
    ap.add_argument(
        "--heartbeat", action="store_true", default=False,
        help="Emit 15ms 800Hz bone-conduction tick after 15s of silence",
    )
    ap.add_argument(
        "--haptic-udp", type=str, default=None,
        help="Fire-and-forget UDP target HOST:PORT for critical haptics",
    )
    ap.add_argument(
        "--telemetry-log", type=str, default=None,
        help="Path to write session .jsonl metrics on exit",
    )
    ap.add_argument(
        "--critical-sound", action=argparse.BooleanOptionalAction, default=True,
        help="Enable/disable critical hazard earcon chimes and urgent prefix (default: --critical-sound; use --no-critical-sound for calm mode)",
    )
    ap.add_argument(
        "--speak-interval", type=float, default=2.0,
        help="Seconds between speech announcements (default: 2.0)",
    )
    ap.add_argument(
        "--absence-reset", type=float, default=1.5,
        help="Seconds before a disappeared object can be re-announced (default: 1.5)",
    )
    ap.add_argument(
        "--min-duration", type=float, default=0.5,
        help="Seconds an object must be continuously visible before announcing (default: 0.5)",
    )
    ap.add_argument(
        "--max-fps", type=float, default=10.0,
        help="Cap inference to a max FPS to prevent AI from backing up the pipeline (default: 10.0)",
    )
    ap.add_argument(
        "--camera", type=str, default="0",
        help="Camera device index (e.g. 0) or IP camera stream URL",
    )
    ap.add_argument(
        "--depth-scale", type=float, default=3.0,
        help="Scale factor to convert relative disparity into rough metric depth (default: 3.0)",
    )
    ap.add_argument(
        "--no-depth", action="store_true",
        help="Disable depth estimation even if weights are present",
    )
    return ap.parse_args()


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()

    # ---- Step 1: Hardware Probe --------------------------------------
    print("=" * 60)
    print("  Blind's Eye — Starting Up (v2.2)")
    print("=" * 60)

    probe = HardwareProbe()
    profile = probe.profile
    hw = probe.detect()
    print(f"[INIT] Hardware tier : {hw['tier']}")
    print(f"[INIT] Platform      : {hw['platform']}")
    print(f"[INIT] CPU           : {hw['cpu']}")
    print(f"[INIT] CUDA          : {hw['cuda']}")
    if hw["gpu"]:
        print(f"[INIT] GPU           : {hw['gpu']} ({hw['vram_gb']} GB)")
    print(f"[INIT] ONNX providers: {hw['onnx_providers']}")
    print()

    # ---- Step 2: Load Models -----------------------------------------
    loader = ModelLoader(profile)

    print("[INIT] Loading YOLO model ...")
    yolo_model = loader.load_yolo()
    print(f"[INIT] YOLO ready: {yolo_model.model_name}")

    depth_session = None
    if not args.no_depth:
        print("[INIT] Loading Depth Anything V2 ...")
        depth_session = loader.load_depth()
        if depth_session is None:
            print("[INIT] Depth model not available — running detection-only mode.")
        else:
            print("[INIT] Depth model ready.")
    else:
        print("[INIT] Depth estimation disabled via --no-depth flag.")
    print()

    # ---- Step 3: Initialize TTS (Dual-Engine: Piper + Kokoro) ---------
    tts_model_path = None
    if args.tts_model:
        tts_model_path = Path(args.tts_model)
    elif args.tts_engine == "piper":
        tts_dir = Path("models") / "tts"
        if tts_dir.exists():
            # If Hindi language requested, prefer hi_IN model if present and configured
            if args.lang in ("hi", "hinglish"):
                hi_candidates = list(tts_dir.glob("hi_IN*.onnx"))
                for c in hi_candidates:
                    if c.with_suffix(".onnx.json").exists():
                        tts_model_path = c
                        break
            if not tts_model_path:
                candidates = [p for p in tts_dir.glob("*.onnx") if not p.name.startswith("kokoro")]
                if candidates:
                    tts_model_path = candidates[0]   # pick first available voice
            if tts_model_path:
                print(f"[INIT] Auto-detected Piper voice: {tts_model_path.name}")

    tts = TTSEngine(
        model_path=tts_model_path,
        engine=args.tts_engine,
        lang=args.lang,
        speech_rate_scale=args.speech_rate,
        stereo=args.stereo,
        heartbeat=args.heartbeat,
        critical_sound=args.critical_sound,
    )
    tts.speak("Blind's Eye is starting up.")
    print("[INIT] TTS engine started.")

    # ---- Step 4: Initialize Vision Pipeline --------------------------
    pipeline = VisionPipeline(
        yolo_model=yolo_model,
        depth_session=depth_session,
        confidence=args.confidence,
        frame_width=args.frame_width,
        depth_scale=args.depth_scale,
        mode=args.mode if args.mode != "auto" else "all",
        depth_res=args.depth_res,
        dynamic_power=args.dynamic_power,
    )
    tracker = AnnouncementTracker(
        absence_reset=args.absence_reset,
        speak_interval=args.speak_interval,
        min_duration=args.min_duration,
        lang=args.lang,
    )
    auto_context = AutoContextManager() if args.mode == "auto" else None
    haptic = UdpHapticEmitter(args.haptic_udp)
    telemetry = TelemetryRecorder()
    print("[INIT] Vision pipeline ready.")
    print()

    # ---- Step 5: Open Camera -----------------------------------------
    cam_src = int(args.camera) if args.camera.isdigit() else args.camera
    print(f"[CAMERA] Opening device {cam_src} ...")
    cap = CameraStream(cam_src)
    tts.set_staleness_fn(lambda: cap.is_stale(1.5))
    time.sleep(1.5)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check camera permissions.")
        tts.speak("Error: camera not available.")
        tts.shutdown()
        sys.exit(1)

    print("[CAMERA] Camera initialized successfully.")
    tts.speak("Camera ready. Beginning detection.")
    print()

    # ---- Step 6: Main Loop -------------------------------------------
    window_name = "Blind's Eye"
    
    # Headless execution detection (FR-5.1)
    headless = args.headless
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        headless = True
        print("[INFO] No DISPLAY detected. Running in headless mode (no video window).")

    dock = HudControlDock(pipeline, tts, tracker, auto_context)
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, dock.on_mouse)

    prev_time = time.time()
    min_frame_time = 1.0 / args.max_fps if args.max_fps > 0 else 0
    camera_stale_alerted = False

    try:
        while True:
            frame_start = time.time()

            # Staleness watchdog (FR-2.2)
            if cap.is_stale(threshold_sec=1.5):
                if not camera_stale_alerted:
                    camera_stale_alerted = True
                    stale_msg = tracker._fmt.format_system_alert("camera_lost")
                    tts.speak(stale_msg)
                time.sleep(0.05)
                continue
            elif camera_stale_alerted:
                camera_stale_alerted = False
                restored_msg = tracker._fmt.format_system_alert("camera_restored")
                tts.speak(restored_msg)

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Process frame (detection + depth in parallel)
            result = pipeline.process_frame(frame)
            
            # Context-aware auto hysteresis (FR-5.2)
            if dock.auto_context_enabled and auto_context:
                switched = auto_context.update(result.detections)
                if switched:
                    pipeline.set_mode(switched)
                    print(f"[MODE] Auto-context switched mode to: {switched}")

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Check for announcements
            msg = tracker.update(
                result.detections,
                depth_frame_id=result.depth_frame_id,
                depth_timestamp=result.depth_timestamp,
                depth_curtain_meters=result.curtain_meters,
                low_light=result.low_light_detected,
            )
            if msg:
                print(f"[SPEAK] {msg}")
                tts.speak(msg)
                if tracker.last_critical_event and args.haptic_udp:
                    haptic.send(tracker.last_critical_event)
            else:
                # If no announcement in this frame, purge queued speech for departed obstacles
                current_labels = {d.label for d in result.detections}
                if "person" not in current_labels:
                    tts.purge_label("person")
                if len(result.detections) == 0:
                    tts.clear_queue()

            # Record telemetry in-memory (FR-5.4)
            telemetry.record(
                yolo_ms=result.yolo_ms,
                depth_ms=result.depth_ms,
                active_tracks=tracker.active_track_count,
                announcement=msg,
                critical_gate=tracker.last_critical_gate,
            )

            # Draw and display (FR-5.1)
            if not headless:
                display_frame = pipeline.draw_overlays(result)
                display_frame = dock.render(display_frame)
                cv2.putText(
                    display_frame, f"FPS: {int(fps)}", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27) or dock.exit_requested:  # 'q', Esc, or dock [EXIT]
                    print("[INFO] Exit requested.")
                    break
                elif key in (ord("c"), ord("C")):
                    new_val = tts.toggle_critical_sound()
                    print(f"[USER] Critical sound toggled: {'ON' if new_val else 'OFF'}")

                # Cleanly exit if the user clicked the 'X' close button on the window
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("[INFO] Exit requested (window closed).")
                        break
                except Exception:
                    break
            else:
                # Slight yield for headless loop
                time.sleep(0.001)

            # Smart FPS cap: sleep only the remaining time to hit target FPS
            elapsed = time.time() - frame_start
            if min_frame_time > 0 and elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received.")
    except Exception as exc:
        print(f"\n[ERROR] Pipeline exception in main loop: {exc}")
        import traceback
        traceback.print_exc()

    finally:
        # Graceful shutdown & Telemetry flush (FR-5.3, FR-5.4)
        print("[SHUTDOWN] Releasing resources ...")
        tts_cache_stats = tts.get_cache_stats()

        if args.telemetry_log:
            print(f"[TELEMETRY] Flushing flight recorder log to: {args.telemetry_log}")
            flushed = telemetry.flush_jsonl(args.telemetry_log)
            print(f"[TELEMETRY] Flushed {flushed} entries to {args.telemetry_log}")
            telemetry.print_summary(tts_stats=tts_cache_stats)
        elif len(telemetry.records) > 0:
            telemetry.print_summary(tts_stats=tts_cache_stats)

        haptic.close()
        tts.speak("Shutting down. Goodbye.")
        time.sleep(1)  # Give TTS time to finish the goodbye
        tts.shutdown()
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print("[SHUTDOWN] Done.")


if __name__ == "__main__":
    main()
