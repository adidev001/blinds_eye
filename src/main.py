"""
Blind's Eye — Application Entry Point & Orchestrator
=====================================================
Initializes the hardware probe, loads tier-appropriate models,
starts the camera, and runs the real-time vision + TTS loop.

Per PRD §6 — this is the top-level ``src/main.py`` that replaces
the monolithic ``real_time_object_detection.py``.

Usage
-----
    conda activate blinded_bythe_beauty
    python -m src.main
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from src.engine import HardwareProbe, ModelLoader
from src.tts_module import TTSEngine
import os
from src.vision import VisionPipeline, AnnouncementTracker
from src.camera import CameraStream


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="blinds-eye",
        description="Blind's Eye — AI-powered real-time spatial awareness assistant",
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
        "--tts-model", type=str, default=None,
        help="Path to a Piper TTS .onnx voice model (auto-detected from models/tts/ if omitted)",
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
    print("  Blind's Eye — Starting Up")
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

    # ---- Step 3: Initialize TTS (Piper offline backend) -----------------
    # Auto-discover a Piper voice model under models/tts/ if not specified.
    from pathlib import Path
    tts_model_path = None
    if args.tts_model:
        tts_model_path = Path(args.tts_model)
    else:
        tts_dir = Path("models") / "tts"
        if tts_dir.exists():
            candidates = list(tts_dir.glob("*.onnx"))
            if candidates:
                tts_model_path = candidates[0]   # pick first available voice
                print(f"[INIT] Auto-detected Piper voice: {tts_model_path.name}")

    tts = TTSEngine(
        model_path=tts_model_path,
        speech_rate_scale=args.speech_rate,
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
    )
    tracker = AnnouncementTracker(
        absence_reset=args.absence_reset,
        speak_interval=args.speak_interval,
    )
    print("[INIT] Vision pipeline ready.")
    print()

    # ---- Step 5: Open Camera -----------------------------------------
    cam_src = int(args.camera) if args.camera.isdigit() else args.camera
    print(f"[CAMERA] Opening device {cam_src} ...")
    cap = CameraStream(cam_src)
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
    
    # Detect if we have a display available (for Docker / Headless Linux)
    headless = False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        headless = True
        print("[INFO] No DISPLAY detected. Running in headless mode (no video window).")
    
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Process frame (detection + depth in parallel)
            result = pipeline.process_frame(frame)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Check for announcements
            msg = tracker.update(result.detections)
            if msg:
                print(f"[SPEAK] {msg}")
                tts.speak(msg)

            # Draw and display
            display_frame = pipeline.draw_overlays(result)
            
            # Draw FPS
            cv2.putText(
                display_frame, f"FPS: {int(fps)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
            )

            if not headless:
                cv2.imshow(window_name, display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Exit requested (q pressed).")
                    break
            else:
                # Slight yield for headless loop
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received.")

    finally:
        # Graceful shutdown
        print("[SHUTDOWN] Releasing resources ...")
        tts.speak("Shutting down. Goodbye.")
        time.sleep(1)  # Give TTS time to finish the goodbye
        tts.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Done.")


if __name__ == "__main__":
    main()
