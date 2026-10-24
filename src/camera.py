import cv2
import threading
import time


class CameraStream:
    """
    Asynchronously reads frames from cv2.VideoCapture in a background thread.
    Features:
    - 2000ms open and read timeouts (FR-2.1)
    - Non-blocking exponential backoff auto-reconnect loop (0.5s, 1.0s, 2.0s)
    - Frame staleness watchdog (FR-2.2) via is_stale()
    - Zero-buffer backlog on high-latency IP camera streams
    """

    def __init__(self, src=0):
        self.src = src
        self.running = True
        self.ret = False
        self.frame = None
        self.last_frame_time = time.monotonic()
        self._lock = threading.Lock()

        self.cap = self._open_capture()

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.ret = True
                self.frame = frame
                self.last_frame_time = time.monotonic()

        self.thread = threading.Thread(
            target=self._update,
            daemon=True,
            name="camera-stream",
        )
        self.thread.start()

    def _open_capture(self) -> cv2.VideoCapture:
        """Initialize VideoCapture and configure non-blocking socket timeouts (FR-2.1)."""
        cap = cv2.VideoCapture(self.src)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
        return cap

    def _update(self):
        reconnect_delay = 0.5
        while self.running:
            if self.cap.isOpened():
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        frame_copy = frame.copy()
                        with self._lock:
                            self.ret = True
                            self.frame = frame_copy
                            self.last_frame_time = time.monotonic()
                        reconnect_delay = 0.5  # reset backoff on success
                    else:
                        with self._lock:
                            self.ret = False
                        self._attempt_reconnect(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2.0, 2.0)
                except Exception as e:
                    print(f"[CAMERA] Stream read error: {e}")
                    with self._lock:
                        self.ret = False
                    self._attempt_reconnect(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2.0, 2.0)
            else:
                self._attempt_reconnect(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2.0, 2.0)

            time.sleep(0.01)

    def _attempt_reconnect(self, delay: float):
        """Execute non-blocking backoff reconnect without hanging the caller."""
        if not self.running:
            return
        time.sleep(delay)
        if not self.running:
            return
        try:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = self._open_capture()
        except Exception as e:
            print(f"[CAMERA] Reconnect attempt failed: {e}")

    def read(self):
        with self._lock:
            return self.ret, self.frame

    def is_stale(self, threshold_sec: float = 1.5) -> bool:
        """Return True if no fresh frame has been decoded within threshold_sec (FR-2.2)."""
        with self._lock:
            return (time.monotonic() - self.last_frame_time) > threshold_sec

    def isOpened(self):
        return self.cap.isOpened() and self.running

    def release(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
