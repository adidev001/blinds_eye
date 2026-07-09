"""
Blind's Eye — Text-to-Speech Module
====================================
High-speed TTS using native Windows COM (SAPI.SpVoice).
This completely eliminates the ~1-second lag caused by spawning 
a pyttsx3 subprocess and runs instantly in a background thread.
"""

import sys
import threading
import queue

class TTSEngine:
    """
    Ultra-low latency text-to-speech engine using native Windows SAPI.
    
    Runs a background daemon thread that continuously pulls messages
    from a bounded queue.
    """

    def __init__(self, speech_rate: int = 170, queue_max: int = 2):
        self._speech_rate = speech_rate
        self._queue_max = queue_max
        self._queue = queue.Queue()
        self._running = True

        self._worker = threading.Thread(
            target=self._speech_loop,
            daemon=True,
            name="tts-worker",
        )
        self._worker.start()

    def speak(self, message: str) -> bool:
        """Enqueue a message for speech output."""
        if not self._running:
            return False
            
        if self._queue.qsize() >= self._queue_max:
            print("[TTS] Queue full — dropping message:", message)
            return False
            
        self._queue.put(message)
        return True

    def shutdown(self) -> None:
        """Signal the worker to stop and wait for it to drain."""
        self._running = False
        self._queue.put("__STOP__")
        self._worker.join(timeout=3.0)
        print("[TTS] Engine shut down.")

    def _speech_loop(self) -> None:
        """Background loop — processes the queue using native win32com."""
        speaker = None
        
        # Initialize COM in the worker thread
        if sys.platform == "win32":
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                # SAPI rate is roughly -10 to +10. 
                # speech_rate 170 -> ~0
                rate_adj = max(-10, min(10, (self._speech_rate - 170) // 10))
                speaker.Rate = rate_adj
            except Exception as e:
                print(f"[TTS] Failed to initialize native Windows TTS: {e}")

        while True:
            try:
                msg = self._queue.get(timeout=1.0)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if msg == "__STOP__":
                self._queue.task_done()
                break

            try:
                if speaker:
                    # '1' = SVSFlagsAsync. 
                    # We wait synchronously in the background thread though.
                    speaker.Speak(msg, 0)
                else:
                    # Fallback if not on Windows (should not be reached)
                    import pyttsx3
                    e = pyttsx3.init()
                    e.say(msg)
                    e.runAndWait()
            except Exception as exc:
                print(f"[TTS ERROR] {exc}")
            finally:
                self._queue.task_done()
