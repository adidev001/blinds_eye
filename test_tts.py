"""Quick smoke test for the Piper TTS module."""
import time
from src.tts_module import TTSEngine

print("--- Smoke Test: Piper TTS ---")
tts = TTSEngine(model_path="models/tts/en_US-amy-low.onnx")
tts.speak("Hello. Piper TTS is working correctly.")
time.sleep(4)

print("--- Testing CRITICAL preemption ---")
tts.speak("Normal message one.")
tts.speak("Normal message two.")
tts.speak("CRITICAL: Stairs directly ahead!")
time.sleep(6)

tts.shutdown()
print("--- Smoke test complete ---")
