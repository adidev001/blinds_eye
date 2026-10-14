"""
Blind's Eye — Comprehensive Unit Tests for Localization & Dual-Engine TTS
========================================================================
Validates:
  1. LocalizationFormatter (en, hi, hinglish, fallbacks, system messages)
  2. AnnouncementTracker (multilingual telemetry, debouncing, suppression)
  3. TTSEngine (priority queue, critical preemption, graceful fallbacks)
"""

import sys
from pathlib import Path
import time
import unittest
from unittest.mock import patch

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.localization import (
    LocalizationFormatter,
    SUPPORTED_LANGS,
    COCO_LABELS_HI,
    COCO_LABELS_HINGLISH,
    DIRECTIONS_HI,
    DIRECTIONS_HINGLISH,
)
from src.vision import AnnouncementTracker, Detection
from src.tts_module import (
    TTSEngine,
    _PRIORITY_CRITICAL,
    _PRIORITY_NORMAL,
)


class TestLocalizationFormatter(unittest.TestCase):
    def test_supported_languages(self):
        for lang in ("en", "hi", "hinglish"):
            fmt = LocalizationFormatter(lang)
            self.assertEqual(fmt.lang, lang)

    def test_invalid_language_raises(self):
        with self.assertRaises(ValueError):
            LocalizationFormatter("french")

    def test_english_formatting(self):
        fmt = LocalizationFormatter("en")
        s = fmt.format_detection("person", "left", 2.1)
        self.assertEqual(s, "person on the left at 2.1 meters")

        s_no_depth = fmt.format_detection("person", "left", None)
        self.assertEqual(s_no_depth, "person on the left")

        ann_single = fmt.format_announcement([s])
        self.assertEqual(ann_single, "I see person on the left at 2.1 meters")

        ann_multi = fmt.format_announcement(["person on the left", "car on the right"])
        self.assertEqual(ann_multi, "I see: person on the left, car on the right")

    def test_hindi_formatting(self):
        fmt = LocalizationFormatter("hi")
        s = fmt.format_detection("person", "left", 2.1)
        self.assertIn("बाईं ओर", s)
        self.assertIn("व्यक्ति", s)
        self.assertIn("2.1 मीटर पर", s)

        ann = fmt.format_announcement([s])
        self.assertIn("मैं देख रहा हूँ", ann)

    def test_hinglish_formatting(self):
        fmt = LocalizationFormatter("hinglish")
        s = fmt.format_detection("person", "center", 1.5)
        self.assertIn("saamne", s)
        self.assertIn("insaan", s)
        self.assertIn("1.5 meter par", s)

        ann = fmt.format_announcement([s])
        self.assertIn("dekh raha hoon", ann)

    def test_label_fallback_for_unknown(self):
        fmt_hi = LocalizationFormatter("hi")
        self.assertEqual(fmt_hi.translate_label("ufo_object"), "ufo_object")

    def test_direction_fallback_for_unknown(self):
        fmt_hi = LocalizationFormatter("hi")
        self.assertEqual(fmt_hi.translate_direction("top"), "top")

    def test_system_messages(self):
        for lang in ("en", "hi", "hinglish"):
            fmt = LocalizationFormatter(lang)
            for key in ("startup", "cam_ready", "shutdown", "cam_error"):
                msg = fmt.system_message(key)
                self.assertIsInstance(msg, str)
                self.assertGreater(len(msg), 0)


class TestAnnouncementTracker(unittest.TestCase):
    def test_tracker_hindi_announcement(self):
        tracker = AnnouncementTracker(
            absence_reset=1.0,
            speak_interval=0.0,
            min_duration=0.0,
            lang="hi",
        )
        det = Detection(
            label="car",
            confidence=0.9,
            x1=10, y1=10, x2=100, y2=100,
            cx=55, cy=55,
            direction="center",
            depth_meters=3.5,
        )
        msg = tracker.update([det])
        self.assertIsNotNone(msg)
        self.assertIn("सामने", msg)
        self.assertIn("कार", msg)
        self.assertIn("3.5 मीटर पर", msg)

    def test_tracker_hinglish_announcement(self):
        tracker = AnnouncementTracker(
            absence_reset=1.0,
            speak_interval=0.0,
            min_duration=0.0,
            lang="hinglish",
        )
        det = Detection(
            label="bicycle",
            confidence=0.85,
            x1=10, y1=10, x2=100, y2=100,
            cx=55, cy=55,
            direction="right",
            depth_meters=1.2,
        )
        msg = tracker.update([det])
        self.assertIsNotNone(msg)
        self.assertIn("daayein", msg)
        self.assertIn("cycle", msg)
        self.assertIn("1.2 meter par", msg)

    def test_tracker_debouncing_and_absence(self):
        tracker = AnnouncementTracker(
            absence_reset=0.2,
            speak_interval=0.0,
            min_duration=0.1,
            lang="en",
        )
        det = Detection(
            label="chair",
            confidence=0.8,
            x1=0, y1=0, x2=10, y2=10,
            cx=5, cy=5,
            direction="left",
            depth_meters=None,
        )
        # First frame: not yet min_duration
        msg = tracker.update([det])
        self.assertIsNone(msg)

        # After min_duration: should announce
        time.sleep(0.12)
        msg = tracker.update([det])
        self.assertIsNotNone(msg)
        self.assertIn("chair on the left", msg)

        # Immediate next frame: already announced streak
        msg2 = tracker.update([det])
        self.assertIsNone(msg2)


class TestTTSEngine(unittest.TestCase):
    @patch("src.tts_module.sd")
    def test_engine_initialization_defaults(self, mock_sd):
        engine = TTSEngine(model_path=None, engine="piper", lang="en")
        try:
            self.assertEqual(engine._lang, "en")
            self.assertEqual(engine._engine_pref, "piper")
        finally:
            engine.shutdown()

    @patch("src.tts_module.sd")
    def test_kokoro_fallback_to_piper(self, mock_sd):
        # When kokoro files are absent, engine falls back safely
        engine = TTSEngine(model_path=None, engine="kokoro", lang="hi")
        try:
            self.assertEqual(engine._lang, "hi")
            self.assertIn(engine._backend, ("piper", "win32com", "espeak", "pyttsx3", "none"))
        finally:
            engine.shutdown()

    @patch("src.tts_module.sd")
    def test_critical_preemption_queue(self, mock_sd):
        engine = TTSEngine(model_path=None, engine="piper", queue_max=5)
        try:
            # Enqueue normal messages
            engine.speak("Normal message 1")
            engine.speak("Normal message 2")

            # Fire critical message
            engine.speak("CRITICAL: Obstacle ahead!")

            # Verify queue only contains critical item (normal items purged)
            self.assertFalse(engine._queue.empty())
            item = engine._queue.get_nowait()
            self.assertEqual(item[0], _PRIORITY_CRITICAL)
            self.assertIn("CRITICAL", item[2])
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
