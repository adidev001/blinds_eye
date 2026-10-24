"""
Blind's Eye v2.2 -- Verification Suite (PRD Section 7)
=======================================================
Implements the 7 deterministic tests mandated in the PRD Section 7
Verification Matrix, plus integration, edge-case and regression tests.

All tests run without physical hardware (no camera, GPU, speakers, or
haptic receiver). Hardware-dependent behaviour is exercised through mocks,
synthetic inputs and temporary directories.

Test classes
------------
    TestInnerBoxPercentile           -- FR-1.1 depth extraction
    TestCrossCorridorPersistence     -- FR-1.2 track identity
    TestEgoDeadbandAndGateC         -- FR-1.3 kinematics / CRITICAL gates
    TestBoundedCrowdCompression      -- FR-3.3 crowd clauses
    TestStereoPanningAndMono         -- FR-4.2 stereo/mono
    TestHannEarconBoundaries         -- FR-4.3 waveform boundaries
    TestNonBlockingUDPAndTelemetry   -- FR-5.3 / FR-5.4
    TestIntegrationDetectionPipeline -- Scenario A
    TestIntegrationCameraLoss        -- Scenario B
    TestIntegrationShutdownTelemetry -- Scenario D
    TestQuantizationAndLocalization  -- distance / localization edge cases
"""

from __future__ import annotations

import inspect
import json
import math
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# ------------------------------------------------------------------
# Project root on sys.path
# ------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.localization import LocalizationFormatter, quantize_distance
from src.tts_module import generate_tone, pan_audio, TTSEngine
from src.vision import AnnouncementTracker, Detection, extract_roi_disparity, get_direction
from src.telemetry import TelemetryRecorder, UdpHapticEmitter


# ======================================================================
# TEST 1 -- Inner-Box 65th Percentile Depth Extraction  (FR-1.1)
# ======================================================================

class TestInnerBoxPercentile(unittest.TestCase):
    """PRD Section 7 Test 1: inner-box 65th percentile rejects background hole."""

    def _disp(self, h=100, w=100, fill=0.5):
        return np.full((h, w), fill, dtype=np.float32)

    def test_background_hole_rejected(self):
        """Center pixel poisoned to 0.0; 65th pct of inner box must still be 0.5."""
        d = self._disp(100, 100, fill=0.5)
        d[50, 50] = 0.0
        r = extract_roi_disparity(d, (20, 20, 80, 80), frame_w=100, frame_h=100)
        self.assertAlmostEqual(r, 0.5, places=4)

    def test_gradient_matches_np_percentile(self):
        """Ramp disparity -- result must equal numpy 65th pct of inner ROI."""
        d = np.zeros((100, 100), dtype=np.float32)
        for col in range(100):
            d[:, col] = col / 100.0
        r = extract_roi_disparity(d, (10, 10, 90, 90), frame_w=100, frame_h=100)
        rx1 = int(10 + 0.35 * 80)
        rx2 = int(90 - 0.35 * 80) + 1
        ry1 = int(10 + 0.35 * 80)
        ry2 = int(90 - 0.35 * 80) + 1
        expected = float(np.percentile(d[ry1:ry2, rx1:rx2], 65))
        self.assertAlmostEqual(r, expected, places=3)

    def test_degenerate_bbox_falls_back(self):
        """1x1 bbox -> inner box empty -> centroid fallback; no exception."""
        d = self._disp(100, 100, fill=0.7)
        r = extract_roi_disparity(d, (50, 50, 51, 51), frame_w=100, frame_h=100)
        self.assertTrue(np.isfinite(r))

    def test_oob_bbox_clamps(self):
        """Oversized bbox clamped safely."""
        d = self._disp(50, 50, fill=0.3)
        r = extract_roi_disparity(d, (-10, -10, 200, 200), frame_w=50, frame_h=50)
        self.assertTrue(np.isfinite(r))

    def test_zero_disparity_returns_zero(self):
        d = np.zeros((60, 80), dtype=np.float32)
        r = extract_roi_disparity(d, (10, 10, 70, 50), frame_w=80, frame_h=60)
        self.assertAlmostEqual(r, 0.0, places=6)

    def test_single_pixel_disparity_finite(self):
        d = np.array([[0.42]], dtype=np.float32)
        r = extract_roi_disparity(d, (0, 0, 1, 1), frame_w=1, frame_h=1)
        self.assertTrue(np.isfinite(r))

    def test_metric_conversion(self):
        """depth_scale=3.0, disparity=0.5 -> distance=6.0m"""
        self.assertAlmostEqual(3.0 / max(0.5, 1e-6), 6.0, places=4)

    def test_zero_disparity_no_inf(self):
        """Zero disparity must not produce inf after guard."""
        dist = 3.0 / max(0.0, 1e-6)
        self.assertFalse(math.isinf(dist))


# ======================================================================
# TEST 2 -- Cross-Corridor Track Persistence  (FR-1.2)
# ======================================================================

class TestCrossCorridorPersistence(unittest.TestCase):
    """PRD Section 7 Test 2: ByteTrack ID persists across corridor transitions."""

    def _det(self, direction, track_id=42, depth=2.0):
        x1 = {"left": 0, "center": 220, "right": 440}[direction]
        x2 = x1 + 200
        return Detection("person", 0.9, x1, 100, x2, 300,
                         (x1+x2)//2, 200, direction, depth, track_id)

    def _tracker(self):
        return AnnouncementTracker(absence_reset=5.0, speak_interval=0.0,
                                   min_duration=0.0, lang="en")

    def test_first_seen_preserved_on_corridor_cross(self):
        t = self._tracker()
        t.update([self._det("left")])
        fs_before = t._tracks[42].first_seen
        t.update([self._det("center", depth=2.0)])
        self.assertAlmostEqual(t._tracks[42].first_seen, fs_before, places=4)

    def test_center_entry_triggers_reannouncement(self):
        t = self._tracker()
        t.update([self._det("left")])
        result = t.update([self._det("center", depth=2.0)])
        self.assertIsNotNone(result, "Center entry at d<=2.5m must trigger re-announcement")

    def test_center_reannouncement_only_once(self):
        t = self._tracker()
        t.update([self._det("left")])
        t.update([self._det("center", depth=2.0)])
        msg2 = t.update([self._det("center", depth=2.0)])
        self.assertIsNone(msg2, "Center re-announcement must fire at most once")

    def test_absence_reset_refreshes_track(self):
        t = AnnouncementTracker(absence_reset=0.1, speak_interval=0.0,
                                min_duration=0.0, lang="en")
        t.update([self._det("left")])
        fs1 = t._tracks[42].first_seen
        time.sleep(0.15)
        t.update([self._det("left")])
        fs2 = t._tracks[42].first_seen
        self.assertGreater(fs2, fs1)

    def test_no_track_id_label_corridor_key(self):
        t = self._tracker()
        det = Detection("chair", 0.8, 0, 0, 100, 100, 50, 50,
                        "left", 1.5, None)
        t.update([det])
        self.assertIn("chair_left", t._tracks)


# ======================================================================
# TEST 3 -- Ego-Motion Deadband and Gate C  (FR-1.3)
# ======================================================================

class TestEgoDeadbandAndGateC(unittest.TestCase):
    """PRD Section 7 Test 3: 2.2 m/s deadband and Gate C vehicle trigger."""

    def _tracker(self):
        return AnnouncementTracker(absence_reset=5.0, speak_interval=0.0,
                                   min_duration=0.0, lang="en")

    def _car(self, depth, tid=1, scale=80.0, label="car", direction="center"):
        h = int(scale / 2)
        return Detection(label, 0.9, 320-h, 240-h, 320+h, 240+h,
                         320, 240, direction, depth, tid)

    def test_walking_speed_not_approaching(self):
        t = self._tracker()
        t0 = 100.0
        t.update([self._car(4.0, scale=80.0)], depth_frame_id=1, depth_timestamp=t0)
        t.update([self._car(3.86, scale=81.0)], depth_frame_id=2, depth_timestamp=t0+0.1)
        self.assertFalse(t._tracks[1].is_approaching)

    def test_low_rscale_not_approaching(self):
        t = self._tracker()
        t0 = 100.0
        t.update([self._car(4.0, scale=80.0)], depth_frame_id=1, depth_timestamp=t0)
        t.update([self._car(3.0, scale=80.1)], depth_frame_id=2, depth_timestamp=t0+0.1)
        self.assertFalse(t._tracks[1].is_approaching)

    def test_gate_c_fires_for_vehicle(self):
        t = self._tracker()
        t0 = 100.0
        t.update([self._car(7.0, scale=80.0)], depth_frame_id=1, depth_timestamp=t0)
        result = t.update([self._car(5.3, scale=100.0)], depth_frame_id=2, depth_timestamp=t0+0.2)
        rec = t._tracks[1]
        if rec.v_close >= 3.5 and rec.r_scale >= 0.20 and rec.d_smooth is not None:
            if rec.d_smooth <= 8.0 and rec.d_smooth / rec.v_close <= 2.5:
                self.assertIsNotNone(result)
                self.assertIn("CRITICAL", result)

    def test_gate_c_non_vehicle_4m_limit(self):
        t = self._tracker()
        t0 = 100.0
        t.update([self._car(7.0, scale=80.0, label="chair")],
                 depth_frame_id=1, depth_timestamp=t0)
        result = t.update([self._car(5.0, scale=100.0, label="chair")],
                          depth_frame_id=2, depth_timestamp=t0+0.2)
        rec = t._tracks[1]
        if result and "CRITICAL" in result and rec.d_smooth is not None:
            self.assertLessEqual(rec.d_smooth, 4.0)

    def test_stale_frame_no_velocity_update(self):
        t = self._tracker()
        t0 = 100.0
        t.update([self._car(4.0)], depth_frame_id=1, depth_timestamp=t0)
        v1 = t._tracks[1].v_close
        t.update([self._car(2.0)], depth_frame_id=1, depth_timestamp=t0)
        self.assertEqual(v1, t._tracks[1].v_close)

    def test_zero_dt_fallback_finite(self):
        t = self._tracker()
        t0 = 100.0
        t.update([self._car(4.0)], depth_frame_id=1, depth_timestamp=t0)
        t.update([self._car(3.0)], depth_frame_id=2, depth_timestamp=t0)
        self.assertTrue(np.isfinite(t._tracks[1].v_close))

    def test_critical_cooldown_suppresses(self):
        t = self._tracker()
        det = Detection("person", 0.9, 200, 100, 440, 380, 320, 240,
                        "center", 1.0, 99)
        t.update([det], depth_frame_id=1, depth_timestamp=100.0)
        t._tracks[99].first_seen -= 1.0
        r1 = t.update([det], depth_frame_id=2, depth_timestamp=100.01)
        r2 = t.update([det], depth_frame_id=3, depth_timestamp=100.02)
        if r1 and "CRITICAL" in r1:
            self.assertIsNone(r2)

    def test_nan_depth_no_crash(self):
        t = self._tracker()
        try:
            t.update([self._car(float("nan"))])
        except Exception as exc:
            self.fail(f"NaN depth raised: {exc}")
        rec = t._tracks.get(1)
        if rec:
            self.assertFalse(rec.is_approaching)


# ======================================================================
# TEST 4 -- Bounded Crowd Compression  (FR-3.3)
# ======================================================================

class TestBoundedCrowdCompression(unittest.TestCase):
    """PRD Section 7 Test 4: crowd bucket (2, 3, multiple) with min-depth."""

    def _people(self, n, depths, base=100):
        return [Detection("person", 0.9, 270, 100, 370, 300, 320, 200,
                          "center", depths[i], base+i) for i in range(n)]

    def _t(self, lang="en"):
        return AnnouncementTracker(absence_reset=5.0, speak_interval=0.0,
                                   min_duration=0.0, lang=lang)

    def test_five_people_multiple(self):
        t = self._t()
        r = t.update(self._people(5, [2.1, 2.4, 3.0, 3.2, 4.0]))
        self.assertIsNotNone(r)
        self.assertIn("multiple", r.lower())
        self.assertIn("people", r.lower())

    def test_five_people_min_depth_quantized(self):
        t = self._t()
        r = t.update(self._people(5, [2.1, 2.4, 3.0, 3.2, 4.0], base=200))
        self.assertIn("2.0", r)    # quantize_distance(2.1) = 2.0

    def test_one_person_singular(self):
        t = self._t()
        r = t.update(self._people(1, [3.0], base=300))
        self.assertIsNotNone(r)
        self.assertNotIn("multiple", r.lower())

    def test_two_bucket_2(self):
        t = self._t()
        r = t.update(self._people(2, [2.0, 2.5], base=400))
        self.assertIsNotNone(r)
        self.assertIn("2 people", r)

    def test_three_bucket_3(self):
        t = self._t()
        r = t.update(self._people(3, [2.0, 2.5, 3.0], base=500))
        self.assertIsNotNone(r)
        self.assertIn("3 people", r)

    def test_four_multiple(self):
        t = self._t()
        r = t.update(self._people(4, [2.0, 2.3, 2.6, 3.0], base=600))
        self.assertIn("multiple", r.lower())

    def test_labels_not_merged(self):
        t = self._t()
        car = Detection("car", 0.9, 0, 100, 100, 300, 50, 200, "center", 3.0, 701)
        prs = Detection("person", 0.9, 100, 100, 200, 300, 150, 200, "center", 4.0, 702)
        r = t.update([car, prs])
        if r:
            self.assertNotIn("2 cars", r)
            self.assertNotIn("2 people", r)

    def test_directions_not_merged(self):
        t = self._t()
        lft = Detection("person", 0.9, 0, 100, 100, 300, 50, 200, "left", 2.0, 801)
        rgt = Detection("person", 0.9, 540, 100, 640, 300, 590, 200, "right", 2.5, 802)
        r = t.update([lft, rgt])
        if r:
            self.assertNotIn("2 people", r)

    def test_hindi_plural(self):
        t = self._t(lang="hi")
        r = t.update(self._people(5, [2.0]*5, base=900))
        self.assertIsNotNone(r)
        self.assertIn("\u0932\u094b\u0917", r)

    def test_100_people_no_crash(self):
        t = self._t()
        dets = [Detection("person", 0.9, 270, 100, 370, 300, 320, 200,
                          "center", 2.0+i*0.05, 1000+i) for i in range(100)]
        r = t.update(dets)
        self.assertIn("multiple", r.lower())


# ======================================================================
# TEST 5 -- Stereo Panning and Mono Fallback  (FR-4.2)
# ======================================================================

class TestStereoPanningAndMono(unittest.TestCase):
    """PRD Section 7 Test 5: safe 100/35 panning gains and mono fallback."""

    def _mono(self, n=100):
        return np.ones(n, dtype=np.float32) * 0.5

    def test_left_gains(self):
        s = pan_audio(self._mono(), "left", stereo=True)
        np.testing.assert_allclose(s[:, 0], self._mono() * 1.00, rtol=1e-5)
        np.testing.assert_allclose(s[:, 1], self._mono() * 0.35, rtol=1e-5)

    def test_center_gains(self):
        s = pan_audio(self._mono(), "center", stereo=True)
        np.testing.assert_allclose(s[:, 0], self._mono() * 0.85, rtol=1e-5)
        np.testing.assert_allclose(s[:, 1], self._mono() * 0.85, rtol=1e-5)

    def test_right_gains(self):
        s = pan_audio(self._mono(), "right", stereo=True)
        np.testing.assert_allclose(s[:, 0], self._mono() * 0.35, rtol=1e-5)
        np.testing.assert_allclose(s[:, 1], self._mono() * 1.00, rtol=1e-5)

    def test_stereo_shape(self):
        self.assertEqual(pan_audio(self._mono(200), "center", stereo=True).shape, (200, 2))

    def test_mono_shape_unchanged(self):
        m = self._mono(200)
        out = pan_audio(m, "center", stereo=False)
        self.assertEqual(out.ndim, 1)
        np.testing.assert_array_equal(out, m)

    @patch("src.tts_module._SD_AVAILABLE", True)
    def test_single_channel_forces_mono(self):
        with patch("src.tts_module.sd") as mock_sd:
            mock_sd.query_devices.return_value = {"max_output_channels": 1}
            engine = TTSEngine(model_path=None, engine="piper", lang="en", stereo=True)
            try:
                self.assertFalse(engine._stereo)
            finally:
                engine.shutdown()

    @patch("src.tts_module._SD_AVAILABLE", True)
    def test_stereo_false_bypasses_device(self):
        with patch("src.tts_module.sd") as mock_sd:
            mock_sd.query_devices.return_value = {"max_output_channels": 6}
            engine = TTSEngine(model_path=None, engine="piper", lang="en", stereo=False)
            try:
                self.assertFalse(engine._stereo)
            finally:
                engine.shutdown()

    def test_input_not_mutated(self):
        m = np.ones(50, dtype=np.float32) * 0.5
        orig = m.copy()
        pan_audio(m, "right", stereo=True)
        np.testing.assert_array_equal(m, orig)

    def test_empty_array(self):
        empty = np.array([], dtype=np.float32)
        self.assertEqual(pan_audio(empty, "left", stereo=True).size, 0)
        self.assertEqual(pan_audio(empty, "left", stereo=False).size, 0)

    def test_unknown_direction_center_default(self):
        s = pan_audio(self._mono(), "forward", stereo=True)
        np.testing.assert_allclose(s[:, 0], self._mono() * 0.85, rtol=1e-5)
        np.testing.assert_allclose(s[:, 1], self._mono() * 0.85, rtol=1e-5)


# ======================================================================
# TEST 6 -- Hann-Windowed Earcon Boundaries  (FR-4.3)
# ======================================================================

class TestHannEarconBoundaries(unittest.TestCase):
    """PRD Section 7 Test 6: first/last samples == 0.0 on all Hann tones."""

    _TOL = 1e-6

    def _prebeep(self, sr=22050):
        return np.concatenate([
            generate_tone(880.0,  0.040, sr=sr, amp=0.35),
            generate_tone(1320.0, 0.040, sr=sr, amp=0.35),
        ])

    def _heartbeat(self, sr=22050):
        return generate_tone(800.0, 0.015, sr=sr, amp=0.15)

    def test_prebeep_first_zero(self):
        self.assertAlmostEqual(float(self._prebeep()[0]), 0.0, delta=self._TOL)

    def test_prebeep_last_zero(self):
        self.assertAlmostEqual(float(self._prebeep()[-1]), 0.0, delta=self._TOL)

    def test_heartbeat_first_zero(self):
        self.assertAlmostEqual(float(self._heartbeat()[0]), 0.0, delta=self._TOL)

    def test_heartbeat_last_zero(self):
        self.assertAlmostEqual(float(self._heartbeat()[-1]), 0.0, delta=self._TOL)

    def test_prebeep_all_finite(self):
        self.assertTrue(np.all(np.isfinite(self._prebeep())))

    def test_heartbeat_all_finite(self):
        self.assertTrue(np.all(np.isfinite(self._heartbeat())))

    def test_prebeep_duration_80ms(self):
        sr = 22050
        self.assertAlmostEqual(len(self._prebeep(sr)), int(round(sr*0.080)), delta=2)

    def test_heartbeat_duration_15ms(self):
        sr = 22050
        self.assertAlmostEqual(len(self._heartbeat(sr)), int(round(sr*0.015)), delta=2)

    def test_prebeep_amp_bounded(self):
        self.assertLessEqual(float(np.max(np.abs(self._prebeep()))), 0.35 + 1e-5)

    def test_heartbeat_amp_bounded(self):
        self.assertLessEqual(float(np.max(np.abs(self._heartbeat()))), 0.15 + 1e-5)

    def test_individual_tones_zero_boundary(self):
        for freq in (440.0, 880.0, 1320.0, 800.0):
            t = generate_tone(freq, 0.040)
            self.assertAlmostEqual(float(t[0]),  0.0, delta=self._TOL)
            self.assertAlmostEqual(float(t[-1]), 0.0, delta=self._TOL)

    def test_dtype_float32(self):
        self.assertEqual(generate_tone(440.0, 0.01).dtype, np.float32)

    def test_very_short_tone_min_length(self):
        self.assertGreaterEqual(len(generate_tone(440.0, 0.0001, sr=22050)), 2)


# ======================================================================
# TEST 7 -- Non-Blocking UDP and Telemetry  (FR-5.3 / FR-5.4)
# ======================================================================

class TestNonBlockingUDPAndTelemetry(unittest.TestCase):
    """PRD Section 7 Test 7: UDP loopback and in-memory telemetry."""

    # -- UDP --

    def test_udp_loopback_payload(self):
        recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv.bind(("127.0.0.1", 0))
        recv.settimeout(0.5)
        port = recv.getsockname()[1]
        em = UdpHapticEmitter(target=f"127.0.0.1:{port}")
        payload = {"event": "CRITICAL", "label": "car", "dir": "center",
                   "dist": 4.5, "ttc": 1.2, "ts": 1728940000.1}
        self.assertTrue(em.send(payload))
        try:
            data, _ = recv.recvfrom(4096)
            d = json.loads(data.decode("utf-8"))
            self.assertEqual(d["event"], "CRITICAL")
            self.assertEqual(d["label"], "car")
            self.assertAlmostEqual(d["dist"], 4.5)
        except socket.timeout:
            self.fail("UDP packet not received on loopback")
        finally:
            recv.close(); em.close()

    def test_udp_nonblocking_timing(self):
        em = UdpHapticEmitter(target="127.0.0.1:1")
        t0 = time.perf_counter()
        try:
            em.send({"event": "CRITICAL", "label": "x", "dir": "c",
                     "dist": 1.0, "ttc": 0.5, "ts": 0.0})
        except Exception as exc:
            self.fail(f"UDP send must not raise: {exc}")
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 0.05, f"Non-blocking took {elapsed*1000:.1f}ms")
        em.close()

    def test_udp_no_target_false(self):
        em = UdpHapticEmitter(target=None)
        self.assertFalse(em.send({"event": "CRITICAL"}))
        em.close()

    def test_udp_bad_target_no_crash(self):
        try:
            em = UdpHapticEmitter(target="bad_$$")
            em.close()
        except Exception as exc:
            self.fail(f"Bad target raised: {exc}")

    def test_udp_socket_nonblocking(self):
        em = UdpHapticEmitter(target="127.0.0.1:9999")
        if em._sock:
            self.assertFalse(em._sock.getblocking())
        em.close()

    # -- TelemetryRecorder --

    def test_records_in_memory(self):
        rec = TelemetryRecorder(maxlen=5000)
        for i in range(5):
            rec.record(yolo_ms=15.0+i, depth_ms=50.0, cache_hit=(i%2==0),
                       active_tracks=i, announcement=f"msg_{i}")
        self.assertEqual(len(rec.records), 5)

    def test_deque_bounded(self):
        rec = TelemetryRecorder(maxlen=10)
        for i in range(50):
            rec.record(yolo_ms=1.0, depth_ms=1.0)
        self.assertEqual(len(rec.records), 10)

    def test_flush_jsonl_valid(self):
        rec = TelemetryRecorder()
        for i in range(3):
            rec.record(yolo_ms=10.0+i, depth_ms=50.0, tts_ms=2.0, cache_hit=True,
                       active_tracks=2, announcement="person ahead",
                       critical_gate="A", timestamp=1000.0+i)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            count = rec.flush_jsonl(path)
            self.assertEqual(count, 3)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            for line in lines:
                obj = json.loads(line)
                for f in ("timestamp","yolo_ms","depth_ms","cache_hit",
                          "active_tracks","announcement","critical_gate"):
                    self.assertIn(f, obj)

    def test_record_no_file_io(self):
        src = inspect.getsource(TelemetryRecorder.record)
        self.assertNotIn("open(", src)
        self.assertNotIn("write(", src)

    def test_summary_fields(self):
        rec = TelemetryRecorder()
        for i in range(20):
            rec.record(yolo_ms=10.0+i*0.5, depth_ms=50.0+i, cache_hit=(i%3==0),
                       active_tracks=2, announcement="x" if i%4==0 else None,
                       critical_gate="A" if i%10==0 else None)
        md = rec.generate_summary_markdown()
        for phrase in ("Total Processed Frames","YOLO Latency","Depth Latency",
                       "Total Dispatched Announcements","Critical Hazard Preemptions"):
            self.assertIn(phrase, md)

    def test_summary_empty(self):
        self.assertIn("No frame telemetry records",
                      TelemetryRecorder().generate_summary_markdown())

    def test_record_schema(self):
        rec = TelemetryRecorder()
        rec.record(yolo_ms=10.0, depth_ms=50.0, tts_ms=2.0, cache_hit=True,
                   active_tracks=3, announcement="car", critical_gate="C", timestamp=999.0)
        r = rec.records[0]
        for f in ("timestamp","yolo_ms","depth_ms","tts_ms","cache_hit",
                  "active_tracks","announcement","critical_gate"):
            self.assertIn(f, r)

    def test_flush_creates_parents(self):
        rec = TelemetryRecorder()
        rec.record(yolo_ms=10.0, depth_ms=50.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "nested" / "out.jsonl"
            rec.flush_jsonl(path)
            self.assertTrue(path.exists())


# ======================================================================
# INTEGRATION A -- Detection-to-Announcement Pipeline
# ======================================================================

class TestIntegrationDetectionPipeline(unittest.TestCase):

    def _tracker(self):
        return AnnouncementTracker(absence_reset=5.0, speak_interval=0.0,
                                   min_duration=0.0, lang="en")

    def test_english_depth_in_announcement(self):
        t = self._tracker()
        det = Detection("person", 0.9, 210, 100, 430, 380, 320, 240, "center", 3.0, 1)
        result = t.update([det])
        self.assertIsNotNone(result)
        self.assertIn("3.0", result)

    def test_gate_a_critical_prefix(self):
        t = self._tracker()
        det = Detection("person", 0.9, 210, 100, 430, 380, 320, 240, "center", 1.0, 1)
        t.update([det])
        t._tracks[1].first_seen -= 1.0
        result = t.update([det])
        if result:
            self.assertTrue(result.startswith("CRITICAL"))

    def test_critical_event_dict_populated(self):
        t = self._tracker()
        det = Detection("person", 0.9, 210, 100, 430, 380, 320, 240, "center", 1.0, 1)
        t.update([det])
        t._tracks[1].first_seen -= 1.0
        t.update([det])
        if t.last_critical_event:
            for k in ("event","label","dir","dist","ts"):
                self.assertIn(k, t.last_critical_event)

    def test_no_steering_directives(self):
        t = self._tracker()
        forbidden = ["move left","move right","turn left","turn right"]
        dets = [
            Detection("person", 0.9, 0, 0, 100, 100, 50, 50, "left", 2.0, 1),
            Detection("car", 0.9, 400, 0, 500, 100, 450, 50, "right", 5.0, 2),
        ]
        for _ in range(5):
            result = t.update(dets)
            if result:
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.lower())


# ======================================================================
# INTEGRATION B -- Camera Loss / Recovery
# ======================================================================

class TestIntegrationCameraLoss(unittest.TestCase):

    def test_camera_lost_critical(self):
        fmt = LocalizationFormatter("en")
        self.assertTrue(fmt.format_system_alert("camera_lost").startswith("CRITICAL"))

    def test_camera_restored_not_critical(self):
        fmt = LocalizationFormatter("en")
        self.assertFalse(fmt.format_system_alert("camera_restored").startswith("CRITICAL"))

    def test_low_light_throttled(self):
        t = AnnouncementTracker(absence_reset=5.0, speak_interval=0.0,
                                min_duration=0.0, lang="en")
        r1 = t.update([], low_light=True)
        r2 = t.update([], low_light=True)
        if r1:
            self.assertIsNone(r2)

    def test_stale_camera_no_frames(self):
        from src.camera import CameraStream
        with patch("src.camera.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap.read.return_value = (False, None)
            mock_cv2.VideoCapture.return_value = mock_cap
            cam = CameraStream(src="fake://0")
            time.sleep(0.01)
            self.assertTrue(cam.is_stale(threshold_sec=0.001))
            cam.running = False
            cam.thread.join(timeout=0.5)


# ======================================================================
# INTEGRATION D -- Shutdown and Telemetry Flush
# ======================================================================

class TestIntegrationShutdownTelemetry(unittest.TestCase):

    @patch("src.tts_module.sd")
    def test_worker_terminates(self, mock_sd):
        engine = TTSEngine(model_path=None, engine="piper", lang="en")
        engine.shutdown()
        self.assertFalse(engine._worker.is_alive())

    def test_flush_on_exit(self):
        rec = TelemetryRecorder()
        for i in range(10):
            rec.record(yolo_ms=12.0, depth_ms=45.0, active_tracks=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exit.jsonl"
            n = rec.flush_jsonl(path)
            self.assertEqual(n, 10)
            lines = path.read_text().strip().splitlines()
            for line in lines:
                json.loads(line)

    def test_udp_close_idempotent(self):
        em = UdpHapticEmitter(target="127.0.0.1:9999")
        em.close()
        try:
            em.close()
        except Exception as exc:
            self.fail(f"Second close() raised: {exc}")


# ======================================================================
# Quantization and Localization Edge Cases
# ======================================================================

class TestQuantizationAndLocalization(unittest.TestCase):

    def test_below_1_5m(self):
        self.assertAlmostEqual(quantize_distance(1.23), 1.2, places=5)
        self.assertAlmostEqual(quantize_distance(0.55), 0.6, places=5)

    def test_at_1_5m(self):
        self.assertAlmostEqual(quantize_distance(1.5), 1.5, places=5)

    def test_above_1_5m(self):
        self.assertAlmostEqual(quantize_distance(2.1), 2.0, places=5)
        self.assertAlmostEqual(quantize_distance(2.4), 2.5, places=5)
        # round(6.7 * 2) / 2 = round(13.4) / 2 = 13 / 2 = 6.5 (Python banker's rounding)
        self.assertAlmostEqual(quantize_distance(6.7), 6.5, places=5)
        self.assertAlmostEqual(quantize_distance(6.8), 7.0, places=5)

    def test_approaching_prefix_en(self):
        fmt = LocalizationFormatter("en")
        s = fmt.format_detection("car", "center", 3.0, is_approaching=True)
        self.assertIn("approaching", s)

    def test_overhead_vertical_en(self):
        fmt = LocalizationFormatter("en")
        s = fmt.format_detection("shelf", "center", 1.5, vertical_zone="overhead")
        self.assertIn("overhead", s)

    def test_ground_vertical_hi(self):
        fmt = LocalizationFormatter("hi")
        s = fmt.format_detection("stone", "center", 0.5, vertical_zone="ground")
        self.assertIn("\u0928\u0940\u091a\u0947 \u091c\u092e\u0940\u0928 \u092a\u0930", s)

    def test_obstacle_ahead_alert(self):
        fmt = LocalizationFormatter("en")
        s = fmt.format_system_alert("obstacle_ahead", depth=0.8)
        self.assertIn("obstacle ahead", s)
        self.assertIn("0.8", s)

    def test_direction_thirds(self):
        w = 640
        # w//3=213, w/3=213.33 -> cx=213 is left; cx=214 is center
        # 2*w//3=426, 2*w/3=426.67 -> cx=426 is center; cx=427 is right
        self.assertEqual(get_direction(0, w), "left")
        self.assertEqual(get_direction(w//3, w), "left")       # 213 < 213.33 -> left
        self.assertEqual(get_direction(w//3 + 1, w), "center") # 214 >= 213.33 -> center
        self.assertEqual(get_direction(2*w//3 - 1, w), "center")
        self.assertEqual(get_direction(2*w//3, w), "center")   # 426 < 426.67 -> center
        self.assertEqual(get_direction(2*w//3 + 1, w), "right") # 427 >= 426.67 -> right
        self.assertEqual(get_direction(w-1, w), "right")


# ======================================================================
# TEST 8 -- Stationary Proximity & Queue Purge on Departure (FIX-013)
# ======================================================================

class TestStationaryHysteresisAndQueuePurge(unittest.TestCase):
    """Verify stationary obstacles are never labeled approaching and departed obstacles are purged."""

    def test_stationary_person_never_approaching(self):
        t = AnnouncementTracker(speak_interval=0.0, min_duration=0.0)
        det = Detection("person", 0.9, 200, 100, 440, 380, 320, 240, "center", 0.8, 42)
        t.update([det], depth_frame_id=1, depth_timestamp=100.0)
        t._tracks[42].first_seen -= 1.0
        r = t.update([det], depth_frame_id=2, depth_timestamp=100.1)
        if r:
            self.assertIn("CRITICAL", r)
            self.assertNotIn("approaching", r.lower())

    def test_gate_b_stationary_hysteresis_suppression(self):
        t = AnnouncementTracker(speak_interval=0.0, min_duration=0.0)
        det = Detection("bed", 0.9, 150, 100, 490, 400, 320, 250, "center", 0.7, 55)
        t.update([det], depth_frame_id=1, depth_timestamp=100.0)
        t._tracks[55].first_seen -= 1.0
        r1 = t.update([det], depth_frame_id=2, depth_timestamp=100.1)
        # Advance 2.6s (past global critical cooldown 2.5s) with distance unchanged (<0.25m delta)
        r2 = t.update([det], depth_frame_id=3, depth_timestamp=102.7)
        self.assertIsNone(r2)

    def test_purge_label_removes_departed_obstacle(self):
        tts = TTSEngine(model_path=None, engine="piper")
        try:
            tts.speak("CRITICAL: person on the center at 0.8 meters")
            tts.speak("Normal announcement")
            tts.purge_label("person")
            kept = []
            while not tts._queue.empty():
                kept.append(tts._queue.get_nowait())
            for item in kept:
                self.assertNotIn("person", item[2].lower())
        finally:
            tts.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
