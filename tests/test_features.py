import unittest
import tempfile
from pathlib import Path

import numpy as np

from data_sim.generator import generate_recording
from data_sim.loader import load_local_records
from features.extract import FEATURE_COUNT, extract_window_features
from features.pipeline import record_to_features
from features.windowing import window_signal


class WindowingTests(unittest.TestCase):
    def test_windows_have_50_percent_overlap(self):
        windows = window_signal(np.arange(256))
        self.assertEqual(windows.shape, (3, 128))
        np.testing.assert_array_equal(windows[0, 64:], windows[1, :64])

    def test_short_signal_has_no_padded_window(self):
        self.assertEqual(window_signal(np.arange(127)).shape, (0, 128))


class FeatureTests(unittest.TestCase):
    def test_feature_vector_has_exactly_18_values(self):
        record = generate_recording(2, sample_count=128, seed=42)
        features = extract_window_features(record.vibration, record.temperature, record.current)
        self.assertEqual(features.shape, (FEATURE_COUNT,))
        self.assertTrue(np.isfinite(features).all())

    def test_record_pipeline_uses_64_sample_hop(self):
        record = generate_recording(0, sample_count=256, seed=42)
        self.assertEqual(record_to_features(record).shape, (3, FEATURE_COUNT))

    def test_loads_headerless_local_bearing_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            state_dir = Path(folder) / "Bearing Wear"
            state_dir.mkdir()
            np.savetxt(state_dir / "run.txt", np.ones((128, 2)), delimiter="\t")
            records = load_local_records(folder)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].label, 1)
        self.assertEqual(records[0].vibration.shape, (128, 3))


if __name__ == "__main__":
    unittest.main()
