import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_sim.dataset import build_feature_dataset


class DatasetTests(unittest.TestCase):
    def test_partial_real_data_is_augmented_for_all_health_states(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "normal.csv"
            samples = np.zeros((256, 3), dtype=np.float32)
            samples[:, 0] = np.sin(np.arange(256) / 4)
            with path.open("w", encoding="utf-8") as stream:
                stream.write("x,y,z,label\n")
                for row in samples:
                    stream.write(f"{row[0]},{row[1]},{row[2]},Normal\n")
            features, labels, source = build_feature_dataset(
                "real", folder, recordings_per_class=2, seed=42
            )
        self.assertEqual(source, "real+synthetic")
        self.assertEqual(features.shape[1], 18)
        self.assertEqual(set(labels), set(range(5)))
        self.assertTrue(all(np.count_nonzero(labels == label) >= 2 for label in range(5)))


if __name__ == "__main__":
    unittest.main()
