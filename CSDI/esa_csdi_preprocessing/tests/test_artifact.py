import hashlib
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "processed" / "mission1_ch41_46_10m_30s"


def array_content_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


class TestESAArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifact = OUTPUT / "aligned.npz"
        manifest = OUTPUT / "manifest.json"
        if not artifact.is_file() or not manifest.is_file():
            raise unittest.SkipTest("Run preprocess.py before validating the artifact")
        with np.load(artifact) as loaded:
            cls.arrays = {name: loaded[name] for name in loaded.files}
        cls.manifest = json.loads(manifest.read_text(encoding="utf-8"))

    def test_schema_and_shapes(self):
        expected = {
            "timestamps_ns",
            "raw_values",
            "normalized_values",
            "observed_mask",
            "clean_mask",
            "label_code",
            "update_mask",
            "split_code",
        }
        self.assertEqual(set(self.arrays), expected)
        rows = len(self.arrays["timestamps_ns"])
        self.assertGreater(rows, 800_000)
        for name in (
            "raw_values",
            "normalized_values",
            "observed_mask",
            "clean_mask",
            "label_code",
            "update_mask",
        ):
            self.assertEqual(self.arrays[name].shape, (rows, 6))
        self.assertEqual(
            self.manifest["schema"]["channel_order"],
            [f"channel_{index}" for index in range(41, 47)],
        )

    def test_regular_time_grid(self):
        differences = np.diff(self.arrays["timestamps_ns"])
        self.assertTrue(np.all(differences == 30_000_000_000))

    def test_masks_and_values(self):
        raw = self.arrays["raw_values"]
        normalized = self.arrays["normalized_values"]
        observed = self.arrays["observed_mask"].astype(bool)
        clean = self.arrays["clean_mask"].astype(bool)
        labels = self.arrays["label_code"]
        self.assertTrue(np.isfinite(raw[observed]).all())
        self.assertTrue(np.isfinite(normalized).all())
        self.assertTrue(np.all(~clean | observed))
        self.assertTrue(np.all(~clean | (labels == 0)))
        self.assertTrue(np.all(self.arrays["observed_mask"][labels == 3] == 0))
        self.assertTrue(np.all(normalized[~observed] == 0))

    def test_chronological_splits(self):
        split = self.arrays["split_code"]
        unique, counts = np.unique(split, return_counts=True)
        self.assertEqual(unique.tolist(), [0, 1, 2])
        self.assertAlmostEqual(counts[0] / len(split), 0.70, places=3)
        self.assertAlmostEqual(counts[1] / len(split), 0.15, places=3)
        self.assertAlmostEqual(counts[2] / len(split), 0.15, places=3)
        self.assertTrue(np.all(np.diff(split.astype(np.int16)) >= 0))

    def test_training_normalization(self):
        normalized = self.arrays["normalized_values"]
        clean = self.arrays["clean_mask"].astype(bool)
        train = self.arrays["split_code"] == 0
        for column in range(6):
            values = normalized[train & clean[:, column], column].astype(np.float64)
            self.assertAlmostEqual(float(values.mean()), 0.0, places=5)
            self.assertAlmostEqual(float(values.std()), 1.0, places=5)

    def test_content_hash(self):
        self.assertEqual(
            array_content_hash(self.arrays),
            self.manifest["array_content_sha256"],
        )


if __name__ == "__main__":
    unittest.main()

