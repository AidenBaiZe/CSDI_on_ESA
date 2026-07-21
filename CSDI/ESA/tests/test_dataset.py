import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_esa import ESAWindowDataset, ESAWindowStore
from experiment import impute_chunked
from main_model import CSDI_Physio


class TestESADataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        cls.config = config
        cls.store = ESAWindowStore.load(
            (ROOT / config["data"]["artifact"]).resolve(),
            (ROOT / config["data"]["manifest"]).resolve(),
        )

    def make_dataset(self, split="train", ratio=None):
        data = self.config["data"]
        return ESAWindowDataset(
            self.store,
            split,
            data["window_length"],
            data["stride"],
            ratio,
            data["mask_seed"],
        )

    def test_expected_window_counts(self):
        self.assertEqual(len(self.make_dataset("train")), 51074)
        self.assertEqual(len(self.make_dataset("validation", 0.5)), 10944)
        self.assertEqual(len(self.make_dataset("test", 0.1)), 10911)

    def test_batch_interface_and_strict_clean_windows(self):
        dataset = self.make_dataset("train")
        sample = dataset[0]
        self.assertEqual(sample["observed_data"].shape, (96, 6))
        self.assertEqual(sample["observed_mask"].shape, (96, 6))
        self.assertEqual(sample["gt_mask"].shape, (96, 6))
        self.assertEqual(sample["timepoints"].shape, (96,))
        self.assertTrue(np.all(sample["observed_mask"] == 1))
        start = int(sample["window_start"])
        self.assertTrue(np.all(self.store.label_code[start : start + 96] == 0))

    def test_fixed_missing_ratios_and_determinism(self):
        for ratio in (0.1, 0.5, 0.9):
            first = self.make_dataset("test", ratio)
            second = self.make_dataset("test", ratio)
            a = first[17]
            b = second[17]
            self.assertTrue(np.array_equal(a["gt_mask"], b["gt_mask"]))
            observed = int(a["observed_mask"].sum())
            masked = int((a["observed_mask"] - a["gt_mask"]).sum())
            self.assertEqual(masked, round(observed * ratio))

    def test_windows_do_not_cross_split_boundaries(self):
        length = self.config["data"]["window_length"]
        for split, code in (("train", 0), ("validation", 1), ("test", 2)):
            dataset = self.make_dataset(split, 0.5 if split != "train" else None)
            starts = dataset.starts
            self.assertTrue(np.all(self.store.split_code[starts] == code))
            self.assertTrue(np.all(self.store.split_code[starts + length - 1] == code))

    def test_partial_policy_keeps_partially_missing_and_drops_all_missing_windows(self):
        data = self.config["data"]
        dataset = ESAWindowDataset(
            self.store,
            "train",
            data["window_length"],
            data["stride"],
            window_policy="partial",
            min_observed_fraction=0.0,
        )
        self.assertGreater(len(dataset), len(self.make_dataset("train")))
        masks = [dataset[index]["observed_mask"] for index in range(len(dataset))]
        self.assertTrue(any(not mask.all() for mask in masks))
        self.assertTrue(all(mask.sum() > 0 for mask in masks))

    def test_model_forward_backward_and_sampling(self):
        config = self.config
        model_config = {
            "model": copy.deepcopy(config["model"]),
            "diffusion": copy.deepcopy(config["diffusion"]),
            "train": copy.deepcopy(config["train"]),
        }
        model_config["diffusion"].update(
            {"layers": 1, "channels": 8, "nheads": 1, "num_steps": 2}
        )
        model_config["model"]["target_strategy"] = "mix"
        model = CSDI_Physio(model_config, "cpu", target_dim=6)
        dataset = self.make_dataset("test", 0.5)
        samples = [dataset[0], dataset[1]]
        batch = {
            key: torch.as_tensor(np.stack([sample[key] for sample in samples]))
            for key in ("observed_data", "observed_mask", "gt_mask", "timepoints")
        }
        loss = model(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        generated, _, target_mask, _, _ = model.evaluate(batch, n_samples=2)
        self.assertEqual(generated.shape, (2, 2, 6, 96))
        self.assertGreater(float(target_mask.sum()), 0.0)
        observed_data, _, observed_time, gt_mask, _, _ = model.process_data(batch)
        side_info = model.get_side_info(observed_time, gt_mask)
        chunked = impute_chunked(
            model, observed_data, gt_mask, side_info, n_samples=3, sample_chunk_size=2
        )
        self.assertEqual(chunked.shape, (2, 3, 6, 96))
        self.assertTrue(torch.isfinite(chunked).all())


class TestHistoricalPatternMasking(unittest.TestCase):
    @staticmethod
    def make_store():
        length = 32
        channels = ("channel_1", "channel_2")
        clean_mask = np.ones((length, len(channels)), dtype=np.uint8)
        clean_mask[8:12, 0] = 0
        clean_mask[10:14, 1] = 0
        return ESAWindowStore(
            normalized_values=np.zeros((length, len(channels)), dtype=np.float32),
            clean_mask=clean_mask,
            label_code=np.zeros((length, len(channels)), dtype=np.uint8),
            split_code=np.zeros(length, dtype=np.uint8),
            timestamps_ns=np.arange(length, dtype=np.int64),
            channel_names=channels,
            means=np.zeros(len(channels), dtype=np.float32),
            stds=np.ones(len(channels), dtype=np.float32),
        )

    def test_historical_pool_is_nonempty_effective_and_deterministic(self):
        kwargs = {
            "store": self.make_store(),
            "split": "train",
            "window_length": 8,
            "stride": 4,
            "window_policy": "partial",
            "use_historical_patterns": True,
            "historical_pattern_seed": 17,
            "historical_min_missing_fraction": 0.1,
            "historical_max_missing_fraction": 0.8,
        }
        first = ESAWindowDataset(**kwargs)
        second = ESAWindowDataset(**kwargs)
        self.assertGreater(first.historical_pattern_starts.size, 0)
        clean_index = int(np.flatnonzero(first.starts == 20)[0])
        a = first[clean_index]
        b = second[clean_index]
        target = a["observed_mask"] - a["hist_mask"]
        target_fraction = float(target.sum() / a["observed_mask"].sum())
        self.assertGreaterEqual(target_fraction, 0.1)
        self.assertLessEqual(target_fraction, 0.8)
        self.assertTrue(np.all(a["hist_mask"] <= a["observed_mask"]))
        self.assertTrue(np.array_equal(a["hist_mask"], b["hist_mask"]))

    def test_model_uses_provided_pattern_and_bounded_random_ratio(self):
        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        model_config = {
            "model": copy.deepcopy(config["model"]),
            "diffusion": copy.deepcopy(config["diffusion"]),
            "train": copy.deepcopy(config["train"]),
        }
        model_config["diffusion"].update(
            {"layers": 1, "channels": 8, "nheads": 1, "num_steps": 2}
        )
        model_config["model"].update(
            {
                "target_strategy": "mix",
                "use_provided_hist_mask": True,
                "mix_random_probability": 0.0,
                "random_mask_min_ratio": 0.2,
                "random_mask_max_ratio": 0.4,
            }
        )
        model = CSDI_Physio(model_config, "cpu", target_dim=2)
        observed = torch.ones((4, 2, 8))
        provided = observed.clone()
        provided[:, :, 2:5] = 0
        condition = model.get_hist_mask(observed, provided)
        self.assertTrue(torch.equal(condition, provided))

        random_condition = model.get_randmask(observed)
        ratios = (observed - random_condition).sum(dim=(1, 2)) / observed.sum(
            dim=(1, 2)
        )
        self.assertTrue(torch.all(ratios >= 0.2 - 1.0 / 16.0))
        self.assertTrue(torch.all(ratios <= 0.4 + 1.0 / 16.0))


if __name__ == "__main__":
    unittest.main()
