from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SplitName = Literal["train", "validation", "test"]
SPLIT_CODES: dict[SplitName, int] = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class ESAWindowStore:
    normalized_values: np.ndarray
    clean_mask: np.ndarray
    label_code: np.ndarray
    split_code: np.ndarray
    timestamps_ns: np.ndarray
    channel_names: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray

    @classmethod
    def load(cls, artifact_path: Path, manifest_path: Path) -> "ESAWindowStore":
        artifact_path = artifact_path.resolve()
        manifest_path = manifest_path.resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"ESA artifact not found: {artifact_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ESA manifest not found: {manifest_path}")

        with np.load(artifact_path) as loaded:
            required = {
                "normalized_values",
                "clean_mask",
                "label_code",
                "split_code",
                "timestamps_ns",
            }
            missing = required - set(loaded.files)
            if missing:
                raise ValueError(f"ESA artifact is missing arrays: {sorted(missing)}")
            arrays = {name: loaded[name] for name in required}

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        channel_names = tuple(manifest["schema"]["channel_order"])
        means = np.asarray(
            [manifest["normalization"]["mean"][name] for name in channel_names],
            dtype=np.float32,
        )
        stds = np.asarray(
            [manifest["normalization"]["std"][name] for name in channel_names],
            dtype=np.float32,
        )
        values = arrays["normalized_values"].astype(np.float32, copy=False)
        expected_shape = (len(arrays["timestamps_ns"]), len(channel_names))
        if values.shape != expected_shape:
            raise ValueError(
                f"normalized_values shape {values.shape} != expected {expected_shape}"
            )
        for name in ("clean_mask", "label_code"):
            if arrays[name].shape != expected_shape:
                raise ValueError(f"{name} shape {arrays[name].shape} != {expected_shape}")
        if arrays["split_code"].shape != (expected_shape[0],):
            raise ValueError("split_code must have one entry per timestamp")
        if not np.isfinite(values).all():
            raise ValueError("normalized_values contains NaN or infinity")

        return cls(
            normalized_values=values,
            clean_mask=arrays["clean_mask"].astype(np.uint8, copy=False),
            label_code=arrays["label_code"].astype(np.uint8, copy=False),
            split_code=arrays["split_code"].astype(np.uint8, copy=False),
            timestamps_ns=arrays["timestamps_ns"].astype(np.int64, copy=False),
            channel_names=channel_names,
            means=means,
            stds=stds,
        )


def build_clean_window_starts(
    store: ESAWindowStore,
    split: SplitName,
    window_length: int,
    stride: int,
    window_policy: str = "strict_clean",
    min_observed_fraction: float = 0.0,
) -> np.ndarray:
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")
    code = SPLIT_CODES[split]
    indices = np.flatnonzero(store.split_code == code)
    if indices.size == 0:
        raise ValueError(f"Split {split} is empty")
    split_start = int(indices[0])
    split_end = int(indices[-1]) + 1
    if split_end - split_start < window_length:
        raise ValueError(f"Split {split} is shorter than one window")
    if not np.all(store.split_code[split_start:split_end] == code):
        raise ValueError(f"Split {split} is not a contiguous time range")

    candidates = np.arange(
        split_start, split_end - window_length + 1, stride, dtype=np.int64
    )
    relative = candidates - split_start
    if window_policy == "strict_clean":
        invalid_timepoint = ~store.clean_mask[split_start:split_end].astype(bool).all(axis=1)
        cumulative = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(invalid_timepoint, dtype=np.int64)]
        )
        usable = cumulative[relative + window_length] == cumulative[relative]
    elif window_policy == "partial":
        if not 0.0 <= min_observed_fraction <= 1.0:
            raise ValueError("min_observed_fraction must be in [0, 1]")
        observed_per_timepoint = store.clean_mask[split_start:split_end].sum(axis=1)
        cumulative = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(observed_per_timepoint, dtype=np.int64)]
        )
        observed_counts = cumulative[relative + window_length] - cumulative[relative]
        minimum = max(
            1,
            int(np.ceil(window_length * len(store.channel_names) * min_observed_fraction)),
        )
        usable = observed_counts >= minimum
    else:
        raise ValueError(f"Unsupported window_policy: {window_policy}")
    return candidates[usable]


def deterministic_condition_mask(
    observed_mask: np.ndarray,
    missing_ratio: float,
    mask_seed: int,
    window_start: int,
) -> np.ndarray:
    if not 0.0 <= missing_ratio < 1.0:
        raise ValueError("missing_ratio must be in [0, 1)")
    condition = observed_mask.astype(np.float32, copy=True)
    observed_indices = np.flatnonzero(condition.reshape(-1) > 0)
    num_masked = int(round(observed_indices.size * missing_ratio))
    if num_masked == 0:
        return condition
    ratio_code = int(round(missing_ratio * 1_000_000))
    sequence = np.random.SeedSequence(
        [int(mask_seed), int(window_start & 0xFFFFFFFF), ratio_code]
    )
    rng = np.random.default_rng(sequence)
    masked = rng.choice(observed_indices, size=num_masked, replace=False)
    flat = condition.reshape(-1)
    flat[masked] = 0.0
    return condition


class ESAWindowDataset(Dataset):
    def __init__(
        self,
        store: ESAWindowStore,
        split: SplitName,
        window_length: int = 96,
        stride: int = 48,
        missing_ratio: float | None = None,
        mask_seed: int = 2026,
        window_policy: str = "strict_clean",
        min_observed_fraction: float = 0.0,
        use_historical_patterns: bool = False,
        historical_pattern_seed: int = 2026,
        historical_min_missing_fraction: float = 0.05,
        historical_max_missing_fraction: float = 0.80,
    ) -> None:
        self.store = store
        self.split = split
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.missing_ratio = missing_ratio
        self.mask_seed = int(mask_seed)
        self.window_policy = str(window_policy)
        self.min_observed_fraction = float(min_observed_fraction)
        self.use_historical_patterns = bool(use_historical_patterns)
        self.historical_pattern_seed = int(historical_pattern_seed)
        self.historical_min_missing_fraction = float(
            historical_min_missing_fraction
        )
        self.historical_max_missing_fraction = float(
            historical_max_missing_fraction
        )
        if not 0.0 < self.historical_min_missing_fraction:
            raise ValueError("historical_min_missing_fraction must be positive")
        if not (
            self.historical_min_missing_fraction
            <= self.historical_max_missing_fraction
            < 1.0
        ):
            raise ValueError(
                "historical missing fractions must satisfy 0 < min <= max < 1"
            )
        self.starts = build_clean_window_starts(
            store,
            split,
            self.window_length,
            self.stride,
            self.window_policy,
            self.min_observed_fraction,
        )
        self.historical_pattern_starts = self._build_historical_pattern_pool()

    def _build_historical_pattern_pool(self) -> np.ndarray:
        if not self.use_historical_patterns:
            return np.empty(0, dtype=np.int64)
        if self.split != "train":
            raise ValueError("Historical pattern transfer is only supported for train")

        total = self.window_length * len(self.store.channel_names)
        missing_counts = np.fromiter(
            (
                total
                - int(
                    self.store.clean_mask[
                        int(start) : int(start) + self.window_length
                    ].sum()
                )
                for start in self.starts
            ),
            dtype=np.int64,
            count=len(self.starts),
        )
        missing_fractions = missing_counts / float(total)
        usable = (
            (missing_fractions >= self.historical_min_missing_fraction)
            & (missing_fractions <= self.historical_max_missing_fraction)
        )
        pool = self.starts[usable]
        if pool.size == 0:
            raise ValueError(
                "No historical patterns satisfy the configured missing-fraction range"
            )
        return pool

    def historical_pattern_summary(self) -> dict[str, float | int | bool]:
        return {
            "enabled": self.use_historical_patterns,
            "pool_size": int(self.historical_pattern_starts.size),
            "min_missing_fraction": self.historical_min_missing_fraction,
            "max_missing_fraction": self.historical_max_missing_fraction,
            "seed": self.historical_pattern_seed,
        }

    def _historical_condition_mask(
        self, observed_mask: np.ndarray, window_start: int
    ) -> np.ndarray:
        if self.historical_pattern_starts.size == 0:
            return observed_mask.copy()

        sequence = np.random.SeedSequence(
            [
                self.historical_pattern_seed,
                int(window_start & 0xFFFFFFFF),
                int((window_start >> 32) & 0xFFFFFFFF),
            ]
        )
        rng = np.random.default_rng(sequence)
        attempts = min(32, int(self.historical_pattern_starts.size))
        candidate_indices = rng.choice(
            self.historical_pattern_starts.size, size=attempts, replace=False
        )
        observed_count = float(observed_mask.sum())
        best_mask = None
        best_distance = float("inf")
        midpoint = 0.5 * (
            self.historical_min_missing_fraction
            + self.historical_max_missing_fraction
        )

        for candidate_index in candidate_indices:
            pattern_start = int(self.historical_pattern_starts[candidate_index])
            pattern = self.store.clean_mask[
                pattern_start : pattern_start + self.window_length
            ].astype(np.float32, copy=False)
            condition = observed_mask * pattern
            target_count = observed_count - float(condition.sum())
            target_fraction = target_count / observed_count
            distance = abs(target_fraction - midpoint)
            if distance < best_distance:
                best_mask = condition.copy()
                best_distance = distance
            if (
                self.historical_min_missing_fraction
                <= target_fraction
                <= self.historical_max_missing_fraction
            ):
                return condition.astype(np.float32, copy=True)

        if best_mask is None:
            raise RuntimeError("Historical pattern pool unexpectedly produced no mask")

        flat_observed = observed_mask.reshape(-1)
        flat_condition = best_mask.reshape(-1)
        observed_indices = np.flatnonzero(flat_observed > 0)
        target_indices = np.flatnonzero(
            (flat_observed > 0) & (flat_condition == 0)
        )
        min_targets = max(
            1,
            int(np.ceil(observed_indices.size * self.historical_min_missing_fraction)),
        )
        max_targets = max(
            min_targets,
            int(np.floor(observed_indices.size * self.historical_max_missing_fraction)),
        )
        if target_indices.size < min_targets:
            available = np.flatnonzero(
                (flat_observed > 0) & (flat_condition > 0)
            )
            additional = rng.choice(
                available, size=min_targets - target_indices.size, replace=False
            )
            flat_condition[additional] = 0.0
        elif target_indices.size > max_targets:
            restored = rng.choice(
                target_indices, size=target_indices.size - max_targets, replace=False
            )
            flat_condition[restored] = 1.0
        return best_mask.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        start = int(self.starts[index])
        end = start + self.window_length
        values = self.store.normalized_values[start:end].astype(np.float32, copy=True)
        observed_mask = self.store.clean_mask[start:end].astype(np.float32, copy=True)
        if self.window_policy == "strict_clean" and not observed_mask.all():
            raise RuntimeError("A non-clean point entered a strict-clean ESA window")
        if observed_mask.sum() == 0:
            raise RuntimeError("An all-missing ESA window entered the dataset")
        values *= observed_mask
        if self.missing_ratio is None:
            gt_mask = observed_mask.copy()
        else:
            gt_mask = deterministic_condition_mask(
                observed_mask, self.missing_ratio, self.mask_seed, start
            )
        hist_mask = (
            self._historical_condition_mask(observed_mask, start)
            if self.use_historical_patterns
            else observed_mask.copy()
        )
        return {
            "observed_data": values,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "hist_mask": hist_mask,
            "timepoints": np.arange(self.window_length, dtype=np.float32),
            "window_start": np.int64(start),
            "timestamp_ns": np.int64(self.store.timestamps_ns[start]),
        }


def make_loader(
    dataset: ESAWindowDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
