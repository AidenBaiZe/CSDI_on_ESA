from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


@dataclass(frozen=True)
class ESAWindowStore:
    split: str
    normalized_values: np.ndarray
    observed_mask: np.ndarray
    update_mask: np.ndarray
    label_code: np.ndarray
    observed_count: np.ndarray
    timestamps_ns: np.ndarray
    channel_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    raw_stds: np.ndarray
    channel_metadata: tuple[dict, ...]

    @classmethod
    def load(cls, processed_dir: Path, split: str) -> "ESAWindowStore":
        processed_dir = processed_dir.resolve()
        manifest_path = processed_dir / "manifest.json"
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Preprocessing manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_dir = processed_dir / split
        required = (
            "normalized_values",
            "observed_mask",
            "update_mask",
            "label_code",
            "observed_count",
            "timestamps_ns",
        )
        paths = {name: split_dir / f"{name}.npy" for name in required}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Processed arrays are missing: {missing}")
        arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
        channel_names = tuple(manifest["schema"]["channel_order"])
        expected = (len(channel_names), len(arrays["timestamps_ns"]))
        for name in ("normalized_values", "observed_mask", "update_mask", "label_code"):
            if arrays[name].shape != expected:
                raise ValueError(f"{name} shape {arrays[name].shape} != {expected}")
        if arrays["observed_count"].shape != (expected[1],):
            raise ValueError("observed_count must contain one value per timestamp")
        means = np.asarray(
            [manifest["normalization"]["mean"][name] for name in channel_names],
            dtype=np.float64,
        )
        scales = np.asarray(
            [manifest["normalization"]["scale"][name] for name in channel_names],
            dtype=np.float64,
        )
        raw_stds = np.asarray(
            [manifest["normalization"]["raw_std"][name] for name in channel_names],
            dtype=np.float64,
        )
        return cls(
            split=split,
            normalized_values=arrays["normalized_values"],
            observed_mask=arrays["observed_mask"],
            update_mask=arrays["update_mask"],
            label_code=arrays["label_code"],
            observed_count=arrays["observed_count"],
            timestamps_ns=arrays["timestamps_ns"],
            channel_names=channel_names,
            means=means,
            scales=scales,
            raw_stds=raw_stds,
            channel_metadata=tuple(manifest["channels"][name] for name in channel_names),
        )


def build_window_starts(
    observed_count: np.ndarray, window_length: int, stride: int
) -> np.ndarray:
    """Return split-local starts, retaining partial windows but dropping all-missing ones."""
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")
    length = int(observed_count.shape[0])
    if length < window_length:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, length - window_length + 1, stride, dtype=np.int64)
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(observed_count, dtype=np.int64)]
    )
    totals = cumulative[starts + window_length] - cumulative[starts]
    return starts[totals > 0]


def evenly_spaced_indices(total: int, requested: int) -> np.ndarray:
    if total <= 0 or requested <= 0:
        raise ValueError("total and requested must be positive")
    if requested > total:
        raise ValueError(f"Cannot select {requested} unique windows from {total}")
    if requested == total:
        return np.arange(total, dtype=np.int64)
    indices = np.rint(np.linspace(0, total - 1, requested)).astype(np.int64)
    if np.unique(indices).size != requested:
        raise RuntimeError("Evenly spaced selection unexpectedly produced duplicates")
    return indices


def deterministic_condition_mask(
    observed_mask: np.ndarray,
    missing_ratio: float,
    seed: int,
    window_start: int,
) -> np.ndarray:
    if not 0.0 <= missing_ratio <= 1.0:
        raise ValueError("missing_ratio must be in [0, 1]")
    observed = observed_mask.astype(np.uint8, copy=False)
    condition = observed.copy()
    observed_indices = np.flatnonzero(observed.reshape(-1))
    target_count = int(round(observed_indices.size * missing_ratio))
    if target_count:
        ratio_code = int(round(missing_ratio * 1_000_000))
        sequence = np.random.SeedSequence(
            [
                int(seed),
                int(window_start & 0xFFFFFFFF),
                int((window_start >> 32) & 0xFFFFFFFF),
                ratio_code,
            ]
        )
        selected = np.random.default_rng(sequence).choice(
            observed_indices, target_count, replace=False
        )
        condition.reshape(-1)[selected] = 0
    return condition


def save_evaluation_protocol(
    path: Path,
    store: ESAWindowStore,
    all_starts: np.ndarray,
    selected_indices: np.ndarray,
    missing_ratio: float,
    seed: int,
    window_length: int,
) -> dict:
    starts = all_starts[selected_indices]
    condition_masks = np.empty(
        (len(starts), window_length, len(store.channel_names)), dtype=np.uint8
    )
    target_counts = np.empty(len(starts), dtype=np.int64)
    observed_counts = np.empty(len(starts), dtype=np.int64)
    for row, start in enumerate(starts):
        observed = store.observed_mask[:, start : start + window_length].T
        condition = deterministic_condition_mask(observed, missing_ratio, seed, int(start))
        condition_masks[row] = condition
        observed_counts[row] = int(observed.sum())
        target_counts[row] = int((observed - condition).sum())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        window_starts=starts,
        window_indices=selected_indices,
        timestamp_ns=np.asarray(store.timestamps_ns[starts], dtype=np.int64),
        condition_masks=condition_masks,
        observed_counts=observed_counts,
        target_counts=target_counts,
        missing_ratio=np.float64(missing_ratio),
        seed=np.int64(seed),
    )
    return {
        "path": str(path.resolve()),
        "windows": int(len(starts)),
        "seed": int(seed),
        "requested_missing_ratio": float(missing_ratio),
        "target_points": int(target_counts.sum()),
        "observed_points": int(observed_counts.sum()),
        "actual_missing_ratio": float(target_counts.sum() / observed_counts.sum()),
    }


class ESAWindowDataset(Dataset):
    def __init__(
        self,
        store: ESAWindowStore,
        window_length: int = 96,
        stride: int = 48,
        starts: Sequence[int] | np.ndarray | None = None,
        condition_masks: np.ndarray | None = None,
    ) -> None:
        self.store = store
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.starts = (
            build_window_starts(store.observed_count, self.window_length, self.stride)
            if starts is None
            else np.asarray(starts, dtype=np.int64)
        )
        self.condition_masks = condition_masks
        if condition_masks is not None:
            expected = (len(self.starts), self.window_length, len(store.channel_names))
            if condition_masks.shape != expected:
                raise ValueError(f"condition_masks shape {condition_masks.shape} != {expected}")
        if self.starts.size and (
            self.starts.min() < 0
            or self.starts.max() + self.window_length > self.store.timestamps_ns.size
        ):
            raise ValueError("A window crosses the selected partition boundary")

    @classmethod
    def from_protocol(
        cls, store: ESAWindowStore, protocol_path: Path, window_length: int = 96
    ) -> "ESAWindowDataset":
        with np.load(protocol_path) as loaded:
            starts = loaded["window_starts"].copy()
            condition_masks = loaded["condition_masks"].copy()
        return cls(
            store,
            window_length=window_length,
            stride=window_length,
            starts=starts,
            condition_masks=condition_masks,
        )

    def __len__(self) -> int:
        return int(self.starts.size)

    def _assemble_item(
        self,
        index: int,
        values: np.ndarray,
        observed: np.ndarray,
        labels: np.ndarray,
        updates: np.ndarray,
    ) -> dict[str, np.ndarray | np.int64]:
        start = int(self.starts[index])
        if observed.sum() == 0:
            raise RuntimeError("An all-natural-missing window entered the dataset")
        values = np.asarray(values, dtype=np.float32).copy()
        observed = np.asarray(observed, dtype=np.float32).copy()
        values *= observed
        if self.condition_masks is None:
            condition = observed.copy()
        else:
            condition = self.condition_masks[index].astype(np.float32, copy=True)
            if np.any(condition > observed):
                raise ValueError("Protocol condition mask contains naturally missing points")
        return {
            "observed_data": values,
            "observed_mask": observed,
            "gt_mask": condition,
            "hist_mask": observed.copy(),
            "timepoints": np.arange(self.window_length, dtype=np.float32),
            "label_code": np.asarray(labels, dtype=np.uint8).copy(),
            "update_mask": np.asarray(updates, dtype=np.uint8).copy(),
            "window_start": np.int64(start),
            "timestamp_ns": np.int64(self.store.timestamps_ns[start]),
        }

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.int64]:
        start = int(self.starts[index])
        stop = start + self.window_length
        return self._assemble_item(
            index,
            self.store.normalized_values[:, start:stop].T,
            self.store.observed_mask[:, start:stop].T,
            self.store.label_code[:, start:stop].T,
            self.store.update_mask[:, start:stop].T,
        )

    def __getitems__(
        self, indices: list[int]
    ) -> list[dict[str, np.ndarray | np.int64]]:
        """Batch memmap reads: one gather per array instead of one read per window."""
        index_array = np.asarray(indices, dtype=np.int64)
        starts = self.starts[index_array]
        positions = starts[:, None] + np.arange(self.window_length, dtype=np.int64)[None, :]
        values = np.asarray(self.store.normalized_values[:, positions]).transpose(1, 2, 0)
        observed = np.asarray(self.store.observed_mask[:, positions]).transpose(1, 2, 0)
        labels = np.asarray(self.store.label_code[:, positions]).transpose(1, 2, 0)
        updates = np.asarray(self.store.update_mask[:, positions]).transpose(1, 2, 0)
        return [
            self._assemble_item(
                int(index), values[row], observed[row], labels[row], updates[row]
            )
            for row, index in enumerate(index_array)
        ]


class CyclicPermutationSampler(Sampler[int]):
    """Finite deterministic no-replacement epochs, resumable by sample offset."""

    def __init__(
        self, dataset_size: int, seed: int, total_samples: int, start_offset: int = 0
    ) -> None:
        if dataset_size <= 0 or total_samples < 0 or start_offset < 0:
            raise ValueError("Invalid sampler dimensions")
        if start_offset > total_samples:
            raise ValueError("start_offset exceeds total_samples")
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.total_samples = int(total_samples)
        self.start_offset = int(start_offset)

    def __len__(self) -> int:
        return self.total_samples - self.start_offset

    def __iter__(self) -> Iterator[int]:
        first_cycle = self.start_offset // self.dataset_size
        offset_in_cycle = self.start_offset % self.dataset_size
        remaining = len(self)
        cycle = first_cycle
        while remaining:
            generator = torch.Generator().manual_seed(self.seed + cycle)
            permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
            if cycle == first_cycle:
                permutation = permutation[offset_in_cycle:]
            take = min(remaining, len(permutation))
            yield from permutation[:take]
            remaining -= take
            cycle += 1


def make_training_loader(
    dataset: ESAWindowDataset,
    batch_size: int,
    seed: int,
    total_steps: int,
    start_step: int = 0,
) -> DataLoader:
    sampler = CyclicPermutationSampler(
        len(dataset),
        seed,
        total_samples=int(total_steps) * int(batch_size),
        start_offset=int(start_step) * int(batch_size),
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def make_evaluation_loader(dataset: ESAWindowDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
