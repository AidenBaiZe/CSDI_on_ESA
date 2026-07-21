from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


FAMILY_NAMES = ("time_block", "channel_dropout", "rectangle")
FAMILY_TO_ID = {name: index for index, name in enumerate(FAMILY_NAMES)}


@dataclass(frozen=True)
class TrainingMaskConfig:
    family_probabilities: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    min_ratio: float = 0.1
    max_ratio: float = 0.9

    def validate(self) -> None:
        if len(self.family_probabilities) != len(FAMILY_NAMES):
            raise ValueError("One probability is required for each mask family")
        if any(value < 0 for value in self.family_probabilities):
            raise ValueError("Mask family probabilities must be non-negative")
        if not math.isclose(sum(self.family_probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("Mask family probabilities must sum to one")
        if not 0.0 <= self.min_ratio <= self.max_ratio <= 1.0:
            raise ValueError("Mask ratios must satisfy 0 <= min <= max <= 1")


def _rand(
    shape: tuple[int, ...] | tuple[()],
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=generator)


def _randint(
    high: int, device: torch.device, generator: torch.Generator | None
) -> int:
    if high <= 1:
        return 0
    return int(torch.randint(high, (1,), device=device, generator=generator).item())


def _ensure_target_and_context(
    condition: torch.Tensor,
    observed: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    observed_indices = torch.nonzero(observed > 0, as_tuple=False)
    if observed_indices.numel() == 0:
        raise ValueError("Cannot mask an all-natural-missing sample")
    target = (observed > 0) & (condition <= 0)
    if not bool(target.any().item()):
        chosen = observed_indices[
            _randint(len(observed_indices), observed.device, generator)
        ]
        condition[tuple(chosen.tolist())] = 0.0
        target = (observed > 0) & (condition <= 0)
    if not bool((condition > 0).any().item()) and int(observed_indices.shape[0]) > 1:
        target_indices = torch.nonzero(target, as_tuple=False)
        chosen = target_indices[
            _randint(len(target_indices), observed.device, generator)
        ]
        condition[tuple(chosen.tolist())] = 1.0
    return condition


def _time_block_condition(
    observed: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    channels, length = observed.shape
    block_length = min(max(int(round(length * ratio)), 1), max(length - 1, 1))
    start = _randint(length - block_length + 1, observed.device, generator)
    condition = observed.clone()
    condition[:, start : start + block_length] = 0.0
    return _ensure_target_and_context(condition, observed, generator)


def _channel_dropout_condition(
    observed: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    channels, _ = observed.shape
    channel_count = min(max(int(round(channels * ratio)), 1), max(channels - 1, 1))
    order = torch.randperm(channels, device=observed.device, generator=generator)
    condition = observed.clone()
    condition[order[:channel_count], :] = 0.0
    return _ensure_target_and_context(condition, observed, generator)


def _rectangle_condition(
    observed: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    channels, length = observed.shape
    side_fraction = math.sqrt(max(ratio, 0.0))
    channel_count = min(
        max(int(round(channels * side_fraction)), 1), max(channels - 1, 1)
    )
    block_length = min(
        max(int(round(length * side_fraction)), 1), max(length - 1, 1)
    )
    channel_order = torch.randperm(
        channels, device=observed.device, generator=generator
    )
    start = _randint(length - block_length + 1, observed.device, generator)
    condition = observed.clone()
    selected = channel_order[:channel_count]
    condition[selected, start : start + block_length] = 0.0
    return _ensure_target_and_context(condition, observed, generator)


def structured_training_condition_mask(
    observed_mask: torch.Tensor,
    config: TrainingMaskConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one independently sampled mask family and severity per batch item.

    Input and output shapes are (batch, channels, time). The returned family ids
    follow ``FAMILY_NAMES`` and the returned ratios are the requested severities.
    """

    config.validate()
    if observed_mask.ndim != 3:
        raise ValueError("observed_mask must have shape (batch, channels, time)")
    device = observed_mask.device
    probabilities = torch.tensor(
        config.family_probabilities, dtype=torch.float64, device=device
    )
    cumulative = torch.cumsum(probabilities, dim=0)
    choices = _rand((observed_mask.shape[0],), device, generator).to(torch.float64)
    family_ids = torch.searchsorted(cumulative, choices, right=False).clamp_max(
        len(FAMILY_NAMES) - 1
    )
    batch, channels, length = observed_mask.shape
    raw_ratios = _rand((batch,), device, generator)
    ratios = config.min_ratio + raw_ratios * (config.max_ratio - config.min_ratio)

    time_lengths = torch.round(ratios * length).long().clamp(1, length - 1)
    time_start_ranges = length - time_lengths + 1
    time_starts = torch.floor(
        _rand((batch,), device, generator) * time_start_ranges
    ).long()
    time_index = torch.arange(length, device=device).view(1, 1, length)
    time_regions = (
        (time_index >= time_starts.view(batch, 1, 1))
        & (time_index < (time_starts + time_lengths).view(batch, 1, 1))
    ).expand(-1, channels, -1)

    channel_counts = torch.round(ratios * channels).long().clamp(1, channels - 1)
    channel_scores = _rand((batch, channels), device, generator)
    channel_ranks = channel_scores.argsort(dim=1).argsort(dim=1)
    channel_regions = (
        channel_ranks < channel_counts.view(batch, 1)
    ).unsqueeze(2).expand(-1, -1, length)

    side_fraction = torch.sqrt(ratios)
    rectangle_channel_counts = (
        torch.round(side_fraction * channels).long().clamp(1, channels - 1)
    )
    rectangle_lengths = (
        torch.round(side_fraction * length).long().clamp(1, length - 1)
    )
    rectangle_start_ranges = length - rectangle_lengths + 1
    rectangle_starts = torch.floor(
        _rand((batch,), device, generator) * rectangle_start_ranges
    ).long()
    rectangle_channel_scores = _rand((batch, channels), device, generator)
    rectangle_channel_ranks = rectangle_channel_scores.argsort(dim=1).argsort(dim=1)
    rectangle_channels = rectangle_channel_ranks < rectangle_channel_counts.view(
        batch, 1
    )
    rectangle_times = (
        (time_index >= rectangle_starts.view(batch, 1, 1))
        & (
            time_index
            < (rectangle_starts + rectangle_lengths).view(batch, 1, 1)
        )
    )
    rectangle_regions = rectangle_channels.unsqueeze(2) & rectangle_times

    regions = torch.where(
        (family_ids == FAMILY_TO_ID["time_block"]).view(batch, 1, 1),
        time_regions,
        torch.where(
            (family_ids == FAMILY_TO_ID["channel_dropout"]).view(batch, 1, 1),
            channel_regions,
            rectangle_regions,
        ),
    )
    conditions = observed_mask * (~regions).to(observed_mask.dtype)

    target_counts = (observed_mask - conditions).sum(dim=(1, 2))
    context_counts = conditions.sum(dim=(1, 2))
    invalid_rows = torch.nonzero(
        (target_counts <= 0) | (context_counts <= 0), as_tuple=False
    ).flatten()
    for row in invalid_rows.tolist():
        conditions[row] = _ensure_target_and_context(
            conditions[row], observed_mask[row], generator
        )
    return conditions, family_ids, ratios


def _numpy_rng(seed: int, window_start: int, family: str) -> np.random.Generator:
    family_code = FAMILY_TO_ID.get(family, 10_000 + sum(map(ord, family)))
    sequence = np.random.SeedSequence(
        [
            int(seed),
            int(window_start & 0xFFFFFFFF),
            int((window_start >> 32) & 0xFFFFFFFF),
            int(family_code),
        ]
    )
    return np.random.default_rng(sequence)


def nested_structured_conditions(
    observed_mask: np.ndarray,
    family: str,
    ratios: Iterable[float],
    seed: int,
    window_start: int,
    return_metadata: bool = False,
) -> dict[float, np.ndarray] | tuple[dict[float, np.ndarray], dict[str, object]]:
    """Build deterministic nested evaluation masks for one window.

    ``observed_mask`` is time-major with shape (time, channels).
    """

    observed = observed_mask.astype(np.uint8, copy=False)
    if observed.ndim != 2:
        raise ValueError("observed_mask must have shape (time, channels)")
    length, channels = observed.shape
    ordered_ratios = sorted({float(value) for value in ratios})
    if not ordered_ratios or ordered_ratios[0] <= 0 or ordered_ratios[-1] >= 1:
        raise ValueError("Evaluation ratios must lie strictly between zero and one")
    rng = _numpy_rng(seed, int(window_start), family)
    regions: dict[float, np.ndarray] = {}

    channel_order = np.arange(channels, dtype=np.int64)
    time_starts: dict[float, int] = {}
    time_lengths: dict[float, int] = {}
    channel_counts: dict[float, int] = {}

    if family == "time_block":
        lengths = {
            ratio: min(max(int(round(ratio * length)), 1), length - 1)
            for ratio in ordered_ratios
        }
        outer_length = lengths[ordered_ratios[-1]]
        outer_start = int(rng.integers(0, length - outer_length + 1))
        for ratio in ordered_ratios:
            block_length = lengths[ratio]
            start = outer_start + (outer_length - block_length) // 2
            region = np.zeros_like(observed, dtype=bool)
            region[start : start + block_length, :] = True
            regions[ratio] = region
            time_starts[ratio] = start
            time_lengths[ratio] = block_length
            channel_counts[ratio] = channels
    elif family == "channel_dropout":
        channel_order = rng.permutation(channels)
        for ratio in ordered_ratios:
            count = min(max(int(round(ratio * channels)), 1), channels - 1)
            region = np.zeros_like(observed, dtype=bool)
            region[:, channel_order[:count]] = True
            regions[ratio] = region
            time_starts[ratio] = 0
            time_lengths[ratio] = length
            channel_counts[ratio] = count
    elif family == "rectangle":
        channel_order = rng.permutation(channels)
        rectangle_channel_counts = {
            ratio: min(
                max(int(round(math.sqrt(ratio) * channels)), 1), channels - 1
            )
            for ratio in ordered_ratios
        }
        lengths = {
            ratio: min(
                max(int(round(math.sqrt(ratio) * length)), 1), length - 1
            )
            for ratio in ordered_ratios
        }
        outer_length = lengths[ordered_ratios[-1]]
        outer_start = int(rng.integers(0, length - outer_length + 1))
        for ratio in ordered_ratios:
            block_length = lengths[ratio]
            start = outer_start + (outer_length - block_length) // 2
            region = np.zeros_like(observed, dtype=bool)
            region[
                start : start + block_length,
                channel_order[: rectangle_channel_counts[ratio]],
            ] = True
            regions[ratio] = region
            time_starts[ratio] = start
            time_lengths[ratio] = block_length
            channel_counts[ratio] = rectangle_channel_counts[ratio]
    else:
        raise ValueError(f"Unsupported structured family: {family}")

    conditions = {}
    previous_target: np.ndarray | None = None
    for ratio in ordered_ratios:
        condition = observed.copy()
        condition[regions[ratio]] = 0
        target = (observed - condition).astype(bool)
        if not target.any() or not condition.any():
            raise ValueError("Structured evaluation mask lacks target or context points")
        if previous_target is not None and not np.all(~previous_target | target):
            raise AssertionError("Structured evaluation targets are not nested")
        previous_target = target
        conditions[ratio] = condition
    if not return_metadata:
        return conditions
    return conditions, {
        "channel_order": channel_order.astype(np.int16),
        "time_starts": time_starts,
        "time_lengths": time_lengths,
        "channel_counts": channel_counts,
    }


def gap_case_condition(
    observed_mask: np.ndarray,
    affected_channels: np.ndarray,
    case: str,
) -> np.ndarray:
    observed = observed_mask.astype(np.uint8, copy=False)
    length, channels = observed.shape
    affected = np.asarray(affected_channels, dtype=np.int64)
    if affected.size == 0 or affected.min() < 0 or affected.max() >= channels:
        raise ValueError("Invalid affected channel indices")
    condition = observed.copy()
    if case == "gap_onset":
        condition[length // 2 :, affected] = 0
    elif case == "gap_sustained":
        condition[:, affected] = 0
    else:
        raise ValueError(f"Unsupported real-gap case: {case}")
    if not (observed - condition).any() or not condition.any():
        raise ValueError("Real-gap case lacks target or context points")
    return condition
