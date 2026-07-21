from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from masking import (  # noqa: E402
    FAMILY_NAMES,
    TrainingMaskConfig,
    gap_case_condition,
    nested_structured_conditions,
    structured_training_condition_mask,
)
from model import StructuredOnlyCSDI  # noqa: E402


RATIOS = (0.1, 0.5, 0.9)


def _targets(observed: np.ndarray, conditions: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    return {ratio: (observed - condition).astype(bool) for ratio, condition in conditions.items()}


def test_training_masks_are_valid_and_preserve_natural_missingness() -> None:
    observed = torch.ones((64, 76, 96))
    observed[:, 0, 0] = 0
    config = TrainingMaskConfig()
    generator = torch.Generator().manual_seed(7)
    condition, families, ratios = structured_training_condition_mask(
        observed, config, generator
    )
    assert condition.shape == observed.shape
    assert torch.all(condition <= observed)
    assert torch.all(condition[:, 0, 0] == 0)
    assert torch.all((observed - condition).sum(dim=(1, 2)) > 0)
    assert torch.all(condition.sum(dim=(1, 2)) > 0)
    assert int(families.min()) >= 0 and int(families.max()) < len(FAMILY_NAMES)
    assert torch.all((ratios >= 0.1) & (ratios <= 0.9))
    assert "random" not in FAMILY_NAMES


def test_training_family_probabilities_are_respected() -> None:
    observed = torch.ones((3000, 8, 12))
    generator = torch.Generator().manual_seed(11)
    _, families, _ = structured_training_condition_mask(
        observed, TrainingMaskConfig(), generator
    )
    fractions = torch.bincount(families, minlength=3).float() / len(families)
    torch.testing.assert_close(
        fractions,
        torch.tensor([1 / 3, 1 / 3, 1 / 3]),
        atol=0.03,
        rtol=0,
    )


def test_each_training_family_has_only_its_declared_structure() -> None:
    observed = torch.ones((1, 76, 96))
    configurations = (
        (TrainingMaskConfig((1.0, 0.0, 0.0), 0.5, 0.5), "time_block"),
        (TrainingMaskConfig((0.0, 1.0, 0.0), 0.5, 0.5), "channel_dropout"),
        (TrainingMaskConfig((0.0, 0.0, 1.0), 0.5, 0.5), "rectangle"),
    )
    for index, (config, family) in enumerate(configurations):
        condition, family_ids, _ = structured_training_condition_mask(
            observed, config, torch.Generator().manual_seed(100 + index)
        )
        target = (observed - condition)[0].bool()
        assert FAMILY_NAMES[int(family_ids[0])] == family
        active_channels = target.any(dim=1)
        active_times = target.any(dim=0)
        if family == "time_block":
            assert int(active_channels.sum()) == 76
            assert int(active_times.sum()) == 48
        elif family == "channel_dropout":
            assert int(active_channels.sum()) == 38
            assert int(active_times.sum()) == 96
        else:
            assert int(active_channels.sum()) == 54
            assert int(active_times.sum()) == 68
        assert int(target.sum()) == int(active_channels.sum() * active_times.sum())


def test_nested_time_blocks_are_contiguous_and_nested() -> None:
    observed = np.ones((96, 76), dtype=np.uint8)
    conditions = nested_structured_conditions(observed, "time_block", RATIOS, 2101, 480)
    targets = _targets(observed, conditions)
    for ratio, expected_length in zip(RATIOS, (10, 48, 86)):
        active_times = np.flatnonzero(targets[ratio].any(axis=1))
        assert active_times.size == expected_length
        np.testing.assert_array_equal(active_times, np.arange(active_times[0], active_times[-1] + 1))
    assert np.all(~targets[0.1] | targets[0.5])
    assert np.all(~targets[0.5] | targets[0.9])


def test_nested_channel_dropout_masks_whole_channels() -> None:
    observed = np.ones((96, 76), dtype=np.uint8)
    conditions = nested_structured_conditions(
        observed, "channel_dropout", RATIOS, 2201, 960
    )
    targets = _targets(observed, conditions)
    for ratio, expected_channels in zip(RATIOS, (8, 38, 68)):
        selected = targets[ratio].any(axis=0)
        assert int(selected.sum()) == expected_channels
        assert np.all(targets[ratio][:, selected])
    assert np.all(~targets[0.1] | targets[0.5])
    assert np.all(~targets[0.5] | targets[0.9])


def test_nested_rectangles_have_contiguous_time_and_nested_area() -> None:
    observed = np.ones((96, 76), dtype=np.uint8)
    conditions = nested_structured_conditions(observed, "rectangle", RATIOS, 2301, 1440)
    targets = _targets(observed, conditions)
    for ratio in RATIOS:
        active_times = np.flatnonzero(targets[ratio].any(axis=1))
        active_channels = np.flatnonzero(targets[ratio].any(axis=0))
        np.testing.assert_array_equal(active_times, np.arange(active_times[0], active_times[-1] + 1))
        assert targets[ratio].sum() == active_times.size * active_channels.size
    assert np.all(~targets[0.1] | targets[0.5])
    assert np.all(~targets[0.5] | targets[0.9])


def test_structured_protocols_are_reproducible() -> None:
    observed = np.ones((96, 76), dtype=np.uint8)
    first = nested_structured_conditions(observed, "rectangle", RATIOS, 2301, 123456)
    second = nested_structured_conditions(observed, "rectangle", RATIOS, 2301, 123456)
    for ratio in RATIOS:
        np.testing.assert_array_equal(first[ratio], second[ratio])


def test_gap_cases_use_affected_channels_only() -> None:
    observed = np.ones((96, 76), dtype=np.uint8)
    affected = np.arange(52)
    onset = gap_case_condition(observed, affected, "gap_onset")
    sustained = gap_case_condition(observed, affected, "gap_sustained")
    onset_target = observed - onset
    sustained_target = observed - sustained
    assert onset_target.sum() == 48 * 52
    assert sustained_target.sum() == 96 * 52
    assert onset_target[:, 52:].sum() == 0
    assert sustained_target[:, 52:].sum() == 0


def test_structured_mask_statistics_survive_state_dict_resume() -> None:
    model_config = {
        "model": {
            "is_unconditional": False,
            "timeemb": 16,
            "featureemb": 4,
            "target_strategy": "structured_only",
        },
        "diffusion": {
            "layers": 1,
            "channels": 8,
            "nheads": 1,
            "diffusion_embedding_dim": 16,
            "beta_start": 0.0001,
            "beta_end": 0.1,
            "num_steps": 2,
            "schedule": "quad",
            "is_linear": False,
        },
    }
    structured_config = {
        "family_probabilities": {
            "time_block": 1 / 3,
            "channel_dropout": 1 / 3,
            "rectangle": 1 / 3,
        },
        "min_ratio": 0.1,
        "max_ratio": 0.9,
    }
    first = StructuredOnlyCSDI(
        model_config, torch.device("cpu"), 76, structured_config
    )
    with torch.no_grad():
        first.mask_family_counts.copy_(torch.tensor([3, 4, 5]))
        first.mask_requested_ratio_sum.copy_(torch.tensor([1.2, 2.0, 3.0]))
        first.mask_target_points.copy_(torch.tensor([100, 200, 300]))
        first.mask_observed_points.copy_(torch.tensor([300, 400, 500]))
    resumed = StructuredOnlyCSDI(
        model_config, torch.device("cpu"), 76, structured_config
    )
    resumed.load_state_dict(first.state_dict())
    assert resumed.mask_statistics() == first.mask_statistics()
