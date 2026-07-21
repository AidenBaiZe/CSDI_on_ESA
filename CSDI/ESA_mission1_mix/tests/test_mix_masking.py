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

from ESA_mission1_mix.model import (  # noqa: E402
    MIX_FAMILY_NAMES,
    RandomStructuredMixCSDI,
)


def _model_config() -> dict:
    return {
        "model": {
            "is_unconditional": False,
            "timeemb": 16,
            "featureemb": 4,
            "target_strategy": "random_structured_mix",
            "random_mask_min_ratio": 0.0,
            "random_mask_max_ratio": 1.0,
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


def _mix_config(random_probability: float) -> dict:
    return {
        "random_probability": random_probability,
        "structured_family_probabilities": {
            "time_block": 1 / 3,
            "channel_dropout": 1 / 3,
            "rectangle": 1 / 3,
        },
        "structured_min_ratio": 0.1,
        "structured_max_ratio": 0.9,
    }


def _make(random_probability: float) -> RandomStructuredMixCSDI:
    return RandomStructuredMixCSDI(
        _model_config(), torch.device("cpu"), 76, _mix_config(random_probability)
    )


def test_mix_masks_preserve_natural_missingness() -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    observed = torch.ones((256, 76, 96))
    observed[:, 0, 0] = 0
    condition, family_ids, ratios = _make(0.5)._mixed_training_condition(observed)
    assert condition.shape == observed.shape
    assert torch.all(condition <= observed)
    assert torch.all(condition[:, 0, 0] == 0)
    assert int(family_ids.min()) >= 0
    assert int(family_ids.max()) < len(MIX_FAMILY_NAMES)
    assert torch.all((ratios >= 0.0) & (ratios <= 1.0))


def test_random_probability_extremes_select_expected_families() -> None:
    observed = torch.ones((128, 76, 96))
    for probability, expected_random in ((1.0, 128), (0.0, 0)):
        torch.manual_seed(11)
        np.random.seed(11)
        _, family_ids, _ = _make(probability)._mixed_training_condition(observed)
        assert int((family_ids == 0).sum()) == expected_random


def test_half_mix_has_expected_family_proportions() -> None:
    torch.manual_seed(19)
    np.random.seed(19)
    observed = torch.ones((6000, 8, 12))
    model = RandomStructuredMixCSDI(
        _model_config(), torch.device("cpu"), 8, _mix_config(0.5)
    )
    _, family_ids, _ = model._mixed_training_condition(observed)
    fractions = torch.bincount(family_ids, minlength=4).float() / len(family_ids)
    torch.testing.assert_close(
        fractions,
        torch.tensor([0.5, 1 / 6, 1 / 6, 1 / 6]),
        atol=0.025,
        rtol=0,
    )


def test_mix_mask_statistics_survive_state_dict_resume() -> None:
    first = _make(0.5)
    with torch.no_grad():
        first.mask_family_counts.copy_(torch.tensor([6, 2, 2, 2]))
        first.mask_ratio_sum.copy_(torch.tensor([3.0, 1.0, 1.0, 1.0]))
        first.mask_target_points.copy_(torch.tensor([60, 20, 20, 20]))
        first.mask_observed_points.copy_(torch.tensor([120, 40, 40, 40]))
    resumed = _make(0.5)
    resumed.load_state_dict(first.state_dict())
    assert resumed.mask_statistics() == first.mask_statistics()


def test_forward_loss_is_finite_and_records_all_windows() -> None:
    torch.manual_seed(23)
    np.random.seed(23)
    model = _make(0.5)
    batch = {
        "observed_data": torch.randn(8, 96, 76),
        "observed_mask": torch.ones(8, 96, 76),
        "gt_mask": torch.ones(8, 96, 76),
        "hist_mask": torch.ones(8, 96, 76),
        "timepoints": torch.arange(96).repeat(8, 1).float(),
    }
    loss = model(batch, is_train=1)
    assert torch.isfinite(loss)
    assert model.mask_statistics()["windows_seen"] == 8
