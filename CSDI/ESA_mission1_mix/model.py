from __future__ import annotations

from typing import Any

import torch

from main_model import CSDI_Physio
from ESA_mission1_structured.masking import (
    FAMILY_NAMES as STRUCTURED_FAMILY_NAMES,
    TrainingMaskConfig,
    structured_training_condition_mask,
)


MIX_FAMILY_NAMES = ("random", *STRUCTURED_FAMILY_NAMES)


class RandomStructuredMixCSDI(CSDI_Physio):
    """CSDI trained with a per-sample mixture of random and structured targets."""

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        target_dim: int,
        mix_config: dict[str, Any],
    ) -> None:
        super().__init__(config, device, target_dim=target_dim)
        self.random_probability = float(mix_config.get("random_probability", 0.5))
        if not 0.0 <= self.random_probability <= 1.0:
            raise ValueError("random_probability must be in [0, 1]")

        probabilities = tuple(
            float(mix_config["structured_family_probabilities"][name])
            for name in STRUCTURED_FAMILY_NAMES
        )
        self.structured_mask_config = TrainingMaskConfig(
            family_probabilities=probabilities,
            min_ratio=float(mix_config.get("structured_min_ratio", 0.1)),
            max_ratio=float(mix_config.get("structured_max_ratio", 0.9)),
        )
        self.structured_mask_config.validate()

        family_count = len(MIX_FAMILY_NAMES)
        self.register_buffer(
            "mask_family_counts", torch.zeros(family_count, dtype=torch.long)
        )
        self.register_buffer(
            "mask_ratio_sum", torch.zeros(family_count, dtype=torch.float64)
        )
        self.register_buffer(
            "mask_target_points", torch.zeros(family_count, dtype=torch.long)
        )
        self.register_buffer(
            "mask_observed_points", torch.zeros(family_count, dtype=torch.long)
        )

    def _mixed_training_condition(
        self, observed_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        random_condition = self.get_randmask(observed_mask)
        structured_condition, structured_ids, structured_ratios = (
            structured_training_condition_mask(
                observed_mask, self.structured_mask_config
            )
        )
        choose_random = (
            torch.rand(observed_mask.shape[0], device=observed_mask.device)
            < self.random_probability
        )
        condition = torch.where(
            choose_random.view(-1, 1, 1),
            random_condition,
            structured_condition,
        )
        family_ids = structured_ids + 1
        family_ids = torch.where(
            choose_random, torch.zeros_like(family_ids), family_ids
        )

        observed_points = observed_mask.sum(dim=(1, 2))
        random_ratios = (observed_mask - random_condition).sum(dim=(1, 2)) / (
            observed_points.clamp_min(1)
        )
        ratios = torch.where(choose_random, random_ratios, structured_ratios)
        return condition, family_ids, ratios

    def forward(self, batch: dict[str, torch.Tensor], is_train: int = 1) -> torch.Tensor:
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
        ) = self.process_data(batch)
        if is_train == 1:
            condition, family_ids, ratios = self._mixed_training_condition(
                observed_mask
            )
            with torch.no_grad():
                self.mask_family_counts += torch.bincount(
                    family_ids, minlength=len(MIX_FAMILY_NAMES)
                ).to(self.mask_family_counts)
                target_per_window = (observed_mask - condition).sum(
                    dim=(1, 2)
                ).long()
                observed_per_window = observed_mask.sum(dim=(1, 2)).long()
                self.mask_ratio_sum.scatter_add_(
                    0, family_ids, ratios.to(torch.float64)
                )
                self.mask_target_points.scatter_add_(
                    0, family_ids, target_per_window
                )
                self.mask_observed_points.scatter_add_(
                    0, family_ids, observed_per_window
                )
        else:
            condition = gt_mask

        side_info = self.get_side_info(observed_tp, condition)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(
            observed_data,
            condition,
            observed_mask,
            side_info,
            is_train,
        )

    def mask_statistics(self) -> dict[str, Any]:
        counts = self.mask_family_counts.detach().cpu().tolist()
        total = int(sum(counts))
        ratio_sums = self.mask_ratio_sum.detach().cpu().tolist()
        observed_by_family = self.mask_observed_points.detach().cpu().tolist()
        target_by_family = self.mask_target_points.detach().cpu().tolist()
        observed = int(sum(observed_by_family))
        target = int(sum(target_by_family))
        return {
            "family_counts": dict(zip(MIX_FAMILY_NAMES, counts)),
            "family_fractions": {
                name: count / total if total else 0.0
                for name, count in zip(MIX_FAMILY_NAMES, counts)
            },
            "windows_seen": total,
            "sampled_ratio_mean": {
                name: ratio_sum / count if count else 0.0
                for name, ratio_sum, count in zip(
                    MIX_FAMILY_NAMES, ratio_sums, counts
                )
            },
            "observed_points_by_family": dict(
                zip(MIX_FAMILY_NAMES, observed_by_family)
            ),
            "target_points_by_family": dict(
                zip(MIX_FAMILY_NAMES, target_by_family)
            ),
            "actual_target_fraction_by_family": {
                name: family_target / family_observed if family_observed else 0.0
                for name, family_target, family_observed in zip(
                    MIX_FAMILY_NAMES, target_by_family, observed_by_family
                )
            },
            "observed_points": observed,
            "target_points": target,
            "actual_target_fraction": target / observed if observed else 0.0,
        }
