from __future__ import annotations

from typing import Any

import torch

from main_model import CSDI_Physio

from masking import FAMILY_NAMES, TrainingMaskConfig, structured_training_condition_mask


class StructuredOnlyCSDI(CSDI_Physio):
    """Mission 1 CSDI trained exclusively with three structured mask families."""

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        target_dim: int,
        structured_config: dict[str, Any],
    ) -> None:
        super().__init__(config, device, target_dim=target_dim)
        probabilities = tuple(
            float(structured_config["family_probabilities"][name])
            for name in FAMILY_NAMES
        )
        self.structured_mask_config = TrainingMaskConfig(
            family_probabilities=probabilities,
            min_ratio=float(structured_config.get("min_ratio", 0.0)),
            max_ratio=float(structured_config.get("max_ratio", 1.0)),
        )
        self.structured_mask_config.validate()
        self.register_buffer(
            "mask_family_counts",
            torch.zeros(len(FAMILY_NAMES), dtype=torch.long),
        )
        self.register_buffer(
            "mask_requested_ratio_sum",
            torch.zeros(len(FAMILY_NAMES), dtype=torch.float64),
        )
        self.register_buffer(
            "mask_target_points",
            torch.zeros(len(FAMILY_NAMES), dtype=torch.long),
        )
        self.register_buffer(
            "mask_observed_points",
            torch.zeros(len(FAMILY_NAMES), dtype=torch.long),
        )

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
            condition, family_ids, requested_ratios = structured_training_condition_mask(
                observed_mask, self.structured_mask_config
            )
            with torch.no_grad():
                self.mask_family_counts += torch.bincount(
                    family_ids, minlength=len(FAMILY_NAMES)
                ).to(self.mask_family_counts)
                target_per_window = (observed_mask - condition).sum(dim=(1, 2)).long()
                observed_per_window = observed_mask.sum(dim=(1, 2)).long()
                self.mask_requested_ratio_sum.scatter_add_(
                    0, family_ids, requested_ratios.to(torch.float64)
                )
                self.mask_target_points.scatter_add_(0, family_ids, target_per_window)
                self.mask_observed_points.scatter_add_(0, family_ids, observed_per_window)
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
        requested_sums = self.mask_requested_ratio_sum.detach().cpu().tolist()
        observed_by_family = self.mask_observed_points.detach().cpu().tolist()
        target_by_family = self.mask_target_points.detach().cpu().tolist()
        observed = int(sum(observed_by_family))
        target = int(sum(target_by_family))
        return {
            "family_counts": dict(zip(FAMILY_NAMES, counts)),
            "family_fractions": {
                name: count / total if total else 0.0
                for name, count in zip(FAMILY_NAMES, counts)
            },
            "windows_seen": total,
            "requested_ratio_mean": {
                name: requested_sum / count if count else 0.0
                for name, requested_sum, count in zip(
                    FAMILY_NAMES, requested_sums, counts
                )
            },
            "observed_points_by_family": dict(
                zip(FAMILY_NAMES, observed_by_family)
            ),
            "target_points_by_family": dict(zip(FAMILY_NAMES, target_by_family)),
            "actual_target_fraction_by_family": {
                name: family_target / family_observed if family_observed else 0.0
                for name, family_target, family_observed in zip(
                    FAMILY_NAMES, target_by_family, observed_by_family
                )
            },
            "observed_points": observed,
            "target_points": target,
            "actual_target_fraction": target / observed if observed else 0.0,
        }
