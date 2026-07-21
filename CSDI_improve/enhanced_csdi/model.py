from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
CSDI_ROOT = HERE.parents[1] / "CSDI"
STRUCTURED_ROOT = CSDI_ROOT / "ESA_mission1_structured"
for source_root in (STRUCTURED_ROOT, CSDI_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from masking import FAMILY_NAMES, structured_training_condition_mask  # noqa: E402
from ESA_mission1_structured.model import StructuredOnlyCSDI  # noqa: E402

try:  # package import
    from .architecture import EnhancedDiffusion, MaskAwareFrequencyEncoder  # type: ignore
except ImportError:  # direct script execution from this directory
    from architecture import EnhancedDiffusion, MaskAwareFrequencyEncoder  # noqa: E402


class EnhancedStructuredCSDI(StructuredOnlyCSDI):
    """Structured-mask CSDI with graph-guided feature attention and STFT context."""

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        target_dim: int,
        structured_config: dict[str, Any],
        graph_adjacency: torch.Tensor,
    ) -> None:
        # CSDI mutates diffusion.side_dim, so keep the caller's configuration clean.
        model_config = copy.deepcopy(config)
        super().__init__(model_config, device, target_dim, structured_config)
        frequency_config = model_config["frequency"]
        self.frequency_encoder = MaskAwareFrequencyEncoder(
            output_dim=int(frequency_config["output_dim"]),
            n_fft=int(frequency_config["n_fft"]),
            hop_length=int(frequency_config["hop_length"]),
            min_coverage=float(frequency_config.get("min_coverage", 0.1)),
        )
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = EnhancedDiffusion(
            config=model_config["diffusion"],
            graph_adjacency=graph_adjacency,
            frequency_dim=int(frequency_config["output_dim"]),
            inputdim=input_dim,
        )

    def get_side_info(
        self,
        observed_tp: torch.Tensor,
        cond_mask: torch.Tensor,
        observed_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observed_data is None:
            raise ValueError(
                "EnhancedStructuredCSDI.get_side_info also requires observed_data "
                "to build mask-aware frequency conditions"
            )
        base_side_info = super().get_side_info(observed_tp, cond_mask)
        frequency_info = self.frequency_encoder(observed_data, cond_mask)
        return torch.cat([base_side_info, frequency_info], dim=1)

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
        side_info = self.get_side_info(observed_tp, condition, observed_data)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, condition, observed_mask, side_info, is_train)

    def evaluate(
        self, batch: dict[str, torch.Tensor], n_samples: int
    ) -> tuple[torch.Tensor, ...]:
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)
        with torch.no_grad():
            condition = gt_mask
            target_mask = observed_mask - condition
            side_info = self.get_side_info(observed_tp, condition, observed_data)
            samples = self.impute(observed_data, condition, side_info, n_samples)
            for row in range(len(cut_length)):
                target_mask[row, ..., : cut_length[row].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp

    def architecture_diagnostics(self) -> dict[str, Any]:
        return {
            **self.diffmodel.diagnostics(),
            "frequency": {
                "output_dim": self.frequency_encoder.output_dim,
                "n_fft": self.frequency_encoder.n_fft,
                "hop_length": self.frequency_encoder.hop_length,
            },
            "graph": {
                "channel_count": int(self.diffmodel.residual_layers[0].graph_adjacency.shape[0]),
                "directed_edges": int(
                    (self.diffmodel.residual_layers[0].graph_adjacency > 0).sum().item()
                ),
            },
        }
