from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
CSDI_ROOT = HERE.parents[1] / "CSDI"
STRUCTURED_ROOT = CSDI_ROOT / "ESA_mission1_structured"
for path in (CSDI_ROOT, STRUCTURED_ROOT, HERE):
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from ESA_mission1_structured.model import StructuredOnlyCSDI  # noqa: E402
from diff_models_graph import GraphBiasedDiffusion  # noqa: E402


class StructuredGraphCSDI(StructuredOnlyCSDI):
    """Structured-Only CSDI with a training-set graph bias on spatial attention."""

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        target_dim: int,
        structured_config: dict[str, Any],
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__(config, device, target_dim, structured_config)
        if adjacency.shape != (target_dim, target_dim):
            raise ValueError(
                f"adjacency shape {tuple(adjacency.shape)} does not match target_dim={target_dim}"
            )
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = GraphBiasedDiffusion(
            config["diffusion"], adjacency=adjacency, inputdim=input_dim
        )

    def graph_diagnostics(self) -> dict[str, Any]:
        return {
            "effective_strength_by_layer": self.diffmodel.graph_strengths(),
            "zero_initialized": True,
        }
