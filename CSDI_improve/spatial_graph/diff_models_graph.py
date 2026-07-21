from __future__ import annotations

import torch
import torch.nn as nn

from diff_models import ResidualBlock, diff_CSDI


class GraphBiasedResidualBlock(ResidualBlock):
    """CSDI residual block with a zero-initialized bias on feature attention."""

    def __init__(
        self,
        side_dim: int,
        channels: int,
        diffusion_embedding_dim: int,
        nheads: int,
        adjacency: torch.Tensor,
        graph_bias_scale: float = 2.0,
        is_linear: bool = False,
    ) -> None:
        if is_linear:
            raise ValueError("graph-biased attention currently requires is_linear=false")
        super().__init__(side_dim, channels, diffusion_embedding_dim, nheads, is_linear)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        self.register_buffer("graph_adjacency", adjacency.detach().float().clone())
        self.graph_strength = nn.Parameter(torch.zeros(()))
        self.graph_bias_scale = float(graph_bias_scale)

    def forward_feature(self, y: torch.Tensor, base_shape: torch.Size) -> torch.Tensor:
        batch, channels, features, length = base_shape
        if features == 1:
            return y
        if features != self.graph_adjacency.shape[0]:
            raise ValueError(
                f"feature count {features} does not match graph size "
                f"{self.graph_adjacency.shape[0]}"
            )
        y = (
            y.reshape(batch, channels, features, length)
            .permute(0, 3, 1, 2)
            .reshape(batch * length, channels, features)
        )
        graph_bias = (
            torch.tanh(self.graph_strength)
            * self.graph_bias_scale
            * self.graph_adjacency.to(dtype=y.dtype)
        )
        y = self.feature_layer(y.permute(2, 0, 1), mask=graph_bias).permute(1, 2, 0)
        return (
            y.reshape(batch, length, channels, features)
            .permute(0, 2, 3, 1)
            .reshape(batch, channels, features * length)
        )


class GraphBiasedDiffusion(diff_CSDI):
    def __init__(self, config: dict, adjacency: torch.Tensor, inputdim: int = 2) -> None:
        super().__init__(config, inputdim=inputdim)
        self.residual_layers = nn.ModuleList(
            [
                GraphBiasedResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    adjacency=adjacency,
                    graph_bias_scale=float(config.get("graph_bias_scale", 2.0)),
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def graph_strengths(self) -> list[float]:
        return [
            float(torch.tanh(layer.graph_strength).detach().cpu())
            for layer in self.residual_layers
        ]
