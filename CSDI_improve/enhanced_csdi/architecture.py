from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CSDI_ROOT = HERE.parents[1] / "CSDI"
if str(CSDI_ROOT) not in sys.path:
    sys.path.insert(0, str(CSDI_ROOT))

from diff_models import (  # noqa: E402
    Conv1d_with_init,
    DiffusionEmbedding,
    ResidualBlock,
)


class MaskAwareFrequencyEncoder(nn.Module):
    """Encode conditional observations into time-aligned STFT features."""

    def __init__(
        self,
        output_dim: int = 32,
        n_fft: int = 64,
        hop_length: int = 16,
        min_coverage: float = 0.1,
    ) -> None:
        super().__init__()
        if n_fft <= 0 or hop_length <= 0 or hop_length > n_fft:
            raise ValueError("require n_fft > 0 and 0 < hop_length <= n_fft")
        self.output_dim = int(output_dim)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.min_coverage = float(min_coverage)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(self.n_fft // 2 + 2, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(
        self,
        conditional_values: torch.Tensor,
        conditional_mask: torch.Tensor,
    ) -> torch.Tensor:
        if conditional_values.shape != conditional_mask.shape:
            raise ValueError("conditional_values and conditional_mask must share shape")
        if conditional_values.ndim != 3:
            raise ValueError("frequency encoder expects [batch, feature, time]")
        batch, features, length = conditional_values.shape
        if length < self.n_fft:
            raise ValueError(
                f"input length {length} is shorter than n_fft={self.n_fft}"
            )

        mask = conditional_mask.to(dtype=conditional_values.dtype)
        values = conditional_values * mask
        flat_values = values.reshape(batch * features, length)
        flat_mask = mask.reshape(batch * features, 1, length)
        spectrum = torch.stft(
            flat_values,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(dtype=flat_values.dtype),
            center=True,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        coverage = F.conv1d(
            F.pad(flat_mask, (self.n_fft // 2, self.n_fft // 2)),
            self.window.to(dtype=flat_values.dtype).view(1, 1, -1),
            stride=self.hop_length,
        ) / self.window.sum().clamp_min(1e-6)
        if coverage.shape[-1] != magnitude.shape[-1]:
            coverage = F.interpolate(
                coverage,
                size=magnitude.shape[-1],
                mode="linear",
                align_corners=False,
            )
        normalized_magnitude = magnitude / coverage.clamp_min(self.min_coverage)
        spectral_features = torch.log1p(normalized_magnitude)
        frame_features = torch.cat([spectral_features, coverage], dim=1)
        frame_features = frame_features.permute(0, 2, 1)
        encoded = self.projection(frame_features).permute(0, 2, 1)
        encoded = F.interpolate(encoded, size=length, mode="linear", align_corners=False)
        return encoded.reshape(batch, features, self.output_dim, length).permute(0, 2, 1, 3)


class EnhancedResidualBlock(ResidualBlock):
    """CSDI block with graph-guided feature attention and frequency conditioning."""

    def __init__(
        self,
        side_dim: int,
        frequency_dim: int,
        channels: int,
        diffusion_embedding_dim: int,
        nheads: int,
        graph_adjacency: torch.Tensor,
        graph_bias_scale: float = 2.0,
        frequency_scale: float = 1.0,
        is_linear: bool = False,
    ) -> None:
        if is_linear:
            raise ValueError("enhanced graph attention requires diffusion.is_linear=false")
        super().__init__(side_dim, channels, diffusion_embedding_dim, nheads, is_linear)
        if graph_adjacency.ndim != 2 or graph_adjacency.shape[0] != graph_adjacency.shape[1]:
            raise ValueError("graph_adjacency must be square")
        adjacency = graph_adjacency.detach().float().clone()
        adjacency.fill_diagonal_(0.0)
        self.register_buffer("graph_adjacency", adjacency)
        self.graph_gate = nn.Parameter(torch.zeros(()))
        self.frequency_gate = nn.Parameter(torch.zeros(()))
        self.graph_bias_scale = float(graph_bias_scale)
        self.frequency_scale = float(frequency_scale)
        self.frequency_projection = Conv1d_with_init(frequency_dim, channels, 1)

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
            torch.tanh(self.graph_gate)
            * self.graph_bias_scale
            * self.graph_adjacency.to(dtype=y.dtype)
        )
        y = self.feature_layer(y.permute(2, 0, 1), mask=graph_bias).permute(1, 2, 0)
        return (
            y.reshape(batch, length, channels, features)
            .permute(0, 2, 3, 1)
            .reshape(batch, channels, features * length)
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_info: torch.Tensor,
        frequency_info: torch.Tensor,
        diffusion_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, features, length = x.shape
        base_shape = x.shape
        flat_x = x.reshape(batch, channels, features * length)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = self.forward_time(flat_x + diffusion_emb, base_shape)
        y = self.forward_feature(y, base_shape)

        frequency = self.frequency_projection(
            frequency_info.reshape(batch, frequency_info.shape[1], features * length)
        )
        y = y + torch.tanh(self.frequency_gate) * self.frequency_scale * frequency
        y = self.mid_projection(y)

        cond = self.cond_projection(
            cond_info.reshape(batch, cond_info.shape[1], features * length)
        )
        gate, filter_value = torch.chunk(y + cond, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_value)
        residual, skip = torch.chunk(self.output_projection(y), 2, dim=1)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class EnhancedDiffusion(nn.Module):
    def __init__(
        self,
        config: dict,
        graph_adjacency: torch.Tensor,
        frequency_dim: int,
        inputdim: int = 2,
    ) -> None:
        super().__init__()
        self.channels = int(config["channels"])
        self.side_dim = int(config["side_dim"])
        self.frequency_dim = int(frequency_dim)
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.residual_layers = nn.ModuleList(
            [
                EnhancedResidualBlock(
                    side_dim=self.side_dim,
                    frequency_dim=self.frequency_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    graph_adjacency=graph_adjacency,
                    graph_bias_scale=float(config.get("graph_bias_scale", 2.0)),
                    frequency_scale=float(config.get("frequency_scale", 1.0)),
                    is_linear=bool(config["is_linear"]),
                )
                for _ in range(int(config["layers"]))
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_info: torch.Tensor,
        diffusion_step: torch.Tensor,
    ) -> torch.Tensor:
        batch, input_dim, features, length = x.shape
        expected_side = self.side_dim + self.frequency_dim
        if cond_info.shape != (batch, expected_side, features, length):
            raise ValueError(
                f"cond_info shape {tuple(cond_info.shape)} != "
                f"{(batch, expected_side, features, length)}"
            )
        base_cond = cond_info[:, : self.side_dim]
        frequency_info = cond_info[:, self.side_dim :]
        hidden = F.relu(
            self.input_projection(x.reshape(batch, input_dim, features * length))
        ).reshape(batch, self.channels, features, length)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skips = []
        for layer in self.residual_layers:
            hidden, skip = layer(hidden, base_cond, frequency_info, diffusion_emb)
            skips.append(skip)
        hidden = torch.stack(skips).sum(dim=0) / math.sqrt(len(skips))
        hidden = F.relu(
            self.output_projection1(hidden.reshape(batch, self.channels, features * length))
        )
        return self.output_projection2(hidden).reshape(batch, features, length)

    def diagnostics(self) -> dict[str, list[float]]:
        return {
            "graph_gate_by_layer": [
                float(torch.tanh(layer.graph_gate).detach().cpu())
                for layer in self.residual_layers
            ],
            "frequency_gate_by_layer": [
                float(torch.tanh(layer.frequency_gate).detach().cpu())
                for layer in self.residual_layers
            ],
        }
