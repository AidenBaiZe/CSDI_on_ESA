from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture import EnhancedDiffusion, MaskAwareFrequencyEncoder  # noqa: E402


def test_frequency_encoder_is_mask_aware() -> None:
    torch.manual_seed(3)
    encoder = MaskAwareFrequencyEncoder(output_dim=6, n_fft=16, hop_length=4)
    values = torch.randn(2, 3, 32)
    mask = torch.ones_like(values)
    mask[:, :, 10:18] = 0
    changed = values.clone()
    changed[:, :, 10:18] = 1000 * torch.randn_like(changed[:, :, 10:18])
    left = encoder(values, mask)
    right = encoder(changed, mask)
    assert left.shape == (2, 6, 3, 32)
    assert torch.isfinite(left).all()
    torch.testing.assert_close(left, right)


def test_enhanced_diffusion_shapes_and_gates() -> None:
    torch.manual_seed(4)
    channels = 5
    adjacency = torch.zeros(channels, channels)
    for row in range(channels):
        adjacency[row, (row + 1) % channels] = 1
    config = {
        "layers": 2,
        "channels": 8,
        "nheads": 1,
        "diffusion_embedding_dim": 16,
        "num_steps": 3,
        "side_dim": 9,
        "is_linear": False,
        "graph_bias_scale": 2.0,
        "frequency_scale": 1.0,
    }
    model = EnhancedDiffusion(config, adjacency, frequency_dim=4, inputdim=2)
    output = model(
        torch.randn(2, 2, channels, 24),
        torch.randn(2, 13, channels, 24),
        torch.tensor([0, 2]),
    )
    assert output.shape == (2, channels, 24)
    assert torch.isfinite(output).all()
    diagnostics = model.diagnostics()
    assert diagnostics["graph_gate_by_layer"] == [0.0, 0.0]
    assert diagnostics["frequency_gate_by_layer"] == [0.0, 0.0]
