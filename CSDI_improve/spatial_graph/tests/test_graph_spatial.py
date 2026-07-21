from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
CSDI_ROOT = HERE.parents[1] / "CSDI"
for path in reversed((HERE, CSDI_ROOT)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diff_models_graph import GraphBiasedResidualBlock  # noqa: E402
from graph import pairwise_pearson, topk_absolute_adjacency  # noqa: E402


def test_pairwise_pearson_respects_missingness() -> None:
    values = np.asarray(
        [[1, 2, 3, 4], [2, 4, 6, 8], [4, 3, 2, 1]], dtype=np.float32
    )
    observed = np.ones_like(values, dtype=np.uint8)
    observed[1, -1] = 0
    correlations, counts = pairwise_pearson(values, observed, chunk_size=2)
    assert correlations[0, 1] > 0.999
    assert correlations[0, 2] < -0.999
    assert counts[0, 1] == 3


def test_topk_graph_is_symmetric_and_has_self_loops() -> None:
    correlations = np.asarray(
        [[1.0, 0.9, 0.1], [0.9, 1.0, -0.8], [0.1, -0.8, 1.0]], dtype=np.float32
    )
    adjacency = topk_absolute_adjacency(
        correlations, top_k=1, min_abs_correlation=0.2
    )
    np.testing.assert_allclose(adjacency, adjacency.T)
    np.testing.assert_allclose(np.diag(adjacency), 1.0)
    assert adjacency[0, 2] == 0.0


def test_graph_bias_is_zero_initialized_and_receives_gradient() -> None:
    block = GraphBiasedResidualBlock(
        5, 8, 16, 1, torch.eye(3), graph_bias_scale=2.0
    )
    assert float(block.graph_strength.detach()) == 0.0
    y = torch.randn(2, 8, 3 * 4, requires_grad=True)
    output = block.forward_feature(y, torch.Size((2, 8, 3, 4)))
    output.square().mean().backward()
    assert block.graph_strength.grad is not None
    assert torch.isfinite(block.graph_strength.grad)
