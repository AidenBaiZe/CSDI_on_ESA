from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def export_validated_difference_graph(
    candidate_path: Path,
    output_path: Path,
    top_k: int = 8,
) -> dict[str, Any]:
    """Export the phase-1 selected difference-Pearson graph for model training."""
    with np.load(candidate_path, allow_pickle=False) as loaded:
        correlations = loaded["diff_pearson"].astype(np.float32, copy=True)
        neighbors = loaded["diff_neighbors"].astype(np.int64, copy=True)
        channel_names = loaded["channel_names"].copy()
    channels = correlations.shape[0]
    if correlations.shape != (channels, channels):
        raise ValueError("diff_pearson must be square")
    if neighbors.shape != (channels, top_k):
        raise ValueError(f"diff_neighbors shape {neighbors.shape} != {(channels, top_k)}")

    # Directed rows match attention queries: each channel receives exactly Top-K keys.
    adjacency = np.zeros((channels, channels), dtype=np.float32)
    for channel, selected in enumerate(neighbors):
        weights = np.abs(correlations[channel, selected])
        maximum = float(weights.max()) if weights.size else 0.0
        if maximum > 0:
            weights = weights / maximum
        adjacency[channel, selected] = weights
    np.fill_diagonal(adjacency, 0.0)
    metadata = {
        "method": "pairwise_complete_first_difference_pearson_directed_topk",
        "source": str(candidate_path.resolve()),
        "source_partition": "first_80_percent_of_training_split",
        "channel_count": int(channels),
        "top_k": int(top_k),
        "directed_edges": int((adjacency > 0).sum()),
        "selection_reason": "phase-1 validation preferred difference Pearson Top-8",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        adjacency=adjacency,
        correlations=correlations,
        neighbors=neighbors,
        channel_names=channel_names,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return {**metadata, "path": str(output_path.resolve())}


def load_graph(path: Path, expected_channels: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        adjacency = loaded["adjacency"].astype(np.float32, copy=True)
    expected = (expected_channels, expected_channels)
    if adjacency.shape != expected:
        raise ValueError(f"graph shape {adjacency.shape} != {expected}")
    if not np.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("graph adjacency must be finite and non-negative")
    np.fill_diagonal(adjacency, 0.0)
    return adjacency
