from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def pairwise_pearson(
    values: np.ndarray,
    observed_mask: np.ndarray,
    chunk_size: int = 131_072,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise-complete Pearson correlations without filling missing data."""
    if values.shape != observed_mask.shape or values.ndim != 2:
        raise ValueError("values and observed_mask must both have shape (channels, time)")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    channels, length = values.shape
    counts = np.zeros((channels, channels), dtype=np.float64)
    sums = np.zeros_like(counts)
    squared_sums = np.zeros_like(counts)
    cross_products = np.zeros_like(counts)

    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        mask = np.asarray(observed_mask[:, start:stop], dtype=np.float64)
        chunk = np.asarray(values[:, start:stop], dtype=np.float64)
        mask *= np.isfinite(chunk)
        chunk = np.where(mask > 0, chunk, 0.0)
        counts += mask @ mask.T
        sums += chunk @ mask.T
        squared_sums += (chunk * chunk) @ mask.T
        cross_products += chunk @ chunk.T

    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = cross_products - sums * sums.T / counts
        variance_left = squared_sums - sums * sums / counts
        denominator = np.sqrt(
            np.maximum(variance_left, 0.0) * np.maximum(variance_left.T, 0.0)
        )
        correlations = covariance / denominator
    correlations[(counts < 2) | ~np.isfinite(correlations)] = 0.0
    correlations = np.clip((correlations + correlations.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlations, 1.0)
    return correlations.astype(np.float32), counts.astype(np.int64)


def topk_absolute_adjacency(
    correlations: np.ndarray,
    top_k: int = 8,
    min_abs_correlation: float = 0.0,
) -> np.ndarray:
    """Build a symmetric weighted graph from each node's strongest correlations."""
    if correlations.ndim != 2 or correlations.shape[0] != correlations.shape[1]:
        raise ValueError("correlations must be a square matrix")
    channels = correlations.shape[0]
    if not 1 <= top_k < channels:
        raise ValueError("top_k must satisfy 1 <= top_k < channel_count")
    if not 0.0 <= min_abs_correlation <= 1.0:
        raise ValueError("min_abs_correlation must be in [0, 1]")

    strengths = np.abs(np.asarray(correlations, dtype=np.float32))
    np.fill_diagonal(strengths, 0.0)
    directed = np.zeros_like(strengths)
    for channel in range(channels):
        order = np.argsort(strengths[channel])[::-1]
        selected = [
            int(neighbor)
            for neighbor in order
            if strengths[channel, neighbor] >= min_abs_correlation
        ][:top_k]
        directed[channel, selected] = strengths[channel, selected]
    adjacency = np.maximum(directed, directed.T)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency.astype(np.float32)


def build_and_save_graph(
    values: np.ndarray,
    observed_mask: np.ndarray,
    channel_names: tuple[str, ...] | list[str],
    output_path: Path,
    top_k: int = 8,
    min_abs_correlation: float = 0.0,
    chunk_size: int = 131_072,
) -> dict[str, Any]:
    correlations, pair_counts = pairwise_pearson(values, observed_mask, chunk_size)
    adjacency = topk_absolute_adjacency(correlations, top_k, min_abs_correlation)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_split": "train",
        "method": "pairwise_complete_pearson_absolute_topk_union",
        "channel_count": int(len(channel_names)),
        "top_k": int(top_k),
        "min_abs_correlation": float(min_abs_correlation),
        "undirected_edges_without_self_loops": int(np.triu(adjacency > 0, 1).sum()),
        "minimum_pair_count": int(pair_counts.min()),
        "maximum_pair_count": int(pair_counts.max()),
    }
    np.savez_compressed(
        output_path,
        adjacency=adjacency,
        correlations=correlations,
        pair_counts=pair_counts,
        channel_names=np.asarray(channel_names),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return {**metadata, "path": str(output_path)}


def load_graph(path: Path, expected_channels: int | None = None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        adjacency = loaded["adjacency"].astype(np.float32, copy=True)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("saved adjacency must be square")
    if expected_channels is not None and adjacency.shape != (
        expected_channels,
        expected_channels,
    ):
        raise ValueError(
            f"graph shape {adjacency.shape} does not match {expected_channels} channels"
        )
    return adjacency
