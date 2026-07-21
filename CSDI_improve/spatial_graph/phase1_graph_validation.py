from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
CSDI_ROOT = HERE.parents[1] / "CSDI"
if str(CSDI_ROOT) not in sys.path:
    sys.path.insert(0, str(CSDI_ROOT))

from ESA_mission1.dataset_esa import ESAWindowStore  # noqa: E402


SEED = 20260719
TOP_K = 8
BUILD_FRACTION = 0.8
STABILITY_BLOCKS = 4
CORRELATION_CHUNK = 131_072
SPEARMAN_SAMPLE = 160_000
RIDGE_SAMPLE = 200_000
VALIDATION_WINDOWS = 128
WINDOW_LENGTH = 96
RIDGE_LAMBDA = 1e-3
METHOD_LABELS = {
    "raw_pearson": "原值 Pearson Top-8",
    "diff_pearson": "一阶差分 Pearson Top-8",
    "spearman": "Spearman Top-8",
    "random": "随机 Top-8",
    "shuffled": "打乱原值图",
    "global_mean": "零值基线",
}
MASK_LABELS = {
    "channel_dropout_50": "通道丢失 50%",
    "rectangle_50": "矩形缺失 50%",
    "time_block_50": "时间块缺失 50%",
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _empty_accumulators(channels: int) -> tuple[np.ndarray, ...]:
    shape = (channels, channels)
    return tuple(np.zeros(shape, dtype=np.float64) for _ in range(4))


def _finalize_correlations(
    counts: np.ndarray,
    sums: np.ndarray,
    squared_sums: np.ndarray,
    cross_products: np.ndarray,
) -> np.ndarray:
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
    return correlations.astype(np.float32)


def streaming_correlations(
    values: np.ndarray,
    observed_mask: np.ndarray,
    start: int,
    stop: int,
    difference: bool = False,
    chunk_size: int = CORRELATION_CHUNK,
) -> np.ndarray:
    """Pairwise-complete correlations on a contiguous slice, optionally first differences."""
    channels = values.shape[0]
    counts, sums, squared_sums, cross_products = _empty_accumulators(channels)
    cursor = start + (1 if difference else 0)
    while cursor < stop:
        end = min(cursor + chunk_size, stop)
        if difference:
            raw = np.asarray(values[:, cursor - 1 : end], dtype=np.float64)
            raw_mask = np.asarray(observed_mask[:, cursor - 1 : end], dtype=bool)
            chunk = np.diff(raw, axis=1)
            mask = raw_mask[:, 1:] & raw_mask[:, :-1] & np.isfinite(chunk)
        else:
            chunk = np.asarray(values[:, cursor:end], dtype=np.float64)
            mask = np.asarray(observed_mask[:, cursor:end], dtype=bool) & np.isfinite(chunk)
        mask_float = mask.astype(np.float64)
        chunk = np.where(mask, chunk, 0.0)
        counts += mask_float @ mask_float.T
        sums += chunk @ mask_float.T
        squared_sums += (chunk * chunk) @ mask_float.T
        cross_products += chunk @ chunk.T
        cursor = end
    return _finalize_correlations(counts, sums, squared_sums, cross_products)


def sampled_spearman(
    values: np.ndarray,
    observed_mask: np.ndarray,
    start: int,
    stop: int,
    sample_size: int = SPEARMAN_SAMPLE,
) -> np.ndarray:
    sample_size = min(sample_size, stop - start)
    indices = np.rint(np.linspace(start, stop - 1, sample_size)).astype(np.int64)
    sample = np.asarray(values[:, indices], dtype=np.float64)
    mask = np.asarray(observed_mask[:, indices], dtype=bool) & np.isfinite(sample)
    ranked = np.zeros_like(sample)
    for channel in range(sample.shape[0]):
        ranked[channel, mask[channel]] = rankdata(sample[channel, mask[channel]], method="average")
    counts = mask.astype(np.float64) @ mask.astype(np.float64).T
    sums = ranked @ mask.astype(np.float64).T
    squared_sums = (ranked * ranked) @ mask.astype(np.float64).T
    cross_products = ranked @ ranked.T
    return _finalize_correlations(counts, sums, squared_sums, cross_products)


def topk_neighbors(correlations: np.ndarray, top_k: int = TOP_K) -> np.ndarray:
    strengths = np.abs(correlations).copy()
    np.fill_diagonal(strengths, -np.inf)
    order = np.argsort(strengths, axis=1)[:, ::-1]
    return order[:, :top_k].astype(np.int64)


def neighbor_edges(neighbors: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for source, row in enumerate(neighbors):
        for target in row:
            left, right = sorted((source, int(target)))
            if left != right:
                edges.add((left, right))
    return edges


def neighbor_jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scores = []
    for left_row, right_row in zip(left, right):
        a, b = set(map(int, left_row)), set(map(int, right_row))
        scores.append(len(a & b) / len(a | b))
    return np.asarray(scores, dtype=np.float64)


def graph_summary(
    method: str,
    correlations: np.ndarray,
    neighbors: np.ndarray,
) -> dict[str, Any]:
    edges = neighbor_edges(neighbors)
    degree = np.zeros(correlations.shape[0], dtype=np.int64)
    weights = []
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
        weights.append(abs(float(correlations[left, right])))
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "directed_neighbors_per_channel": int(neighbors.shape[1]),
        "undirected_edges": len(edges),
        "degree_min": int(degree.min()),
        "degree_median": float(np.median(degree)),
        "degree_max": int(degree.max()),
        "edge_abs_correlation_min": float(np.min(weights)),
        "edge_abs_correlation_median": float(np.median(weights)),
        "edge_abs_correlation_max": float(np.max(weights)),
    }


def stability_analysis(
    values: np.ndarray,
    observed_mask: np.ndarray,
    build_stop: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundaries = np.rint(np.linspace(0, build_stop, STABILITY_BLOCKS + 1)).astype(np.int64)
    rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    for method, difference in (("raw_pearson", False), ("diff_pearson", True)):
        block_correlations = []
        block_neighbors = []
        for block in range(STABILITY_BLOCKS):
            log(f"计算 {METHOD_LABELS[method]} 稳定性块 {block + 1}/{STABILITY_BLOCKS}")
            correlations = streaming_correlations(
                values,
                observed_mask,
                int(boundaries[block]),
                int(boundaries[block + 1]),
                difference=difference,
            )
            block_correlations.append(correlations)
            block_neighbors.append(topk_neighbors(correlations))
        per_channel_all = [[] for _ in range(values.shape[0])]
        for left, right in itertools.combinations(range(STABILITY_BLOCKS), 2):
            left_edges = neighbor_edges(block_neighbors[left])
            right_edges = neighbor_edges(block_neighbors[right])
            node_scores = neighbor_jaccard(block_neighbors[left], block_neighbors[right])
            for channel, score in enumerate(node_scores):
                per_channel_all[channel].append(float(score))
            upper = np.triu_indices(values.shape[0], 1)
            weight_rho = spearmanr(
                block_correlations[left][upper], block_correlations[right][upper]
            ).statistic
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "block_pair": f"{left + 1}-{right + 1}",
                    "edge_jaccard": len(left_edges & right_edges) / len(left_edges | right_edges),
                    "mean_channel_neighbor_jaccard": float(node_scores.mean()),
                    "median_channel_neighbor_jaccard": float(np.median(node_scores)),
                    "correlation_weight_spearman": float(weight_rho),
                }
            )
        for channel, scores in enumerate(per_channel_all):
            channel_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "channel_index": channel,
                    "channel": f"channel_{channel + 1}",
                    "mean_neighbor_jaccard": float(np.mean(scores)),
                    "median_neighbor_jaccard": float(np.median(scores)),
                    "min_neighbor_jaccard": float(np.min(scores)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(channel_rows)


def shuffled_neighbors(neighbors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permutation = rng.permutation(neighbors.shape[0])
    shuffled = np.empty_like(neighbors)
    for old_target in range(neighbors.shape[0]):
        shuffled[permutation[old_target]] = permutation[neighbors[old_target]]
    return shuffled


def random_neighbors(channels: int, top_k: int, rng: np.random.Generator) -> np.ndarray:
    result = np.empty((channels, top_k), dtype=np.int64)
    for channel in range(channels):
        choices = np.delete(np.arange(channels), channel)
        result[channel] = rng.choice(choices, size=top_k, replace=False)
    return result


def fit_neighbor_ridge(
    values: np.ndarray,
    observed_mask: np.ndarray,
    build_stop: int,
    method_neighbors: dict[str, np.ndarray],
    sample_size: int = RIDGE_SAMPLE,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    indices = np.rint(np.linspace(0, build_stop - 1, min(sample_size, build_stop))).astype(np.int64)
    sample = np.asarray(values[:, indices], dtype=np.float64)
    mask = np.asarray(observed_mask[:, indices], dtype=bool) & np.isfinite(sample)
    models: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, neighbors in method_neighbors.items():
        log(f"拟合邻居 Ridge：{METHOD_LABELS[method]}")
        coefficients = np.zeros((sample.shape[0], neighbors.shape[1]), dtype=np.float64)
        intercepts = np.zeros(sample.shape[0], dtype=np.float64)
        for channel, selected in enumerate(neighbors):
            valid = mask[channel] & mask[selected].all(axis=0)
            x = sample[selected][:, valid].T
            y = sample[channel, valid]
            if y.size < selected.size + 2:
                continue
            x_mean = x.mean(axis=0)
            y_mean = y.mean()
            centered = x - x_mean
            gram = centered.T @ centered
            scale = max(float(np.trace(gram) / selected.size), 1.0)
            beta = np.linalg.solve(
                gram + np.eye(selected.size) * RIDGE_LAMBDA * scale,
                centered.T @ (y - y_mean),
            )
            coefficients[channel] = beta
            intercepts[channel] = y_mean - x_mean @ beta
        models[method] = (coefficients, intercepts)
    return models


def validation_starts(build_stop: int, total: int) -> np.ndarray:
    first = build_stop
    last = total - WINDOW_LENGTH
    if last < first:
        raise ValueError("validation partition is shorter than one window")
    return np.rint(np.linspace(first, last, VALIDATION_WINDOWS)).astype(np.int64)


def condition_masks(
    observed: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    channels, length = observed.shape
    outputs: dict[str, np.ndarray] = {}

    condition = observed.copy()
    selected = rng.choice(channels, size=round(channels * 0.5), replace=False)
    condition[selected, :] = False
    outputs["channel_dropout_50"] = condition

    condition = observed.copy()
    selected_channels = rng.choice(channels, size=54, replace=False)
    width = 68
    start = int(rng.integers(0, length - width + 1))
    condition[np.ix_(selected_channels, np.arange(start, start + width))] = False
    outputs["rectangle_50"] = condition

    condition = observed.copy()
    width = round(length * 0.5)
    start = int(rng.integers(0, length - width + 1))
    condition[:, start : start + width] = False
    outputs["time_block_50"] = condition
    return outputs


def evaluate_reconstruction(
    values: np.ndarray,
    observed_mask: np.ndarray,
    build_stop: int,
    method_neighbors: dict[str, np.ndarray],
    models: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = list(method_neighbors) + ["global_mean"]
    families = list(MASK_LABELS)
    channels = values.shape[0]
    totals = {
        (method, family): {"sse": 0.0, "sae": 0.0, "count": 0}
        for method in methods
        for family in families
    }
    channel_totals = {
        (method, family): {
            "sse": np.zeros(channels),
            "sae": np.zeros(channels),
            "count": np.zeros(channels, dtype=np.int64),
        }
        for method in methods
        for family in families
    }
    window_rows: list[dict[str, Any]] = []
    starts = validation_starts(build_stop, values.shape[1])
    for window_index, start in enumerate(starts):
        if window_index % 32 == 0:
            log(f"代理重建窗口 {window_index + 1}/{len(starts)}")
        stop = int(start + WINDOW_LENGTH)
        window = np.asarray(values[:, start:stop], dtype=np.float64)
        observed = np.asarray(observed_mask[:, start:stop], dtype=bool) & np.isfinite(window)
        rng = np.random.default_rng(SEED + int(start))
        masks = condition_masks(observed, rng)
        for family, condition in masks.items():
            target = observed & ~condition
            for method in methods:
                predictions = np.zeros_like(window)
                if method != "global_mean":
                    neighbors = method_neighbors[method]
                    coefficients, intercepts = models[method]
                    for channel, selected in enumerate(neighbors):
                        features = window[selected].copy()
                        available = condition[selected]
                        features[~available] = 0.0
                        predictions[channel] = intercepts[channel] + coefficients[channel] @ features
                error = predictions - window
                selected_error = error[target]
                stats = totals[(method, family)]
                stats["sse"] += float(np.square(selected_error).sum())
                stats["sae"] += float(np.abs(selected_error).sum())
                stats["count"] += int(selected_error.size)
                window_rows.append(
                    {
                        "window_index": window_index,
                        "window_start": int(start),
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "mask_family": family,
                        "mask_label": MASK_LABELS[family],
                        "normalized_rmse": float(np.sqrt(np.square(selected_error).mean())),
                        "normalized_mae": float(np.abs(selected_error).mean()),
                        "evaluation_points": int(selected_error.size),
                    }
                )
                channel_stats = channel_totals[(method, family)]
                channel_stats["sse"] += np.square(error * target).sum(axis=1)
                channel_stats["sae"] += np.abs(error * target).sum(axis=1)
                channel_stats["count"] += target.sum(axis=1)

    rows = []
    channel_rows = []
    for method in methods:
        for family in families:
            stats = totals[(method, family)]
            count = stats["count"]
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "mask_family": family,
                    "mask_label": MASK_LABELS[family],
                    "normalized_rmse": math.sqrt(stats["sse"] / count),
                    "normalized_mae": stats["sae"] / count,
                    "evaluation_points": count,
                    "windows": len(starts),
                }
            )
            detail = channel_totals[(method, family)]
            for channel in range(channels):
                channel_count = int(detail["count"][channel])
                channel_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "mask_family": family,
                        "mask_label": MASK_LABELS[family],
                        "channel_index": channel,
                        "channel": f"channel_{channel + 1}",
                        "normalized_rmse": (
                            math.sqrt(detail["sse"][channel] / channel_count)
                            if channel_count
                            else np.nan
                        ),
                        "normalized_mae": (
                            detail["sae"][channel] / channel_count
                            if channel_count
                            else np.nan
                        ),
                        "evaluation_points": channel_count,
                    }
                )
    summary = pd.DataFrame(rows)
    random_rmse = summary[summary["method"] == "random"].set_index("mask_family")[
        "normalized_rmse"
    ]
    mean_rmse = summary[summary["method"] == "global_mean"].set_index("mask_family")[
        "normalized_rmse"
    ]
    summary["rmse_change_vs_random_pct"] = summary.apply(
        lambda row: 100.0
        * (row["normalized_rmse"] / random_rmse[row["mask_family"]] - 1.0),
        axis=1,
    )
    summary["rmse_change_vs_global_mean_pct"] = summary.apply(
        lambda row: 100.0
        * (row["normalized_rmse"] / mean_rmse[row["mask_family"]] - 1.0),
        axis=1,
    )
    return summary, pd.DataFrame(channel_rows), pd.DataFrame(window_rows)


def paired_bootstrap_vs_random(
    window_metrics: pd.DataFrame,
    draws: int = 10_000,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 91)
    rows = []
    for family in MASK_LABELS:
        family_frame = window_metrics[window_metrics["mask_family"] == family]
        pivot = family_frame.pivot(
            index="window_index", columns="method", values="normalized_rmse"
        ).dropna()
        random_values = pivot["random"].to_numpy()
        sample_indices = rng.integers(0, len(pivot), size=(draws, len(pivot)))
        for method in ("raw_pearson", "diff_pearson", "spearman", "shuffled"):
            method_values = pivot[method].to_numpy()
            paired_change = 100.0 * (method_values / random_values - 1.0)
            bootstrap_means = paired_change[sample_indices].mean(axis=1)
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "mask_family": family,
                    "mask_label": MASK_LABELS[family],
                    "windows": len(pivot),
                    "paired_window_mean_change_pct": float(paired_change.mean()),
                    "paired_window_median_change_pct": float(np.median(paired_change)),
                    "ci95_low_pct": float(np.quantile(bootstrap_means, 0.025)),
                    "ci95_high_pct": float(np.quantile(bootstrap_means, 0.975)),
                    "win_fraction": float((method_values < random_values).mean()),
                }
            )
    return pd.DataFrame(rows)


def edge_diagnostics(
    channel_names: tuple[str, ...],
    correlations: dict[str, np.ndarray],
    neighbors: dict[str, np.ndarray],
    update_fraction: np.ndarray,
) -> pd.DataFrame:
    rows = []
    raw_edges = neighbor_edges(neighbors["raw_pearson"])
    diff_edges = neighbor_edges(neighbors["diff_pearson"])
    spearman_edges = neighbor_edges(neighbors["spearman"])
    for left, right in sorted(raw_edges):
        raw_r = float(correlations["raw_pearson"][left, right])
        diff_r = float(correlations["diff_pearson"][left, right])
        rank_r = float(correlations["spearman"][left, right])
        rows.append(
            {
                "left_index": left,
                "right_index": right,
                "left_channel": channel_names[left],
                "right_channel": channel_names[right],
                "edge": f"{channel_names[left]} ↔ {channel_names[right]}",
                "raw_pearson": raw_r,
                "abs_raw_pearson": abs(raw_r),
                "diff_pearson": diff_r,
                "abs_diff_pearson": abs(diff_r),
                "spearman": rank_r,
                "abs_spearman": abs(rank_r),
                "in_diff_top8_union": (left, right) in diff_edges,
                "in_spearman_top8_union": (left, right) in spearman_edges,
                "left_update_fraction": float(update_fraction[left]),
                "right_update_fraction": float(update_fraction[right]),
                "min_update_fraction": float(min(update_fraction[left], update_fraction[right])),
                "raw_only_suspect": bool(
                    (left, right) not in diff_edges
                    and abs(diff_r) < 0.1
                    and abs(raw_r) >= 0.8
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_raw_pearson", ascending=False)


def cross_method_overlap(neighbors: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for left, right in itertools.combinations(("raw_pearson", "diff_pearson", "spearman"), 2):
        left_edges = neighbor_edges(neighbors[left])
        right_edges = neighbor_edges(neighbors[right])
        node_scores = neighbor_jaccard(neighbors[left], neighbors[right])
        rows.append(
            {
                "left_method": left,
                "right_method": right,
                "comparison": f"{METHOD_LABELS[left]} vs {METHOD_LABELS[right]}",
                "edge_jaccard": len(left_edges & right_edges) / len(left_edges | right_edges),
                "mean_channel_neighbor_jaccard": float(node_scores.mean()),
                "median_channel_neighbor_jaccard": float(np.median(node_scores)),
            }
        )
    return pd.DataFrame(rows)


def save_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    database = output_dir / "phase1_analysis.sqlite"
    if database.exists():
        database.unlink()
    with sqlite3.connect(database) as connection:
        for name, frame in tables.items():
            frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
            frame.to_sql(name, connection, index=False, if_exists="replace")


def _source(source_id: str, label: str, table: str, generated_at: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": f"SELECT * FROM {table}",
            "description": label,
            "tables_used": [f"phase1_analysis.sqlite:{table}"],
            "filters": [
                "ESA Mission 1 training split only",
                "first 80% chronological build partition",
                "last 20% chronological validation partition",
            ],
            "executed_at": generated_at,
        },
    }


def build_artifact(
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
    summary: dict[str, Any],
) -> None:
    generated_at = summary["generated_at"]
    sources = [
        _source("stability_source", "连续时段图稳定性", "graph_stability", generated_at),
        _source("edge_source", "原值图边的差分与秩相关诊断", "edge_diagnostics", generated_at),
        _source("reconstruction_source", "训练集留出段邻居重建代理", "reconstruction_metrics", generated_at),
        _source("graph_source", "候选图结构统计", "graph_method_summary", generated_at),
    ]

    stability_chart = (
        tables["stability_summary"]
        [["method_label", "mean_edge_jaccard", "mean_channel_neighbor_jaccard"]]
        .round(6)
        .to_dict("records")
    )
    reconstruction_wide = (
        tables["reconstruction_metrics"]
        .pivot(index="mask_label", columns="method_label", values="normalized_rmse")
        .reset_index()
    )
    reconstruction_wide.columns.name = None
    reconstruction_rows = reconstruction_wide.round(6).to_dict("records")
    edge_rows = (
        tables["edge_diagnostics"]
        [[
            "edge",
            "raw_pearson",
            "abs_raw_pearson",
            "diff_pearson",
            "abs_diff_pearson",
            "spearman",
            "min_update_fraction",
            "raw_only_suspect",
        ]]
        .round(6)
        .to_dict("records")
    )
    suspect_rows = (
        tables["edge_diagnostics"]
        .sort_values(["raw_only_suspect", "abs_raw_pearson"], ascending=[False, False])
        .head(15)
        [[
            "edge",
            "raw_pearson",
            "diff_pearson",
            "spearman",
            "min_update_fraction",
            "raw_only_suspect",
        ]]
        .round(6)
        .to_dict("records")
    )
    graph_rows = tables["graph_method_summary"].round(6).to_dict("records")

    decision = summary["decision"]
    summary_text = (
        f"## 技术摘要\n\n"
        f"**结论：{decision['headline']}**\n\n"
        f"- 原值 Pearson 图的跨时段平均边 Jaccard 为 **{summary['stability']['raw_mean_edge_jaccard']:.3f}**，"
        f"一阶差分图为 **{summary['stability']['diff_mean_edge_jaccard']:.3f}**。\n"
        f"- 原值图中 **{summary['edge_diagnostics']['raw_only_suspect_edges']} / {summary['edge_diagnostics']['raw_edges']}** 条边符合“高原值相关、弱差分支持且不在差分 Top-8”判据。\n"
        f"- 在通道丢失 50% 代理实验中，原值图相对随机图 RMSE 变化为 "
        f"**{summary['reconstruction']['raw_vs_random_channel_dropout_pct']:+.2f}%**；差分图为 "
        f"**{summary['reconstruction']['diff_vs_random_channel_dropout_pct']:+.2f}%**。差分图按窗口配对变化的 "
        f"95% Bootstrap 区间为 **[{summary['reconstruction']['diff_dropout_ci95_low_pct']:+.2f}%, "
        f"{summary['reconstruction']['diff_dropout_ci95_high_pct']:+.2f}%]**。\n"
        f"- 这些结果只能验证统计图的稳定性与线性邻居预测价值，不能证明物理因果关系，也不能替代 Graph-CSDI 正式消融。"
    )

    recommendation_text = "## 是否进入 Graph-CSDI\n\n" + "\n".join(
        f"- {item}" for item in decision["actions"]
    )
    limitation_text = (
        "## 限制、稳健性与不确定性\n\n"
        "- 所有建图、参数选择和代理验证均限制在 Mission 1 训练段；最后 20% 只是训练集内部的连续留出段。\n"
        "- 邻居重建使用 Ridge 线性代理和归一化零值处理不可用邻居，只回答“同一时刻图邻居是否提供预测信号”，不等价于扩散模型能力。\n"
        "- Spearman 图使用均匀抽取的 160k 个训练时间点；Pearson 与差分 Pearson 使用完整连续数据。\n"
        "- 30 秒零阶保持会放大长期平台段的一致性；差分图和通道更新率用于识别该风险，但不能完全排除共同工况造成的非因果相关。"
    )

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "ESA Mission 1 空间通道图第一阶段验证",
            "description": "静态通道图的时段稳定性、零阶保持伪相关与邻居重建价值",
            "generatedAt": generated_at,
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# ESA Mission 1 空间通道图第一阶段验证", "layout": "full"},
                {"id": "summary", "type": "markdown", "body": summary_text, "layout": "full"},
                {"id": "stability_text", "type": "markdown", "sourceId": "stability_source", "body": "## 静态图在连续时段中有多稳定\n\n边 Jaccard 衡量不同时间块的无向 Top-8 边重合，通道邻居 Jaccard 衡量每个目标通道的定向邻居重合。权重排序一致但 Top-8 重合较低，意味着相关结构整体相近但边界邻居不稳定。", "layout": "full"},
                {"id": "stability_chart_block", "type": "chart", "chartId": "stability_chart", "layout": "full"},
                {"id": "edge_text", "type": "markdown", "sourceId": "edge_source", "body": "## 原值相关中存在多少平台段风险\n\n横轴是原始归一化数值的 Pearson 相关，纵轴是一阶差分相关。右下区域代表原值高度同步、变化量却缺少同步支持的边，应视为零阶保持或共同状态造成的候选伪相关，而不是直接解释为物理连接。", "layout": "full"},
                {"id": "edge_chart_block", "type": "chart", "chartId": "edge_chart", "layout": "full"},
                {"id": "suspect_text", "type": "markdown", "sourceId": "edge_source", "body": "## 优先人工检查的原值图边\n\n下表把满足候选伪相关判据的边排在前面，并保留差分、Spearman 与最低更新率用于复核。", "layout": "full"},
                {"id": "suspect_table_block", "type": "table", "tableId": "suspect_table", "layout": "full"},
                {"id": "reconstruction_text", "type": "markdown", "sourceId": "reconstruction_source", "body": "## 图邻居能否恢复被遮挡通道\n\n每种候选图用前 80% 训练段拟合逐通道 Ridge，在后 20% 连续留出段的 128 个窗口上施加三类 50% 结构化遮挡。真实图只有稳定优于随机图和打乱图，才说明具体边结构具有增量价值。", "layout": "full"},
                {"id": "reconstruction_chart_block", "type": "chart", "chartId": "reconstruction_chart", "layout": "full"},
                {"id": "graph_text", "type": "markdown", "sourceId": "graph_source", "body": "## 三种候选图的结构差异\n\n所有方法都固定每个目标通道选择 8 个定向邻居，再做无向并集统计；因此节点度数可以大于 8。", "layout": "full"},
                {"id": "graph_table_block", "type": "table", "tableId": "graph_table", "layout": "full"},
                {"id": "scope", "type": "markdown", "body": "## 范围、数据与指标定义\n\n分析对象为 ESA Mission 1 的 76 个归一化遥测通道，30 秒时间网格。训练段前 80% 用于建图和拟合代理模型，后 20% 用作连续留出验证。Pearson 使用成对完整观测；一阶差分要求相邻两个时间点均有观测。Top-8 按相关系数绝对值选择，因此正相关和负相关都被视为可用依赖。", "layout": "full"},
                {"id": "method", "type": "markdown", "body": "## 方法\n\n稳定性使用建图段内四个连续等长块，共比较 6 个块对。候选伪相关判据为：原值图边的 |Pearson|≥0.8、|差分 Pearson|<0.1，且该边不属于差分 Top-8 无向并集。代理重建在每个通道上使用固定 8 邻居的 Ridge 回归；被遮挡或自然缺失的邻居在归一化空间置零。随机图与打乱图使用固定种子 20260719。", "layout": "full"},
                {"id": "limitations", "type": "markdown", "body": limitation_text, "layout": "full"},
                {"id": "recommendation", "type": "markdown", "body": recommendation_text, "layout": "full"},
                {"id": "questions", "type": "markdown", "body": "## 后续问题\n\n- 如果原值图和差分图各有优势，Graph Encoder 应使用单图、双关系图还是可学习混合？\n- 通道更新率极低时，邻居关系应按状态相关还是变化相关定义？\n- 图通过代理验证后，真实 Graph-CSDI 的提升是否来自边结构，而不是新增参数？这需要随机图和打乱图消融回答。", "layout": "full"},
            ],
            "charts": [
                {
                    "id": "stability_chart",
                    "title": "连续时段 Top-8 图稳定性",
                    "subtitle": "建图段四等分的 6 个块对均值；1 表示完全一致",
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "stability_chart",
                    "sourceId": "stability_source",
                    "encodings": {
                        "x": {"field": "method_label", "type": "nominal", "label": "候选图"},
                        "y": {"fields": ["mean_edge_jaccard", "mean_channel_neighbor_jaccard"], "type": "quantitative", "aggregate": "none", "label": "Jaccard"},
                    },
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "edge_chart",
                    "title": "原值图边的原值相关与差分相关",
                    "subtitle": "每个点是一条原值 Pearson Top-8 无向边；颜色标记候选伪相关",
                    "type": "scatter",
                    "intent": "relationship",
                    "dataset": "edge_diagnostics",
                    "sourceId": "edge_source",
                    "encodings": {
                        "x": {"field": "abs_raw_pearson", "type": "quantitative", "label": "|原值 Pearson|"},
                        "y": {"field": "abs_diff_pearson", "type": "quantitative", "label": "|一阶差分 Pearson|"},
                        "color": {"field": "raw_only_suspect", "type": "nominal", "label": "候选伪相关"},
                        "tooltip": [
                            {"field": "edge", "type": "nominal", "label": "通道边"},
                            {"field": "spearman", "type": "quantitative", "label": "Spearman"},
                            {"field": "min_update_fraction", "type": "quantitative", "format": "percent", "label": "最低更新率"},
                        ],
                    },
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "reconstruction_chart",
                    "title": "训练集连续留出段的邻居重建 RMSE",
                    "subtitle": "128 个 96 步窗口；越低越好；所有数值在通道归一化空间计算",
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "reconstruction_wide",
                    "sourceId": "reconstruction_source",
                    "encodings": {
                        "x": {"field": "mask_label", "type": "nominal", "label": "遮挡协议"},
                        "y": {"fields": list(reconstruction_wide.columns[1:]), "type": "quantitative", "aggregate": "none", "label": "归一化 RMSE"},
                    },
                    "valueFormat": "number",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "suspect_table",
                    "title": "原值图边诊断（前 15 条）",
                    "subtitle": "候选伪相关优先，再按 |原值 Pearson| 降序",
                    "dataset": "suspect_edges",
                    "sourceId": "edge_source",
                    "defaultSort": {"field": "raw_only_suspect", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "edge", "label": "通道边"},
                        {"field": "raw_pearson", "label": "原值 Pearson", "format": "number"},
                        {"field": "diff_pearson", "label": "差分 Pearson", "format": "number"},
                        {"field": "spearman", "label": "Spearman", "format": "number"},
                        {"field": "min_update_fraction", "label": "最低更新率", "format": "percent"},
                        {"field": "raw_only_suspect", "label": "候选伪相关", "type": "boolean"},
                    ],
                },
                {
                    "id": "graph_table",
                    "title": "候选图结构统计",
                    "subtitle": "建图段前 80%；Top-8 定向邻居的无向并集",
                    "dataset": "graph_summary",
                    "sourceId": "graph_source",
                    "defaultSort": {"field": "method_label", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "method_label", "label": "方法"},
                        {"field": "undirected_edges", "label": "无向边", "format": "number"},
                        {"field": "degree_min", "label": "最小度", "format": "number"},
                        {"field": "degree_median", "label": "中位度", "format": "number"},
                        {"field": "degree_max", "label": "最大度", "format": "number"},
                        {"field": "edge_abs_correlation_median", "label": "边 |r| 中位数", "format": "number"},
                    ],
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "stability_chart": stability_chart,
                "edge_diagnostics": edge_rows,
                "suspect_edges": suspect_rows,
                "reconstruction_wide": reconstruction_rows,
                "graph_summary": graph_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    (output_dir / "phase1_artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def make_decision(
    stability_summary: pd.DataFrame,
    edge_frame: pd.DataFrame,
    reconstruction: pd.DataFrame,
) -> dict[str, Any]:
    indexed = reconstruction.set_index(["method", "mask_family"])
    raw_dropout = float(indexed.loc[("raw_pearson", "channel_dropout_50"), "rmse_change_vs_random_pct"])
    diff_dropout = float(indexed.loc[("diff_pearson", "channel_dropout_50"), "rmse_change_vs_random_pct"])
    raw_rectangle = float(indexed.loc[("raw_pearson", "rectangle_50"), "rmse_change_vs_random_pct"])
    diff_rectangle = float(indexed.loc[("diff_pearson", "rectangle_50"), "rmse_change_vs_random_pct"])
    raw_stability = float(
        stability_summary.set_index("method").loc["raw_pearson", "mean_edge_jaccard"]
    )
    suspect_fraction = float(edge_frame["raw_only_suspect"].mean())

    useful_raw = raw_dropout <= -2.0 or raw_rectangle <= -1.0
    useful_diff = diff_dropout <= -2.0 or diff_rectangle <= -1.0
    stable_raw = raw_stability >= 0.35
    low_suspect = suspect_fraction <= 0.25
    if useful_raw and stable_raw and low_suspect:
        headline = "原值 Pearson 图具备进入 Graph-CSDI 小规模消融的条件，但仍需随机图对照"
        actions = [
            "冻结当前 Top-8 原值图，进入 Graph-only 3k–5k 步筛选实验。",
            "同时训练度数匹配的随机图和标签打乱图，排除新增参数效应。",
            "正式实验前保留差分图作为结构敏感性对照。",
        ]
    elif useful_diff:
        headline = "原值图证据不足，优先使用一阶差分图进入小规模 Graph-CSDI 筛选"
        actions = [
            "暂不把当前原值 Pearson Top-8 图作为主图。",
            "以一阶差分 Top-8 图做 Graph-only 3k–5k 步筛选，并配随机/打乱图。",
            "保留原值图作为消融，以检验状态相关与变化相关哪种更适合遥测恢复。",
        ]
    else:
        headline = "当前静态相关图尚未显示足够的邻居恢复价值，不建议立即进入完整 Graph-CSDI"
        actions = [
            "先检查低更新率通道和候选伪相关边，重新定义图关系。",
            "尝试更新事件相关、滞后相关或按通道类型分图，再重复本阶段代理验证。",
            "在代理图未优于随机图前，不启动 32k 步 Graph-CSDI 正式训练。",
        ]
    return {
        "headline": headline,
        "actions": actions,
        "criteria": {
            "raw_proxy_useful": useful_raw,
            "diff_proxy_useful": useful_diff,
            "raw_stability_pass": stable_raw,
            "raw_suspect_fraction_pass": low_suspect,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ESA Mission 1 channel graphs")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=CSDI_ROOT / "ESA_mission1" / "data" / "processed",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "phase1_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log("读取 Mission 1 训练段")
    store = ESAWindowStore.load(args.processed_dir.resolve(), "train")
    values, observed = store.normalized_values, store.observed_mask
    build_stop = int(values.shape[1] * BUILD_FRACTION)
    rng = np.random.default_rng(SEED)

    log("计算建图段原值 Pearson")
    raw_corr = streaming_correlations(values, observed, 0, build_stop, difference=False)
    log("计算建图段一阶差分 Pearson")
    diff_corr = streaming_correlations(values, observed, 0, build_stop, difference=True)
    log("计算建图段 Spearman（均匀抽样）")
    rank_corr = sampled_spearman(values, observed, 0, build_stop)
    correlations = {
        "raw_pearson": raw_corr,
        "diff_pearson": diff_corr,
        "spearman": rank_corr,
    }
    neighbors = {method: topk_neighbors(corr) for method, corr in correlations.items()}
    neighbors["random"] = random_neighbors(values.shape[0], TOP_K, rng)
    neighbors["shuffled"] = shuffled_neighbors(neighbors["raw_pearson"], rng)

    graph_summaries = pd.DataFrame(
        [
            graph_summary(method, correlations[method], neighbors[method])
            for method in ("raw_pearson", "diff_pearson", "spearman")
        ]
    )
    overlap = cross_method_overlap(neighbors)
    stability, channel_stability = stability_analysis(values, observed, build_stop)
    stability_summary = (
        stability.groupby(["method", "method_label"], as_index=False)
        .agg(
            mean_edge_jaccard=("edge_jaccard", "mean"),
            min_edge_jaccard=("edge_jaccard", "min"),
            mean_channel_neighbor_jaccard=("mean_channel_neighbor_jaccard", "mean"),
            mean_correlation_weight_spearman=("correlation_weight_spearman", "mean"),
        )
    )
    log("计算训练段通道更新率")
    update_fraction = np.asarray(store.update_mask[:, :build_stop].mean(axis=1), dtype=np.float64)
    edges = edge_diagnostics(store.channel_names, correlations, neighbors, update_fraction)

    ridge_neighbors = {key: neighbors[key] for key in ("raw_pearson", "diff_pearson", "spearman", "random", "shuffled")}
    ridge_models = fit_neighbor_ridge(values, observed, build_stop, ridge_neighbors)
    reconstruction, reconstruction_detail, reconstruction_windows = evaluate_reconstruction(
        values, observed, build_stop, ridge_neighbors, ridge_models
    )
    reconstruction_bootstrap = paired_bootstrap_vs_random(reconstruction_windows)

    tables = {
        "graph_method_summary": graph_summaries,
        "cross_method_overlap": overlap,
        "graph_stability": stability,
        "stability_summary": stability_summary,
        "channel_neighbor_stability": channel_stability,
        "edge_diagnostics": edges,
        "reconstruction_metrics": reconstruction,
        "reconstruction_channel_metrics": reconstruction_detail,
        "reconstruction_window_metrics": reconstruction_windows,
        "reconstruction_bootstrap": reconstruction_bootstrap,
    }
    save_tables(output_dir, tables)

    stability_index = stability_summary.set_index("method")
    reconstruction_index = reconstruction.set_index(["method", "mask_family"])
    bootstrap_index = reconstruction_bootstrap.set_index(["method", "mask_family"])
    decision = make_decision(stability_summary, edges, reconstruction)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": {
            "split": "train",
            "channels": int(values.shape[0]),
            "timestamps": int(values.shape[1]),
            "build_timestamps": build_stop,
            "validation_timestamps": int(values.shape[1] - build_stop),
            "grid_seconds": 30,
        },
        "parameters": {
            "top_k": TOP_K,
            "stability_blocks": STABILITY_BLOCKS,
            "spearman_sample": SPEARMAN_SAMPLE,
            "ridge_sample": RIDGE_SAMPLE,
            "validation_windows": VALIDATION_WINDOWS,
            "window_length": WINDOW_LENGTH,
            "seed": SEED,
        },
        "stability": {
            "raw_mean_edge_jaccard": float(stability_index.loc["raw_pearson", "mean_edge_jaccard"]),
            "diff_mean_edge_jaccard": float(stability_index.loc["diff_pearson", "mean_edge_jaccard"]),
            "raw_mean_channel_neighbor_jaccard": float(stability_index.loc["raw_pearson", "mean_channel_neighbor_jaccard"]),
            "diff_mean_channel_neighbor_jaccard": float(stability_index.loc["diff_pearson", "mean_channel_neighbor_jaccard"]),
        },
        "edge_diagnostics": {
            "raw_edges": int(len(edges)),
            "raw_only_suspect_edges": int(edges["raw_only_suspect"].sum()),
            "raw_only_suspect_fraction": float(edges["raw_only_suspect"].mean()),
            "raw_edges_in_diff_top8_fraction": float(edges["in_diff_top8_union"].mean()),
            "raw_edges_in_spearman_top8_fraction": float(edges["in_spearman_top8_union"].mean()),
        },
        "reconstruction": {
            "raw_vs_random_channel_dropout_pct": float(reconstruction_index.loc[("raw_pearson", "channel_dropout_50"), "rmse_change_vs_random_pct"]),
            "diff_vs_random_channel_dropout_pct": float(reconstruction_index.loc[("diff_pearson", "channel_dropout_50"), "rmse_change_vs_random_pct"]),
            "raw_vs_random_rectangle_pct": float(reconstruction_index.loc[("raw_pearson", "rectangle_50"), "rmse_change_vs_random_pct"]),
            "diff_vs_random_rectangle_pct": float(reconstruction_index.loc[("diff_pearson", "rectangle_50"), "rmse_change_vs_random_pct"]),
            "diff_dropout_ci95_low_pct": float(bootstrap_index.loc[("diff_pearson", "channel_dropout_50"), "ci95_low_pct"]),
            "diff_dropout_ci95_high_pct": float(bootstrap_index.loc[("diff_pearson", "channel_dropout_50"), "ci95_high_pct"]),
            "diff_rectangle_ci95_low_pct": float(bootstrap_index.loc[("diff_pearson", "rectangle_50"), "ci95_low_pct"]),
            "diff_rectangle_ci95_high_pct": float(bootstrap_index.loc[("diff_pearson", "rectangle_50"), "ci95_high_pct"]),
        },
        "decision": decision,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "candidate_graphs_top8.npz",
        raw_pearson=raw_corr,
        diff_pearson=diff_corr,
        spearman=rank_corr,
        raw_neighbors=neighbors["raw_pearson"],
        diff_neighbors=neighbors["diff_pearson"],
        spearman_neighbors=neighbors["spearman"],
        random_neighbors=neighbors["random"],
        shuffled_neighbors=neighbors["shuffled"],
        channel_names=np.asarray(store.channel_names),
    )
    build_artifact(output_dir, tables, summary)
    (output_dir / "source_notes.md").write_text(
        "# Source and QA notes\n\n"
        "- Technical report required structure: title, technical summary, findings, scope/definitions, methodology, limitations/robustness, recommendations, further questions.\n"
        "- Chart map: stability grouped bar; raw-vs-difference edge scatter; reconstruction grouped bar; candidate graph and suspect-edge audit tables.\n"
        "- Pearson analyses use the full chronological build data. Spearman uses a deterministic evenly spaced sample.\n"
        "- Test split is not read by this script.\n",
        encoding="utf-8",
    )
    log(f"完成：{decision['headline']}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
