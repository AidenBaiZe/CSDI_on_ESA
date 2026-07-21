from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal, stats


SAMPLE_SECONDS = 30
HORIZONS = (96, 192, 384, 1440, 2880)
WINDOWS_PER_HORIZON = 96
MASK_WINDOWS_PER_HORIZON = 64
MASK_HORIZONS = (96, 192, 384)
MASK_RATIOS = (0.1, 0.5, 0.9)
SEED = 20260719
MIN_NOMINAL_FRACTION = 0.99
MIN_ACTIVE_STD = 1.0e-4


@dataclass(frozen=True)
class Paths:
    source_root: Path
    output_root: Path

    @property
    def train_dir(self) -> Path:
        return self.source_root / "ESA_mission1" / "data" / "processed" / "train"

    @property
    def manifest(self) -> Path:
        return self.source_root / "ESA_mission1" / "data" / "processed" / "manifest.json"

    @property
    def channel_errors(self) -> Path:
        return (
            self.source_root
            / "ESA_mission1_structured"
            / "results"
            / "mission1_structured_only_seed1"
            / "evaluation"
            / "channel_metrics.csv"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-only frequency feasibility analysis for ESA Mission 1"
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path(r"F:\2026小学期\CSDI")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path(r"F:\2026小学期\CSDI_improve")
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_inputs(paths: Paths) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {
        name: np.load(paths.train_dir / f"{name}.npy", mmap_mode="r")
        for name in (
            "normalized_values",
            "observed_mask",
            "label_code",
            "update_mask",
            "timestamps_ns",
        )
    }
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    values = arrays["normalized_values"]
    if values.shape[0] != 76:
        raise ValueError(f"Expected 76 channels, found {values.shape}")
    if any(array.shape[-1] != values.shape[-1] for array in arrays.values()):
        raise ValueError("Processed training arrays have inconsistent time lengths")
    return arrays, manifest


def channel_metadata(manifest: dict[str, Any]) -> pd.DataFrame:
    records = []
    for index in range(1, 77):
        item = manifest["channels"][f"channel_{index}"]
        records.append(
            {
                "channel_id": index,
                "channel": f"channel_{index}",
                "value_type": item["value_type"],
                "differenced": bool(item["differenced"]),
                "near_constant": bool(item["near_constant"]),
                "clean_unique_count": int(item["clean_unique_count"]),
                "normalization_scale": float(item["normalization_scale"]),
                "global_train_update_fraction": float(item["train"]["update_fraction"]),
                "global_train_observed_fraction": float(item["train"]["observed_fraction"]),
            }
        )
    return pd.DataFrame.from_records(records)


def evenly_spaced_starts(length: int, horizon: int, count: int) -> np.ndarray:
    max_start = length - horizon
    if max_start < 0:
        raise ValueError("Horizon exceeds source length")
    return np.unique(np.rint(np.linspace(0, max_start, count)).astype(np.int64))


def fill_rows(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    x = np.arange(values.shape[1])
    for row in range(values.shape[0]):
        good = valid[row]
        if good.all():
            continue
        if good.sum() < 2:
            result[row] = 0.0
            continue
        result[row, ~good] = np.interp(x[~good], x[good], result[row, good])
    return result


def spectra(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    detrended = signal.detrend(batch, axis=1, type="linear")
    window = signal.windows.hann(batch.shape[1], sym=False)
    transformed = np.fft.rfft(detrended * window[None, :], axis=1)
    amplitude = np.abs(transformed)[:, 1:]
    power = amplitude**2
    return amplitude, power


def per_window_spectral_metrics(
    batch: np.ndarray, update_fraction: np.ndarray, horizon: int
) -> pd.DataFrame:
    amplitude, power = spectra(batch)
    total = power.sum(axis=1)
    active = (np.std(batch, axis=1) > MIN_ACTIVE_STD) & (total > 1.0e-16)
    safe_total = np.where(total > 0, total, 1.0)
    normalized = power / safe_total[:, None]
    entropy_denominator = math.log(max(power.shape[1], 2))
    entropy = -np.sum(
        np.where(normalized > 0, normalized * np.log(normalized + 1.0e-30), 0.0),
        axis=1,
    ) / entropy_denominator
    top3 = np.sort(normalized, axis=1)[:, -min(3, normalized.shape[1]) :].sum(axis=1)
    dominant_bin = np.argmax(power, axis=1) + 1
    dominant_period_seconds = horizon * SAMPLE_SECONDS / dominant_bin
    peak_to_median = np.max(power, axis=1) / (np.median(power, axis=1) + 1.0e-30)
    boundary = dominant_period_seconds >= 0.75 * horizon * SAMPLE_SECONDS
    high_start = max(0, int(math.floor(power.shape[1] * 0.75)))
    high_share = power[:, high_start:].sum(axis=1) / safe_total
    return pd.DataFrame(
        {
            "active": active,
            "spectral_entropy": entropy,
            "top3_power_share": top3,
            "dominant_period_seconds": dominant_period_seconds,
            "peak_to_median_power": peak_to_median,
            "boundary_peak": boundary,
            "high_frequency_share": high_share,
            "update_fraction": update_fraction,
        }
    )


def analyze_horizons(
    arrays: dict[str, np.ndarray], metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = arrays["normalized_values"]
    observed = arrays["observed_mask"]
    labels = arrays["label_code"]
    updates = arrays["update_mask"]
    length = values.shape[1]
    window_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        starts = evenly_spaced_starts(length, horizon, WINDOWS_PER_HORIZON)
        offsets = np.arange(horizon, dtype=np.int64)
        positions = starts[:, None] + offsets[None, :]
        for channel_index in range(values.shape[0]):
            raw = np.asarray(values[channel_index, positions], dtype=np.float64)
            valid = (
                np.asarray(observed[channel_index, positions], dtype=bool)
                & (np.asarray(labels[channel_index, positions]) == 0)
            )
            nominal_fraction = valid.mean(axis=1)
            eligible = nominal_fraction >= MIN_NOMINAL_FRACTION
            if not eligible.any():
                metric = pd.DataFrame()
            else:
                filled = fill_rows(raw[eligible], valid[eligible])
                update_fraction = np.asarray(
                    updates[channel_index, positions][eligible], dtype=np.float64
                ).mean(axis=1)
                metric = per_window_spectral_metrics(filled, update_fraction, horizon)
                eligible_indices = np.flatnonzero(eligible)
                for local_index, source_index in enumerate(eligible_indices):
                    row = metric.iloc[local_index]
                    window_records.append(
                        {
                            "channel_id": channel_index + 1,
                            "channel": f"channel_{channel_index + 1}",
                            "horizon_steps": horizon,
                            "window_minutes": horizon * SAMPLE_SECONDS / 60.0,
                            "window_index": int(source_index),
                            "window_start": int(starts[source_index]),
                            "nominal_fraction": float(nominal_fraction[source_index]),
                            **{key: json_safe(value) for key, value in row.to_dict().items()},
                        }
                    )

            active_metric = metric[metric["active"]] if not metric.empty else metric
            meta = metadata.iloc[channel_index]
            if active_metric.empty:
                summary = {
                    "eligible_windows": int(eligible.sum()),
                    "active_windows": 0,
                    "active_window_fraction": 0.0,
                    "median_spectral_entropy": np.nan,
                    "median_top3_power_share": np.nan,
                    "median_dominant_period_seconds": np.nan,
                    "dominant_period_iqr_octaves": np.nan,
                    "dominant_period_stability": 0.0,
                    "median_peak_to_median_power": np.nan,
                    "boundary_peak_fraction": np.nan,
                    "median_high_frequency_share": np.nan,
                    "median_update_fraction": np.nan,
                }
            else:
                periods = active_metric["dominant_period_seconds"].to_numpy(float)
                log_periods = np.log2(periods)
                median_log = float(np.median(log_periods))
                q25, q75 = np.quantile(log_periods, [0.25, 0.75])
                summary = {
                    "eligible_windows": int(eligible.sum()),
                    "active_windows": int(len(active_metric)),
                    "active_window_fraction": float(len(active_metric) / max(int(eligible.sum()), 1)),
                    "median_spectral_entropy": float(active_metric["spectral_entropy"].median()),
                    "median_top3_power_share": float(active_metric["top3_power_share"].median()),
                    "median_dominant_period_seconds": float(np.median(periods)),
                    "dominant_period_iqr_octaves": float(q75 - q25),
                    "dominant_period_stability": float(
                        np.mean(np.abs(log_periods - median_log) <= 0.25)
                    ),
                    "median_peak_to_median_power": float(
                        active_metric["peak_to_median_power"].median()
                    ),
                    "boundary_peak_fraction": float(active_metric["boundary_peak"].mean()),
                    "median_high_frequency_share": float(
                        active_metric["high_frequency_share"].median()
                    ),
                    "median_update_fraction": float(active_metric["update_fraction"].median()),
                }
            summary.update(
                {
                    "channel_id": channel_index + 1,
                    "channel": f"channel_{channel_index + 1}",
                    "horizon_steps": horizon,
                    "window_minutes": horizon * SAMPLE_SECONDS / 60.0,
                    "value_type": meta["value_type"],
                    "differenced": bool(meta["differenced"]),
                }
            )
            score_parts = (
                summary["median_top3_power_share"],
                1.0 - summary["median_spectral_entropy"]
                if math.isfinite(summary["median_spectral_entropy"])
                else np.nan,
                summary["dominant_period_stability"],
                summary["active_window_fraction"],
                1.0 - summary["boundary_peak_fraction"]
                if math.isfinite(summary["boundary_peak_fraction"])
                else np.nan,
            )
            summary["periodicity_score"] = (
                float(np.prod(score_parts)) if all(math.isfinite(x) for x in score_parts) else np.nan
            )
            summary["candidate_periodic"] = bool(
                summary["eligible_windows"] >= 24
                and summary["active_window_fraction"] >= 0.25
                and math.isfinite(summary["median_top3_power_share"])
                and summary["median_top3_power_share"] >= 0.35
                and summary["median_spectral_entropy"] <= 0.75
                and summary["dominant_period_stability"] >= 0.50
                and summary["boundary_peak_fraction"] < 0.50
                and summary["median_update_fraction"] >= 0.05
            )
            summary["strong_periodic"] = bool(
                summary["candidate_periodic"]
                and summary["median_top3_power_share"] >= 0.50
                and summary["median_spectral_entropy"] <= 0.65
                and summary["dominant_period_stability"] >= 0.60
                and summary["boundary_peak_fraction"] < 0.25
                and summary["active_window_fraction"] >= 0.50
            )
            summaries.append(summary)

    summary_frame = pd.DataFrame.from_records(summaries)
    window_frame = pd.DataFrame.from_records(window_records)
    return summary_frame, window_frame


def mask_proxy(values: np.ndarray, keep: np.ndarray, method: str) -> np.ndarray:
    if method == "zero":
        return np.where(keep, values, 0.0)
    if method == "linear":
        return fill_rows(values, keep)
    raise ValueError(method)


def normalized_log_spectrum(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amplitude, power = spectra(batch)
    log_amplitude = np.log1p(amplitude)
    norm = np.linalg.norm(log_amplitude, axis=1, keepdims=True)
    normalized = log_amplitude / np.maximum(norm, 1.0e-30)
    dominant_bin = np.argmax(power, axis=1) + 1
    safe_total = np.maximum(power.sum(axis=1), 1.0e-30)
    high_start = max(0, int(math.floor(power.shape[1] * 0.75)))
    high_share = power[:, high_start:].sum(axis=1) / safe_total
    return normalized, dominant_bin, high_share


def analyze_mask_robustness(arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    values = arrays["normalized_values"]
    observed = arrays["observed_mask"]
    labels = arrays["label_code"]
    length = values.shape[1]
    detail_records: list[dict[str, Any]] = []

    for horizon in MASK_HORIZONS:
        starts = evenly_spaced_starts(length, horizon, MASK_WINDOWS_PER_HORIZON)
        positions = starts[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
        for channel_index in range(values.shape[0]):
            raw = np.asarray(values[channel_index, positions], dtype=np.float64)
            natural_valid = (
                np.asarray(observed[channel_index, positions], dtype=bool)
                & (np.asarray(labels[channel_index, positions]) == 0)
            )
            eligible = natural_valid.mean(axis=1) >= MIN_NOMINAL_FRACTION
            if not eligible.any():
                continue
            full = fill_rows(raw[eligible], natural_valid[eligible])
            active = np.std(full, axis=1) > MIN_ACTIVE_STD
            full = full[active]
            natural_valid_active = natural_valid[eligible][active]
            if len(full) == 0:
                continue
            full_spectrum, full_peak, full_high = normalized_log_spectrum(full)
            for ratio in MASK_RATIOS:
                block = max(1, min(horizon - 2, int(round(horizon * ratio))))
                max_start = horizon - block - 1
                min_start = 1
                starts_mask = rng.integers(min_start, max_start + 1, size=len(full))
                artificial_keep = np.ones_like(natural_valid_active, dtype=bool)
                for row, start in enumerate(starts_mask):
                    artificial_keep[row, start : start + block] = False
                condition = natural_valid_active & artificial_keep
                for method in ("zero", "linear"):
                    proxy = mask_proxy(full, condition, method)
                    proxy_spectrum, proxy_peak, proxy_high = normalized_log_spectrum(proxy)
                    cosine = np.sum(full_spectrum * proxy_spectrum, axis=1)
                    ratio_peak = np.maximum(proxy_peak / full_peak, full_peak / proxy_peak)
                    peak_preserved = ratio_peak <= 1.25
                    high_inflation = proxy_high - full_high
                    for row in range(len(full)):
                        detail_records.append(
                            {
                                "channel_id": channel_index + 1,
                                "channel": f"channel_{channel_index + 1}",
                                "horizon_steps": horizon,
                                "window_minutes": horizon * SAMPLE_SECONDS / 60.0,
                                "missing_ratio": ratio,
                                "method": method,
                                "spectral_cosine": float(cosine[row]),
                                "dominant_peak_preserved": bool(peak_preserved[row]),
                                "high_frequency_inflation": float(high_inflation[row]),
                            }
                        )

    detail = pd.DataFrame.from_records(detail_records)
    aggregate = (
        detail.groupby(["horizon_steps", "window_minutes", "missing_ratio", "method"], as_index=False)
        .agg(
            comparisons=("spectral_cosine", "size"),
            median_spectral_cosine=("spectral_cosine", "median"),
            q25_spectral_cosine=("spectral_cosine", lambda x: float(np.quantile(x, 0.25))),
            q75_spectral_cosine=("spectral_cosine", lambda x: float(np.quantile(x, 0.75))),
            dominant_peak_preservation_rate=("dominant_peak_preserved", "mean"),
            median_high_frequency_inflation=("high_frequency_inflation", "median"),
        )
    )
    aggregate["context"] = aggregate.apply(
        lambda row: f"{int(row.horizon_steps)}步 / {int(round(row.missing_ratio * 100))}%", axis=1
    )
    return aggregate, detail


def add_error_association(
    paths: Paths, horizon_summary: pd.DataFrame, selected_horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = pd.read_csv(paths.channel_errors)
    errors = errors[
        (errors["model_kind"] == "structured_only_csdi")
        & (errors["protocol_key"] == "time_block_missing_50")
        & (errors["method"] == "csdi")
        & (errors["scale"] == "normalized")
        & (errors["bucket"] == "all")
    ].copy()
    errors["channel_id"] = errors["channel"].str.extract(r"(\d+)").astype(int)
    spectral = horizon_summary[horizon_summary["horizon_steps"] == selected_horizon].copy()
    joined = spectral.merge(
        errors[["channel_id", "eval_points", "rmse", "mae", "crps"]],
        on="channel_id",
        how="left",
    )
    records = []
    for group_name, group in (
        ("all_channels", joined),
        ("continuous", joined[joined["value_type"] == "continuous"]),
        ("discrete_like", joined[joined["value_type"] == "discrete_like"]),
        ("non_differenced", joined[~joined["differenced"]]),
    ):
        clean = group[["periodicity_score", "mae", "rmse"]].dropna()
        for error_metric in ("mae", "rmse"):
            if len(clean) >= 3 and clean["periodicity_score"].nunique() > 1:
                coefficient, pvalue = stats.spearmanr(
                    clean["periodicity_score"], clean[error_metric]
                )
            else:
                coefficient, pvalue = np.nan, np.nan
            records.append(
                {
                    "group": group_name,
                    "error_metric": error_metric,
                    "channels": len(clean),
                    "spearman_rho": float(coefficient),
                    "p_value": float(pvalue),
                    "selected_horizon_steps": selected_horizon,
                }
            )
    return joined, pd.DataFrame.from_records(records)


def choose_horizon(
    horizon_summary: pd.DataFrame, mask_aggregate: pd.DataFrame
) -> tuple[int, dict[str, Any]]:
    counts = (
        horizon_summary.groupby("horizon_steps")
        .agg(
            candidate_channels=("candidate_periodic", "sum"),
            strong_channels=("strong_periodic", "sum"),
            median_score=("periodicity_score", "median"),
        )
        .reset_index()
    )
    proxy = mask_aggregate[
        (mask_aggregate["missing_ratio"] == 0.5) & (mask_aggregate["method"] == "linear")
    ][["horizon_steps", "median_spectral_cosine", "dominant_peak_preservation_rate"]]
    eligible = counts[counts["horizon_steps"].isin(MASK_HORIZONS)].merge(
        proxy, on="horizon_steps", how="left"
    )
    maximum = int(eligible["candidate_channels"].max())
    acceptable = eligible[
        (eligible["candidate_channels"] >= max(1, math.ceil(0.9 * maximum)))
        & (eligible["median_spectral_cosine"] >= 0.70)
    ]
    if acceptable.empty:
        selected = int(
            eligible.sort_values(
                ["candidate_channels", "median_spectral_cosine", "horizon_steps"],
                ascending=[False, False, True],
            ).iloc[0]["horizon_steps"]
        )
    else:
        selected = int(acceptable.sort_values("horizon_steps").iloc[0]["horizon_steps"])
    row = eligible[eligible["horizon_steps"] == selected].iloc[0].to_dict()
    return selected, json_safe(row)


def horizon_overview(horizon_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        horizon_summary.groupby(["horizon_steps", "window_minutes"], as_index=False)
        .agg(
            candidate_channels=("candidate_periodic", "sum"),
            strong_channels=("strong_periodic", "sum"),
            median_periodicity_score=("periodicity_score", "median"),
            median_top3_power_share=("median_top3_power_share", "median"),
            median_spectral_entropy=("median_spectral_entropy", "median"),
            median_stability=("dominant_period_stability", "median"),
            median_active_fraction=("active_window_fraction", "median"),
            median_boundary_fraction=("boundary_peak_fraction", "median"),
        )
    )
    type_counts = (
        horizon_summary[horizon_summary["candidate_periodic"]]
        .groupby(["horizon_steps", "value_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(
            columns={
                "continuous": "continuous_candidates",
                "discrete_like": "discrete_candidates",
            }
        )
    )
    for field in ("continuous_candidates", "discrete_candidates"):
        if field not in type_counts:
            type_counts[field] = 0
    grouped = grouped.merge(
        type_counts[["horizon_steps", "continuous_candidates", "discrete_candidates"]],
        on="horizon_steps",
        how="left",
    )
    grouped[["continuous_candidates", "discrete_candidates"]] = grouped[
        ["continuous_candidates", "discrete_candidates"]
    ].fillna(0).astype(int)
    grouped["context"] = grouped["horizon_steps"].map(lambda x: f"{int(x)}步")
    return grouped


def recommended_parameters(selected_horizon: int) -> pd.DataFrame:
    if selected_horizon <= 192:
        n_fft = 64
    elif selected_horizon <= 384:
        n_fft = 128
    else:
        n_fft = 256
    hop = n_fft // 4
    rows = []
    for candidate in (32, 64, 96, 128, 256):
        if candidate > selected_horizon:
            continue
        candidate_hop = candidate // 4
        rows.append(
            {
                "context_steps": selected_horizon,
                "context_minutes": selected_horizon * SAMPLE_SECONDS / 60.0,
                "n_fft": candidate,
                "hop_length": candidate_hop,
                "frame_minutes": candidate * SAMPLE_SECONDS / 60.0,
                "hop_minutes": candidate_hop * SAMPLE_SECONDS / 60.0,
                "frequency_bins": candidate // 2 + 1,
                "center_false_frames": 1 + (selected_horizon - candidate) // candidate_hop,
                "recommended": candidate == n_fft,
            }
        )
    return pd.DataFrame.from_records(rows)


def write_sqlite(output: Path, tables: dict[str, pd.DataFrame]) -> None:
    if output.exists():
        output.unlink()
    with sqlite3.connect(output) as connection:
        for name, frame in tables.items():
            frame.to_sql(name, connection, if_exists="replace", index=False)


def source(source_id: str, label: str, table: str, generated_at: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "query": {
            "engine": "SQLite",
            "language": "sql",
            "sql": f"SELECT * FROM {table}",
            "description": label,
            "tables_used": [f"analysis.sqlite:{table}"],
            "filters": ["ESA Mission 1 training split only", "nominal observed fraction >= 99%"],
            "executed_at": generated_at,
        },
    }


def build_artifact(
    generated_at: str,
    horizon_frame: pd.DataFrame,
    overview: pd.DataFrame,
    mask_aggregate: pd.DataFrame,
    joined_errors: pd.DataFrame,
    associations: pd.DataFrame,
    parameters: pd.DataFrame,
    selected_horizon: int,
) -> dict[str, Any]:
    selected = horizon_frame[horizon_frame["horizon_steps"] == selected_horizon].copy()
    selected = selected.sort_values("periodicity_score", ascending=False)
    candidates = selected[selected["candidate_periodic"]]
    strong_count = int(selected["strong_periodic"].sum())
    candidate_count = int(selected["candidate_periodic"].sum())
    continuous_candidates = int(
        ((selected["candidate_periodic"]) & (selected["value_type"] == "continuous")).sum()
    )
    discrete_candidates = int(
        ((selected["candidate_periodic"]) & (selected["value_type"] == "discrete_like")).sum()
    )
    current_continuous_candidates = int(
        (
            (horizon_frame["horizon_steps"] == 96)
            & horizon_frame["candidate_periodic"]
            & (horizon_frame["value_type"] == "continuous")
        ).sum()
    )
    current_count = int(
        horizon_frame[horizon_frame["horizon_steps"] == 96]["candidate_periodic"].sum()
    )
    proxy_row = mask_aggregate[
        (mask_aggregate["horizon_steps"] == selected_horizon)
        & (mask_aggregate["missing_ratio"] == 0.5)
        & (mask_aggregate["method"] == "linear")
    ].iloc[0]
    zero_row = mask_aggregate[
        (mask_aggregate["horizon_steps"] == selected_horizon)
        & (mask_aggregate["missing_ratio"] == 0.5)
        & (mask_aggregate["method"] == "zero")
    ].iloc[0]
    assoc_row = associations[
        (associations["group"] == "all_channels")
        & (associations["error_metric"] == "mae")
    ].iloc[0]
    recommended = parameters[parameters["recommended"]].iloc[0]
    top_names = "、".join(
        candidates.head(8)["channel_id"].astype(int).map(lambda x: str(x)).tolist()
    ) or "无"
    association_text = (
        f"Spearman ρ={assoc_row.spearman_rho:.2f}（p={assoc_row.p_value:.3g}）"
        if math.isfinite(float(assoc_row.spearman_rho))
        else "无法稳定估计"
    )

    sources = [
        source("horizon_source", "训练集多尺度频谱统计", "channel_horizon_metrics", generated_at),
        source("mask_source", "遮挡代理频谱保真度", "mask_robustness", generated_at),
        source("error_source", "Structured-Only CSDI 连续块误差关联", "error_join", generated_at),
        source("parameter_source", "候选 STFT 参数与时间分辨率", "recommended_parameters", generated_at),
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# ESA Mission 1 频率可利用性分析", "layout": "full"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "horizon_source",
            "body": (
                "## 技术摘要\n\n"
                f"- **现有证据支持做通道自适应的频率分支实验，但不支持把频率条件统一强加到全部通道。** "
                f"当前窗口识别出 {current_count} 个候选周期通道；综合频谱集中度、跨窗口稳定性和遮挡代理保真度，"
                f"建议第一版使用 {selected_horizon} 步（{selected_horizon * SAMPLE_SECONDS / 60:.0f} 分钟）条件上下文。\n"
                f"- **推荐上下文中有 {candidate_count} 个候选周期通道，其中 {discrete_candidates} 个是离散状态通道，"
                f"连续通道只有 {continuous_candidates} 个；当前 96 步上下文的连续候选为 {current_continuous_candidates} 个。** "
                f"排名靠前的通道为 {top_names}。周期峰可能反映状态切换或驻留模式，因此模型必须使用通道门控并分类型消融。\n"
                f"- **50% 连续遮挡时，线性代理频谱的中位余弦相似度为 {proxy_row.median_spectral_cosine:.3f}，"
                f"零填充为 {zero_row.median_spectral_cosine:.3f}。** 频率分支必须从 condition mask 构造代理序列，"
                "并显式携带 mask/reliability。\n"
                f"- **周期性强弱与现有连续块 MAE 的关联为 {association_text}。** 这只是后验假设生成，"
                "不能证明频率条件一定降低误差；最终结论仍需同遮挡策略、同训练预算的消融实验。"
            ),
            "layout": "full",
        },
        {
            "id": "horizon_text",
            "type": "markdown",
            "sourceId": "horizon_source",
            "body": (
                "## 48 分钟窗口不是充分的频率上下文\n\n"
                "下图按连续与离散状态通道比较不同上下文中满足候选阈值的通道数。候选阈值同时要求：窗口内有足够动态、"
                "前三频率功率集中、谱熵较低、主频跨窗口稳定、主峰不长期贴在最低非零频点，且原始更新率不低于 5%。"
                "因此它比单独查看 FFT 最大峰更保守。**如果更长上下文显著增加候选通道数，应该扩展频率条件上下文，"
                "而不是扩大 CSDI 的 96 步插补目标。**"
            ),
            "layout": "full",
        },
        {"id": "horizon_chart_block", "type": "chart", "chartId": "horizon_chart", "layout": "full"},
        {
            "id": "channel_text",
            "type": "markdown",
            "sourceId": "horizon_source",
            "body": (
                "## 周期信息集中在部分通道，而不是全体通道\n\n"
                "周期性分数综合了功率集中度、低谱熵、主频稳定性、有效动态窗口比例和边界峰惩罚。"
                "高分通道适合优先观察频率增强收益；低分通道应依赖零初始化门控保持接近原始 CSDI。"
                f"当前 {candidate_count} 个候选中有 {discrete_candidates} 个是离散状态通道。它们的周期峰可能代表状态驻留或模式切换，"
                "而不一定是连续遥测中的正弦周期，后续必须按通道类型分别报告。"
            ),
            "layout": "full",
        },
        {"id": "channel_chart_block", "type": "chart", "chartId": "channel_chart", "layout": "full"},
        {
            "id": "mask_text",
            "type": "markdown",
            "sourceId": "mask_source",
            "body": (
                "## 代理序列的选择会直接改变频谱可信度\n\n"
                "对每个正常训练窗口人工加入连续遮挡，再只利用保留点构造代理序列。余弦相似度越接近 1，"
                "说明代理的 log-magnitude 频谱越接近完整窗口。线性插值通常比零填充更少制造人为高频，"
                "但在 90% 遮挡下仍可能只保留粗略趋势。**模型输入应同时包含 condition mask，避免把低可信度频谱当成真实观测。**"
            ),
            "layout": "full",
        },
        {"id": "mask_chart_block", "type": "chart", "chartId": "mask_chart", "layout": "full"},
        {
            "id": "error_text",
            "type": "markdown",
            "sourceId": "error_source",
            "body": (
                "## 现有误差只支持提出假设，不支持宣称收益\n\n"
                "散点图把推荐上下文的周期性分数与 Structured-Only CSDI 在 50% 连续时间块协议下的通道 MAE 对齐。"
                f"总体秩相关为 {association_text}。连续通道的 MAE 关联弱于离散通道，说明总体关系部分由通道类型驱动。"
                "强相关意味着可优先检查高周期通道，弱相关则意味着频率模块可能只改善局部场景。"
                "由于误差来自固定测试协议且目前只有 seed=1，本关联不得用于选择最终超参数或作为因果结论。"
            ),
            "layout": "full",
        },
        {"id": "error_chart_block", "type": "chart", "chartId": "error_chart", "layout": "full"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## 范围、数据与指标定义\n\n"
                "分析主体只读取 Mission 1 官方训练段（2000-01-01 至 2007-01-01）、76 个遥测通道和 30 秒规则网格。"
                "每个尺度在七年训练段均匀抽取 96 个窗口；窗口至少 99% 为正常且可观测点才进入频谱统计。"
                "少量无效点仅用同一窗口内的正常点线性补齐。异常、稀有事件和通信中断不作为周期证据。\n\n"
                "- **前三频率功率占比**：去线性趋势、Hann 加窗后，非零频率中功率最大的三个频点占总功率比例。\n"
                "- **归一化谱熵**：频率功率分布的熵，0 表示高度集中，1 表示接近平坦。\n"
                "- **主频稳定性**：窗口主周期落在通道中位主周期 ±0.25 octave 内的比例。\n"
                "- **边界峰比例**：主周期接近窗口长度的窗口比例，用于惩罚趋势或分辨率不足造成的伪周期。\n"
                "- **代理频谱相似度**：完整窗口与 mask-aware 代理窗口的 log-magnitude 频谱余弦相似度。"
            ),
            "layout": "full",
        },
        {
            "id": "method",
            "type": "markdown",
            "body": (
                "## 方法与稳健性检查\n\n"
                "每个窗口先做线性去趋势，再乘 Hann 窗并计算实数 FFT；DC 分量不参与主频、谱熵和集中度计算。"
                "候选周期通道必须同时通过集中度、熵、稳定性、动态窗口比例、边界峰和更新率阈值。"
                "遮挡稳健性在 96/192/384 步上下文、10%/50%/90% 连续块中比较零填充与线性插值。"
                "现有插补误差只作为独立的后验关联检查。全部采样起点、遮挡随机数种子和阈值固定在分析脚本中。"
            ),
            "layout": "full",
        },
        {
            "id": "parameters_text",
            "type": "markdown",
            "sourceId": "parameter_source",
            "body": (
                "## 第一版频率条件参数建议\n\n"
                f"建议保持插补目标长度 96 不变，把频率条件上下文扩展为 {selected_horizon} 步，"
                f"初始采用 `n_fft={int(recommended.n_fft)}`、`hop_length={int(recommended.hop_length)}`、"
                "`log1p(|STFT|)`，仅从 condition mask 可见值构造线性代理。频率编码后对齐到 96 个目标时间点，"
                "通过零初始化门控注入残差块。下表给出同一上下文下候选参数的时间与频率分辨率，最终参数仍需烟雾测试和消融确认。"
            ),
            "layout": "full",
        },
        {"id": "parameter_table_block", "type": "table", "tableId": "parameter_table", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 局限、不确定性与下一步\n\n"
                "- 30 秒规则网格由零阶保持形成，低频功率和状态驻留可能被放大；报告用 update fraction 和边界峰惩罚降低、但不能完全消除该风险。\n"
                "- FFT 峰只描述窗口内重复结构，不证明该结构在任务运行模式切换后保持稳定。\n"
                "- 离散状态通道和一阶差分通道需要单独解释；频率特征不应统一强加到全部通道。\n"
                "- 测试误差关联只有 seed=1，且属于后验分析，不参与最终参数选择。\n\n"
                "**下一步实验：**以 Structured-Only CSDI 为匹配基线，固定遮挡协议和 32,000 步预算，依次比较同参数量控制组、"
                "全频幅值、低频分支和高频残差；至少运行 3 个随机种子，并按候选周期通道、非周期通道、连续/离散/差分通道分别报告 MAE、RMSE 和 CRPS。"
            ),
            "layout": "full",
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 仍需回答的问题\n\n"
                "1. 更长频率上下文能否在不扩大扩散目标窗口的情况下稳定对齐到中心 96 步？\n"
                "2. 周期候选通道的收益是否来自真实频率信息，而不是额外参数量或平滑代理？\n"
                "3. 频率门控是否会自动压低离散状态、差分和低稳定性通道的贡献？\n"
                "4. 连续块位于窗口边缘时，单边代理频谱的可靠性应如何编码？"
            ),
            "layout": "full",
        },
    ]

    mask_chart_data = mask_aggregate.copy()
    mask_chart_data["method_label"] = mask_chart_data["method"].map(
        {"linear": "线性代理", "zero": "零填充"}
    )
    mask_chart_data["series_key"] = mask_chart_data.apply(
        lambda r: f"{int(r.horizon_steps)}步-{r.method_label}", axis=1
    )
    mask_wide = (
        mask_chart_data.pivot_table(
            index="missing_ratio", columns="series_key", values="median_spectral_cosine"
        )
        .reset_index()
    )
    mask_wide["missing_label"] = mask_wide["missing_ratio"].map(
        lambda x: f"{int(round(x * 100))}%"
    )
    mask_fields = [c for c in mask_wide.columns if c not in ("missing_ratio", "missing_label")]

    top_channels = selected.head(15).copy()
    top_channels["channel_label"] = top_channels["channel_id"].map(lambda x: f"通道 {int(x)}")
    joined_plot = joined_errors.copy()
    joined_plot["channel_label"] = joined_plot["channel_id"].map(lambda x: f"通道 {int(x)}")

    charts = [
        {
            "id": "horizon_chart",
            "title": "不同条件上下文的周期候选通道数",
            "subtitle": "训练段均匀窗口；候选阈值同时约束集中度、熵、稳定性、边界峰和更新率",
            "type": "bar",
            "intent": "comparison",
            "dataset": "horizon_overview",
            "sourceId": "horizon_source",
            "encodings": {
                "x": {"field": "context", "type": "nominal", "label": "条件上下文"},
                "y": {
                    "fields": ["continuous_candidates", "discrete_candidates"],
                    "type": "quantitative",
                    "aggregate": "none",
                    "label": "通道数",
                },
                "tooltip": [
                    {"field": "median_periodicity_score", "type": "quantitative", "label": "中位周期性分数"},
                    {"field": "window_minutes", "type": "quantitative", "label": "窗口分钟"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "channel_chart",
            "title": "推荐上下文的周期性分数前 15 个通道",
            "subtitle": f"{selected_horizon} 步上下文；分数越高表示集中、稳定且非边界峰的周期结构越明显",
            "type": "horizontalBar",
            "intent": "ranking",
            "dataset": "top_channels",
            "sourceId": "horizon_source",
            "encodings": {
                "x": {"field": "channel_label", "type": "nominal", "label": "通道"},
                "y": {"field": "periodicity_score", "type": "quantitative", "label": "周期性分数"},
                "tooltip": [
                    {"field": "median_dominant_period_seconds", "type": "quantitative", "label": "中位主周期（秒）"},
                    {"field": "dominant_period_stability", "type": "quantitative", "format": "percent", "label": "主频稳定性"},
                    {"field": "value_type", "type": "nominal", "label": "通道类型"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "mask_chart",
            "title": "连续遮挡下代理频谱的中位余弦相似度",
            "subtitle": "完整窗口与代理 log-magnitude 频谱；1 表示完全一致",
            "type": "bar",
            "intent": "comparison",
            "dataset": "mask_wide",
            "sourceId": "mask_source",
            "encodings": {
                "x": {"field": "missing_label", "type": "nominal", "label": "连续遮挡率"},
                "y": {
                    "fields": mask_fields,
                    "type": "quantitative",
                    "aggregate": "none",
                    "label": "频谱余弦相似度",
                },
            },
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "error_chart",
            "title": "周期性分数与 50% 连续块插补 MAE",
            "subtitle": f"Structured-Only CSDI，seed=1；周期性分数来自 {selected_horizon} 步训练窗口",
            "type": "scatter",
            "intent": "relationship",
            "dataset": "error_join",
            "sourceId": "error_source",
            "encodings": {
                "x": {"field": "periodicity_score", "type": "quantitative", "label": "周期性分数"},
                "y": {"field": "mae", "type": "quantitative", "label": "归一化 MAE"},
                "color": {"field": "value_type", "type": "nominal", "label": "通道类型"},
                "tooltip": [
                    {"field": "channel_label", "type": "nominal", "label": "通道"},
                    {"field": "value_type", "type": "nominal", "label": "类型"},
                    {"field": "candidate_periodic", "type": "nominal", "label": "周期候选"},
                    {"field": "eval_points", "type": "quantitative", "label": "评价点"},
                ],
            },
            "valueFormat": "number",
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "parameter_table",
            "title": "推荐上下文下的 STFT 候选参数",
            "subtitle": "时间分辨率与频率分辨率的实现折中",
            "dataset": "recommended_parameters",
            "sourceId": "parameter_source",
            "defaultSort": {"field": "n_fft", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "n_fft", "label": "n_fft", "format": "number"},
                {"field": "hop_length", "label": "hop_length", "format": "number"},
                {"field": "frame_minutes", "label": "单帧分钟", "format": "number"},
                {"field": "hop_minutes", "label": "步进分钟", "format": "number"},
                {"field": "frequency_bins", "label": "频率 bins", "format": "number"},
                {"field": "center_false_frames", "label": "无 padding 帧数", "format": "number"},
                {"field": "recommended", "label": "第一版推荐", "type": "boolean"},
            ],
        }
    ]

    snapshot_datasets = {
        "horizon_overview": overview.to_dict(orient="records"),
        "top_channels": top_channels.to_dict(orient="records"),
        "mask_wide": mask_wide.to_dict(orient="records"),
        "error_join": joined_plot.to_dict(orient="records"),
        "recommended_parameters": parameters.to_dict(orient="records"),
    }
    snapshot_datasets = json_safe(snapshot_datasets)
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "ESA Mission 1 频率可利用性分析",
            "description": "训练集限定的多尺度周期性、遮挡代理频谱保真度与模型参数建议。",
            "generatedAt": generated_at,
            "sources": sources,
            "blocks": blocks,
            "charts": charts,
            "tables": tables,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": snapshot_datasets,
            "accessIssues": [],
        },
        "sources": sources,
    }


def write_notes(
    output_root: Path,
    selected_horizon: int,
    selected_details: dict[str, Any],
    row_counts: dict[str, int],
) -> None:
    notes = f"""# Frequency analysis supporting notes

## Reporting job

- Question: Does ESA Mission 1 contain stable frequency structure that can justify a mask-aware frequency-conditioned CSDI branch?
- Audience: technical.
- Scope: Mission 1 training split only for spectral design evidence; fixed seed-1 test error is secondary hypothesis-generation evidence.
- Decision: choose a conditioning context and a first STFT configuration without changing the 96-step imputation target.

## Required structure mapping

1. Title -> report title block.
2. Technical summary -> `technical_summary`.
3. Key findings with visual evidence -> horizon, channel, mask and error sections.
4. Scope, data and definitions -> `scope`.
5. Methodology -> `method`.
6. Limitations and robustness -> `limitations`.
7. Recommended next steps -> parameter section and limitations.
8. Further questions -> `questions`.

## Selected configuration

- Conditioning context: {selected_horizon} steps ({selected_horizon * SAMPLE_SECONDS / 60:.0f} minutes).
- Selection details: `{json.dumps(selected_details, ensure_ascii=False)}`
- Fixed seed: {SEED}
- Minimum nominal observed fraction: {MIN_NOMINAL_FRACTION:.0%}

## Chart map

| Section | Question | Family / type | Fields | Claim | Palette |
|---|---|---|---|---|---|
| Horizon | How much context exposes usable periodicity? | comparison / grouped bar | context, candidate/strong counts | 48 minutes may be insufficient | single blue root + neutral |
| Channels | Which channels show the clearest stable frequency structure? | ranking / horizontal bar | channel, periodicity score | frequency value is channel-specific | single blue root |
| Mask proxy | Does a condition-only proxy preserve spectrum? | comparison / grouped bar | missing ratio, context-method series, cosine | linear proxy is safer than zero fill | hard two-root cap |
| Error association | Are periodic channels currently hard to impute? | relationship / scatter | periodicity score, MAE | association is hypothesis-generating only | single blue root |

## Evidence inventory

{json.dumps(row_counts, ensure_ascii=False, indent=2)}

## Quantitative visual omission notes

- Exact candidate thresholds and parameter alternatives use narrative/table evidence because exact lookup matters more than another chart.
- Per-window spectra are preserved in CSV but omitted from the reader because tens of thousands of rows would not improve the decision.
"""
    (output_root / "source_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = Paths(args.source_root.resolve(), args.output_root.resolve())
    paths.output_root.mkdir(parents=True, exist_ok=True)
    arrays, manifest = read_inputs(paths)
    metadata = channel_metadata(manifest)

    horizon_frame, window_frame = analyze_horizons(arrays, metadata)
    mask_aggregate, mask_detail = analyze_mask_robustness(arrays)
    selected_horizon, selected_details = choose_horizon(horizon_frame, mask_aggregate)
    overview = horizon_overview(horizon_frame)
    joined_errors, associations = add_error_association(
        paths, horizon_frame, selected_horizon
    )
    parameters = recommended_parameters(selected_horizon)

    tables = {
        "channel_metadata": metadata,
        "channel_horizon_metrics": horizon_frame,
        "window_spectral_metrics": window_frame,
        "horizon_overview": overview,
        "mask_robustness": mask_aggregate,
        "mask_robustness_detail": mask_detail,
        "error_join": joined_errors,
        "error_associations": associations,
        "recommended_parameters": parameters,
    }
    for name, frame in tables.items():
        frame.to_csv(paths.output_root / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_sqlite(paths.output_root / "analysis.sqlite", tables)

    generated_at = datetime.now().astimezone().isoformat()
    artifact = build_artifact(
        generated_at,
        horizon_frame,
        overview,
        mask_aggregate,
        joined_errors,
        associations,
        parameters,
        selected_horizon,
    )
    (paths.output_root / "artifact.json").write_text(
        json.dumps(json_safe(artifact), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "generated_at": generated_at,
        "source": "ESA Mission 1 processed training split",
        "selected_horizon_steps": selected_horizon,
        "selected_horizon_minutes": selected_horizon * SAMPLE_SECONDS / 60.0,
        "selection_details": selected_details,
        "candidate_channels": int(
            horizon_frame[
                (horizon_frame["horizon_steps"] == selected_horizon)
                & horizon_frame["candidate_periodic"]
            ].shape[0]
        ),
        "strong_channels": int(
            horizon_frame[
                (horizon_frame["horizon_steps"] == selected_horizon)
                & horizon_frame["strong_periodic"]
            ].shape[0]
        ),
        "continuous_candidate_channels": int(
            horizon_frame[
                (horizon_frame["horizon_steps"] == selected_horizon)
                & horizon_frame["candidate_periodic"]
                & (horizon_frame["value_type"] == "continuous")
            ].shape[0]
        ),
        "discrete_candidate_channels": int(
            horizon_frame[
                (horizon_frame["horizon_steps"] == selected_horizon)
                & horizon_frame["candidate_periodic"]
                & (horizon_frame["value_type"] == "discrete_like")
            ].shape[0]
        ),
        "recommended_parameters": json_safe(
            parameters[parameters["recommended"]].iloc[0].to_dict()
        ),
        "error_associations": json_safe(associations.to_dict(orient="records")),
        "row_counts": {name: int(len(frame)) for name, frame in tables.items()},
    }
    (paths.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_notes(
        paths.output_root,
        selected_horizon,
        selected_details,
        summary["row_counts"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
