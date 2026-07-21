from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


MODEL_LABELS = {
    "random_csdi": "Random-CSDI",
    "structured_only_csdi": "Structured-Only-CSDI",
}
FAMILY_LABELS = {
    "random": "随机散点",
    "time_block": "连续时间块",
    "channel_dropout": "整窗通道块",
    "rectangle": "通道–时间块",
    "real_gap_case": "真实通信中断型",
}
BUCKET_LABELS = {
    "all": "全部目标",
    "nominal": "正常点",
    "anomaly_rare": "异常/稀有点",
    "update": "真实更新点",
    "held": "保持值点",
}


def _configure_chinese_font() -> None:
    candidates = (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\Deng.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    )
    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            return
    raise FileNotFoundError("No Chinese-capable report font is installed")


def _resolve(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _enrich_existing_random(
    result: dict[str, Any], percent: int, actual_ratio: float
) -> dict[str, Any]:
    metadata = {
        "model_kind": "random_csdi",
        "protocol_key": f"random_missing_{percent:02d}",
        "mask_family": "random",
        "severity": f"{percent}%",
        "requested_missing_ratio": percent / 100,
        "actual_missing_ratio": actual_ratio,
    }
    result.update(metadata)
    result["headline_normalized"].update(metadata)
    for collection in ("channel_rows", "group_rows"):
        for row in result[collection]:
            row.update(metadata)
    return result


def _existing_random_results(
    config: dict[str, Any], config_path: Path
) -> list[dict[str, Any]]:
    metrics_dir = (
        config_path.parent.parent
        / "ESA_mission1"
        / "results"
        / "mission1_random_seed1"
        / "evaluation"
    )
    protocol_dir = _resolve(config["models"]["random_protocol_dir"], config_path)
    results = []
    for percent in (10, 50, 90):
        metrics_path = metrics_dir / f"metrics_missing_{percent}.json"
        protocol_path = protocol_dir / f"missing_{percent:02d}.npz"
        if not metrics_path.is_file() or not protocol_path.is_file():
            raise FileNotFoundError(
                f"Existing Random-CSDI result is incomplete for {percent}%"
            )
        with np.load(protocol_path) as loaded:
            actual_ratio = float(
                loaded["target_counts"].sum() / loaded["observed_counts"].sum()
            )
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        results.append(_enrich_existing_random(result, percent, actual_ratio))
    return results


def _collect_results(
    config: dict[str, Any], config_path: Path
) -> list[dict[str, Any]]:
    results = _existing_random_results(config, config_path)
    evaluation = _resolve(config["evaluation"]["output_dir"], config_path)
    for model_kind in ("random_csdi", "structured_only_csdi"):
        directory = evaluation / "metrics" / model_kind
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name in {"summary.json", "progress.json"}:
                continue
            results.append(json.loads(path.read_text(encoding="utf-8")))
    unique = {
        (result["model_kind"], result["protocol_key"]): result
        for result in results
    }
    return list(unique.values())


def _headline_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for row in result["group_rows"]:
            if (
                row["scale"] == "normalized"
                and row["bucket"] == "all"
                and row["group"] == "all_channels"
            ):
                rows.append(dict(row))
    return rows


def _comparison_rows(headline: pd.DataFrame) -> list[dict[str, Any]]:
    csdi = headline[headline["method"] == "csdi"]
    rows = []
    for protocol_key, group in csdi.groupby("protocol_key"):
        lookup = {row.model_kind: row for row in group.itertuples()}
        if set(MODEL_LABELS).issubset(lookup):
            old = lookup["random_csdi"]
            new = lookup["structured_only_csdi"]
            rows.append(
                {
                    "protocol_key": protocol_key,
                    "mask_family": new.mask_family,
                    "severity": new.severity,
                    "random_rmse": old.rmse,
                    "structured_rmse": new.rmse,
                    "rmse_improvement_fraction": (old.rmse - new.rmse) / old.rmse,
                    "random_mae": old.mae,
                    "structured_mae": new.mae,
                    "mae_improvement_fraction": (old.mae - new.mae) / old.mae,
                    "interpretation": (
                        "random_missing_cost"
                        if new.mask_family == "random"
                        else "structured_training_benefit"
                    ),
                }
            )
    return sorted(rows, key=lambda row: row["protocol_key"])


def _plot_severity_curves(headline: pd.DataFrame, output: Path) -> None:
    data = headline[
        (headline["method"] == "csdi")
        & headline["mask_family"].isin(
            ["random", "time_block", "channel_dropout", "rectangle"]
        )
        & (headline["requested_missing_ratio"] > 0)
    ].copy()
    data["severity_percent"] = data["requested_missing_ratio"] * 100
    families = ["random", "time_block", "channel_dropout", "rectangle"]
    colors = {"random_csdi": "#2463A6", "structured_only_csdi": "#D97706"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for axis, family in zip(axes.flat, families):
        subset = data[data["mask_family"] == family]
        for model_kind, group in subset.groupby("model_kind"):
            group = group.sort_values("severity_percent")
            axis.plot(
                group["severity_percent"],
                group["rmse"],
                marker="o",
                linewidth=2,
                color=colors[model_kind],
                label=MODEL_LABELS[model_kind],
            )
        axis.set_title(FAMILY_LABELS[family])
        axis.set_ylabel("归一化 RMSE")
        axis.set_xticks([10, 50, 90])
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("人工遮盖率 (%)")
    axes[1, 1].set_xlabel("人工遮盖率 (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("结构训练收益与随机缺失代价", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_method_comparison(headline: pd.DataFrame, output: Path) -> None:
    data = headline[
        (headline["model_kind"] == "structured_only_csdi")
        & headline["mask_family"].isin(
            ["time_block", "channel_dropout", "rectangle"]
        )
        & (headline["requested_missing_ratio"] > 0)
    ].copy()
    data["severity_percent"] = data["requested_missing_ratio"] * 100
    methods = {
        "csdi": ("CSDI", "#2463A6"),
        "forward_fill": ("前向填充", "#A16207"),
        "linear_interpolation": ("线性插值", "#6B7280"),
    }
    families = ["time_block", "channel_dropout", "rectangle"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, family in zip(axes, families):
        subset = data[data["mask_family"] == family]
        for method, (label, color) in methods.items():
            group = subset[subset["method"] == method].sort_values(
                "severity_percent"
            )
            axis.plot(
                group["severity_percent"],
                group["mae"],
                marker="o",
                label=label,
                color=color,
            )
        axis.set_title(FAMILY_LABELS[family])
        axis.set_xlabel("人工遮盖率 (%)")
        axis.set_xticks([10, 50, 90])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("归一化 MAE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Structured-Only-CSDI 与无泄漏基线", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_channel_heatmap(channel_rows: list[dict[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(channel_rows)
    data = frame[
        (frame["model_kind"] == "structured_only_csdi")
        & (frame["method"] == "csdi")
        & (frame["scale"] == "normalized")
        & (frame["bucket"] == "all")
        & (frame["mask_family"] != "random")
    ].copy()
    protocol_order = [
        f"{family}_missing_{percent:02d}"
        for family in ("time_block", "channel_dropout", "rectangle")
        for percent in (10, 50, 90)
    ] + ["gap_onset", "gap_sustained"]
    channel_order = sorted(
        data["channel"].unique(), key=lambda value: int(value.rsplit("_", 1)[1])
    )
    table = data.pivot(index="channel", columns="protocol_key", values="mae").reindex(
        index=channel_order, columns=protocol_order
    )
    fig, axis = plt.subplots(figsize=(16, 20))
    image = axis.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(table.index)), table.index, fontsize=7)
    axis.set_title("76 通道归一化 MAE 热力图")
    fig.colorbar(image, ax=axis, label="MAE")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_bucket_heatmap(group_rows: list[dict[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(group_rows)
    data = frame[
        (frame["model_kind"] == "structured_only_csdi")
        & (frame["method"] == "csdi")
        & (frame["scale"] == "normalized")
        & (frame["group"] == "all_channels")
        & (frame["mask_family"] != "random")
    ].copy()
    buckets = ["all", "nominal", "anomaly_rare", "update", "held"]
    protocols = sorted(data["protocol_key"].unique())
    table = data.pivot(index="bucket", columns="protocol_key", values="mae").reindex(
        index=buckets, columns=protocols
    )
    fig, axis = plt.subplots(figsize=(16, 4.5))
    image = axis.imshow(table.to_numpy(), aspect="auto", cmap="YlOrRd")
    axis.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
    axis.set_yticks(
        range(len(table.index)), [BUCKET_LABELS.get(value, value) for value in table.index]
    )
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            value = table.iloc[row, column]
            if pd.notna(value):
                axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=7)
    axis.set_title("正常/异常与更新/保持分桶归一化 MAE")
    fig.colorbar(image, ax=axis, label="MAE")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_training_curves(
    config: dict[str, Any], config_path: Path, output: Path
) -> None:
    random_log = (
        config_path.parent.parent
        / "ESA_mission1"
        / "results"
        / "mission1_random_seed1"
        / "training"
        / "train_log.jsonl"
    )
    structured_log = (
        _resolve(config["models"]["structured_output_dir"], config_path)
        / "train_log.jsonl"
    )
    fig, axis = plt.subplots(figsize=(11, 5))
    for path, label, color in (
        (random_log, "Random-CSDI", "#2463A6"),
        (structured_log, "Structured-Only-CSDI", "#D97706"),
    ):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        frame = pd.DataFrame(rows)
        axis.plot(
            frame["step"],
            frame["loss"].rolling(250, min_periods=1).mean(),
            label=label,
            color=color,
            linewidth=1.8,
        )
    for milestone in (24000, 28800):
        axis.axvline(milestone, linestyle="--", color="#777", alpha=0.7)
    axis.set_title("训练扩散损失（250 步移动平均）")
    axis.set_xlabel("训练步数")
    axis.set_ylabel("扩散损失")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_report(config: dict[str, Any], config_path: Path) -> None:
    _configure_chinese_font()
    root = _resolve(config["evaluation"]["result_root"], config_path)
    results = _collect_results(config, config_path)
    channel_rows = [row for result in results for row in result["channel_rows"]]
    group_rows = [row for result in results for row in result["group_rows"]]
    headline_rows = _headline_rows(results)
    headline = pd.DataFrame(headline_rows)
    comparisons = _comparison_rows(headline)
    expected_common = {
        "random_missing_10",
        "random_missing_50",
        "random_missing_90",
        *config["evaluation"]["random_baseline_protocols"],
    }
    found_common = {row["protocol_key"] for row in comparisons}
    if found_common != expected_common:
        raise ValueError(
            f"Paired model comparison is incomplete: {sorted(expected_common - found_common)}"
        )

    evaluation = root / "evaluation"
    _write_csv(evaluation / "channel_metrics.csv", channel_rows)
    _write_csv(evaluation / "group_metrics.csv", group_rows)
    _write_csv(evaluation / "headline_metrics.csv", headline_rows)
    _write_csv(evaluation / "paired_model_comparisons.csv", comparisons)

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_severity_curves(headline, figures / "severity_and_transfer_cost.png")
    _plot_method_comparison(headline, figures / "structured_method_comparison.png")
    _plot_channel_heatmap(channel_rows, figures / "channel_mae_heatmap.png")
    _plot_bucket_heatmap(group_rows, figures / "bucket_mae_heatmap.png")
    _plot_training_curves(config, config_path, figures / "training_comparison.png")

    csdi = headline[headline["method"] == "csdi"].copy()
    summary = {
        "experiment": "ESA Mission 1 pure structured masking",
        "training_mask_families": ["time_block", "channel_dropout", "rectangle"],
        "random_point_masks_used_for_training": False,
        "structured_model_protocols": int(
            csdi[csdi["model_kind"] == "structured_only_csdi"]["protocol_key"].nunique()
        ),
        "random_model_new_structured_protocols": int(
            csdi[
                (csdi["model_kind"] == "random_csdi")
                & (csdi["mask_family"] != "random")
            ]["protocol_key"].nunique()
        ),
        "paired_model_comparisons": comparisons,
        "random_missing_cost_protocols": [
            row for row in comparisons if row["interpretation"] == "random_missing_cost"
        ],
        "structured_training_benefit_protocols": [
            row
            for row in comparisons
            if row["interpretation"] == "structured_training_benefit"
        ],
        "artifacts": {
            "report": str((root / "report_zh.md").resolve()),
            "headline_csv": str((evaluation / "headline_metrics.csv").resolve()),
            "paired_comparison_csv": str(
                (evaluation / "paired_model_comparisons.csv").resolve()
            ),
            "channel_csv": str((evaluation / "channel_metrics.csv").resolve()),
            "group_csv": str((evaluation / "group_metrics.csv").resolve()),
        },
    }
    _write_json(root / "summary.json", summary)

    structured = csdi[csdi["model_kind"] == "structured_only_csdi"].sort_values(
        ["mask_family", "requested_missing_ratio", "protocol_key"]
    )
    headline_table = [
        "| 遮盖协议 | 类型 | 严重度 | 目标点 | RMSE | MAE | CRPS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in structured.itertuples():
        headline_table.append(
            f"| {row.protocol_key} | {FAMILY_LABELS.get(row.mask_family, row.mask_family)} | "
            f"{row.severity} | {int(row.eval_points):,} | {row.rmse:.6f} | "
            f"{row.mae:.6f} | {row.crps:.6f} |"
        )
    comparison_table = [
        "| 协议 | 解释 | Random RMSE | Structured RMSE | RMSE 改进率 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        interpretation = (
            "随机缺失代价" if row["interpretation"] == "random_missing_cost" else "结构训练收益"
        )
        comparison_table.append(
            f"| {row['protocol_key']} | {interpretation} | {row['random_rmse']:.6f} | "
            f"{row['structured_rmse']:.6f} | {row['rmse_improvement_fraction']:.2%} |"
        )

    report = f"""# ESA Mission 1 CSDI 纯结构性遮盖实验报告

## 实验口径

Structured-Only-CSDI 的训练目标全部由连续时间块、整窗通道块和通道–时间块产生，三类等概率，严重度从 0.1–0.9 连续均匀抽取；训练中不使用随机散点遮盖。随机 10%/50%/90% 只用于测试结构训练的泛化代价。模型固定使用第 32,000 步检查点，不使用测试集选模。

## Structured-Only-CSDI headline

{chr(10).join(headline_table)}

## 结构收益与随机缺失代价

正改进率表示 Structured-Only-CSDI 的误差更低；负值表示纯结构训练带来退化。

{chr(10).join(comparison_table)}

## 图表

![结构收益与随机代价](figures/severity_and_transfer_cost.png)

![模型与基线](figures/structured_method_comparison.png)

![76 通道热力图](figures/channel_mae_heatmap.png)

![指标分桶](figures/bucket_mae_heatmap.png)

![训练曲线](figures/training_comparison.png)

## 解释限制

- 输入为 76 个遥测通道，不包含 ESA-ADB 官方输入中的 11 条高优先级 telecommand。
- 通道 4–11 的原始尺度指标处于差分空间，不能与其余通道的原值空间误差直接汇总解释。
- anomaly/rare event 保留为观测及潜在目标；communication gap 和非有限值作为自然缺失，不计入人工目标。
- 真实 gap 压力测试只使用训练段 4 次事件得到的 52 通道集合，没有读取测试标签。
- CRPS 基于每窗 50 个生成样本，只与相同采样数的实验直接比较。
"""
    (root / "report_zh.md").write_text(report, encoding="utf-8")
