from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment import write_json
from preprocess import resolve_path


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_table(group: pd.DataFrame, scale: str, bucket: str, group_name: str) -> pd.DataFrame:
    selected = group[
        (group["scale"] == scale)
        & (group["bucket"] == bucket)
        & (group["group"] == group_name)
    ].copy()
    return selected.sort_values(["missing_ratio", "method"])


def _plot_training(training_dir: Path, figure_dir: Path) -> str | None:
    path = training_dir / "train_log.jsonl"
    if not path.is_file():
        return None
    log = pd.read_json(path, lines=True)
    if log.empty:
        return None
    window = min(250, max(1, len(log) // 100))
    smooth = log["loss"].rolling(window, min_periods=1).mean()
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(log["step"], log["loss"], color="#9ecae1", linewidth=0.45, alpha=0.45)
    axis.plot(log["step"], smooth, color="#08519c", linewidth=1.5, label=f"moving mean ({window})")
    axis.axvline(24000, color="#e6550d", linestyle="--", linewidth=1)
    axis.axvline(28800, color="#e6550d", linestyle="--", linewidth=1)
    axis.set(xlabel="Training step", ylabel="Diffusion loss", title="CSDI training curve")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    output = figure_dir / "training_curve.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output.name


def _plot_method_comparison(group: pd.DataFrame, figure_dir: Path) -> str:
    selected = _metric_table(group, "normalized", "all", "all_channels")
    ratios = sorted(selected["missing_ratio"].unique())
    methods = ["csdi", "forward_fill", "linear_interpolation"]
    labels = ["CSDI", "Forward fill", "Linear interpolation"]
    x = np.arange(len(ratios))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for index, (method, label) in enumerate(zip(methods, labels)):
        values = [
            float(selected[(selected["method"] == method) & (selected["missing_ratio"] == ratio)]["rmse"].iloc[0])
            for ratio in ratios
        ]
        axis.bar(x + (index - 1) * width, values, width, label=label)
    axis.set_xticks(x, [f"{ratio:.0%}" for ratio in ratios])
    axis.set(xlabel="Artificial missing ratio", ylabel="Normalized RMSE", title="Imputation performance")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    output = figure_dir / "method_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output.name


def _plot_heatmap(channel: pd.DataFrame, figure_dir: Path) -> str:
    selected = channel[
        (channel["method"] == "csdi")
        & (channel["scale"] == "normalized")
        & (channel["bucket"] == "all")
    ].copy()
    ratios = sorted(selected["missing_ratio"].unique())
    names = [f"channel_{index}" for index in range(1, 77)]
    matrix = np.full((len(ratios), len(names)), np.nan, dtype=np.float64)
    for row, ratio in enumerate(ratios):
        lookup = selected[selected["missing_ratio"] == ratio].set_index("channel")["rmse"]
        matrix[row] = [float(lookup[name]) if name in lookup.index else np.nan for name in names]
    fig, axis = plt.subplots(figsize=(15, 3.8))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(ratios)), [f"{ratio:.0%}" for ratio in ratios])
    axis.set_xticks(np.arange(0, 76, 5), [str(index + 1) for index in range(0, 76, 5)])
    axis.set(xlabel="Telemetry channel", ylabel="Missing ratio", title="Per-channel CSDI normalized RMSE")
    fig.colorbar(image, ax=axis, label="RMSE")
    fig.tight_layout()
    output = figure_dir / "channel_heatmap.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output.name


def _plot_buckets(group: pd.DataFrame, figure_dir: Path) -> str:
    selected = group[
        (group["method"] == "csdi")
        & (group["scale"] == "normalized")
        & (group["group"] == "all_channels")
    ].copy()
    ratios = sorted(selected["missing_ratio"].unique())
    buckets = ["all", "nominal", "anomaly_rare", "update", "held"]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for bucket in buckets:
        values = []
        for ratio in ratios:
            row = selected[(selected["bucket"] == bucket) & (selected["missing_ratio"] == ratio)]
            values.append(float(row["rmse"].iloc[0]) if not row.empty and pd.notna(row["rmse"].iloc[0]) else np.nan)
        axis.plot(ratios, values, marker="o", label=bucket)
    axis.set_xticks(ratios, [f"{ratio:.0%}" for ratio in ratios])
    axis.set(xlabel="Artificial missing ratio", ylabel="Normalized RMSE", title="CSDI error by target bucket")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    fig.tight_layout()
    output = figure_dir / "bucket_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output.name


def generate_report(config: dict[str, Any], config_path: Path) -> Path:
    result_dir = resolve_path(config["train"]["output_dir"], config_path).parent
    training_dir = resolve_path(config["train"]["output_dir"], config_path)
    evaluation_dir = resolve_path(config["evaluation"]["output_dir"], config_path)
    processed_dir = resolve_path(config["data"]["processed_dir"], config_path)
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    training = json.loads((training_dir / "training_summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads((evaluation_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    channel = pd.read_csv(evaluation_dir / "channel_metrics.csv")
    group = pd.read_csv(evaluation_dir / "group_metrics.csv")
    headline = _metric_table(group, "normalized", "all", "all_channels")
    for _, row in headline.iterrows():
        if not _finite(row["rmse"]) or not _finite(row["mae"]):
            raise FloatingPointError("A headline RMSE/MAE is non-finite")
        if row["method"] == "csdi" and not _finite(row["crps"]):
            raise FloatingPointError("A headline CSDI CRPS is non-finite")

    figure_dir = result_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "training": _plot_training(training_dir, figure_dir),
        "methods": _plot_method_comparison(group, figure_dir),
        "heatmap": _plot_heatmap(channel, figure_dir),
        "buckets": _plot_buckets(group, figure_dir),
    }
    restored_train = sum(
        int(metadata["train"]["restored_annotation_points"])
        for metadata in manifest["channels"].values()
    )
    restored_test = sum(
        int(metadata["test"]["restored_annotation_points"])
        for metadata in manifest["channels"].values()
    )
    near_constant = [
        name for name, metadata in manifest["channels"].items() if metadata["near_constant"]
    ]
    discrete = [
        name for name, metadata in manifest["channels"].items() if metadata["value_type"] == "discrete_like"
    ]
    summary_rows = []
    for _, row in headline.iterrows():
        summary_rows.append(
            {
                "missing_ratio": float(row["missing_ratio"]),
                "method": row["method"],
                "eval_points": int(row["eval_points"]),
                "rmse_normalized": float(row["rmse"]),
                "mae_normalized": float(row["mae"]),
                "crps_normalized": float(row["crps"]) if pd.notna(row["crps"]) else None,
            }
        )
    summary = {
        "experiment": "ESA Mission 1 full telemetry CSDI random masking",
        "input_dimensions": 76,
        "official_esa_adb_input_dimensions": 87,
        "telecommands_included": False,
        "split_at": config["data"]["split_at"],
        "fixed_training_step": int(evaluation["fixed_model_step"]),
        "validation_used": False,
        "nsample": int(config["evaluation"]["nsample"]),
        "formal_window_count": int(evaluation["benchmark"]["formal_window_count"]),
        "fallback_to_128_windows": bool(evaluation["benchmark"]["fallback_triggered"]),
        "headline": summary_rows,
        "near_constant_channels": near_constant,
        "discrete_like_channels": discrete,
        "differenced_channels": [f"channel_{index}" for index in range(4, 12)],
        "restored_annotation_points": {"train": restored_train, "test": restored_test},
        "artifacts": {
            "channel_metrics_csv": str((evaluation_dir / "channel_metrics.csv").resolve()),
            "group_metrics_csv": str((evaluation_dir / "group_metrics.csv").resolve()),
            "report": str((result_dir / "report_zh.md").resolve()),
            "figures": {key: str((figure_dir / value).resolve()) if value else None for key, value in figures.items()},
        },
    }
    write_json(result_dir / "summary.json", summary)

    method_names = {
        "csdi": "CSDI",
        "forward_fill": "前向填充",
        "linear_interpolation": "线性插值",
    }
    table_lines = ["| 缺失率 | 方法 | 目标点数 | 归一化 RMSE | 归一化 MAE | 归一化 CRPS |", "|---:|---|---:|---:|---:|---:|"]
    for row in summary_rows:
        crps = "—" if row["crps_normalized"] is None else f"{row['crps_normalized']:.6f}"
        table_lines.append(
            f"| {row['missing_ratio']:.0%} | {method_names[row['method']]} | {row['eval_points']:,} | "
            f"{row['rmse_normalized']:.6f} | {row['mae_normalized']:.6f} | {crps} |"
        )
    fallback_note = (
        "单批基准推算超过 9 小时，因此按预设降级为 128 个窗口。"
        if summary["fallback_to_128_windows"]
        else "单批基准推算未超过 9 小时，正式评估使用 256 个窗口。"
    )
    report = f"""# ESA Mission 1 全量 CSDI 随机遮盖实验报告

## 结论

本实验使用 ESA Mission 1 归档中的 76 个遥测通道，在 ESA-ADB 官方 84/84 个月边界上训练和测试。模型固定训练到第 32,000 步；未设置验证集，也没有用测试段选择模型。下表以归一化空间指标为主，便于跨通道比较。

{chr(10).join(table_lines)}

## 数据与预处理口径

- 输入为 76 维遥测，不含 11 条 priority≥3 telecommand；因此不等同于 ESA-ADB 官方 87 维输入。
- 原始时间戳 `<= 2007-01-01` 进入训练段，`> 2007-01-01` 进入测试段；两个 30 秒网格各含 7,364,161 个时间点，窗口不会跨界。
- 通道 4–11 在原始不规则采样上先做 `np.diff(value, append=value[-1])`，再做 30 秒零阶保持。这 8 个通道的原始尺度误差是差分空间单位，不能与其余 68 个通道的原值空间误差直接合并解释。
- 已复刻异常/稀有样本还原，训练段共还原 {restored_train:,} 点，测试段共还原 {restored_test:,} 点；communication gap 不还原。
- 全局 `ffill().bfill()` 后，头部 bfill 点视为有效观测，但属于 `update_mask == 0` 的保持桶。
- 归一化统计只使用训练段有限 nominal 点。近常数通道仍减 clean mean，但除数降级为 1.0。本次近常数通道数为 {len(near_constant)}，离散型通道数为 {len(discrete)}。

## 训练与测试协议

- 窗口长度 96、步长 48；允许自然缺失，剔除全自然缺失窗口。
- 条件式 CSDI 使用 PhysioNet 的 random target strategy，每个训练窗口独立从 `[0,1]` 均匀抽取遮盖率。
- 随机种子为 1，批量 16，固定 32,000 次更新；学习率在 24,000 和 28,800 步各乘 0.1，每 3,200 步保存检查点。
- 真实异常段在 clean-only 归一化后仍可达到很大幅度。为不裁剪数据又避免参数溢出，反向传播采用全局梯度范数裁剪 1.0，裁剪前范数逐步记录在训练日志中。
- 若极端异常输入仍产生 NaN/Inf 梯度元素，只将对应梯度元素置零并记录精确数量；输入值、目标值和评估值均不裁剪。
- 若梯度元素均有限但 float32 聚合范数溢出，该步梯度整体缩放为零并记录溢出标记，避免把不可靠方向写入参数。
- 三档测试缺失率分别独立抽取掩码，不构造嵌套关系；同档的窗口、种子、条件掩码和真实目标数已保存。{fallback_note}
- CSDI 每窗生成 50 个样本。CRPS 使用 0.05–0.95 的 19 个分位数，并在排序后按 `q × (n-1)` 做线性插值；只应与同样使用 50 个生成样本的结果直接比较。
- 前向填充和线性插值基线仅访问遮盖后可见值，边界按预设回退，不读取人工遮盖目标。

## 分桶结果与图表

![训练曲线](figures/{figures['training']})

![方法对比](figures/{figures['methods']})

![逐通道热力图](figures/{figures['heatmap']})

![分桶结果](figures/{figures['buckets']})

完整的逐通道长表见 `evaluation/channel_metrics.csv`，通道组及 nominal/anomaly+rare、update/held 分桶长表见 `evaluation/group_metrics.csv`。每行都保存真实评估点数；空桶不伪造指标。

## 解释限制

- 随机遮盖会遮盖大量 ZOH 保持值，慢通道可能比真实更新点更容易插补；因此必须结合 update 与 held 两个桶解释总体指标。
- anomaly/rare 点来自稀少事件，误差方差可能明显大于 nominal 点。
- “原始尺度总体”混合了不同物理单位，本报告只把归一化指标作为 headline；原始尺度结果保留在长表中用于逐通道查看。
"""
    report_path = result_dir / "report_zh.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    with (Path(__file__).with_name("config.yaml")).open("r", encoding="utf-8") as handle:
        import yaml

        loaded = yaml.safe_load(handle)
    print(generate_report(loaded, Path(__file__).with_name("config.yaml")))
