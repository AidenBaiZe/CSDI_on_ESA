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


def _plot_training(training_dir: Path, figure_dir: Path, milestones: list[int]) -> str | None:
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
    for milestone in milestones:
        axis.axvline(milestone, color="#e6550d", linestyle="--", linewidth=1)
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


def _plot_heatmap(channel: pd.DataFrame, figure_dir: Path, channel_names: list[str]) -> str:
    selected = channel[
        (channel["method"] == "csdi")
        & (channel["scale"] == "normalized")
        & (channel["bucket"] == "all")
    ].copy()
    ratios = sorted(selected["missing_ratio"].unique())
    matrix = np.full((len(ratios), len(channel_names)), np.nan, dtype=np.float64)
    for row, ratio in enumerate(ratios):
        lookup = selected[selected["missing_ratio"] == ratio].set_index("channel")["rmse"]
        matrix[row] = [
            float(lookup[name]) if name in lookup.index else np.nan for name in channel_names
        ]
    fig, axis = plt.subplots(figsize=(15, 3.8))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(ratios)), [f"{ratio:.0%}" for ratio in ratios])
    tick_step = max(1, len(channel_names) // 20)
    ticks = np.arange(0, len(channel_names), tick_step)
    axis.set_xticks(ticks, [channel_names[index].removeprefix("channel_") for index in ticks])
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

    channel_names = list(manifest["schema"]["channel_order"])
    milestones = [int(value) for value in config["train"]["lr_milestones"]]
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "training": _plot_training(training_dir, figure_dir, milestones),
        "methods": _plot_method_comparison(group, figure_dir),
        "heatmap": _plot_heatmap(channel, figure_dir, channel_names),
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
    factorized = [
        name for name, metadata in manifest["channels"].items() if metadata.get("factorized")
    ]
    differenced_ids = manifest["esa_adb_compatibility"]["differenced_channel_ids"]
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
        "experiment": "ESA Mission 2 full telemetry CSDI random masking",
        "input_dimensions": len(channel_names),
        "official_esa_adb_input_dimensions": 104,
        "telecommands_included": False,
        "split_at": config["data"]["split_at"],
        "resample_seconds": int(config["data"]["resample_seconds"]),
        "fixed_training_step": int(evaluation["fixed_model_step"]),
        "validation_used": False,
        "nsample": int(config["evaluation"]["nsample"]),
        "formal_window_count": int(evaluation["benchmark"]["formal_window_count"]),
        "fallback_to_fewer_windows": bool(evaluation["benchmark"]["fallback_triggered"]),
        "headline": summary_rows,
        "near_constant_channels": near_constant,
        "discrete_like_channels": discrete,
        "factorized_channels": factorized,
        "differenced_channels": [f"channel_{index}" for index in differenced_ids],
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
    requested_windows = int(config["evaluation"]["requested_windows"])
    fallback_windows = int(config["evaluation"]["fallback_windows"])
    fallback_note = (
        f"单批基准推算超过 {config['evaluation']['max_projected_hours']} 小时，"
        f"因此按预设降级为 {fallback_windows} 个窗口。"
        if summary["fallback_to_fewer_windows"]
        else f"单批基准推算未超预算，正式评估使用 {requested_windows} 个窗口。"
    )
    total_steps = int(config["train"]["steps"])
    milestones_text = "、".join(f"{value:,}" for value in milestones)
    differenced_count = len(differenced_ids)
    non_differenced_count = len(channel_names) - differenced_count
    report = f"""# ESA Mission 2 全量 CSDI 随机遮盖实验报告

## 结论

本实验使用 ESA Mission 2 归档中的 {len(channel_names)} 个遥测通道，在 ESA-ADB 官方 21/21 个月边界
（划分点 {config["data"]["split_at"]}）上训练和测试。模型固定训练到第 {total_steps:,} 步；
未设置验证集，也没有用测试段选择模型。下表以归一化空间指标为主，便于跨通道比较。

{chr(10).join(table_lines)}

## 数据与预处理口径

- 输入为 {len(channel_names)} 维遥测，不含 priority≥3 telecommand；因此不等同于 ESA-ADB 官方 104 维输入。
- 原始时间戳 `<= {config["data"]["split_at"][:10]}` 进入训练段，之后进入测试段；训练网格
  {int(config["data"]["expected_train_timepoints"]):,} 点、测试网格 {int(config["data"]["expected_test_timepoints"]):,} 点
  （{config["data"]["resample_seconds"]} 秒零阶保持），窗口不会跨界。
- 通道 {differenced_ids[0]}–{differenced_ids[-1]} 在原始不规则采样上先做 `np.diff(value, append=value[-1])`，
  再做零阶保持。这 {differenced_count} 个通道的原始尺度误差是差分空间单位，不能与其余 {non_differenced_count} 个通道的原值空间误差直接合并解释。
- 非数值（字符串）通道按官方脚本 `pd.factorize` 转为类别整数；本次共 {len(factorized)} 个此类通道。
- Mission 2 标注只含 anomaly 与 rare event 两类，官方预处理没有 communication gap 分支，
  因此自然缺失仅可能来自非有限值。已复刻异常/稀有样本还原，训练段共还原 {restored_train:,} 点，测试段共还原 {restored_test:,} 点。
- 全局 `ffill().bfill()` 后，头部 bfill 点视为有效观测，但属于 `update_mask == 0` 的保持桶。
- 归一化统计只使用训练段有限 nominal 点。近常数通道仍减 clean mean，但除数降级为 1.0。
  本次近常数通道数为 {len(near_constant)}，离散型通道数为 {len(discrete)}。

## 训练与测试协议

- 窗口长度 {config["data"]["window_length"]}、步长 {config["data"]["stride"]}；允许自然缺失，剔除全自然缺失窗口。
- 条件式 CSDI 使用 PhysioNet 的 random target strategy，每个训练窗口独立从 `[0,1]` 均匀抽取遮盖率。
- 随机种子为 {config["train"]["seed"]}，批量 {config["train"]["batch_size"]}，固定 {total_steps:,} 次更新；
  学习率在 {milestones_text} 步各乘 0.1，每 {int(config["train"]["checkpoint_interval"]):,} 步保存检查点。
- 反向传播采用全局梯度范数裁剪 {config["train"]["gradient_clip_norm"]}，裁剪前范数逐步记录在训练日志中；
  NaN/Inf 梯度元素只做置零并记录精确数量，输入值、目标值和评估值均不裁剪。
- 三档测试缺失率分别独立抽取掩码，不构造嵌套关系；同档的窗口、种子、条件掩码和真实目标数已保存。{fallback_note}
- CSDI 每窗生成 {config["evaluation"]["nsample"]} 个样本。CRPS 使用 0.05–0.95 的 19 个分位数，
  并在排序后按 `q × (n-1)` 做线性插值；只应与同样使用 {config["evaluation"]["nsample"]} 个生成样本的结果直接比较。
- 前向填充和线性插值基线仅访问遮盖后可见值，边界按预设回退，不读取人工遮盖目标。

## 分桶结果与图表

![训练曲线](figures/{figures['training']})

![方法对比](figures/{figures['methods']})

![逐通道热力图](figures/{figures['heatmap']})

![分桶结果](figures/{figures['buckets']})

完整的逐通道长表见 `evaluation/channel_metrics.csv`，通道组及 nominal/anomaly+rare、update/held 分桶长表见
`evaluation/group_metrics.csv`。每行都保存真实评估点数；空桶不伪造指标。

## 解释限制

- 随机遮盖会遮盖大量 ZOH 保持值，慢通道可能比真实更新点更容易插补；因此必须结合 update 与 held 两个桶解释总体指标。
- anomaly/rare 点来自稀少事件，误差方差可能明显大于 nominal 点。
- “原始尺度总体”混合了不同物理单位，本报告只把归一化指标作为 headline；原始尺度结果保留在长表中用于逐通道查看。
- 本实验为 18 秒网格，与 Mission 1 的 30 秒网格在同一缺失率下任务难度不同，跨任务只能做定性比较。
"""
    report_path = result_dir / "report_zh.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    with (Path(__file__).with_name("config.yaml")).open("r", encoding="utf-8") as handle:
        import yaml

        loaded = yaml.safe_load(handle)
    print(generate_report(loaded, Path(__file__).with_name("config.yaml")))
