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
    "mix_csdi": "Random-Structured-Mix-CSDI",
}
MODEL_COLORS = {
    "random_csdi": "#2463A6",
    "structured_only_csdi": "#D97706",
    "mix_csdi": "#14866D",
}
FAMILY_LABELS = {
    "random": "随机散点",
    "time_block": "连续时间块",
    "channel_dropout": "整窗通道块",
    "rectangle": "通道–时间块",
    "real_gap_case": "训练通信中断型压力测试",
}


def _resolve(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    return (
        (config_path.parent / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _configure_chinese_font() -> None:
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\Deng.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=str(path)
            ).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def _protocol_lookup(config: dict[str, Any], config_path: Path) -> dict[str, dict]:
    path = _resolve(config["evaluation"]["output_dir"], config_path)
    path = path / "protocols" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Protocol manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {row["key"]: row for row in manifest["records"]}


def _enrich_result(
    result: dict[str, Any], model_kind: str, record: dict[str, Any]
) -> dict[str, Any]:
    metadata = {
        "model_kind": model_kind,
        "protocol_key": record["key"],
        "mask_family": record["mask_family"],
        "severity": record["severity"],
        "requested_missing_ratio": record["requested_missing_ratio"],
        "actual_missing_ratio": record["actual_missing_ratio"],
    }
    result.update(metadata)
    result["headline_normalized"].update(metadata)
    for collection in ("channel_rows", "group_rows"):
        for row in result[collection]:
            row.update(metadata)
    return result


def _load_metric_directory(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    results = []
    for path in sorted(directory.glob("*.json")):
        if path.name in {"summary.json", "progress.json"}:
            continue
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def _collect_results(
    config: dict[str, Any], config_path: Path
) -> list[dict[str, Any]]:
    protocols = _protocol_lookup(config, config_path)
    random_evaluation = _resolve(
        config["references"]["random_evaluation_dir"], config_path
    )
    structured_evaluation = _resolve(
        config["references"]["structured_evaluation_dir"], config_path
    )
    mix_evaluation = _resolve(config["evaluation"]["output_dir"], config_path)

    results: list[dict[str, Any]] = []
    for percent in (10, 50, 90):
        key = f"random_missing_{percent:02d}"
        path = random_evaluation / f"metrics_missing_{percent}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Random reference result is missing: {path}")
        results.append(
            _enrich_result(
                json.loads(path.read_text(encoding="utf-8")),
                "random_csdi",
                protocols[key],
            )
        )

    results.extend(
        _load_metric_directory(structured_evaluation / "metrics" / "random_csdi")
    )
    results.extend(
        _load_metric_directory(
            structured_evaluation / "metrics" / "structured_only_csdi"
        )
    )
    results.extend(_load_metric_directory(mix_evaluation / "metrics" / "mix_csdi"))
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
    rows: list[dict[str, Any]] = []
    for protocol_key, group in csdi.groupby("protocol_key"):
        lookup = {row.model_kind: row for row in group.itertuples()}
        if "mix_csdi" not in lookup or "structured_only_csdi" not in lookup:
            continue
        mix = lookup["mix_csdi"]
        structured = lookup["structured_only_csdi"]
        row: dict[str, Any] = {
            "protocol_key": protocol_key,
            "mask_family": mix.mask_family,
            "severity": mix.severity,
            "structured_rmse": structured.rmse,
            "mix_rmse": mix.rmse,
            "mix_vs_structured_rmse_improvement_fraction": (
                structured.rmse - mix.rmse
            )
            / structured.rmse,
            "structured_mae": structured.mae,
            "mix_mae": mix.mae,
            "mix_vs_structured_mae_improvement_fraction": (
                structured.mae - mix.mae
            )
            / structured.mae,
        }
        if "random_csdi" in lookup:
            random = lookup["random_csdi"]
            row.update(
                {
                    "random_rmse": random.rmse,
                    "mix_vs_random_rmse_improvement_fraction": (
                        random.rmse - mix.rmse
                    )
                    / random.rmse,
                    "random_mae": random.mae,
                    "mix_vs_random_mae_improvement_fraction": (
                        random.mae - mix.mae
                    )
                    / random.mae,
                }
            )
        rows.append(row)
    return sorted(rows, key=lambda row: row["protocol_key"])


def _plot_three_model_comparison(headline: pd.DataFrame, output: Path) -> None:
    data = headline[headline["method"] == "csdi"]
    common = []
    for protocol, group in data.groupby("protocol_key"):
        if set(MODEL_LABELS).issubset(set(group["model_kind"])):
            common.append(protocol)
    common.sort()
    if not common:
        return
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    x = np.arange(len(common))
    width = 0.25
    for offset, model_kind in enumerate(MODEL_LABELS):
        lookup = data[data["model_kind"] == model_kind].set_index("protocol_key")
        for axis, metric in zip(axes, ("rmse", "mae")):
            axis.bar(
                x + (offset - 1) * width,
                [lookup.loc[key, metric] for key in common],
                width,
                label=MODEL_LABELS[model_kind],
                color=MODEL_COLORS[model_kind],
            )
            axis.set_ylabel(metric.upper())
            axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3)
    axes[1].set_xticks(x, common, rotation=35, ha="right")
    fig.suptitle("相同固定协议上的三模型比较（归一化空间）")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_report(config: dict[str, Any], config_path: Path) -> None:
    _configure_chinese_font()
    root = _resolve(config["evaluation"]["result_root"], config_path)
    results = _collect_results(config, config_path)
    headline_rows = _headline_rows(results)
    headline = pd.DataFrame(headline_rows)
    comparisons = _comparison_rows(headline)

    mix_protocols = set(
        headline.loc[
            (headline["model_kind"] == "mix_csdi")
            & (headline["method"] == "csdi"),
            "protocol_key",
        ]
    )
    expected = {
        "random_missing_10",
        "random_missing_50",
        "random_missing_90",
        "time_block_missing_10",
        "time_block_missing_50",
        "time_block_missing_90",
        "channel_dropout_missing_10",
        "channel_dropout_missing_50",
        "channel_dropout_missing_90",
        "rectangle_missing_10",
        "rectangle_missing_50",
        "rectangle_missing_90",
        "gap_onset",
        "gap_sustained",
    }
    if mix_protocols != expected:
        raise ValueError(
            f"Mix evaluation is incomplete: {sorted(expected - mix_protocols)}"
        )

    evaluation = root / "evaluation"
    channel_rows = [row for result in results for row in result["channel_rows"]]
    group_rows = [row for result in results for row in result["group_rows"]]
    _write_csv(evaluation / "headline_metrics.csv", headline_rows)
    _write_csv(evaluation / "channel_metrics.csv", channel_rows)
    _write_csv(evaluation / "group_metrics.csv", group_rows)
    _write_csv(evaluation / "mix_comparisons.csv", comparisons)

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_three_model_comparison(headline, figures / "three_model_comparison.png")

    mix = headline[
        (headline["model_kind"] == "mix_csdi")
        & (headline["method"] == "csdi")
    ].sort_values(["mask_family", "requested_missing_ratio", "protocol_key"])
    mix_table = [
        "| 协议 | 类型 | 严重度 | 目标点 | RMSE | MAE | CRPS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in mix.itertuples():
        mix_table.append(
            f"| {row.protocol_key} | {FAMILY_LABELS.get(row.mask_family, row.mask_family)} | "
            f"{row.severity} | {int(row.eval_points):,} | {row.rmse:.6f} | "
            f"{row.mae:.6f} | {row.crps:.6f} |"
        )

    comparison_table = [
        "| 协议 | Random RMSE | Structured RMSE | Mix RMSE | Mix vs Random | Mix vs Structured |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        random_rmse = row.get("random_rmse")
        random_text = f"{random_rmse:.6f}" if random_rmse is not None else "—"
        versus_random = row.get("mix_vs_random_rmse_improvement_fraction")
        versus_random_text = (
            f"{versus_random:.2%}" if versus_random is not None else "—"
        )
        comparison_table.append(
            f"| {row['protocol_key']} | {random_text} | {row['structured_rmse']:.6f} | "
            f"{row['mix_rmse']:.6f} | {versus_random_text} | "
            f"{row['mix_vs_structured_rmse_improvement_fraction']:.2%} |"
        )

    report = f"""# ESA Mission 1 Random–Structured Mix CSDI 实验报告

## 实验口径

每个训练窗口以 `{float(config['mix_mask']['random_probability']):.0%}` 概率使用随机散点遮盖；其余窗口在连续时间块、整窗通道块和通道–时间矩形块之间按配置概率选择。该实验是 Random–Structured Mix，不是论文原版 Historical Mix。模型固定训练到第 {int(config['train']['steps']):,} 步，不使用测试集选模。

## Mix-CSDI headline

{chr(10).join(mix_table)}

## 与 Random / Structured-Only 的固定协议比较

正改进率表示 Mix-CSDI 的 RMSE 更低。

{chr(10).join(comparison_table)}

![三模型比较](figures/three_model_comparison.png)

## 解释限制

- 当前配置只有 seed={int(config['train']['seed'])}；最终论文结论需要多个训练种子。
- 输入为 76 个遥测通道，不包含 11 条高优先级 telecommand。
- anomaly/rare event 保留为观测及潜在目标；communication gap 和非有限值作为自然缺失。
- gap 协议是从训练段通信中断构造的测试压力场景，测试段本身没有自然 gap 事件。
- CRPS 基于每窗 {int(config['evaluation']['nsample'])} 个生成样本，只与相同采样数实验比较。
"""
    (root / "report_zh.md").write_text(report, encoding="utf-8")

    _write_json(
        root / "summary.json",
        {
            "experiment": "ESA Mission 1 random-structured mixed masking",
            "random_probability": float(config["mix_mask"]["random_probability"]),
            "structured_family_probabilities": config["mix_mask"][
                "structured_family_probabilities"
            ],
            "mix_model_protocols": len(mix_protocols),
            "comparisons": comparisons,
            "artifacts": {
                "report": str((root / "report_zh.md").resolve()),
                "headline_csv": str((evaluation / "headline_metrics.csv").resolve()),
                "comparison_csv": str((evaluation / "mix_comparisons.csv").resolve()),
                "figure": str((figures / "three_model_comparison.png").resolve()),
            },
        },
    )
