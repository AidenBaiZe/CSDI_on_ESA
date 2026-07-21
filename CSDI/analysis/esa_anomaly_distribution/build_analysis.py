from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "esa_csdi_preprocessing/data/processed/mission1_ch41_46_10m_30s/aligned.npz"
MANIFEST = ROOT / "esa_csdi_preprocessing/data/processed/mission1_ch41_46_10m_30s/manifest.json"
EXPERIMENT_DIR = ROOT / "esa_csdi_experiment"
CHANNELS = [f"channel_{index}" for index in range(41, 47)]
SPLIT_NAMES = {0: "训练集", 1: "验证集", 2: "测试集"}
LABEL_NAMES = {0: "正常", 1: "异常", 2: "稀有事件", 3: "通信中断"}


def contiguous_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return starts, ends


def build_results() -> dict[str, object]:
    with np.load(ARTIFACT) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}

    timestamps = pd.to_datetime(arrays["timestamps_ns"])
    labels = arrays["label_code"]
    splits = arrays["split_code"]
    observed = arrays["observed_mask"]
    clean = arrays["clean_mask"]
    total_scalars = int(labels.size)
    total_timepoints = int(labels.shape[0])

    overall = {
        "timepoints": total_timepoints,
        "scalar_values": total_scalars,
        "observed_zero_count": int((observed == 0).sum()),
        "anomaly_count": int((labels == 1).sum()),
        "rare_count": int((labels == 2).sum()),
        "communication_gap_count": int((labels == 3).sum()),
        "abnormal_scalar_count": int((labels > 0).sum()),
        "abnormal_scalar_rate": float((labels > 0).mean()),
        "abnormal_timepoint_count": int((labels > 0).any(axis=1).sum()),
        "abnormal_timepoint_rate": float((labels > 0).any(axis=1).mean()),
    }

    channel_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    for column, channel in enumerate(CHANNELS):
        abnormal = labels[:, column] > 0
        starts, ends = contiguous_runs(abnormal)
        lengths = ends - starts + 1
        channel_rows.append(
            {
                "channel": channel.replace("channel_", "通道 "),
                "normal_count": int((labels[:, column] == 0).sum()),
                "anomaly_count": int((labels[:, column] == 1).sum()),
                "rare_count": int((labels[:, column] == 2).sum()),
                "abnormal_count": int(abnormal.sum()),
                "abnormal_rate": float(abnormal.mean()),
                "episode_count": int(lengths.size),
                "median_duration_minutes": float(np.median(lengths) * 0.5),
                "mean_duration_minutes": float(np.mean(lengths) * 0.5),
                "max_duration_hours": float(np.max(lengths) / 120),
            }
        )
        for start, end in zip(starts, ends):
            episode_rows.append(
                {
                    "channel": channel,
                    "start": timestamps[start].isoformat(),
                    "end": timestamps[end].isoformat(),
                    "duration_minutes": float((end - start + 1) * 0.5),
                    "label_types": ",".join(
                        LABEL_NAMES[int(code)]
                        for code in np.unique(labels[start : end + 1, column])
                        if int(code) > 0
                    ),
                }
            )

    month_index = pd.PeriodIndex(timestamps, freq="M").astype(str)
    monthly_rows: list[dict[str, object]] = []
    for month in pd.unique(month_index):
        selected = month_index == month
        scalar_denominator = int(selected.sum() * labels.shape[1])
        monthly_rows.append(
            {
                "month": month,
                "timepoints": int(selected.sum()),
                "anomaly_count": int((labels[selected] == 1).sum()),
                "rare_count": int((labels[selected] == 2).sum()),
                "abnormal_count": int((labels[selected] > 0).sum()),
                "abnormal_rate": float((labels[selected] > 0).sum() / scalar_denominator),
                "affected_timepoints": int((labels[selected] > 0).any(axis=1).sum()),
                "affected_timepoint_rate": float((labels[selected] > 0).any(axis=1).mean()),
            }
        )

    split_rows: list[dict[str, object]] = []
    for split_code, split_name in SPLIT_NAMES.items():
        selected = splits == split_code
        split_rows.append(
            {
                "split": split_name,
                "timepoints": int(selected.sum()),
                "anomaly_count": int((labels[selected] == 1).sum()),
                "rare_count": int((labels[selected] == 2).sum()),
                "abnormal_count": int((labels[selected] > 0).sum()),
                "abnormal_rate": float((labels[selected] > 0).mean()),
                "affected_timepoints": int((labels[selected] > 0).any(axis=1).sum()),
            }
        )

    simultaneous_counts = (labels > 0).sum(axis=1)
    simultaneous_rows = [
        {
            "abnormal_channels": int(count),
            "timepoints": int((simultaneous_counts == count).sum()),
            "share_of_abnormal_timepoints": float(
                (simultaneous_counts == count).sum() / (simultaneous_counts > 0).sum()
            )
            if count > 0
            else 0.0,
        }
        for count in range(1, 7)
    ]

    sys.path.insert(0, str(EXPERIMENT_DIR))
    from dataset_esa import (  # noqa: PLC0415
        ESAWindowDataset,
        ESAWindowStore,
        build_clean_window_starts,
        deterministic_condition_mask,
    )

    store = ESAWindowStore.load(ARTIFACT, MANIFEST)
    window_length = 96
    stride = 48
    window_rows: list[dict[str, object]] = []
    natural_window_rows: list[dict[str, object]] = []
    for split_code, split in enumerate(("train", "validation", "test")):
        split_size = int((splits == split_code).sum())
        candidate_count = ((split_size - window_length) // stride) + 1
        clean_starts = build_clean_window_starts(store, split, window_length, stride)
        window_rows.append(
            {
                "split": SPLIT_NAMES[split_code],
                "candidate_windows": int(candidate_count),
                "usable_clean_windows": int(clean_starts.size),
                "excluded_windows": int(candidate_count - clean_starts.size),
                "excluded_rate": float(1 - clean_starts.size / candidate_count),
            }
        )
        missing_rates = np.asarray(
            [(labels[start : start + window_length] > 0).mean() for start in np.arange(
                int(np.flatnonzero(splits == split_code)[0]),
                int(np.flatnonzero(splits == split_code)[-1]) + 2 - window_length,
                stride,
                dtype=np.int64,
            )],
            dtype=np.float64,
        )
        affected_rates = missing_rates[missing_rates > 0]
        natural_window_rows.append(
            {
                "split": SPLIT_NAMES[split_code],
                "candidate_windows": int(missing_rates.size),
                "windows_with_missing": int((missing_rates > 0).sum()),
                "partial_missing_windows": int(((missing_rates > 0) & (missing_rates < 1)).sum()),
                "fully_missing_windows": int((missing_rates == 1).sum()),
                "affected_window_rate": float((missing_rates > 0).mean()),
                "median_missing_rate_affected": float(np.median(affected_rates)) if affected_rates.size else 0.0,
                "mean_missing_rate_affected": float(np.mean(affected_rates)) if affected_rates.size else 0.0,
                "max_missing_rate": float(missing_rates.max()) if missing_rates.size else 0.0,
            }
        )

    artificial_rows: list[dict[str, object]] = []
    test_dataset = ESAWindowDataset(store, "test", window_length, stride, 0.1, 2026)
    points_per_window = window_length * len(CHANNELS)
    for ratio in (0.1, 0.5, 0.9):
        by_channel = np.zeros(len(CHANNELS), dtype=np.int64)
        by_position = np.zeros(window_length, dtype=np.int64)
        for start in test_dataset.starts:
            condition = deterministic_condition_mask(
                np.ones((window_length, len(CHANNELS)), dtype=np.float32),
                ratio,
                2026,
                int(start),
            )
            missing = condition == 0
            by_channel += missing.sum(axis=0)
            by_position += missing.sum(axis=1)
        masked_per_window = int(round(points_per_window * ratio))
        total_masked = int(by_channel.sum())
        for channel_index, channel in enumerate(CHANNELS):
            artificial_rows.append(
                {
                    "requested_ratio": ratio,
                    "actual_ratio": masked_per_window / points_per_window,
                    "windows": int(test_dataset.starts.size),
                    "masked_per_window": masked_per_window,
                    "total_masked_window_cells": total_masked,
                    "channel": channel.replace("channel_", "通道 "),
                    "channel_masked_count": int(by_channel[channel_index]),
                    "channel_share": float(by_channel[channel_index] / total_masked),
                    "position_min_count": int(by_position.min()),
                    "position_max_count": int(by_position.max()),
                }
            )

    return {
        "overall": overall,
        "channels": channel_rows,
        "episodes": episode_rows,
        "monthly": monthly_rows,
        "splits": split_rows,
        "simultaneous": simultaneous_rows,
        "windows": window_rows,
        "natural_windows": natural_window_rows,
        "artificial": artificial_rows,
    }


def save_tables(results: dict[str, object]) -> None:
    database_path = OUT / "analysis.sqlite"
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(database_path)
    for name in ("channels", "episodes", "monthly", "splits", "simultaneous", "windows", "natural_windows", "artificial"):
        frame = pd.DataFrame(results[name])
        frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
        frame.to_sql(name, connection, index=False)
    connection.close()
    (OUT / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_notebook(results: dict[str, object]) -> None:
    overall = results["overall"]
    def markdown(source: str) -> dict[str, object]:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

    def code(source: str) -> dict[str, object]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
        },
        "cells": [
        markdown(
            "# ESA 子集异常与实验缺失分布分析\n\n"
            "## tl;dr\n\n"
            f"- 原始观测掩码中空值为 **{overall['observed_zero_count']:,}**；ESA 标注异常不是原始缺测。\n"
            f"- ESA 异常/稀有事件共 **{overall['abnormal_scalar_count']:,}** 个通道-时间单元，占 **{overall['abnormal_scalar_rate']:.2%}**。\n"
            f"- 至少一个通道异常的时间点为 **{overall['abnormal_timepoint_count']:,}**，占 **{overall['abnormal_timepoint_rate']:.2%}**。\n"
            "- 当前 CSDI 加载器只使用全通道正常窗口；评估缺失由 10%/50%/90% 随机掩蔽另外生成。"
        ),
        markdown(
            "## Context & Methods\n\n"
            "### Key Assumptions\n\n"
            "- ESA 标签码 1、2 分别视为异常和稀有事件，均归入“不干净/不可训练”位置。\n"
            "- 异常率以 878,400 × 6 个通道-时间单元为分母；时间点影响率以 878,400 个时间戳为分母。\n"
            "- 连续异常段以 30 秒相邻标签连续为定义。人工缺失统计按重叠窗口中的评估单元计数。"
        ),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n\n"
            "HERE = Path.cwd()\n"
            "if not (HERE / 'summary.json').exists():\n"
            "    HERE = Path('analysis/esa_anomaly_distribution')\n"
            "results = json.loads((HERE / 'summary.json').read_text(encoding='utf-8'))\n"
            "channels = pd.read_csv(HERE / 'channels.csv')\n"
            "monthly = pd.read_csv(HERE / 'monthly.csv')\n"
            "splits = pd.read_csv(HERE / 'splits.csv')\n"
            "windows = pd.read_csv(HERE / 'windows.csv')\n"
            "natural_windows = pd.read_csv(HERE / 'natural_windows.csv')\n"
            "artificial = pd.read_csv(HERE / 'artificial.csv')\n"
            "results['overall']"
        ),
        markdown("## Data\n\n### 通道级异常概况"),
        code("channels"),
        markdown("### 月度分布"),
        code("monthly"),
        markdown("## Results\n\n### 按数据划分的异常分布"),
        code("splits"),
        markdown("### 严格正常窗口保留情况"),
        code("windows"),
        markdown("### 如果把 ESA 异常保留为窗口内自然缺失"),
        code("natural_windows"),
        markdown("### CSDI 人工缺失分布"),
        code("artificial"),
        markdown(
            "## Takeaways\n\n"
            "1. ESA 标签异常与 CSDI 人工缺失是两层不同机制：前者用于排除窗口，后者才是模型评价目标。\n"
            "2. 异常分布并不均匀，验证集没有 ESA 标签异常，测试集异常集中在 10 月。\n"
            "3. 大多数受影响时间点六个通道同步异常，若按单元数报告必须同时给出去重后的受影响时间点数。\n"
            "4. 人工掩蔽在通道和窗口位置上近似均匀，但 10% 和 90% 因每窗整数取整分别对应 10.069% 和 89.931%。"
        ),
        ],
    }
    notebook_path = OUT / "esa_anomaly_distribution_analysis.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    reloaded = json.loads(notebook_path.read_text(encoding="utf-8"))
    if reloaded.get("nbformat") != 4 or not reloaded.get("cells"):
        raise ValueError("Generated notebook failed structural validation")


def build_report_artifact(results: dict[str, object]) -> None:
    overall = results["overall"]
    monthly = results["monthly"]
    channels = results["channels"]
    splits = results["splits"]
    windows = results["windows"]
    natural_windows = results["natural_windows"]
    artificial_summary = []
    for ratio in (0.1, 0.5, 0.9):
        rows = [row for row in results["artificial"] if row["requested_ratio"] == ratio]
        artificial_summary.append(
            {
                "requested_ratio": ratio,
                "actual_ratio": rows[0]["actual_ratio"],
                "masked_per_window": rows[0]["masked_per_window"],
                "windows": rows[0]["windows"],
                "total_masked_window_cells": rows[0]["total_masked_window_cells"],
                "min_channel_share": min(row["channel_share"] for row in rows),
                "max_channel_share": max(row["channel_share"] for row in rows),
            }
        )

    generated_at = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    source = {
        "id": "esa_processed",
        "label": "ESA Mission 1 预处理数据（通道 41–46，2000-01 至 2000-11）",
        "query": {
            "engine": "Python/NumPy",
            "language": "python",
            "description": "读取 aligned.npz 的标签、观测、清洁和划分数组，并按通道、月份、划分与连续异常段聚合。",
            "tables_used": ["ESA Mission 1: mission1_ch41_46_10m_30s/aligned.npz"],
            "filters": ["channels 41-46", "2000-01-01 00:00:30 through 2000-11-01 00:00:00", "30-second grid"],
            "metric_definitions": [
                "异常单元 = label_code in {1,2}",
                "异常单元率 = 异常通道-时间单元 / (878400 × 6)",
                "受影响时间点 = 任一通道 label_code in {1,2}",
                "人工缺失 = observed_mask - gt_mask within strict-clean CSDI windows",
            ],
            "executed_at": generated_at,
        },
    }
    experiment_source = {
        "id": "esa_experiment",
        "label": "ESA CSDI 实验配置与数据加载逻辑",
        "query": {
            "engine": "Python",
            "language": "python",
            "description": "复用 dataset_esa.py 的严格清洁窗口和 deterministic_condition_mask 逻辑。",
            "tables_used": ["esa_csdi_experiment/dataset_esa.py", "esa_csdi_experiment/config.yaml"],
            "filters": ["window_length=96", "stride=48", "mask_seed=2026", "test split"],
            "metric_definitions": ["每个窗口含 96 × 6 = 576 个通道-时间单元", "掩蔽数 = round(576 × missing_ratio)"],
            "executed_at": generated_at,
        },
    }

    def sql_source(source_id: str, label: str, table: str, description: str) -> dict[str, object]:
        return {
            "id": source_id,
            "label": label,
            "query": {
                "engine": "SQLite",
                "language": "sql",
                "sql": f"SELECT * FROM {table}",
                "description": description,
                "tables_used": [f"analysis.sqlite:{table}"],
                "filters": ["ESA Mission 1 channels 41-46", "2000-01 through 2000-11", "30-second grid"],
                "executed_at": generated_at,
            },
        }

    monthly_source = sql_source("monthly_source", "ESA monthly anomaly aggregates", "monthly", "Monthly anomaly and rare-event distribution.")
    channel_source = sql_source("channel_source", "ESA channel anomaly aggregates", "channels", "Channel-level anomaly rates and episode statistics.")
    split_source = sql_source("split_source", "ESA split anomaly aggregates", "splits", "Anomaly distribution across train, validation, and test ranges.")
    window_source = sql_source("window_source", "ESA strict-clean window statistics", "windows", "Candidate-window retention under the current strict-clean loader.")
    natural_window_source = sql_source("natural_window_source", "ESA natural-missing window statistics", "natural_windows", "Window missingness when ESA anomalies are treated as natural missing values.")
    artificial_source = sql_source("artificial_source", "ESA artificial masking statistics", "artificial", "Deterministic artificial-mask distribution over test windows.")
    all_sources = [
        source,
        experiment_source,
        monthly_source,
        channel_source,
        split_source,
        window_source,
        natural_window_source,
        artificial_source,
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# ESA 子集异常与实验缺失分布", "layout": "full"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "esa_processed",
            "layout": "full",
            "body": (
                "## Executive Summary\n\n"
                f"- **原始数据没有真正空值。** `observed_mask` 中未观测单元为 **0**；ESA 标签异常/稀有事件共有 **{overall['abnormal_scalar_count']:,}** 个通道-时间单元，占 **{overall['abnormal_scalar_rate']:.2%}**。\n"
                f"- **异常高度同步且集中成段。** 至少一个通道异常的时间点只有 **{overall['abnormal_timepoint_count']:,}** 个（**{overall['abnormal_timepoint_rate']:.2%}**），其中 **{results['simultaneous'][-1]['share_of_abnormal_timepoints']:.2%}** 是六通道同时异常。\n"
                "- **时间划分并不均衡。** 验证集没有 ESA 标注异常；测试集异常全部落在 10 月，因此验证损失不能代表模型在真实异常附近的表现。\n"
                "- **你设想的训练方式与当前代码不一致。** 你的定义是把 ESA 异常保留为窗口内自然缺失；当前加载器却删除所有含异常窗口，再按 10%/50%/90% 随机遮盖正常值。"
            ),
        },
        {
            "id": "definitions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 两类“缺失”必须分开理解\n\n"
                "**按你的实验定义，ESA 标签异常属于派生自然缺失。** 六个通道原始文件没有空值；标签码 1/2 指示需要剔除的异常或稀有事件，剔除后形成 1.815% 的缺失掩码。\n\n"
                "**人工缺失是另一层评价掩码。** 在剩余可观测位置上再随机遮盖 10%/50%/90%，用于计算插补误差。自然缺失没有可信真值，通常不能直接作为 RMSE/MAE 的评价目标。"
            ),
        },
        {
            "id": "monthly_text",
            "type": "markdown",
            "sourceId": "esa_processed",
            "layout": "full",
            "body": (
                "## 异常只集中在少数月份\n\n"
                "2 月和 7 月主要是稀有事件，3–4 月主要是异常，6 月有少量异常，10 月同时出现异常和少量稀有事件；1 月、5 月、8 月和9 月没有标签异常。"
            ),
        },
        {"id": "monthly_chart_block", "type": "chart", "chartId": "monthly_chart", "layout": "full"},
        {
            "id": "channel_text",
            "type": "markdown",
            "sourceId": "esa_processed",
            "layout": "full",
            "body": (
                "## 六个通道的异常比例接近，但并非独立\n\n"
                "各通道异常单元率约为 1.78%–1.86%，表面上十分接近。真正重要的是同步性：绝大多数受影响时间点六个通道同时被标注，因此把六个通道的异常数简单相加会夸大独立异常事件数量。"
            ),
        },
        {"id": "channel_chart_block", "type": "chart", "chartId": "channel_chart", "layout": "full"},
        {
            "id": "split_text",
            "type": "markdown",
            "sourceId": "esa_processed",
            "layout": "full",
            "body": (
                "## 时间切分造成异常覆盖偏斜\n\n"
                "训练集承载大部分标签异常，验证集完全没有标签异常，测试集的异常都位于 10 月。严格清洁窗口筛选又会删除任何触碰标签异常的候选窗口，所以模型训练和评价数据本质上来自正常区间。"
            ),
        },
        {"id": "split_table_block", "type": "table", "tableId": "split_table", "layout": "full"},
        {"id": "window_table_block", "type": "table", "tableId": "window_table", "layout": "full"},
        {
            "id": "natural_window_text",
            "type": "markdown",
            "sourceId": "esa_processed",
            "layout": "full",
            "body": (
                "## 直接保留异常为窗口内缺失会产生大量全缺失窗口\n\n"
                "虽然全局缺失率只有 1.815%，缺失并非随机散点，而是跨通道同步的长块。训练集 285 个含缺失候选窗口中有 254 个完全缺失；测试集 81 个含缺失窗口中有 56 个完全缺失。因此更合适的 Air Quality 式实现是：保留部分缺失窗口，但继续剔除全缺失窗口。"
            ),
        },
        {"id": "natural_window_table_block", "type": "table", "tableId": "natural_window_table", "layout": "full"},
        {
            "id": "mask_text",
            "type": "markdown",
            "sourceId": "esa_experiment",
            "layout": "full",
            "body": (
                "## 人工缺失近似均匀，但存在整数取整\n\n"
                "每个窗口共有 576 个单元。10%、50%、90% 分别实际遮盖 58、288、518 个，所以实际比例是 10.069%、50.000%、89.931%。固定随机种子让结果可复现，六通道获得的遮盖份额都接近 1/6。"
            ),
        },
        {"id": "mask_table_block", "type": "table", "tableId": "mask_table", "layout": "full"},
        {
            "id": "next_steps",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 建议的实验表述与下一步\n\n"
                "1. 将 `label_code in {1,2}` 称为 **ESA 标注异常/排除位置**，不要直接称为原始缺失值。\n"
                "2. 在实现中令 `observed_mask = clean_mask`，保留部分缺失窗口；仅剔除观测数为 0 的全缺失窗口。\n"
                "3. 将正常位置上的额外随机遮盖称为 **人工缺失/插补目标**，并报告每窗实际遮盖数。\n"
                "4. 补充按异常事件或月份分层的验证，避免“验证集无自然缺失、测试集自然缺失集中在10月”带来的分布偏差。"
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 进一步问题\n\n"
                "- 论文目标是插补正常遥测中的随机缺测，还是恢复异常期间的遥测值？这两种任务需要不同的数据窗口定义。\n"
                "- 是否需要把稀有事件与异常分别作为不可训练位置，还是保留稀有事件用于泛化测试？"
            ),
        },
        {
            "id": "caveats",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Caveats and Assumptions\n\n"
                "统计以 30 秒网格上的标签码为准；连续异常段是标签连续段，并不等同于独立根因事件。人工缺失按重叠窗口中的单元计数，因此同一原始时间点可能在不同窗口中重复出现。"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "ESA 子集异常与实验缺失分布",
            "description": "通道 41–46、2000 年 1 月至 11 月：ESA 标签异常与 CSDI 人工缺失的区分和分布。",
            "generatedAt": generated_at,
            "sources": all_sources,
            "blocks": blocks,
            "charts": [
                {
                    "id": "monthly_chart",
                    "title": "月度 ESA 标签分布",
                    "subtitle": "2000 年 1–11 月，按通道-时间单元计数",
                    "type": "stackedBar",
                    "intent": "composition",
                    "dataset": "monthly",
                    "sourceId": "monthly_source",
                    "encodings": {
                        "x": {"field": "month", "type": "ordinal", "label": "月份"},
                        "y": {"fields": ["anomaly_count", "rare_count"], "type": "quantitative", "aggregate": "none", "label": "标签单元数"},
                        "tooltip": [
                            {"field": "affected_timepoints", "type": "quantitative", "label": "受影响时间点"},
                            {"field": "abnormal_rate", "type": "quantitative", "format": "percent", "label": "异常单元率"},
                        ],
                    },
                    "palette": {"kind": "categorical", "name": "blue-orange"},
                    "valueFormat": "compact",
                    "layout": "full",
                    "surface": {"surface": "card", "viewMode": "both"},
                },
                {
                    "id": "channel_chart",
                    "title": "各通道 ESA 标签异常率",
                    "subtitle": "异常与稀有事件合计，占各通道 878,400 个时间点的比例",
                    "type": "horizontalBar",
                    "intent": "comparison",
                    "dataset": "channels",
                    "sourceId": "channel_source",
                    "encodings": {
                        "x": {"field": "channel", "type": "nominal", "label": "通道"},
                        "y": {"field": "abnormal_rate", "type": "quantitative", "format": "percent", "label": "异常率"},
                        "tooltip": [
                            {"field": "abnormal_count", "type": "quantitative", "label": "异常单元数"},
                            {"field": "episode_count", "type": "quantitative", "label": "连续异常段数"},
                        ],
                    },
                    "valueFormat": "percent",
                    "palette": {"kind": "sequential", "name": "blue"},
                    "layout": "full",
                    "surface": {"surface": "card", "viewMode": "both"},
                },
            ],
            "tables": [
                {
                    "id": "split_table",
                    "title": "训练、验证、测试区间的 ESA 标签分布",
                    "subtitle": "按时间顺序 70%/15%/15% 划分；精确值用于核对分布偏斜",
                    "dataset": "splits",
                    "sourceId": "split_source",
                    "defaultSort": {"field": "timepoints", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "split", "label": "划分", "type": "text"},
                        {"field": "timepoints", "label": "时间点", "format": "number"},
                        {"field": "anomaly_count", "label": "异常单元", "format": "number"},
                        {"field": "rare_count", "label": "稀有事件单元", "format": "number"},
                        {"field": "abnormal_rate", "label": "异常单元率", "format": "percent"},
                        {"field": "affected_timepoints", "label": "受影响时间点", "format": "number"},
                    ],
                },
                {
                    "id": "window_table",
                    "title": "严格清洁窗口筛选结果",
                    "subtitle": "96 步窗口、48 步步长；触碰任一 ESA 标签异常的候选窗口被排除",
                    "dataset": "windows",
                    "sourceId": "window_source",
                    "defaultSort": {"field": "candidate_windows", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "split", "label": "划分", "type": "text"},
                        {"field": "candidate_windows", "label": "候选窗口", "format": "number"},
                        {"field": "usable_clean_windows", "label": "可用正常窗口", "format": "number"},
                        {"field": "excluded_windows", "label": "排除窗口", "format": "number"},
                        {"field": "excluded_rate", "label": "排除率", "format": "percent"},
                    ],
                },
                {
                    "id": "mask_table",
                    "title": "测试集人工缺失比例",
                    "subtitle": "2,663 个严格正常窗口；每窗 576 个单元，固定随机种子 2026",
                    "dataset": "artificial_summary",
                    "sourceId": "artificial_source",
                    "defaultSort": {"field": "requested_ratio", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "requested_ratio", "label": "设定比例", "format": "percent"},
                        {"field": "actual_ratio", "label": "实际比例", "format": "percent"},
                        {"field": "masked_per_window", "label": "每窗遮盖单元", "format": "number"},
                        {"field": "total_masked_window_cells", "label": "窗口单元总计", "format": "number"},
                        {"field": "min_channel_share", "label": "最小通道份额", "format": "percent"},
                        {"field": "max_channel_share", "label": "最大通道份额", "format": "percent"},
                    ],
                },
                {
                    "id": "natural_window_table",
                    "title": "异常作为自然缺失时的窗口分布",
                    "subtitle": "96 步窗口、48 步步长；部分缺失窗口与全缺失窗口分开统计",
                    "dataset": "natural_windows",
                    "sourceId": "natural_window_source",
                    "defaultSort": {"field": "candidate_windows", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "split", "label": "划分", "type": "text"},
                        {"field": "candidate_windows", "label": "候选窗口", "format": "number"},
                        {"field": "windows_with_missing", "label": "含缺失窗口", "format": "number"},
                        {"field": "partial_missing_windows", "label": "部分缺失窗口", "format": "number"},
                        {"field": "fully_missing_windows", "label": "全缺失窗口", "format": "number"},
                        {"field": "affected_window_rate", "label": "含缺失窗口占比", "format": "percent"},
                        {"field": "mean_missing_rate_affected", "label": "受影响窗口平均缺失率", "format": "percent"},
                    ],
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "monthly": monthly,
                "channels": channels,
                "splits": splits,
                "windows": windows,
                "natural_windows": natural_windows,
                "artificial_summary": artificial_summary,
            },
            "accessIssues": [],
        },
        "sources": all_sources,
    }
    (OUT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = build_results()
    save_tables(results)
    build_notebook(results)
    build_report_artifact(results)
    print(json.dumps(results["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
