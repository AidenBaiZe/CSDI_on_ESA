from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ZIP_ROOT = ROOT / "datasets/ESA Anomaly Dataset"
RAW_ROOT = OUT / "interim/ESA-Mission2/channels"
RECOMMENDED = [9, 10, 13, 14, 74, 86]


def read_labels(mission: str) -> pd.DataFrame:
    archive = ZIP_ROOT / f"{mission}.zip"
    with ZipFile(archive) as loaded:
        labels = pd.read_csv(io.BytesIO(loaded.read(f"{mission}/labels.csv")))
        types = pd.read_csv(io.BytesIO(loaded.read(f"{mission}/anomaly_types.csv")))
    labels = labels.merge(types[["ID", "Category"]], on="ID", how="left")
    labels["StartTime"] = pd.to_datetime(labels["StartTime"], utc=True).dt.tz_localize(None)
    labels["EndTime"] = pd.to_datetime(labels["EndTime"], utc=True).dt.tz_localize(None)
    return labels


def build_mask(labels: pd.DataFrame, channels: list[int], start: str, end: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    grid = pd.date_range(start, end, freq="30s")
    mask = np.zeros((len(grid), len(channels)), dtype=bool)
    for column, channel in enumerate(channels):
        selected = labels[labels["Channel"].eq(f"channel_{channel}")]
        for row in selected.itertuples():
            left = int(grid.searchsorted(row.StartTime.ceil("30s"), side="left"))
            right = int(grid.searchsorted(row.EndTime.floor("30s"), side="right"))
            if right > left:
                mask[left:right, column] = True
    return grid, mask


def summarize_mask(mask: np.ndarray, window_length: int = 96, stride: int = 48) -> dict[str, float | int]:
    missing_by_time = mask.sum(axis=1)
    starts = np.arange(0, len(mask) - window_length + 1, stride, dtype=np.int64)
    cumulative = np.concatenate([[0], np.cumsum(missing_by_time, dtype=np.int64)])
    missing_by_window = cumulative[starts + window_length] - cumulative[starts]
    full_size = window_length * mask.shape[1]
    affected = missing_by_window > 0
    return {
        "timepoints": int(len(mask)),
        "missing_rate": float(mask.mean()),
        "affected_timepoint_rate": float((missing_by_time > 0).mean()),
        "all_channel_sync_share": float((missing_by_time == mask.shape[1]).sum() / (missing_by_time > 0).sum()) if (missing_by_time > 0).any() else 0.0,
        "candidate_windows": int(starts.size),
        "partial_windows": int(((missing_by_window > 0) & (missing_by_window < full_size)).sum()),
        "fully_missing_windows": int((missing_by_window == full_size).sum()),
        "affected_windows": int(affected.sum()),
        "full_share_affected": float((missing_by_window == full_size).sum() / affected.sum()) if affected.any() else 0.0,
        "mean_missing_rate_affected": float((missing_by_window[affected] / full_size).mean()) if affected.any() else 0.0,
    }


def correlation_for(channels: list[int]) -> pd.DataFrame:
    series = {}
    for channel in channels:
        path = RAW_ROOT / f"channel_{channel}.zip"
        loaded = pd.read_pickle(path)
        values = loaded.iloc[:, 0] if isinstance(loaded, pd.DataFrame) else loaded
        values = values[~values.index.duplicated(keep="last")].sort_index()
        series[f"通道 {channel}"] = values.resample("10min").last().ffill()
    return pd.DataFrame(series).dropna().corr()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mission1 = read_labels("ESA-Mission1")
    mission2 = read_labels("ESA-Mission2")
    mission3 = read_labels("ESA-Mission3")

    option_specs = [
        ("M1 当前 10个月", mission1, [41, 42, 43, 44, 45, 46], "2000-01-01 00:00:30", "2000-11-01 00:00:00"),
        ("M1 扩至 42个月", mission1, [41, 42, 43, 44, 45, 46], "2000-01-01 00:00:30", "2003-06-30 23:59:30"),
        ("M1 扩至 14年", mission1, [41, 42, 43, 44, 45, 46], "2000-01-01 00:00:30", "2013-12-31 23:59:30"),
        ("M2 同组高频通道", mission2, [9, 10, 12, 13, 14, 20], "2000-01-01 00:00:30", "2003-06-30 23:59:30"),
        ("M2 推荐六通道", mission2, RECOMMENDED, "2000-01-01 00:00:30", "2003-06-30 23:59:30"),
    ]

    option_rows = []
    masks = {}
    for option, labels, channels, start, end in option_specs:
        grid, mask = build_mask(labels, channels, start, end)
        masks[option] = mask
        row = summarize_mask(mask)
        row.update({
            "option": option,
            "channels": ", ".join(map(str, channels)),
            "start": start,
            "end": end,
            "duration_months": round((pd.Timestamp(end) - pd.Timestamp(start)).days / 30.4375, 1),
        })
        option_rows.append(row)

    options = pd.DataFrame(option_rows)
    option_order = [item[0] for item in option_specs]
    options["option"] = pd.Categorical(options["option"], option_order, ordered=True)
    options = options.sort_values("option").reset_index(drop=True)

    split_rows = []
    for option in ("M1 当前 10个月", "M2 推荐六通道"):
        mask = masks[option]
        for split, left, right in (("训练集", 0.0, 0.7), ("验证集", 0.7, 0.85), ("测试集", 0.85, 1.0)):
            subset = mask[int(len(mask) * left) : int(len(mask) * right)]
            metrics = summarize_mask(subset)
            split_rows.append({
                "option": option,
                "split": split,
                "missing_rate": metrics["missing_rate"],
                "partial_windows": metrics["partial_windows"],
                "fully_missing_windows": metrics["fully_missing_windows"],
                "affected_windows": metrics["affected_windows"],
            })
    splits = pd.DataFrame(split_rows)

    correlation = correlation_for(RECOMMENDED)
    correlation_long = correlation.stack().rename("correlation").reset_index()
    correlation_long.columns = ["channel_x", "channel_y", "correlation"]
    pair_values = correlation.to_numpy()[np.triu_indices(len(RECOMMENDED), 1)]

    mission_quality = pd.DataFrame([
        {
            "mission": "Mission 1",
            "label_rows": len(mission1),
            "events": mission1["ID"].nunique(),
            "missing_channel_labels": int(mission1["Channel"].isna().sum()),
            "negative_duration_labels": int((mission1["EndTime"] < mission1["StartTime"]).sum()),
            "assessment": "标签可用，但当前通道缺失高度同步且长块化",
        },
        {
            "mission": "Mission 2",
            "label_rows": len(mission2),
            "events": mission2["ID"].nunique(),
            "missing_channel_labels": int(mission2["Channel"].isna().sum()),
            "negative_duration_labels": int((mission2["EndTime"] < mission2["StartTime"]).sum()),
            "assessment": "标签干净，事件密集且多数短于 6 小时",
        },
        {
            "mission": "Mission 3",
            "label_rows": len(mission3),
            "events": mission3["ID"].nunique(),
            "missing_channel_labels": int(mission3["Channel"].isna().sum()),
            "negative_duration_labels": int((mission3["EndTime"] < mission3["StartTime"]).sum()),
            "assessment": "通信中断丰富，但需先修复标签质量问题",
        },
    ])

    recommended = options[options["option"].astype(str).eq("M2 推荐六通道")].iloc[0]
    summary = {
        "recommendation": "Switch to ESA Mission 2 channels 9, 10, 13, 14, 74, 86 and use the full 2000-01-01 to 2003-06-30 range.",
        "recommended_missing_rate": float(recommended["missing_rate"]),
        "recommended_partial_windows": int(recommended["partial_windows"]),
        "recommended_fully_missing_windows": int(recommended["fully_missing_windows"]),
        "recommended_sync_share": float(recommended["all_channel_sync_share"]),
        "mean_absolute_pairwise_correlation": float(np.abs(pair_values).mean()),
        "median_absolute_pairwise_correlation": float(np.median(np.abs(pair_values))),
        "estimated_size_multiple_vs_current": float(recommended["timepoints"] / options.iloc[0]["timepoints"]),
        "options": json.loads(options.to_json(orient="records", force_ascii=False)),
        "splits": json.loads(splits.to_json(orient="records", force_ascii=False)),
        "mission_quality": json.loads(mission_quality.to_json(orient="records", force_ascii=False)),
    }

    options.to_csv(OUT / "options.csv", index=False, encoding="utf-8-sig")
    splits.to_csv(OUT / "splits.csv", index=False, encoding="utf-8-sig")
    correlation_long.to_csv(OUT / "correlation.csv", index=False, encoding="utf-8-sig")
    mission_quality.to_csv(OUT / "mission_quality.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    database = OUT / "analysis.sqlite"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    options.to_sql("options", connection, index=False)
    splits.to_sql("splits", connection, index=False)
    correlation_long.to_sql("correlation", connection, index=False)
    mission_quality.to_sql("mission_quality", connection, index=False)
    connection.close()

    render_static(options, splits, correlation, mission_quality, summary)
    build_artifact(options, splits, correlation_long, mission_quality, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def render_static(options, splits, correlation, mission_quality, summary):
    blue, light, orange, ink, grey, grid = "#2878B5", "#9BC4E2", "#F28E2B", "#24313D", "#A7B0B7", "#DDE3E8"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "text.color": ink, "axes.labelcolor": ink,
        "xtick.color": ink, "ytick.color": ink, "axes.edgecolor": "#B9C2C9",
    })
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.86, hspace=0.44, wspace=0.28)
    fig.suptitle("ESA 子集选择：扩大 Mission 1 还是切换 Mission 2", fontsize=20, fontweight="bold", x=0.04, ha="left")
    fig.text(0.04, 0.945, "统一口径：6 通道、30 秒网格、96 步窗口、48 步步长", fontsize=11, color="#5E6B75")

    labels = [str(value) for value in options["option"]]
    ax = axes[0, 0]
    rates = options["missing_rate"].to_numpy() * 100
    colors = [orange if label == "M2 推荐六通道" else blue for label in labels]
    bars = ax.barh(labels, rates, color=colors, edgecolor="white")
    ax.set_title("A. 方案总体派生缺失率", loc="left", fontsize=14, fontweight="bold", pad=28)
    ax.text(0, 1.015, "标签异常/稀有事件单元 ÷ 全部通道-时间单元", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.set_xlabel("缺失率"); ax.xaxis.set_major_formatter(mtick.PercentFormatter()); ax.set_xlim(0, max(rates) * 1.3)
    ax.grid(axis="x", color=grid); ax.set_axisbelow(True); ax.invert_yaxis()
    for bar, value in zip(bars, rates): ax.text(value + 0.04, bar.get_y() + bar.get_height()/2, f"{value:.2f}%", va="center", fontsize=9)

    ax = axes[0, 1]
    partial = options["partial_windows"].to_numpy(); full = options["fully_missing_windows"].to_numpy()
    y = np.arange(len(labels))
    ax.barh(y, partial, color=light, label="部分缺失窗口")
    ax.barh(y, full, left=partial, color=orange, label="全缺失窗口")
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel("受影响窗口数")
    ax.set_title("B. 受影响窗口的可用性", loc="left", fontsize=14, fontweight="bold", pad=28)
    ax.text(0, 1.015, "Air Quality 式训练需要部分缺失窗口，不能依赖全缺失窗口", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.grid(axis="x", color=grid); ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    current = splits[splits.option.eq("M1 当前 10个月")].set_index("split")
    rec = splits[splits.option.eq("M2 推荐六通道")].set_index("split")
    order = ["训练集", "验证集", "测试集"]; x = np.arange(3); width = 0.34
    b1 = ax.bar(x-width/2, current.loc[order,"missing_rate"]*100, width, color=blue, label="M1 当前")
    b2 = ax.bar(x+width/2, rec.loc[order,"missing_rate"]*100, width, color=orange, label="M2 推荐")
    ax.set_xticks(x, order); ax.set_ylabel("缺失率"); ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("C. 时间划分中的缺失率", loc="left", fontsize=14, fontweight="bold", pad=28)
    ax.text(0, 1.015, "按时间顺序 70% / 15% / 15% 划分", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.grid(axis="y", color=grid); ax.set_axisbelow(True); ax.legend(frameon=False)
    for bars in (b1,b2):
        for bar in bars: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{bar.get_height():.2f}%", ha="center", fontsize=8.5)

    ax = axes[1, 1]
    image = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(6), correlation.columns, rotation=45, ha="right"); ax.set_yticks(range(6), correlation.index)
    ax.set_title("D. M2 推荐通道的遥测相关性", loc="left", fontsize=14, fontweight="bold", pad=28)
    ax.text(0, 1.015, "10 分钟重采样后的 Pearson 相关系数", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    for i in range(6):
        for j in range(6): ax.text(j, i, f"{correlation.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(correlation.iloc[i,j])>.55 else ink)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for axis in axes.flat: axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
    fig.text(0.04,0.025,"建议：切换到 Mission 2 的通道 9、10、13、14、74、86，并使用完整 42 个月；扩大当前 Mission 1 不会解决全缺失长块问题。",fontsize=10.5)
    fig.savefig(OUT/"selection_overview.png",dpi=180,bbox_inches="tight",facecolor="white")


def build_artifact(options, splits, correlation_long, mission_quality, summary):
    generated_at = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    def source(sid, table, description):
        return {"id":sid,"label":description,"query":{"engine":"SQLite","language":"sql","sql":f"SELECT * FROM {table}","description":description,"tables_used":[f"analysis.sqlite:{table}"],"executed_at":generated_at}}
    sources=[source("options_source","options","ESA subset option metrics"),source("splits_source","splits","Chronological split metrics"),source("corr_source","correlation","Recommended-channel correlations"),source("quality_source","mission_quality","Mission label-quality profile")]
    option_rows=json.loads(options.to_json(orient="records",force_ascii=False)); split_rows=json.loads(splits.to_json(orient="records",force_ascii=False)); corr_rows=json.loads(correlation_long.to_json(orient="records",force_ascii=False)); quality_rows=json.loads(mission_quality.to_json(orient="records",force_ascii=False))
    artifact={"surface":"report","manifest":{"version":1,"surface":"report","title":"ESA 子集选择建议","description":"比较扩大 Mission 1 与切换 Mission 2 的数据适用性。","generatedAt":generated_at,"sources":sources,
    "blocks":[
      {"id":"title","type":"markdown","body":"# ESA 子集选择建议","layout":"full"},
      {"id":"summary","type":"markdown","sourceId":"options_source","body":"## Technical Summary\n\n- **建议切换到 Mission 2，而不是只扩大当前 Mission 1。** 推荐通道为 **9、10、13、14、74、86**，时间范围为 **2000年1月1日至2003年6月30日**。\n- **推荐方案有 3,689 个部分缺失窗口、0 个全缺失窗口。** 当前 Mission 1 扩大到14年后仍有约90%的受影响窗口完全缺失。\n- **缺失率在时间划分中更均衡。** 推荐方案训练/验证/测试缺失率约为1.47%/1.47%/2.00%，避免当前验证集缺失率为0。","layout":"full"},
      {"id":"option_text","type":"markdown","sourceId":"options_source","body":"## 扩大 Mission 1 不能修复缺失结构\n\n扩大时间范围会增加样本量，但六通道同步长块仍然存在。核心问题不是数据量不足，而是通道组合和事件结构不适合48分钟窗口。","layout":"full"},
      {"id":"option_chart_block","type":"chart","chartId":"option_chart","layout":"full"},
      {"id":"window_text","type":"markdown","sourceId":"options_source","body":"## Mission 2 推荐组合把全部受影响窗口变成部分缺失\n\n通道组合同时保留了可学习的通道对关系，并避免六通道同时完全缺失。自然缺失率仍约1.55%，训练时可继续在正常位置增加人工掩蔽。","layout":"full"},
      {"id":"window_chart_block","type":"chart","chartId":"window_chart","layout":"full"},
      {"id":"split_text","type":"markdown","sourceId":"splits_source","body":"## 推荐方案的验证集不再是零缺失\n\n按时间顺序划分后，三个区间都有自然缺失样本，更适合早停、模型选择和最终测试。","layout":"full"},
      {"id":"split_chart_block","type":"chart","chartId":"split_chart","layout":"full"},
      {"id":"quality_text","type":"markdown","sourceId":"quality_source","body":"## Mission 3 暂不作为首选\n\nMission 3 的通信中断与无效片段更接近自然缺失，但标签包含156条空通道记录和18条结束时间早于开始时间的记录。未先定义全局无效区间的应用规则并修复时间边界前，不适合作为第一选择。","layout":"full"},
      {"id":"quality_table_block","type":"table","tableId":"quality_table","layout":"full"},
      {"id":"method","type":"markdown","body":"## Scope, Data, and Method\n\n所有方案统一映射到30秒时间网格，并将异常、稀有事件和通信中断标签视为派生缺失。窗口长度为96步（48分钟），步长为48步（24分钟）。部分缺失窗口定义为缺失单元数介于1和575之间；全缺失窗口为576个单元全部缺失。遥测相关性使用10分钟重采样后的 Pearson 系数，仅用于检查通道间是否存在可学习关系。","layout":"full"},
      {"id":"next","type":"markdown","body":"## Recommended Next Steps\n\n1. 新建 Mission 2 预处理配置：通道 9、10、13、14、74、86，完整42个月，30秒网格。\n2. 令 `observed_mask = clean_mask`，保留部分缺失窗口，仅删除观测数为0的窗口。\n3. 训练阶段继续使用 random 或 mix 掩蔽，评价只计算人工遮盖的正常值。\n4. 先做1–2个epoch烟雾测试，再决定是否对53k训练窗口采样或减少epoch。","layout":"full"},
      {"id":"limits","type":"markdown","body":"## Limitations and Further Questions\n\n标签分析不能证明所有通道在物理意义上属于同一子系统；当前证据只确认时间覆盖一致、存在若干强相关通道对、缺失窗口结构更适合插补。正式实验前应结合 channels.csv 或任务文档核对通道语义。完整42个月约为当前子集的4.2倍，训练成本也会相应增加。","layout":"full"}
    ],
    "charts":[
      {"id":"option_chart","title":"候选方案总体派生缺失率","subtitle":"统一6通道、30秒网格","type":"horizontalBar","intent":"comparison","dataset":"options","sourceId":"options_source","encodings":{"x":{"field":"option","type":"nominal","label":"方案"},"y":{"field":"missing_rate","type":"quantitative","format":"percent","label":"缺失率"},"tooltip":[{"field":"partial_windows","type":"quantitative","label":"部分缺失窗口"},{"field":"fully_missing_windows","type":"quantitative","label":"全缺失窗口"}]},"valueFormat":"percent","layout":"full"},
      {"id":"window_chart","title":"候选方案受影响窗口构成","subtitle":"部分缺失与全缺失窗口数量","type":"stackedBar","intent":"composition","dataset":"options","sourceId":"options_source","encodings":{"x":{"field":"option","type":"nominal","label":"方案"},"y":{"fields":["partial_windows","fully_missing_windows"],"type":"quantitative","aggregate":"none","label":"窗口数"}},"valueFormat":"compact","layout":"full"},
      {"id":"split_chart","title":"当前与推荐方案的时间划分缺失率","subtitle":"训练/验证/测试按时间顺序70%/15%/15%","type":"bar","intent":"comparison","dataset":"split_wide","sourceId":"splits_source","encodings":{"x":{"field":"split","type":"nominal","label":"划分"},"y":{"fields":["current_rate","recommended_rate"],"type":"quantitative","aggregate":"none","format":"percent","label":"缺失率"}},"valueFormat":"percent","layout":"full"}
    ],
    "tables":[{"id":"quality_table","title":"三项 Mission 标签质量概况","subtitle":"用于判断是否能直接进入缺失分布建模","dataset":"mission_quality","sourceId":"quality_source","defaultSort":{"field":"mission","direction":"asc"},"density":"spacious","layout":"full","columns":[{"field":"mission","label":"数据集","type":"text"},{"field":"label_rows","label":"标签行","format":"number"},{"field":"events","label":"事件数","format":"number"},{"field":"missing_channel_labels","label":"空通道标签","format":"number"},{"field":"negative_duration_labels","label":"负时长标签","format":"number"},{"field":"assessment","label":"判断","type":"text"}]}]},
    "snapshot":{"version":1,"generatedAt":generated_at,"status":"ready","datasets":{"options":option_rows,"splits":split_rows,"split_wide":make_split_wide(splits),"correlation":corr_rows,"mission_quality":quality_rows},"accessIssues":[]},"sources":sources}
    (OUT/"artifact.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding="utf-8")


def make_split_wide(splits):
    pivot=splits.pivot(index="split",columns="option",values="missing_rate").reset_index()
    pivot=pivot.rename(columns={"M1 当前 10个月":"current_rate","M2 推荐六通道":"recommended_rate"})
    return json.loads(pivot.to_json(orient="records",force_ascii=False))


if __name__ == "__main__":
    main()
