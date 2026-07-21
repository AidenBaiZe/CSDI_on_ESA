from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = OUT / "interim/ESA-Mission2/channels"
CHANNELS = [9, 10, 13, 14, 74, 86]
START = pd.Timestamp("2000-01-01 00:00:30")
END = pd.Timestamp("2003-06-30 23:59:30")


def load_labels() -> pd.DataFrame:
    with ZipFile(ROOT / "datasets/ESA Anomaly Dataset/ESA-Mission2.zip") as loaded:
        labels = pd.read_csv(io.BytesIO(loaded.read("ESA-Mission2/labels.csv")))
        types = pd.read_csv(io.BytesIO(loaded.read("ESA-Mission2/anomaly_types.csv")))
    labels = labels.merge(types[["ID", "Category"]], on="ID", how="left")
    labels["StartTime"] = pd.to_datetime(labels["StartTime"], utc=True).dt.tz_localize(None)
    labels["EndTime"] = pd.to_datetime(labels["EndTime"], utc=True).dt.tz_localize(None)
    return labels[labels["Channel"].isin([f"channel_{item}" for item in CHANNELS])].copy()


def load_series(channel: int) -> pd.Series:
    loaded = pd.read_pickle(RAW / f"channel_{channel}.zip")
    series = loaded.iloc[:, 0] if isinstance(loaded, pd.DataFrame) else loaded
    return series[~series.index.duplicated(keep="last")].sort_index()


def main() -> None:
    labels = load_labels()
    grid = pd.date_range(START, END, freq="30s")
    mask = np.zeros((len(grid), len(CHANNELS)), dtype=bool)
    series_10m = {}
    profiles = []

    for column, channel in enumerate(CHANNELS):
        selected = labels[labels["Channel"].eq(f"channel_{channel}")]
        for row in selected.itertuples():
            left = int(grid.searchsorted(row.StartTime.ceil("30s")))
            right = int(grid.searchsorted(row.EndTime.floor("30s"), side="right"))
            mask[left:right, column] = True

        raw = load_series(channel)
        sampled = raw.resample("10min").last().ffill()
        series_10m[channel] = sampled
        duration_hours = (selected["EndTime"] - selected["StartTime"]).dt.total_seconds() / 3600
        change_rate = float((sampled.diff().abs() > 0).mean())
        profiles.append({
            "channel": f"通道 {channel}",
            "raw_points": int(len(raw)),
            "unique_values": int(raw.nunique()),
            "missing_rate": float(mask[:, column].mean()),
            "events": int(selected["ID"].nunique()),
            "anomaly_events": int(selected.loc[selected["Category"].eq("Anomaly"), "ID"].nunique()),
            "rare_events": int(selected.loc[selected["Category"].eq("Rare Event"), "ID"].nunique()),
            "median_event_minutes": float(duration_hours.median() * 60),
            "p90_event_hours": float(duration_hours.quantile(0.9)),
            "max_event_hours": float(duration_hours.max()),
            "ten_min_change_rate": change_rate,
            "ten_min_lag1_autocorrelation": float(sampled.autocorr(lag=1)),
            "ten_min_daily_autocorrelation": float(sampled.autocorr(lag=144)),
        })

    data_10m = pd.DataFrame(series_10m).dropna()
    correlation = data_10m.corr()
    profile = pd.DataFrame(profiles)
    overlap = pd.DataFrame({
        "missing_channels": np.arange(7),
        "timepoints": [(mask.sum(axis=1) == number).sum() for number in range(7)],
    })
    overlap["share_all_timepoints"] = overlap["timepoints"] / len(mask)
    affected_total = int((mask.sum(axis=1) > 0).sum())
    overlap["share_affected_timepoints"] = np.where(
        overlap["missing_channels"] > 0, overlap["timepoints"] / affected_total, 0.0
    )

    profile.to_csv(OUT / "channel_patterns.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(OUT / "missing_overlap.csv", index=False, encoding="utf-8-sig")
    correlation.to_csv(OUT / "recommended_correlation_matrix.csv", encoding="utf-8-sig")

    summary = {
        "channels": CHANNELS,
        "profile": json.loads(profile.to_json(orient="records", force_ascii=False)),
        "correlation": correlation.to_dict(),
        "affected_timepoints": affected_total,
        "six_channel_sync_share": float(overlap.loc[overlap["missing_channels"].eq(6), "share_affected_timepoints"].iloc[0]),
        "four_or_fewer_missing_share": float(overlap.loc[overlap["missing_channels"].between(1, 4), "share_affected_timepoints"].sum()),
        "example_range": ["2000-04-21 12:00", "2000-04-25 00:00"],
    }
    (OUT / "channel_patterns_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render(profile, overlap, correlation, series_10m, labels)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def render(profile, overlap, correlation, series_10m, labels):
    blue, orange, ink, grid_color = "#2878B5", "#F28E2B", "#24313D", "#DDE3E8"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "text.color": ink,
        "axes.labelcolor": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "axes.edgecolor": "#B9C2C9",
    })
    fig = plt.figure(figsize=(15, 12))
    layout = fig.add_gridspec(4, 2, left=0.07, right=0.985, bottom=0.08, top=0.88, hspace=0.7, wspace=0.23)
    fig.suptitle("Mission 2 推荐六通道：遥测值与标签缺失模式", fontsize=20, fontweight="bold", x=0.04, ha="left")
    fig.text(0.04, 0.94, "通道 9、10、13、14、74、86｜橙色阴影表示 ESA 标签异常或稀有事件", fontsize=11, color="#5E6B75")

    zoom_start = pd.Timestamp("2000-04-21 12:00")
    zoom_end = pd.Timestamp("2000-04-25 00:00")
    for index, channel in enumerate(CHANNELS):
        ax = fig.add_subplot(layout[index // 2, index % 2])
        series = series_10m[channel].loc[zoom_start:zoom_end]
        robust_scale = series.quantile(0.75) - series.quantile(0.25)
        normalized = (series - series.median()) / (robust_scale if robust_scale > 0 else series.std())
        ax.plot(normalized.index, normalized, color=blue, linewidth=1.0)
        selected = labels[
            labels["Channel"].eq(f"channel_{channel}")
            & labels["StartTime"].lt(zoom_end)
            & labels["EndTime"].gt(zoom_start)
        ]
        for row in selected.itertuples():
            ax.axvspan(max(row.StartTime, zoom_start), min(row.EndTime, zoom_end), color=orange, alpha=0.24, linewidth=0)
        row = profile[profile["channel"].eq(f"通道 {channel}")].iloc[0]
        ax.set_title(f"通道 {channel}｜缺失率 {row.missing_rate:.2%}｜事件 {int(row.events)} 个", loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("稳健标准化值")
        ax.grid(axis="y", color=grid_color, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(layout[3, 0])
    rates = profile["missing_rate"] * 100
    bars = ax.bar(profile["channel"], rates, color=[blue, blue, blue, blue, orange, orange], edgecolor="white")
    ax.set_title("完整42个月的通道级派生缺失率", loc="left", fontsize=13, fontweight="bold", pad=20)
    ax.text(0, 1.015, "前四个通道标签密集，74/86更多充当条件上下文", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.set_ylabel("缺失率")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, rates.max() * 1.25)
    ax.grid(axis="y", color=grid_color); ax.set_axisbelow(True)
    for bar, value in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.06, f"{value:.2f}%", ha="center", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(layout[3, 1])
    affected = overlap[overlap["missing_channels"] > 0]
    bars = ax.bar(affected["missing_channels"].astype(str), affected["share_affected_timepoints"] * 100,
                  color=[blue, blue, blue, blue, orange, orange], edgecolor="white")
    ax.set_title("受影响时间点的同时缺失通道数", loc="left", fontsize=13, fontweight="bold", pad=20)
    ax.text(0, 1.015, "只有0.55%的受影响时间点六通道同时缺失", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.set_xlabel("同时缺失通道数"); ax.set_ylabel("占受影响时间点比例")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", color=grid_color); ax.set_axisbelow(True)
    for bar, value in zip(bars, affected["share_affected_timepoints"] * 100):
        if value > 1: ax.text(bar.get_x()+bar.get_width()/2, value+0.5, f"{value:.1f}%", ha="center", fontsize=8.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.text(0.04, 0.025, "模式总结：9/10是连续高分辨率信号，13/14是另一组相关遥测，74/86呈强逆向阶梯状态；标签主要落在前四个通道，因此多数窗口保留可观测上下文。", fontsize=10.5)
    fig.savefig(OUT / "mission2_channel_patterns.png", dpi=180, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
