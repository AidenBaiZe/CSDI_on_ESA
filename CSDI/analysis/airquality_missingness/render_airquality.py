from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/pm25/Code/STMVL/SampleData"
GROUND = DATA / "pm25_ground.txt"
MASKED = DATA / "pm25_missing.txt"
OUTPUT = OUT / "airquality_missingness_overview.png"

BLUE = "#2878B5"
BLUE_LIGHT = "#9BC4E2"
ORANGE = "#F28E2B"
INK = "#24313D"
GREY = "#A7B0B7"
GRID = "#DDE3E8"


def window_missing_rates(frame: pd.DataFrame, months: list[int], test_mode: bool) -> np.ndarray:
    rates: list[float] = []
    for month in months:
        missing = frame[frame.index.month == month].isna().to_numpy()
        starts = range(0, len(missing), 36) if test_mode else range(0, len(missing) - 36 + 1)
        for start in starts:
            window = missing[start : start + 36]
            if len(window) < 36:
                window = np.vstack([window, np.zeros((36 - len(window), missing.shape[1]), dtype=bool)])
            rates.append(float(window.mean()))
    return np.asarray(rates)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ground = pd.read_csv(GROUND, index_col="datetime", parse_dates=True)
    masked = pd.read_csv(MASKED, index_col="datetime", parse_dates=True)
    extra_mask = ground.notna() & masked.isna()

    monthly_rows = []
    for month in range(1, 13):
        selected = ground.index.month == month
        natural_rate = float(ground.loc[selected].isna().to_numpy().mean())
        observed_count = int(ground.loc[selected].notna().to_numpy().sum())
        additional_count = int(extra_mask.loc[selected].to_numpy().sum())
        monthly_rows.append({
            "month": month,
            "timepoints": int(selected.sum()),
            "natural_missing_rate": natural_rate,
            "additional_mask_rate_of_observed": additional_count / observed_count,
            "natural_missing_count": int(ground.loc[selected].isna().to_numpy().sum()),
            "additional_mask_count": additional_count,
        })
    monthly = pd.DataFrame(monthly_rows)

    station = pd.DataFrame({
        "station": ground.columns,
        "natural_missing_rate": ground.isna().mean().to_numpy(),
        "natural_missing_count": ground.isna().sum().to_numpy(),
    }).sort_values("natural_missing_rate")

    split_months = {
        "训练集": [2, 4, 5, 7, 8, 10, 11],
        "验证集": [1],
        "测试集": [3, 6, 9, 12],
    }
    split_rows = []
    window_rates: dict[str, np.ndarray] = {}
    for split, months in split_months.items():
        selected = ground.index.month.isin(months)
        natural_rate = float(ground.loc[selected].isna().to_numpy().mean())
        observed_count = int(ground.loc[selected].notna().to_numpy().sum())
        additional_count = int(extra_mask.loc[selected].to_numpy().sum())
        rates = window_missing_rates(ground, months, test_mode=(split == "测试集"))
        window_rates[split] = rates
        split_rows.append({
            "split": split,
            "months": ",".join(map(str, months)),
            "timepoints": int(selected.sum()),
            "natural_missing_rate": natural_rate,
            "evaluation_mask_rate_of_observed": additional_count / observed_count if split != "训练集" else None,
            "window_count": int(rates.size),
            "windows_with_missing": int((rates > 0).sum()),
            "fully_missing_windows": int((rates == 1).sum()),
            "median_window_missing_rate": float(np.median(rates)),
            "p90_window_missing_rate": float(np.quantile(rates, 0.9)),
            "max_window_missing_rate": float(rates.max()),
        })
    splits = pd.DataFrame(split_rows)

    summary = {
        "shape": list(ground.shape),
        "start": ground.index.min().isoformat(),
        "end": ground.index.max().isoformat(),
        "natural_missing_count": int(ground.isna().to_numpy().sum()),
        "natural_missing_rate": float(ground.isna().to_numpy().mean()),
        "additional_mask_count": int(extra_mask.to_numpy().sum()),
        "additional_mask_rate_of_observed": float(extra_mask.to_numpy().sum() / ground.notna().to_numpy().sum()),
        "station_rate_min": float(station["natural_missing_rate"].min()),
        "station_rate_median": float(station["natural_missing_rate"].median()),
        "station_rate_max": float(station["natural_missing_rate"].max()),
        "splits": split_rows,
    }
    monthly.to_csv(OUT / "monthly.csv", index=False, encoding="utf-8-sig")
    station.to_csv(OUT / "stations.csv", index=False, encoding="utf-8-sig")
    splits.to_csv(OUT / "splits.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": "#B9C2C9",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.11, top=0.86, hspace=0.42, wspace=0.24)
    fig.suptitle("Air Quality：PM2.5 自然缺失与评价遮盖分布", fontsize=20, fontweight="bold", x=0.04, ha="left")
    fig.text(0.04, 0.945, "北京 36 个监测站｜2014-05-01 至 2015-04-30｜每小时采样", fontsize=11, color="#5E6B75")

    ax = axes[0, 0]
    monthly_rate = monthly["natural_missing_rate"] * 100
    bars = ax.bar([f"{month:02d}" for month in monthly["month"]], monthly_rate, color=[ORANGE if value >= 20 else BLUE for value in monthly_rate], edgecolor="white")
    ax.set_title("A. 月度自然缺失率", loc="left", fontsize=14, fontweight="bold", pad=29)
    ax.text(0, 1.015, "pm25_ground.txt 中的空值 ÷ 当月全部站点-小时单元", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.set_xlabel("月份")
    ax.set_ylabel("缺失率")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, monthly_rate.max() * 1.18)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, monthly_rate):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax = axes[0, 1]
    station_rate = station["natural_missing_rate"] * 100
    bars = ax.bar(np.arange(1, len(station) + 1), station_rate, color=BLUE, width=0.8)
    ax.axhline(station_rate.median(), color=ORANGE, linestyle="--", linewidth=1.8, label=f"中位数 {station_rate.median():.1f}%")
    ax.set_title("B. 监测站自然缺失率分布", loc="left", fontsize=14, fontweight="bold", pad=29)
    ax.text(0, 1.015, "36 个站点按缺失率从低到高排列", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.set_xlabel("站点排序")
    ax.set_ylabel("缺失率")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, station_rate.max() * 1.18)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.text(1, station_rate.iloc[0] + 0.8, f"最低 {station_rate.iloc[0]:.1f}%", ha="left", fontsize=9)
    ax.text(36, station_rate.iloc[-1] + 0.8, f"最高 {station_rate.iloc[-1]:.1f}%", ha="right", fontsize=9)

    ax = axes[1, 0]
    positions = np.arange(len(splits))
    width = 0.34
    natural_rates = splits["natural_missing_rate"].to_numpy() * 100
    eval_rates = splits["evaluation_mask_rate_of_observed"].to_numpy(dtype=float) * 100
    bars_natural = ax.bar(positions - width / 2, natural_rates, width, color=BLUE, label="自然缺失")
    bars_eval = ax.bar(positions + width / 2, eval_rates, width, color=ORANGE, label="评价额外遮盖")
    ax.set_title("C. 数据划分中的缺失率", loc="left", fontsize=14, fontweight="bold", pad=29)
    ax.text(0, 1.015, "默认 validationindex=0；训练掩码由 mix/historical 策略动态生成", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.set_xticks(positions, splits["split"])
    ax.set_ylabel("缺失率")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, np.nanmax(np.r_[natural_rates, eval_rates]) * 1.25)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for bars_group in (bars_natural, bars_eval):
        for bar in bars_group:
            value = bar.get_height()
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    labels = list(window_rates)
    values = [window_rates[label] * 100 for label in labels]
    box = ax.boxplot(values, labels=labels, patch_artist=True, showfliers=False, widths=0.58,
                     medianprops={"color": INK, "linewidth": 1.8},
                     whiskerprops={"color": "#6E7C87"}, capprops={"color": "#6E7C87"})
    for patch, color in zip(box["boxes"], [BLUE_LIGHT, GREY, ORANGE]):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
    ax.set_title("D. 36 小时窗口自然缺失率分布", loc="left", fontsize=14, fontweight="bold", pad=29)
    ax.text(0, 1.015, "训练/验证为滑动窗口；测试为不重叠窗口", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.set_ylabel("窗口缺失率")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 55)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for index, rates in enumerate(values, start=1):
        ax.text(index, min(np.quantile(rates, 0.9) + 2.5, 52), f"中位数 {np.median(rates):.1f}%", ha="center", fontsize=9)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        0.04, 0.02,
        f"结论：Air Quality 自然缺失率为 {summary['natural_missing_rate']:.2%}，明显高于 ESA 的 1.815%；缺失分散在站点和窗口中，没有全缺失的 36 小时窗口。",
        fontsize=10.5,
    )
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
