from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "esa_missingness_overview.png"

BLUE = "#2878B5"
BLUE_LIGHT = "#9BC4E2"
ORANGE = "#F28E2B"
INK = "#24313D"
GRID = "#DDE3E8"
GREY = "#A7B0B7"


def add_value_labels(axis, bars, formatter, pad=3):
    for bar in bars:
        value = bar.get_height() if bar.get_height() else bar.get_width()
        if bar.get_width() > bar.get_height() and bar.get_width() > 0.2:
            axis.annotate(
                formatter(value),
                (bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(pad, 0), textcoords="offset points", va="center", ha="left",
                fontsize=9, color=INK,
            )
        else:
            axis.annotate(
                formatter(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height()),
                xytext=(0, pad), textcoords="offset points", va="bottom", ha="center",
                fontsize=9, color=INK,
            )


def main():
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

    monthly = pd.read_csv(HERE / "monthly.csv")
    channels = pd.read_csv(HERE / "channels.csv")
    splits = pd.read_csv(HERE / "splits.csv")
    natural = pd.read_csv(HERE / "natural_windows.csv")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.11, top=0.86, hspace=0.42, wspace=0.24)
    fig.suptitle("ESA 子集：将标注异常视为自然缺失后的分布", fontsize=20, fontweight="bold", x=0.04, ha="left")
    fig.text(0.04, 0.945, "Mission 1 通道 41–46｜2000-01-01 至 2000-11-01｜30 秒时间网格", fontsize=11, color="#5E6B75")

    # A. Monthly distribution: discrete bars reveal zero months and concentration.
    ax = axes[0, 0]
    month_labels = [item[5:] for item in monthly["month"]]
    rates = monthly["abnormal_rate"] * 100
    colors = [ORANGE if value >= 3 else BLUE for value in rates]
    bars = ax.bar(month_labels, rates, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("A. 月度自然缺失率", loc="left", fontsize=14, fontweight="bold", pad=29)
    ax.text(0, 1.015, "异常与稀有事件单元 ÷ 当月全部通道-时间单元", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.set_xlabel("月份（2000 年）")
    ax.set_ylabel("缺失率")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, max(rates) * 1.22)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, rates):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.2f}%", ha="center", va="bottom", fontsize=8.5)

    # B. Per-channel rates: horizontal bars support close comparisons.
    ax = axes[0, 1]
    channel_rates = channels["abnormal_rate"] * 100
    bars = ax.barh(channels["channel"], channel_rates, color=BLUE, edgecolor="white")
    ax.set_title("B. 各通道自然缺失率", loc="left", fontsize=14, fontweight="bold", pad=29)
    ax.text(0, 1.015, "各通道分母均为 878,400 个时间点", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.set_xlabel("缺失率")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlim(0, 2.15)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, channel_rates):
        ax.text(value + 0.025, bar.get_y() + bar.get_height() / 2, f"{value:.3f}%", va="center", fontsize=9)
    ax.invert_yaxis()

    # C. Split comparison: zero validation rate is the important contrast.
    ax = axes[1, 0]
    split_rates = splits["abnormal_rate"] * 100
    bars = ax.bar(splits["split"], split_rates, color=[BLUE, GREY, ORANGE], edgecolor="white", width=0.62)
    ax.set_title("C. 数据划分中的自然缺失率", loc="left", fontsize=14, fontweight="bold", pad=29)
    ax.text(0, 1.015, "时间顺序划分 70% / 15% / 15%", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.set_ylabel("缺失率")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, max(split_rates) * 1.28)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, split_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.3f}%", ha="center", va="bottom", fontsize=10)

    # D. Window composition: affected windows are split into partial vs fully missing.
    ax = axes[1, 1]
    partial = natural["partial_missing_windows"]
    full = natural["fully_missing_windows"]
    bars_partial = ax.bar(natural["split"], partial, color=BLUE_LIGHT, edgecolor="white", label="部分缺失窗口")
    bars_full = ax.bar(natural["split"], full, bottom=partial, color=ORANGE, edgecolor="white", label="全缺失窗口")
    ax.set_title("D. 含自然缺失窗口的构成", loc="left", fontsize=14, fontweight="bold", pad=29)
    ax.text(0, 1.015, "96 步窗口、48 步步长；仅统计含 ESA 标签异常的窗口", transform=ax.transAxes, fontsize=9.5, color="#5E6B75")
    ax.set_ylabel("窗口数")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    for index, (p, f) in enumerate(zip(partial, full)):
        if p:
            ax.text(index, p / 2, str(int(p)), ha="center", va="center", fontsize=9, color=INK)
        if f:
            ax.text(index, p + f / 2, str(int(f)), ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(index, p + f + 6, f"合计 {int(p + f)}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max((partial + full).max() * 1.2, 10))

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.text(
        0.04, 0.015,
        "结论：总体缺失率仅 1.815%，但缺失集中在少数月份，且 94.91% 的受影响时间点为六通道同步缺失；多数受影响窗口完全缺失。",
        fontsize=10.5, color=INK,
    )
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
