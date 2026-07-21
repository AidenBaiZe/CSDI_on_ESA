from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CHANNELS = ["Ch 9", "Ch 10", "Ch 13", "Ch 14", "Ch 74", "Ch 86"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot an ESA CSDI experiment summary")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output = (args.output or result_dir / "mix_performance.png").resolve()

    with (result_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    history = []
    with (result_dir / "train_log.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            history.append(json.loads(line))

    metric_rows = [summary["metrics"][f"missing_{ratio}"] for ratio in (10, 50, 90)]
    ratios = np.array([10, 50, 90])
    train_loss = np.array([row["train_loss"] for row in history])
    epochs = np.array([row["epoch"] for row in history])

    manifest_path = (
        result_dir.parents[2]
        / "esa_csdi_preprocessing"
        / "data"
        / "processed"
        / "mission2_ch9_10_13_14_74_86_42m_30s"
        / "manifest.json"
    )
    with manifest_path.open("r", encoding="utf-8") as handle:
        data_manifest = json.load(handle)
    stds = np.array(list(data_manifest["normalization"]["std"].values()))
    channel_rmse = np.array(
        [
            [channel["rmse_original"] for channel in row["per_channel"]]
            for row in metric_rows
        ]
    ) / stds[None, :]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#172033",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.84, bottom=0.20, wspace=0.16)
    fig.patch.set_facecolor("white")
    fig.suptitle("ESA Mission 2 — CSDI Mix pilot performance", fontsize=16, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, train_loss, color="#2563eb", linewidth=2.2, marker="o", markersize=4)
    valid = [(row["epoch"], row["valid_loss"]) for row in history if row["valid_loss"] is not None]
    if valid:
        ax.scatter(*zip(*valid), color="#f59e0b", marker="D", s=55, zorder=3, label="Validation")
        ax.legend(frameon=False, loc="upper right")
    ax.set_title("Training convergence", loc="left", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Diffusion loss")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)

    ax = axes[1]
    series = [
        ("RMSE", [row["rmse_normalized"] for row in metric_rows], "#2563eb", "o", "-"),
        ("MAE", [row["mae_normalized"] for row in metric_rows], "#f59e0b", "s", "--"),
        ("CRPS", [row["crps_normalized"] for row in metric_rows], "#64748b", "^", ":"),
    ]
    for label, values, color, marker, linestyle in series:
        ax.plot(ratios, values, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=2.2, markersize=6)
        for x, y in zip(ratios, values):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set_title("Robustness to artificial missingness", loc="left", fontweight="bold")
    ax.set_xlabel("Artificial missing ratio")
    ax.set_ylabel("Normalized error (lower is better)")
    ax.set_xticks(ratios, ["10%", "50%", "90%"])
    ax.set_ylim(0, 0.35)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[2]
    x = np.arange(len(CHANNELS))
    width = 0.24
    colors = ["#93c5fd", "#2563eb", "#f59e0b"]
    hatches = ["", "//", "xx"]
    for index, ratio in enumerate(ratios):
        ax.bar(
            x + (index - 1) * width,
            channel_rmse[index],
            width,
            label=f"{ratio}%",
            color=colors[index],
            edgecolor="#334155",
            linewidth=0.6,
            hatch=hatches[index],
        )
    ax.set_title("Error by channel", loc="left", fontweight="bold")
    ax.set_xlabel("Mission 2 channel")
    ax.set_ylabel("Normalized RMSE")
    ax.set_xticks(x, CHANNELS, rotation=0)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.legend(title="Missing", frameon=False, ncol=3, loc="upper left")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        0.035,
        "Pilot scope: 20 epochs × 500 train batches; 10 sampled imputations; 10 test batches (~640 windows) per ratio. Seed 1.",
        fontsize=8.5,
        color="#64748b",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
