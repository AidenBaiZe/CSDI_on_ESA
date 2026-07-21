import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RATIOS = [0.1, 0.5, 0.9]
FOLDS = [0, 1, 2, 3, 4]
PAPER_MAE = {0.1: 0.284, 0.5: 0.368, 0.9: 0.517}
PAPER_SE = {0.1: 0.001, 0.5: 0.002, 0.9: 0.002}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize BRITS healthcare fold metrics.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--out",
        type=str,
        default="baseline_results/brits/summary_brits_healthcare.csv",
    )
    return parser.parse_args()


def ratio_text(ratio):
    return str(ratio)


def metric_candidates(ratio, seed, fold):
    base = REPO_ROOT / "baseline_results" / "brits" / f"physio_missing{ratio_text(ratio)}_seed{seed}_fold{fold}"
    candidates = [base / "run_patience30" / "metrics.json"]
    if ratio == 0.1 and fold == 0:
        candidates.append(base / "run_epoch1000" / "metrics.json")
    return candidates


def read_metric(ratio, seed, fold):
    for path in metric_candidates(ratio, seed, fold):
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            return path, metrics
    return None, None


def main():
    args = parse_args()
    rows = []
    summary_rows = []

    for ratio in RATIOS:
        fold_maes = []
        for fold in FOLDS:
            path, metrics = read_metric(ratio, args.seed, fold)
            if metrics is None:
                rows.append({
                    "type": "fold",
                    "missing_ratio": ratio,
                    "fold": fold,
                    "status": "missing",
                    "metrics_path": "",
                    "test_mae": "",
                    "valid_mae": "",
                    "best_epoch": "",
                    "stopped_epoch": "",
                    "early_stopped": "",
                    "paper_mae": PAPER_MAE[ratio],
                    "paper_se": PAPER_SE[ratio],
                })
                continue

            test_mae = float(metrics["test"]["mae"])
            valid_mae = float(metrics["valid"]["mae"])
            fold_maes.append(test_mae)
            rows.append({
                "type": "fold",
                "missing_ratio": ratio,
                "fold": fold,
                "status": "complete",
                "metrics_path": str(path.relative_to(REPO_ROOT)),
                "test_mae": test_mae,
                "valid_mae": valid_mae,
                "best_epoch": metrics.get("best_epoch", metrics.get("best_epoch_from_log", "")),
                "stopped_epoch": metrics.get("stopped_epoch", metrics.get("stopped_after_epoch", "")),
                "early_stopped": metrics.get("early_stopped", ""),
                "paper_mae": PAPER_MAE[ratio],
                "paper_se": PAPER_SE[ratio],
            })

        if len(fold_maes) == 5:
            maes = np.asarray(fold_maes, dtype=np.float64)
            mean_mae = float(maes.mean())
            se_mae = float(maes.std(ddof=1) / math.sqrt(len(maes)))
            status = "complete"
        else:
            mean_mae = ""
            se_mae = ""
            status = f"incomplete ({len(fold_maes)}/5)"

        summary_rows.append({
            "type": "summary",
            "missing_ratio": ratio,
            "fold": "all",
            "status": status,
            "metrics_path": "",
            "test_mae": mean_mae,
            "valid_mae": "",
            "best_epoch": "",
            "stopped_epoch": "",
            "early_stopped": "",
            "paper_mae": PAPER_MAE[ratio],
            "paper_se": PAPER_SE[ratio],
        })

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "type",
        "missing_ratio",
        "fold",
        "status",
        "metrics_path",
        "test_mae",
        "valid_mae",
        "best_epoch",
        "stopped_epoch",
        "early_stopped",
        "paper_mae",
        "paper_se",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(summary_rows)

    print(out_path)
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
