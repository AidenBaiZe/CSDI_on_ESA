import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER = {
    0.1: "0.700(0.002)",
    0.5: "0.676(0.003)",
    0.9: "0.761(0.010)",
}


def ratio_text(ratio):
    return f"{ratio:g}"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Latent ODE CSDI Table 4 results.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "baseline_results" / "latent_ode" / "summary_latent_ode_table4.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    summary_rows = []
    for ratio in args.ratios:
        crps_values = []
        for fold in args.folds:
            base = (
                REPO_ROOT
                / "baseline_results"
                / "latent_ode"
                / f"physio_interp_missing{ratio_text(ratio)}_seed{args.seed}_fold{fold}"
            )
            metrics_path = base / "metrics.json"
            row = {
                "kind": "fold",
                "missing_ratio": ratio,
                "fold": fold,
                "path": str(metrics_path),
                "paper_crps": PAPER.get(ratio),
            }
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                test = metrics.get("test", {})
                row.update(
                    {
                        "crps": test.get("crps"),
                        "mae": test.get("mae"),
                        "rmse": test.get("rmse"),
                        "eval_points": test.get("eval_points"),
                        "best_epoch": metrics.get("best_epoch"),
                        "best_valid_crps": metrics.get("best_valid_crps"),
                        "status": metrics.get("status"),
                    }
                )
                if test.get("crps") is not None:
                    crps_values.append(float(test["crps"]))
            else:
                row["status"] = "missing"
            rows.append(row)

        if crps_values:
            arr = np.asarray(crps_values, dtype=float)
            se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan")
            summary_rows.append(
                {
                    "kind": "summary",
                    "missing_ratio": ratio,
                    "fold": "all_available",
                    "crps": float(arr.mean()),
                    "se_crps": se,
                    "num_folds": len(arr),
                    "paper_crps": PAPER.get(ratio),
                    "status": "partial" if len(arr) < len(args.folds) else "completed",
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows + summary_rows for key in row.keys()})
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows + summary_rows:
            writer.writerow(row)
    print(output)


if __name__ == "__main__":
    main()
