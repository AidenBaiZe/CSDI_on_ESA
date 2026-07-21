import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RATIOS = [0.1, 0.5, 0.9]
FOLDS = [0, 1, 2, 3, 4]


def parse_args():
    parser = argparse.ArgumentParser(description="Run BRITS healthcare grid with early stopping.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hid-size", type=int, default=108)
    parser.add_argument("--impute-weight", type=float, default=0.3)
    parser.add_argument("--label-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--include-reused-fold0", action="store_true")
    return parser.parse_args()


def ratio_text(ratio):
    return str(ratio)


def data_dir_for(ratio, seed, fold):
    return REPO_ROOT / "baseline_results" / "brits" / f"physio_missing{ratio_text(ratio)}_seed{seed}_fold{fold}"


def run_command(command, cwd, stdout_path=None, stderr_path=None):
    print(" ".join(str(x) for x in command), flush=True)
    if stdout_path is None:
        subprocess.run(command, cwd=cwd, check=True)
        return

    with open(stdout_path, "w", encoding="utf-8") as stdout, open(stderr_path, "w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, check=True)


def prepare_data(ratio, seed, fold):
    outdir = data_dir_for(ratio, seed, fold)
    expected = [outdir / "train.jsonl", outdir / "valid.jsonl", outdir / "test.jsonl", outdir / "data_manifest.json"]
    if all(path.is_file() for path in expected):
        print(f"data exists: ratio={ratio} fold={fold}", flush=True)
        return outdir

    command = [
        sys.executable,
        "repro_baselines/brits/prepare_brits_physio.py",
        "--missing-ratio",
        ratio_text(ratio),
        "--seed",
        str(seed),
        "--nfold",
        str(fold),
        "--outdir",
        str(outdir),
    ]
    run_command(command, REPO_ROOT)
    return outdir


def run_job(args, ratio, fold):
    data_dir = prepare_data(ratio, args.seed, fold)
    output_dir = data_dir / "run_patience30"
    metrics_path = output_dir / "metrics.json"

    if metrics_path.is_file() and not args.rerun_existing:
        print(f"skip existing metrics: ratio={ratio} fold={fold}", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "repro_baselines/brits/run_brits_physio.py",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--batch-size",
        str(args.batch_size),
        "--hid-size",
        str(args.hid_size),
        "--impute-weight",
        str(args.impute_weight),
        "--label-weight",
        str(args.label_weight),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    run_command(command, REPO_ROOT, output_dir / "train.log", output_dir / "train.err.log")


def main():
    args = parse_args()
    for ratio in RATIOS:
        for fold in FOLDS:
            if ratio == 0.1 and fold == 0 and not args.include_reused_fold0:
                print("reuse existing result: ratio=0.1 fold=0", flush=True)
                continue
            run_job(args, ratio, fold)

    summary_command = [sys.executable, "repro_baselines/brits/summarize_brits_results.py", "--seed", str(args.seed)]
    run_command(summary_command, REPO_ROOT)


if __name__ == "__main__":
    main()
