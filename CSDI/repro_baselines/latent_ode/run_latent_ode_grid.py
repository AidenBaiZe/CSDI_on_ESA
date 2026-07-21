import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


def ratio_text(ratio):
    return f"{ratio:g}"


def parse_args():
    parser = argparse.ArgumentParser(description="Queue Latent ODE CSDI Table 4 runs.")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.9])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--valid-num-samples", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-ratio", type=float, default=None)
    parser.add_argument(
        "--log",
        default=str(REPO_ROOT / "baseline_results" / "latent_ode" / "grid_strict.log"),
    )
    return parser.parse_args()


def output_dir(ratio, seed, fold):
    return (
        REPO_ROOT
        / "baseline_results"
        / "latent_ode"
        / f"physio_interp_missing{ratio_text(ratio)}_seed{seed}_fold{fold}"
    )


def checkpoint_for(ratio, seed, fold):
    return output_dir(ratio, seed, fold) / "best_model.pt"


def log_line(log_path, payload):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()


def launch_run(args, ratio, fold, log_path):
    outdir = output_dir(ratio, args.seed, fold)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_for(args.checkpoint_ratio, args.seed, fold) if args.checkpoint_ratio is not None else None
    if args.eval_only and checkpoint is None:
        raise ValueError("--eval-only requires --checkpoint-ratio")
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    cmd = [
        str(PYTHON),
        "repro_baselines/latent_ode/run_latent_ode_physio.py",
        "--missing-ratio",
        str(ratio),
        "--seed",
        str(args.seed),
        "--nfold",
        str(fold),
        "--output-dir",
        str(outdir),
        "--batch-size",
        str(args.batch_size),
        "--valid-num-samples",
        str(args.valid_num_samples),
        "--num-samples",
        str(args.num_samples),
        "--device",
        args.device,
        "--save-predictions",
    ]
    if args.eval_batch_size is not None:
        cmd.extend(["--eval-batch-size", str(args.eval_batch_size)])
    if args.eval_only:
        cmd.extend(["--eval-only", "--checkpoint", str(checkpoint)])
        stdout_name = "eval.log"
        stderr_name = "eval.err.log"
    else:
        cmd.extend(["--epochs", str(args.epochs)])
        if checkpoint is not None:
            cmd.extend(["--init-checkpoint", str(checkpoint)])
        stdout_name = "train.log"
        stderr_name = "train.err.log"
    stdout = open(outdir / stdout_name, "w", encoding="utf-8")
    stderr = open(outdir / stderr_name, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=stdout, stderr=stderr)
    payload = {
        "event": "started",
        "time": datetime.now().isoformat(timespec="seconds"),
        "pid": proc.pid,
        "ratio": ratio,
        "fold": fold,
        "outdir": str(outdir),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "eval_only": args.eval_only,
        "cmd": cmd,
    }
    log_line(log_path, payload)
    return {"proc": proc, "stdout": stdout, "stderr": stderr, "ratio": ratio, "fold": fold, "outdir": outdir}


def close_run(run):
    run["stdout"].close()
    run["stderr"].close()


def main():
    args = parse_args()
    log_path = Path(args.log)
    log_line(
        log_path,
        {
            "event": "grid_started",
            "time": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "python": str(PYTHON),
        },
    )
    tasks = []
    for ratio in args.ratios:
        for fold in args.folds:
            outdir = output_dir(ratio, args.seed, fold)
            if args.skip_existing and (outdir / "metrics.json").exists():
                log_line(
                    log_path,
                    {
                        "event": "skipped_existing",
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "ratio": ratio,
                        "fold": fold,
                        "outdir": str(outdir),
                    },
                )
                continue
            tasks.append((ratio, fold))

    active = []
    while tasks or active:
        while tasks and len(active) < args.max_parallel:
            ratio, fold = tasks.pop(0)
            active.append(launch_run(args, ratio, fold, log_path))
            time.sleep(2)

        still_active = []
        for run in active:
            ret = run["proc"].poll()
            if ret is None:
                still_active.append(run)
                continue
            close_run(run)
            log_line(
                log_path,
                {
                    "event": "finished",
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "returncode": ret,
                    "ratio": run["ratio"],
                    "fold": run["fold"],
                    "outdir": str(run["outdir"]),
                    "metrics_exists": (run["outdir"] / "metrics.json").exists(),
                },
            )
            if ret != 0:
                log_line(
                    log_path,
                    {
                        "event": "aborting_after_failure",
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "ratio": run["ratio"],
                        "fold": run["fold"],
                    },
                )
                for other in still_active:
                    other["proc"].terminate()
                    close_run(other)
                return ret
        active = still_active
        time.sleep(30)

    summarize = subprocess.run(
        [str(PYTHON), "repro_baselines/latent_ode/summarize_latent_ode_results.py", "--seed", str(args.seed)],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    log_line(
        log_path,
        {
            "event": "summary",
            "time": datetime.now().isoformat(timespec="seconds"),
            "returncode": summarize.returncode,
            "stdout": summarize.stdout.strip(),
            "stderr": summarize.stderr.strip(),
        },
    )
    return summarize.returncode


if __name__ == "__main__":
    raise SystemExit(main())
