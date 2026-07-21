import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GPVAE_ROOT = REPO_ROOT / "external" / "GP-VAE"


def parse_args():
    parser = argparse.ArgumentParser(description="Run original GP-VAE PhysioNet baseline.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--model-type", default="gp-vae")
    parser.add_argument("--latent-dim", type=int, default=35)
    parser.add_argument("--encoder-sizes", default="128,128")
    parser.add_argument("--decoder-sizes", default="256,256")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=1.005)
    parser.add_argument("--length-scale", type=float, default=7.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--kernel", default="cauchy")
    parser.add_argument("--banded-covar", action="store_true", default=True)
    parser.add_argument("--no-banded-covar", dest="banded_covar", action="store_false")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--intra-op-threads", type=int, default=4)
    parser.add_argument("--inter-op-threads", type=int, default=2)
    parser.add_argument("--omp-num-threads", type=int, default=4)
    return parser.parse_args()


def run(cmd, cwd, log_path, err_path, env):
    with open(log_path, "w", encoding="utf-8") as stdout, open(err_path, "w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=stdout, stderr=stderr, text=True, env=env)
    return proc.returncode


def read_results_tsv(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[0]


def find_run_dir(models_dir):
    candidates = [p for p in models_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No GP-VAE output directory found under {models_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def compute_metrics(data_file, run_dir):
    data = np.load(data_file)
    samples = np.load(run_dir / "imputed_samples_no_gt.npy")
    eval_bool = data["m_test_artificial"].astype(bool)
    target_points = data["x_test_full"].astype(np.float32)[eval_bool]
    sample_points = np.transpose(samples, (0, 2, 3, 1))[eval_bool]
    median = np.median(sample_points, axis=1)
    errors = np.abs(median - target_points)
    sq_errors = (median - target_points) ** 2
    crps = calc_quantile_crps_points(target_points, sample_points)
    return {
        "mae": float(errors.mean()),
        "mse": float(sq_errors.mean()),
        "crps": float(crps),
        "eval_points": int(eval_bool.sum()),
        "num_samples": int(samples.shape[1]),
    }


def calc_quantile_crps_points(target, forecast):
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = np.sum(np.abs(target))
    if denom == 0:
        raise ValueError("CRPS denominator is zero")

    crps = 0.0
    for q in quantiles:
        q_pred = np.quantile(forecast, q, axis=1)
        q_loss = 2.0 * np.sum(
            np.abs((q_pred - target) * ((target <= q_pred).astype(np.float32) - q))
        )
        crps += q_loss / denom
    return crps / len(quantiles)


def calc_quantile_crps(target, forecast, eval_points):
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = np.sum(np.abs(target * eval_points))
    if denom == 0:
        raise ValueError("CRPS denominator is zero")

    crps = 0.0
    for q in quantiles:
        q_pred = np.quantile(forecast, q, axis=1)
        q_loss = 2.0 * np.sum(
            np.abs((q_pred - target) * eval_points * ((target <= q_pred).astype(np.float32) - q))
        )
        crps += q_loss / denom
    return crps / len(quantiles)


def git_commit(path):
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(path), text=True)
        return out.strip()
    except Exception:
        return None


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    data_file = Path(args.data_file).resolve()
    exp_name = output_dir.name
    cmd = [
        args.python,
        "-u",
        "train.py",
        "--model_type",
        args.model_type,
        "--data_type",
        "physionet",
        "--exp_name",
        exp_name,
        "--basedir",
        str(models_dir),
        "--data_dir",
        str(data_file),
        "--seed",
        str(args.seed),
        "--num_epochs",
        str(args.num_epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_samples",
        str(args.num_samples),
        "--latent_dim",
        str(args.latent_dim),
        "--encoder_sizes",
        args.encoder_sizes,
        "--decoder_sizes",
        args.decoder_sizes,
        "--window_size",
        str(args.window_size),
        "--sigma",
        str(args.sigma),
        "--length_scale",
        str(args.length_scale),
        "--beta",
        str(args.beta),
        "--kernel",
        args.kernel,
    ]
    if args.banded_covar:
        cmd.append("--banded_covar")

    env = os.environ.copy()
    env["TF_NUM_INTRAOP_THREADS"] = str(args.intra_op_threads)
    env["TF_NUM_INTEROP_THREADS"] = str(args.inter_op_threads)
    env["OMP_NUM_THREADS"] = str(args.omp_num_threads)
    env["KMP_AFFINITY"] = "disabled"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if platform.system() == "Windows":
        conda_env = Path(args.python).resolve().parent
        conda_paths = [
            conda_env,
            conda_env / "Library" / "mingw-w64" / "bin",
            conda_env / "Library" / "usr" / "bin",
            conda_env / "Library" / "bin",
            conda_env / "Scripts",
        ]
        env["PATH"] = os.pathsep.join(str(p) for p in conda_paths) + os.pathsep + env.get("PATH", "")
        env.setdefault("CONDA_PREFIX", str(conda_env))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": cmd,
        "cwd": str(GPVAE_ROOT),
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "args": vars(args),
        "gpvae_commit": git_commit(GPVAE_ROOT),
        "csdi_commit": git_commit(REPO_ROOT),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "tf_num_intraop_threads": env["TF_NUM_INTRAOP_THREADS"],
            "tf_num_interop_threads": env["TF_NUM_INTEROP_THREADS"],
            "omp_num_threads": env["OMP_NUM_THREADS"],
        },
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    returncode = run(cmd, GPVAE_ROOT, output_dir / "train.log", output_dir / "train.err.log", env)
    if returncode != 0:
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"status": "failed", "returncode": returncode}, f, indent=2)
        raise SystemExit(returncode)

    run_dir = find_run_dir(models_dir)
    results = read_results_tsv(run_dir / "results.tsv")
    test = compute_metrics(data_file, run_dir)
    metrics = {
        "status": "completed",
        "run_dir": str(run_dir),
        "test": test,
        "table2_metric": {
            "name": "CRPS",
            "value": test["crps"],
            "paper_reference": {
                "table": "CSDI Table 2",
                "baseline": "GP-VAE",
                "healthcare_10_percent_missing": "0.574(0.003)",
                "healthcare_50_percent_missing": "0.774(0.004)",
                "healthcare_90_percent_missing": "0.998(0.001)"
            }
        },
        "gpvae_results_tsv": {
            "nll": float(results["NLL"]),
            "mse": float(results["MSE"]),
            "auprc": float(results["AUPRC"]),
            "auroc": float(results["AUROC"]),
        },
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
