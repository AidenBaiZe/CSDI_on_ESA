import argparse
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_physio import Physio_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run CSDI Table 2 Multitask GP baseline on PhysioNet.")
    parser.add_argument("--missing-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nfold", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--training-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def split_indices(num_records, seed, nfold):
    indlist = np.arange(num_records)
    np.random.seed(seed)
    np.random.shuffle(indlist)

    start = int(nfold * 0.2 * num_records)
    end = int((nfold + 1) * 0.2 * num_records)
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = int(num_records * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]
    return train_index, valid_index, test_index


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
    return float(crps / len(quantiles))


def git_commit(path):
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(path), text=True)
        return out.strip()
    except Exception:
        return None


def make_inputs(mask):
    times, feats = np.nonzero(mask)
    if len(times) == 0:
        return None, times, feats
    x = np.stack([times.astype(np.float32) / 47.0, feats.astype(np.float32)], axis=1)
    return x, times, feats


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import gpytorch

    class MTGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, rank):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            time_kernel = gpytorch.kernels.RBFKernel(active_dims=[0])
            task_kernel = gpytorch.kernels.IndexKernel(num_tasks=35, rank=rank, active_dims=[1])
            self.covar_module = gpytorch.kernels.ScaleKernel(time_kernel * task_kernel)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    requested_device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = Physio_Dataset(missing_ratio=args.missing_ratio, seed=args.seed)
    train_index, valid_index, test_index = split_indices(len(dataset), args.seed, args.nfold)
    if args.max_records is not None:
        test_index = test_index[: args.max_records]

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "command": sys.argv,
        "csdi_commit": git_commit(REPO_ROOT),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(requested_device),
            "gpytorch": gpytorch.__version__,
        },
        "splits": {
            "train": int(len(train_index)),
            "valid": int(len(valid_index)),
            "test": int(len(test_index)),
        },
        "kernel": "ScaleKernel(RBFKernel(time) * IndexKernel(feature, rank=10))",
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    all_targets = []
    all_samples = []
    per_record = []
    failures = []

    for pos, record_index in enumerate(test_index, start=1):
        observed = dataset.observed_values[record_index].astype(np.float32)
        observed_mask = dataset.observed_masks[record_index].astype(bool)
        gt_mask = dataset.gt_masks[record_index].astype(bool)
        eval_mask = observed_mask & ~gt_mask

        train_x_np, train_t, train_f = make_inputs(gt_mask)
        test_x_np, test_t, test_f = make_inputs(eval_mask)
        if train_x_np is None or test_x_np is None:
            failures.append({"record_index": int(record_index), "reason": "empty_train_or_eval"})
            continue

        train_y_np = observed[train_t, train_f]
        target_np = observed[test_t, test_f]
        train_x = torch.as_tensor(train_x_np, dtype=dtype, device=requested_device)
        train_y = torch.as_tensor(train_y_np, dtype=dtype, device=requested_device)
        test_x = torch.as_tensor(test_x_np, dtype=dtype, device=requested_device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(requested_device, dtype=dtype)
        model = MTGPModel(train_x, train_y, likelihood, args.rank).to(requested_device, dtype=dtype)
        likelihood.noise = torch.tensor(0.05, dtype=dtype, device=requested_device)

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        last_loss = None
        try:
            for _ in range(args.training_steps):
                optimizer.zero_grad(set_to_none=True)
                with gpytorch.settings.cholesky_jitter(1e-4):
                    output = model(train_x)
                    loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())

            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
                posterior = likelihood(model(test_x))
                sample_tensor = posterior.sample(torch.Size([args.num_samples])).transpose(0, 1)
            samples_np = sample_tensor.detach().cpu().numpy().astype(np.float32)
        except Exception as exc:
            failures.append({"record_index": int(record_index), "reason": repr(exc)})
            continue

        all_targets.append(target_np.astype(np.float32))
        all_samples.append(samples_np)
        median = np.median(samples_np, axis=1)
        per_record.append(
            {
                "position": int(pos),
                "record_index": int(record_index),
                "train_points": int(len(train_y_np)),
                "eval_points": int(len(target_np)),
                "loss": last_loss,
                "mae": float(np.mean(np.abs(median - target_np))),
                "mse": float(np.mean((median - target_np) ** 2)),
            }
        )
        loss_text = "none" if last_loss is None else f"{last_loss:.4f}"
        print(
            f"record {pos}/{len(test_index)} index={record_index} "
            f"train={len(train_y_np)} eval={len(target_np)} loss={loss_text}",
            flush=True,
        )

    if not all_targets:
        raise RuntimeError("No records were successfully evaluated")

    targets = np.concatenate(all_targets, axis=0)
    samples = np.concatenate(all_samples, axis=0)
    med = np.median(samples, axis=1)
    test = {
        "mae": float(np.mean(np.abs(med - targets))),
        "mse": float(np.mean((med - targets) ** 2)),
        "crps": calc_quantile_crps_points(targets, samples),
        "eval_points": int(len(targets)),
        "num_samples": int(samples.shape[1]),
        "num_records": int(len(per_record)),
        "failures": int(len(failures)),
    }
    metrics = {
        "status": "completed",
        "test": test,
        "per_record": per_record,
        "failures": failures,
        "paper_reference": {
            "table": "CSDI Table 2",
            "baseline": "Multitask GP",
            "healthcare_10_percent_missing": "0.489(0.005)",
            "healthcare_50_percent_missing": "0.581(0.003)",
            "healthcare_90_percent_missing": "0.942(0.010)",
        },
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if args.save_predictions:
        np.savez_compressed(output_dir / "predictions.npz", targets=targets, samples=samples)
    print(json.dumps(metrics["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
