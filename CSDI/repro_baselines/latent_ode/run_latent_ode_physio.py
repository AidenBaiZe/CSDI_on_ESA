import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
LATENT_ODE_ROOT = REPO_ROOT / "external" / "latent_ode"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LATENT_ODE_ROOT) not in sys.path:
    sys.path.insert(0, str(LATENT_ODE_ROOT))

from dataset_physio import attributes, get_idlist  # noqa: E402
from lib.create_latent_ode_model import create_LatentODE_model  # noqa: E402
import lib.utils as latent_utils  # noqa: E402


N_FEATURES = len(attributes)
MAX_MINUTE = 48 * 60 - 1
PAPER_CRPS = {0.1: "0.700(0.002)", 0.5: "0.676(0.003)", 0.9: "0.761(0.010)"}
PAPER_MAE = {0.1: "0.522(0.002)", 0.5: "0.506(0.003)", 0.9: "0.578(0.009)"}
PAPER_RMSE = {0.1: "0.799(0.012)", 0.5: "0.783(0.012)", 0.9: "0.865(0.017)"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Latent ODE on the CSDI Table 4 PhysioNet interpolation protocol."
    )
    parser.add_argument("--missing-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nfold", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--valid-num-samples", type=int, default=10)
    parser.add_argument("--valid-interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-valid-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=None)

    # Latent ODE hyperparameters from the original PhysioNet README command.
    parser.add_argument("--latents", type=int, default=20)
    parser.add_argument("--rec-dims", type=int, default=40)
    parser.add_argument("--rec-layers", type=int, default=3)
    parser.add_argument("--gen-layers", type=int, default=3)
    parser.add_argument("--units", type=int, default=50)
    parser.add_argument("--gru-units", type=int, default=50)
    parser.add_argument("--z0-encoder", choices=["odernn", "rnn"], default="odernn")
    parser.add_argument("--poisson", action="store_true")
    parser.add_argument("--use-classif", action="store_true")
    parser.add_argument("--linear-classif", action="store_true")
    parser.add_argument("--train-target", choices=["all", "heldout"], default="all")
    return parser.parse_args()


def git_commit(path):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(path), text=True).strip()
    except Exception:
        return None


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


def minute_from_time(value):
    hour, minute = map(int, str(value).split(":"))
    total = hour * 60 + minute
    if total < 0:
        return None
    return min(total, MAX_MINUTE)


def load_outcomes(path):
    outcomes = {}
    if not path.exists():
        return outcomes
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        outcomes[str(int(row["RecordID"]))] = float(row["In-hospital_death"])
    return outcomes


def parse_record(record_id, outcomes):
    path = REPO_ROOT / "data" / "physio" / "set-a" / f"{record_id}.txt"
    df = pd.read_csv(path)
    df = df[df["Parameter"].isin(attributes)].copy()
    if len(df) == 0:
        return None

    feature_to_idx = {name: idx for idx, name in enumerate(attributes)}
    by_minute = {}
    for _, row in df.iterrows():
        minute = minute_from_time(row["Time"])
        if minute is None:
            continue
        feat = feature_to_idx[row["Parameter"]]
        by_minute.setdefault(minute, {})[feat] = float(row["Value"])

    if not by_minute:
        return None

    times = np.asarray(sorted(by_minute), dtype=np.int64)
    values = np.zeros((len(times), N_FEATURES), dtype=np.float32)
    mask = np.zeros((len(times), N_FEATURES), dtype=np.float32)
    for t_pos, minute in enumerate(times):
        for feat, value in by_minute[minute].items():
            values[t_pos, feat] = value
            mask[t_pos, feat] = 1.0

    return {
        "record_id": str(record_id),
        "times": times,
        "values": values,
        "mask": mask,
        "label": outcomes.get(str(record_id), np.nan),
    }


def load_records():
    outcomes = load_outcomes(REPO_ROOT / "data" / "physio" / "Outcomes-a.txt")
    records = []
    skipped = []
    for record_id in get_idlist():
        parsed = parse_record(record_id, outcomes)
        if parsed is None:
            skipped.append(str(record_id))
        else:
            records.append(parsed)
    return records, skipped


def normalize_records(records):
    stacked_values = []
    stacked_masks = []
    for rec in records:
        stacked_values.append(rec["values"])
        stacked_masks.append(rec["mask"])
    values = np.concatenate(stacked_values, axis=0)
    masks = np.concatenate(stacked_masks, axis=0)
    mean = np.zeros(N_FEATURES, dtype=np.float32)
    std = np.ones(N_FEATURES, dtype=np.float32)
    for idx in range(N_FEATURES):
        observed = values[:, idx][masks[:, idx] == 1]
        if len(observed) > 0:
            mean[idx] = observed.mean()
            std_val = observed.std()
            std[idx] = std_val if std_val > 0 else 1.0
    for rec in records:
        rec["values"] = ((rec["values"] - mean) / std * rec["mask"]).astype(np.float32)
    return mean, std


def choose_eval_timepoints(times, missing_ratio, rng):
    n_times = len(times)
    n_eval = int(n_times * missing_ratio)
    if n_eval <= 0:
        return np.zeros(n_times, dtype=bool)
    chosen = rng.choice(np.arange(n_times), size=n_eval, replace=False)
    mask = np.zeros(n_times, dtype=bool)
    mask[chosen] = True
    return mask


class PhysioInterpolationDataset(Dataset):
    def __init__(self, records, indices, missing_ratio, seed, split_name, train_target="all"):
        self.records = records
        self.indices = list(map(int, indices))
        self.missing_ratio = float(missing_ratio)
        self.seed = int(seed)
        self.split_name = split_name
        self.train_target = train_target
        self.epoch = 0

        self.fixed_eval = {}
        rng = np.random.default_rng(seed)
        for idx, rec in enumerate(records):
            self.fixed_eval[idx] = choose_eval_timepoints(rec["times"], missing_ratio, rng)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.indices)

    def _train_eval_mask(self, dataset_index, rec):
        rng = np.random.default_rng(self.seed + self.epoch * 100_003 + dataset_index)
        return choose_eval_timepoints(rec["times"], self.missing_ratio, rng)

    def __getitem__(self, position):
        dataset_index = self.indices[position]
        rec = self.records[dataset_index]
        if self.split_name == "train":
            eval_tp = self._train_eval_mask(dataset_index, rec)
            if self.train_target == "all":
                target_time_mask = np.ones(len(rec["times"]), dtype=bool)
            else:
                target_time_mask = eval_tp
        else:
            eval_tp = self.fixed_eval[dataset_index]
            target_time_mask = eval_tp

        cond_time_mask = ~eval_tp
        if not cond_time_mask.any():
            cond_time_mask[np.argmin(rec["times"])] = True

        observed_mask = rec["mask"].astype(np.float32)
        cond_mask = observed_mask * cond_time_mask[:, None].astype(np.float32)
        target_mask = observed_mask * target_time_mask[:, None].astype(np.float32)

        return {
            "dataset_index": dataset_index,
            "record_id": rec["record_id"],
            "times": rec["times"].astype(np.int64),
            "values": rec["values"].astype(np.float32),
            "observed_mask": observed_mask,
            "cond_mask": cond_mask.astype(np.float32),
            "target_mask": target_mask.astype(np.float32),
            "label": np.asarray([rec["label"]], dtype=np.float32),
        }


def make_collate(device):
    def collate(batch):
        cond_times = sorted(set(int(t) for item in batch for t in item["times"][item["cond_mask"].sum(1) > 0]))
        pred_times = sorted(set(int(t) for item in batch for t in item["times"][item["target_mask"].sum(1) > 0]))
        if not cond_times:
            cond_times = sorted(set(int(t) for item in batch for t in item["times"]))
        if not pred_times:
            pred_times = sorted(set(int(t) for item in batch for t in item["times"]))

        cond_pos = {minute: pos for pos, minute in enumerate(cond_times)}
        pred_pos = {minute: pos for pos, minute in enumerate(pred_times)}
        batch_size = len(batch)
        observed_data = np.zeros((batch_size, len(cond_times), N_FEATURES), dtype=np.float32)
        observed_mask = np.zeros_like(observed_data)
        data_to_predict = np.zeros((batch_size, len(pred_times), N_FEATURES), dtype=np.float32)
        mask_predicted_data = np.zeros_like(data_to_predict)
        labels = np.zeros((batch_size, 1), dtype=np.float32)

        for b, item in enumerate(batch):
            labels[b] = item["label"]
            for src_pos, minute in enumerate(item["times"]):
                if minute in cond_pos:
                    dst = cond_pos[minute]
                    observed_data[b, dst] = item["values"][src_pos] * item["cond_mask"][src_pos]
                    observed_mask[b, dst] = item["cond_mask"][src_pos]
                if minute in pred_pos:
                    dst = pred_pos[minute]
                    data_to_predict[b, dst] = item["values"][src_pos]
                    mask_predicted_data[b, dst] = item["target_mask"][src_pos]

        cond_tp = torch.as_tensor(np.asarray(cond_times, dtype=np.float32) / float(MAX_MINUTE), device=device)
        pred_tp = torch.as_tensor(np.asarray(pred_times, dtype=np.float32) / float(MAX_MINUTE), device=device)
        return {
            "record_ids": [item["record_id"] for item in batch],
            "dataset_indices": [int(item["dataset_index"]) for item in batch],
            "observed_data": torch.as_tensor(observed_data, device=device),
            "observed_tp": cond_tp,
            "data_to_predict": torch.as_tensor(data_to_predict, device=device),
            "tp_to_predict": pred_tp,
            "observed_mask": torch.as_tensor(observed_mask, device=device),
            "mask_predicted_data": torch.as_tensor(mask_predicted_data, device=device),
            "labels": torch.as_tensor(labels, device=device),
            "mode": "interp",
        }

    return collate


def calc_quantile_crps_points(target, forecast):
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = np.sum(np.abs(target))
    if denom == 0:
        raise ValueError("CRPS denominator is zero")
    crps = 0.0
    for q in quantiles:
        q_pred = np.quantile(forecast, q, axis=1)
        q_loss = 2.0 * np.sum(np.abs((q_pred - target) * ((target <= q_pred).astype(np.float32) - q)))
        crps += q_loss / denom
    return float(crps / len(quantiles))


def evaluate(model, loader, args, device, num_samples, save_predictions_path=None):
    model.eval()
    all_targets = []
    all_samples = []
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            results = model.compute_all_losses(batch, n_traj_samples=3, kl_coef=1.0)
            total_loss += float(results["loss"].detach().cpu())
            total_batches += 1
            pred_x, _ = model.get_reconstruction(
                batch["tp_to_predict"],
                batch["observed_data"],
                batch["observed_tp"],
                mask=batch["observed_mask"],
                n_traj_samples=num_samples,
                mode="interp",
            )
            target_mask = batch["mask_predicted_data"].bool()
            if target_mask.sum() == 0:
                continue
            target = batch["data_to_predict"][target_mask].detach().cpu().numpy().astype(np.float32)
            samples = pred_x[:, target_mask].transpose(0, 1).detach().cpu().numpy().astype(np.float32)
            all_targets.append(target)
            all_samples.append(samples)

    if not all_targets:
        raise RuntimeError("No evaluation targets found.")
    targets = np.concatenate(all_targets, axis=0)
    samples = np.concatenate(all_samples, axis=0)
    med = np.median(samples, axis=1)
    metrics = {
        "loss": float(total_loss / max(total_batches, 1)),
        "crps": calc_quantile_crps_points(targets, samples),
        "mae": float(np.mean(np.abs(med - targets))),
        "mse": float(np.mean((med - targets) ** 2)),
        "rmse": float(math.sqrt(np.mean((med - targets) ** 2))),
        "eval_points": int(len(targets)),
        "num_samples": int(samples.shape[1]),
    }
    if save_predictions_path is not None:
        np.savez_compressed(save_predictions_path, targets=targets, samples=samples)
    return metrics


def make_model_args(args):
    return SimpleNamespace(
        latents=args.latents,
        poisson=args.poisson,
        units=args.units,
        gen_layers=args.gen_layers,
        rec_dims=args.rec_dims,
        rec_layers=args.rec_layers,
        z0_encoder=args.z0_encoder,
        gru_units=args.gru_units,
        classif=args.use_classif,
        linear_classif=args.linear_classif,
        dataset="physionet",
    )


def make_manifest(args, records, skipped, splits, mean, std, device):
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
    }
    try:
        import torchdiffeq

        environment["torchdiffeq"] = getattr(torchdiffeq, "__version__", "unknown")
    except Exception:
        environment["torchdiffeq"] = None
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "command": sys.argv,
        "csdi_commit": git_commit(REPO_ROOT),
        "latent_ode_commit": git_commit(LATENT_ODE_ROOT),
        "environment": environment,
        "data_protocol": {
            "paper_table": "CSDI Table 4",
            "task": "healthcare interpolation",
            "raw_dataset": "PhysioNet Challenge 2012 set-a",
            "features": attributes,
            "num_features": N_FEATURES,
            "time_grid": "minute-level irregular time series, 48*60 possible minutes",
            "normalization": "CSDI healthcare zero mean / unit variance over observed set-a values",
            "masking": "missing_ratio selects observed time points, not individual feature values",
            "train_target": args.train_target,
        },
        "num_records": len(records),
        "skipped_record_ids": skipped,
        "splits": {
            "train": int(len(splits["train"])),
            "valid": int(len(splits["valid"])),
            "test": int(len(splits["test"])),
        },
        "normalization": {
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
        },
    }


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_history_from_log(path):
    if not path.exists():
        return []
    pattern = re.compile(
        r"epoch=(?P<epoch>\d+)\s+train_loss=(?P<train>[-+0-9.eE]+)\s+"
        r"valid_crps=(?P<valid>[-+0-9.eE]+)\s+best=(?P<best>[-+0-9.eE]+)\s+"
        r"no_improve=(?P<no_improve>\d+)"
    )
    history = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        history.append(
            {
                "epoch": int(match.group("epoch")),
                "train_loss": float(match.group("train")),
                "valid": {"crps": float(match.group("valid"))},
                "best_valid_crps": float(match.group("best")),
                "no_improve": int(match.group("no_improve")),
            }
        )
    return history


def split_mask_stats(dataset):
    cond_timepoints = 0
    eval_timepoints = 0
    observed_timepoints = 0
    cond_values = 0
    eval_values = 0
    observed_values = 0
    for pos in range(len(dataset)):
        item = dataset[pos]
        obs_tp = item["observed_mask"].sum(axis=1) > 0
        cond_tp = item["cond_mask"].sum(axis=1) > 0
        eval_tp = item["target_mask"].sum(axis=1) > 0
        observed_timepoints += int(obs_tp.sum())
        cond_timepoints += int(cond_tp.sum())
        eval_timepoints += int(eval_tp.sum())
        observed_values += int(item["observed_mask"].sum())
        cond_values += int(item["cond_mask"].sum())
        eval_values += int(item["target_mask"].sum())
    return {
        "num_records": len(dataset),
        "observed_timepoints": observed_timepoints,
        "cond_timepoints": cond_timepoints,
        "cond_timepoint_ratio": float(cond_timepoints / observed_timepoints) if observed_timepoints else None,
        "eval_timepoints": eval_timepoints,
        "eval_timepoint_ratio": float(eval_timepoints / observed_timepoints) if observed_timepoints else None,
        "observed_values": observed_values,
        "cond_values": cond_values,
        "cond_value_ratio": float(cond_values / observed_values) if observed_values else None,
        "eval_values": eval_values,
        "eval_value_ratio": float(eval_values / observed_values) if observed_values else None,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    records, skipped = load_records()
    mean, std = normalize_records(records)
    train_index, valid_index, test_index = split_indices(len(records), args.seed, args.nfold)
    if args.max_train_records is not None:
        train_index = train_index[: args.max_train_records]
    if args.max_valid_records is not None:
        valid_index = valid_index[: args.max_valid_records]
    if args.max_test_records is not None:
        test_index = test_index[: args.max_test_records]
    splits = {"train": train_index, "valid": valid_index, "test": test_index}

    manifest = make_manifest(args, records, skipped, splits, mean, std, device)
    write_json(output_dir / "run_manifest.json", manifest)
    env_manifest_path = REPO_ROOT / "baseline_results" / "latent_ode" / "env_manifest.json"
    env_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        env_manifest_path,
        {
            "created_at": manifest["created_at"],
            "csdi_commit": manifest["csdi_commit"],
            "latent_ode_commit": manifest["latent_ode_commit"],
            "environment": manifest["environment"],
        },
    )

    collate = make_collate(device)
    train_dataset = PhysioInterpolationDataset(
        records, train_index, args.missing_ratio, args.seed, "train", train_target=args.train_target
    )
    valid_dataset = PhysioInterpolationDataset(records, valid_index, args.missing_ratio, args.seed, "valid")
    test_dataset = PhysioInterpolationDataset(records, test_index, args.missing_ratio, args.seed, "test")
    write_json(
        output_dir / "data_manifest.json",
        {
            "num_records": len(records),
            "raw_set_a_records": len(get_idlist()),
            "skipped_record_ids": skipped,
            "splits": {
                "train": split_mask_stats(train_dataset),
                "valid": split_mask_stats(valid_dataset),
                "test": split_mask_stats(test_dataset),
            },
            "missing_ratio": args.missing_ratio,
            "seed": args.seed,
            "nfold": args.nfold,
        },
    )
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate)

    model_args = make_model_args(args)
    obsrv_std = torch.tensor([0.01], device=device)
    z0_prior = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
    model = create_LatentODE_model(model_args, N_FEATURES, z0_prior, obsrv_std, device, n_labels=1).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    checkpoint_path = output_dir / "best_model.pt"

    if args.eval_only:
        load_path = Path(args.checkpoint).resolve() if args.checkpoint else checkpoint_path
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        test_predictions_path = output_dir / "predictions.npz" if args.save_predictions else None
        test_metrics = evaluate(
            model, test_loader, args, device, num_samples=args.num_samples, save_predictions_path=test_predictions_path
        )
        history = read_history_from_log(output_dir / "train.log")
        best_epoch = None
        best_valid = None
        if history:
            best_row = min(history, key=lambda row: row["valid"]["crps"])
            best_epoch = best_row["epoch"]
            best_valid = best_row["valid"]["crps"]
        metrics = {
            "status": "evaluated_checkpoint",
            "eval_only": True,
            "checkpoint": str(load_path),
            "early_stopped": True,
            "stopped_epoch": history[-1]["epoch"] if history else None,
            "best_epoch": best_epoch,
            "best_valid_crps": best_valid,
            "history": history,
            "test": test_metrics,
            "paper_reference": {
                "table": "CSDI Table 4",
                "baseline": "Latent ODE",
                "crps": PAPER_CRPS.get(args.missing_ratio),
                "appendix_mae": PAPER_MAE.get(args.missing_ratio),
                "appendix_rmse": PAPER_RMSE.get(args.missing_ratio),
            },
        }
        write_json(output_dir / "metrics.json", metrics)
        print(json.dumps(metrics["test"], indent=2), flush=True)
        return

    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint).resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {init_path}")
        init_checkpoint = torch.load(init_path, map_location=device)
        model.load_state_dict(init_checkpoint["state_dict"])
        print(f"loaded_init_checkpoint={init_path}", flush=True)

    best_valid = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    num_batches = len(train_loader)
    global_itr = 0

    for epoch in range(args.epochs):
        model.train()
        train_dataset.set_epoch(epoch)
        train_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            global_itr += 1
            latent_utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr / 10.0)
            wait_until_kl_inc = 10
            if global_itr // max(num_batches, 1) < wait_until_kl_inc:
                kl_coef = 0.0
            else:
                kl_coef = 1.0 - 0.99 ** (global_itr // max(num_batches, 1) - wait_until_kl_inc)
            optimizer.zero_grad(set_to_none=True)
            train_res = model.compute_all_losses(batch, n_traj_samples=3, kl_coef=kl_coef)
            loss = train_res["loss"]
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            batch_count += 1
            if args.max_batches_per_epoch is not None and batch_count >= args.max_batches_per_epoch:
                break

        avg_train_loss = train_loss / max(batch_count, 1)
        row = {"epoch": epoch, "train_loss": avg_train_loss}
        should_validate = (epoch + 1) % args.valid_interval == 0 or epoch == args.epochs - 1
        if should_validate:
            valid_metrics = evaluate(model, valid_loader, args, device, num_samples=args.valid_num_samples)
            row["valid"] = valid_metrics
            current_valid = valid_metrics["crps"]
            improved = current_valid < best_valid - args.min_delta
            if improved:
                best_valid = current_valid
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save({"args": vars(args), "state_dict": model.state_dict()}, checkpoint_path)
            else:
                epochs_without_improvement += args.valid_interval
            print(
                f"epoch={epoch} train_loss={avg_train_loss:.6f} "
                f"valid_crps={current_valid:.6f} best={best_valid:.6f} "
                f"no_improve={epochs_without_improvement}",
                flush=True,
            )
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                history.append(row)
                break
        else:
            print(f"epoch={epoch} train_loss={avg_train_loss:.6f}", flush=True)
        history.append(row)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    test_predictions_path = output_dir / "predictions.npz" if args.save_predictions else None
    test_metrics = evaluate(
        model, test_loader, args, device, num_samples=args.num_samples, save_predictions_path=test_predictions_path
    )

    stopped_epoch = history[-1]["epoch"] if history else None
    metrics = {
        "status": "completed",
        "early_stopped": stopped_epoch is not None and stopped_epoch < args.epochs - 1,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "best_valid_crps": best_valid,
        "history": history,
        "test": test_metrics,
        "paper_reference": {
            "table": "CSDI Table 4",
            "baseline": "Latent ODE",
            "crps": PAPER_CRPS.get(args.missing_ratio),
            "appendix_mae": PAPER_MAE.get(args.missing_ratio),
            "appendix_rmse": PAPER_RMSE.get(args.missing_ratio),
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
