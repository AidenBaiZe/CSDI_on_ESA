import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate original BRITS on CSDI PhysioNet split.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--brits-dir", type=str, default="external/BRITS")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hid-size", type=int, default=108)
    parser.add_argument("--impute-weight", type=float, default=0.3)
    parser.add_argument("--label-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


class BritsJsonlDataset(Dataset):
    def __init__(self, path, is_train):
        self.path = Path(path)
        self.is_train = float(is_train)
        with open(self.path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        rec["is_train"] = self.is_train
        return rec


def collate_fn(records):
    def to_tensor_dict(direction_records):
        return {
            "values": torch.tensor([r["values"] for r in direction_records], dtype=torch.float32),
            "forwards": torch.tensor([r["forwards"] for r in direction_records], dtype=torch.float32),
            "masks": torch.tensor([r["masks"] for r in direction_records], dtype=torch.float32),
            "deltas": torch.tensor([r["deltas"] for r in direction_records], dtype=torch.float32),
            "evals": torch.tensor([r["evals"] for r in direction_records], dtype=torch.float32),
            "eval_masks": torch.tensor([r["eval_masks"] for r in direction_records], dtype=torch.float32),
        }

    forward = [r["forward"] for r in records]
    backward = [r["backward"] for r in records]
    return {
        "forward": to_tensor_dict(forward),
        "backward": to_tensor_dict(backward),
        "labels": torch.tensor([r["label"] for r in records], dtype=torch.float32),
        "is_train": torch.tensor([r["is_train"] for r in records], dtype=torch.float32),
        "record_ids": torch.tensor([r["record_id"] for r in records], dtype=torch.long),
    }


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_brits_model(brits_dir):
    brits_dir = Path(brits_dir).resolve()
    sys.path.insert(0, str(brits_dir))
    sys.path.insert(0, str(brits_dir / "models"))
    import brits  # noqa: PLC0415

    return brits.Model


def make_loader(path, is_train, batch_size, shuffle, num_workers):
    dataset = BritsJsonlDataset(path, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )


def evaluate(model, loader, device):
    model.eval()
    evals = []
    imputations = []
    labels = []
    preds = []
    record_ids = []

    with torch.no_grad():
        for batch in loader:
            record_ids.extend(batch["record_ids"].cpu().numpy().tolist())
            batch = move_to_device(batch, device)
            ret = model.run_on_batch(batch, None)

            pred = ret["predictions"].detach().cpu().numpy().reshape(-1)
            label = ret["labels"].detach().cpu().numpy().reshape(-1)
            eval_masks = ret["eval_masks"].detach().cpu().numpy()
            eval_values = ret["evals"].detach().cpu().numpy()
            imputed = ret["imputations"].detach().cpu().numpy()

            preds.extend(pred.tolist())
            labels.extend(label.tolist())
            evals.extend(eval_values[np.where(eval_masks == 1)].tolist())
            imputations.extend(imputed[np.where(eval_masks == 1)].tolist())

    evals = np.asarray(evals, dtype=np.float64)
    imputations = np.asarray(imputations, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    preds = np.asarray(preds, dtype=np.float64)

    mae = float(np.abs(evals - imputations).mean()) if len(evals) else float("nan")
    mre = float(np.abs(evals - imputations).sum() / np.abs(evals).sum()) if len(evals) else float("nan")
    auc = float(metrics.roc_auc_score(labels, preds)) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "mae": mae,
        "mre": mre,
        "auc": auc,
        "eval_points": int(len(evals)),
        "num_records": int(len(record_ids)),
    }


def git_output(args, cwd):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def write_manifest(args, outdir, best_epoch, best_valid, final_test):
    brits_dir = Path(args.brits_dir)
    manifest = {
        "command": " ".join(sys.argv),
        "data_dir": str(Path(args.data_dir).resolve()),
        "brits_dir": str(brits_dir.resolve()),
        "brits_commit": git_output(["rev-parse", "HEAD"], brits_dir),
        "csdi_commit": git_output(["rev-parse", "HEAD"], Path.cwd()),
        "csdi_status_short": git_output(["status", "--short"], Path.cwd()),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_valid": best_valid,
        "final_test": final_test,
    }
    with open(outdir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    outdir = Path(args.output_dir) if args.output_dir else data_dir
    outdir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    test_path = data_dir / "test.jsonl"
    for path in [train_path, valid_path, test_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    Model = import_brits_model(args.brits_dir)
    model = Model(args.hid_size, args.impute_weight, args.label_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = make_loader(train_path, 1, args.batch_size, True, args.num_workers)
    valid_loader = make_loader(valid_path, 0, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_path, 0, args.batch_size, False, args.num_workers)

    best_valid_mae = float("inf")
    best_epoch = -1
    stopped_epoch = -1
    epochs_without_improvement = 0
    best_path = outdir / "model_best.pth"
    max_epochs = args.max_epochs if args.max_epochs is not None else args.epochs

    for epoch in range(max_epochs):
        model.train()
        run_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            ret = model.run_on_batch(batch, optimizer, epoch)
            run_loss += ret["loss"].item()

        valid_metrics = evaluate(model, valid_loader, device)
        avg_loss = run_loss / max(1, len(train_loader))
        print(
            f"epoch={epoch} train_loss={avg_loss:.6f} "
            f"valid_mae={valid_metrics['mae']:.6f} valid_auc={valid_metrics['auc']:.6f}",
            flush=True,
        )

        if valid_metrics["mae"] < best_valid_mae - args.min_delta:
            best_valid_mae = valid_metrics["mae"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1

        if args.patience is not None and epochs_without_improvement >= args.patience:
            stopped_epoch = epoch
            print(
                f"early_stop epoch={epoch} best_epoch={best_epoch} "
                f"best_valid_mae={best_valid_mae:.6f} patience={args.patience}",
                flush=True,
            )
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    valid_metrics = evaluate(model, valid_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    torch.save(model.state_dict(), outdir / "model_final_loaded_best.pth")

    metrics_payload = {
        "early_stopped": stopped_epoch >= 0,
        "stopped_epoch": stopped_epoch if stopped_epoch >= 0 else max_epochs - 1,
        "best_epoch": best_epoch,
        "best_valid_mae": best_valid_mae,
        "max_epochs": max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "valid": valid_metrics,
        "test": test_metrics,
        "target_paper_mae_10_missing": 0.284,
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    write_manifest(args, outdir, best_epoch, valid_metrics, test_metrics)
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
