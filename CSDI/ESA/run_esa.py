from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from main_model import CSDI_Physio  # noqa: E402

from dataset_esa import ESAWindowDataset, ESAWindowStore, make_loader  # noqa: E402
from experiment import evaluate_model, set_global_seed, train_model, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSDI on a preprocessed ESA subset")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument(
        "--mode", choices=("smoke", "baseline", "evaluate"), default="smoke"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--nsample", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--missing-ratios", type=float, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def resolve_from_config(path_value: str, config_path: Path) -> Path:
    return (config_path.parent / path_value).resolve()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, path


def build_model_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "diffusion": copy.deepcopy(config["diffusion"]),
        "model": copy.deepcopy(config["model"]),
        "train": copy.deepcopy(config["train"]),
    }


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_global_seed(args.seed)

    data_config = config["data"]
    store = ESAWindowStore.load(
        resolve_from_config(data_config["artifact"], config_path),
        resolve_from_config(data_config["manifest"], config_path),
    )
    length = int(data_config["window_length"])
    stride = int(data_config["stride"])
    mask_seed = int(data_config["mask_seed"])
    window_policy = str(data_config.get("window_policy", "strict_clean"))
    min_observed_fraction = float(data_config.get("min_observed_fraction", 0.0))
    historical_config = data_config.get("historical_patterns", {})
    use_historical_patterns = bool(historical_config.get("enabled", False))
    train_dataset = None
    valid_dataset = None
    train_loader = None
    valid_loader = None
    if args.mode != "evaluate":
        train_dataset = ESAWindowDataset(
            store,
            "train",
            length,
            stride,
            missing_ratio=None,
            window_policy=window_policy,
            min_observed_fraction=min_observed_fraction,
            use_historical_patterns=use_historical_patterns,
            historical_pattern_seed=int(historical_config.get("seed", mask_seed)),
            historical_min_missing_fraction=float(
                historical_config.get("min_missing_fraction", 0.05)
            ),
            historical_max_missing_fraction=float(
                historical_config.get("max_missing_fraction", 0.80)
            ),
        )
        valid_dataset = ESAWindowDataset(
            store,
            "validation",
            length,
            stride,
            missing_ratio=float(data_config["validation_missing_ratio"]),
            mask_seed=mask_seed,
            window_policy=window_policy,
            min_observed_fraction=min_observed_fraction,
        )
        train_loader = make_loader(
            train_dataset, int(config["train"]["batch_size"]), True, args.seed
        )
        valid_loader = make_loader(
            valid_dataset, int(config["evaluation"]["batch_size"]), False, args.seed
        )

    if args.mode == "smoke":
        mode_config = config["smoke"]
        epochs = args.epochs or int(mode_config["epochs"])
        nsample = args.nsample or int(mode_config["nsample"])
        sample_chunk_size = int(mode_config["sample_chunk_size"])
        ratios = [float(value) for value in mode_config["missing_ratios"]]
        max_train_batches = args.max_train_batches or int(
            mode_config["train_batches_per_epoch"]
        )
        max_valid_batches = int(mode_config["valid_batches"])
        max_eval_batches = args.max_eval_batches or int(mode_config["eval_batches"])
    else:
        epochs = args.epochs or int(config["train"]["epochs"])
        nsample = args.nsample or int(config["evaluation"]["nsample"])
        sample_chunk_size = int(config["evaluation"]["sample_chunk_size"])
        ratios = [float(value) for value in data_config["evaluation_missing_ratios"]]
        max_train_batches = args.max_train_batches
        max_valid_batches = int(config["train"]["valid_batches"])
        max_eval_batches = args.max_eval_batches
    if args.missing_ratios is not None:
        ratios = [float(value) for value in args.missing_ratios]
    if any(not 0.0 <= ratio < 1.0 for ratio in ratios):
        raise ValueError("All missing ratios must be in [0, 1)")
    if args.mode == "evaluate" and args.checkpoint is None:
        raise ValueError("--checkpoint is required in evaluate mode")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else HERE / "results" / f"{args.mode}_seed{args.seed}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    model_config = build_model_config(config)
    model = CSDI_Physio(model_config, args.device, target_dim=len(store.channel_names)).to(
        args.device
    )
    run_manifest = {
        "mode": args.mode,
        "seed": args.seed,
        "device": args.device,
        "window_length": length,
        "stride": stride,
        "window_policy": window_policy,
        "min_observed_fraction": min_observed_fraction,
        "window_counts": {
            "train": None if train_dataset is None else len(train_dataset),
            "validation": None if valid_dataset is None else len(valid_dataset),
        },
        "historical_patterns": (
            None
            if train_dataset is None
            else train_dataset.historical_pattern_summary()
        ),
        "epochs": epochs,
        "nsample": nsample,
        "max_train_batches_per_epoch": max_train_batches,
        "max_valid_batches": max_valid_batches,
        "max_eval_batches_per_ratio": max_eval_batches,
        "missing_ratios": ratios,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint.resolve()),
        "model_config": model_config,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    if args.mode == "evaluate":
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        try:
            state = torch.load(checkpoint, map_location=args.device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=args.device)
        model.load_state_dict(state)
        training = {"skipped": True, "checkpoint": str(checkpoint)}
    else:
        training = train_model(
            model,
            train_loader,
            valid_loader,
            config["train"],
            output_dir,
            epochs=epochs,
            max_train_batches=max_train_batches,
            max_valid_batches=max_valid_batches,
        )
        best_checkpoint = output_dir / "model_best.pth"
        if best_checkpoint.is_file():
            try:
                state = torch.load(
                    best_checkpoint, map_location=args.device, weights_only=True
                )
            except TypeError:
                state = torch.load(best_checkpoint, map_location=args.device)
            model.load_state_dict(state)
            training["evaluated_checkpoint"] = str(best_checkpoint)

    metrics_by_ratio: dict[str, Any] = {}
    for ratio in ratios:
        test_dataset = ESAWindowDataset(
            store,
            "test",
            length,
            stride,
            missing_ratio=ratio,
            mask_seed=mask_seed,
            window_policy=window_policy,
            min_observed_fraction=min_observed_fraction,
        )
        test_loader = make_loader(
            test_dataset,
            int(config["evaluation"]["batch_size"]),
            False,
            args.seed,
        )
        metrics = evaluate_model(
            model,
            test_loader,
            nsample,
            store.means,
            store.stds,
            sample_chunk_size,
            max_batches=max_eval_batches,
        )
        metrics["missing_ratio"] = ratio
        metrics["test_windows"] = len(test_dataset)
        ratio_name = f"missing_{int(round(ratio * 100))}"
        metrics_by_ratio[ratio_name] = metrics
        write_json(output_dir / f"metrics_{ratio_name}.json", metrics)

    summary = {
        "run": run_manifest,
        "training": {key: value for key, value in training.items() if key != "history"},
        "metrics": metrics_by_ratio,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
