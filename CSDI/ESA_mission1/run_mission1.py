from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_model import CSDI_Physio  # noqa: E402

from dataset_esa import (  # noqa: E402
    ESAWindowDataset,
    ESAWindowStore,
    build_window_starts,
    evenly_spaced_indices,
    make_evaluation_loader,
    make_training_loader,
    save_evaluation_protocol,
)
from experiment import (  # noqa: E402
    evaluate_model,
    load_checkpoint,
    set_global_seed,
    train_fixed_steps,
    write_csv,
    write_json,
)
from preprocess import preprocess, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ESA Mission 1 CSDI experiment")
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "train", "benchmark", "evaluate", "report", "full"),
        default="full",
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle), path


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(config: dict[str, Any], device: torch.device) -> CSDI_Physio:
    model_config = {
        "model": copy.deepcopy(config["model"]),
        "diffusion": copy.deepcopy(config["diffusion"]),
    }
    return CSDI_Physio(model_config, device, target_dim=76).to(device)


def result_root(config: dict[str, Any], config_path: Path) -> Path:
    return resolve_path(config["train"]["output_dir"], config_path).parent


def record_sources(config: dict[str, Any], config_path: Path) -> None:
    destination = result_root(config, config_path) / "provenance"
    destination.mkdir(parents=True, exist_ok=True)
    source_paths = [
        HERE / "config.yaml",
        HERE / "preprocess.py",
        HERE / "dataset_esa.py",
        HERE / "experiment.py",
        HERE / "run_mission1.py",
        HERE / "report_results.py",
        ROOT / "main_model.py",
        ROOT / "diff_models.py",
    ]
    records = {}
    for path in source_paths:
        if path.is_file():
            records[str(path.resolve())] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
    shutil.copy2(config_path, destination / "config_used.yaml")
    write_json(
        destination / "source_manifest.json",
        {
            "created_unix_time": time.time(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "files": records,
        },
    )


def load_stores(config: dict[str, Any], config_path: Path) -> tuple[ESAWindowStore, ESAWindowStore]:
    processed = resolve_path(config["data"]["processed_dir"], config_path)
    return ESAWindowStore.load(processed, "train"), ESAWindowStore.load(processed, "test")


def run_preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    train_store, test_store = load_stores(config, config_path)
    length = int(config["data"]["window_length"])
    stride = int(config["data"]["stride"])
    train_starts = build_window_starts(train_store.observed_count, length, stride)
    test_starts = build_window_starts(test_store.observed_count, length, stride)
    expected_points = int(config["data"]["expected_timepoints_per_split"])
    if train_store.timestamps_ns.size != expected_points or test_store.timestamps_ns.size != expected_points:
        raise ValueError("Processed split length does not match the official ESA-ADB 84-month grid")
    split_at = np.datetime64(config["data"]["split_at"], "ns").astype(np.int64)
    if int(train_store.timestamps_ns[-1]) != int(split_at):
        raise ValueError("Training grid does not end at the official split boundary")
    if int(test_store.timestamps_ns[0]) != int(split_at):
        raise ValueError("Test grid does not start at the official split boundary")
    if train_starts.size == 0 or test_starts.size == 0:
        raise ValueError("No usable windows were produced")
    if train_starts[-1] + length > train_store.timestamps_ns.size:
        raise ValueError("A training window crosses its partition")
    if test_starts[-1] + length > test_store.timestamps_ns.size:
        raise ValueError("A test window crosses its partition")
    payload = {
        "official_split_at": config["data"]["split_at"],
        "train_timepoints": int(train_store.timestamps_ns.size),
        "test_timepoints": int(test_store.timestamps_ns.size),
        "window_length": length,
        "stride": stride,
        "train_windows": int(train_starts.size),
        "test_windows": int(test_starts.size),
        "training_passes_at_32000_steps": (
            int(config["train"]["steps"]) * int(config["train"]["batch_size"]) / train_starts.size
        ),
        "all_missing_windows_removed_train": int(
            (train_store.timestamps_ns.size - length) // stride + 1 - train_starts.size
        ),
        "all_missing_windows_removed_test": int(
            (test_store.timestamps_ns.size - length) // stride + 1 - test_starts.size
        ),
    }
    destination = result_root(config, config_path)
    write_json(destination / "preflight.json", payload)
    record_sources(config, config_path)
    return payload


def create_protocols(
    config: dict[str, Any],
    config_path: Path,
    test_store: ESAWindowStore,
    window_count: int,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    length = int(config["data"]["window_length"])
    stride = int(config["data"]["stride"])
    all_starts = build_window_starts(test_store.observed_count, length, stride)
    selected = evenly_spaced_indices(len(all_starts), window_count)
    protocol_dir = resolve_path(evaluation["output_dir"], config_path) / "protocols"
    ratios = [float(value) for value in evaluation["missing_ratios"]]
    seeds = [int(value) for value in evaluation["mask_seeds"]]
    if len(ratios) != len(seeds):
        raise ValueError("evaluation.missing_ratios and mask_seeds must have equal length")
    records = {}
    target_arrays = {}
    for ratio, seed in zip(ratios, seeds):
        key = f"missing_{int(round(ratio * 100)):02d}"
        path = protocol_dir / f"{key}.npz"
        records[key] = save_evaluation_protocol(
            path, test_store, all_starts, selected, ratio, seed, length
        )
        with np.load(path) as loaded:
            observed = np.empty_like(loaded["condition_masks"])
            for row, start in enumerate(loaded["window_starts"]):
                observed[row] = test_store.observed_mask[:, int(start) : int(start) + length].T
            target_arrays[key] = observed - loaded["condition_masks"]
    independence = {}
    keys = list(target_arrays)
    for left_index in range(len(keys)):
        for right_index in range(left_index + 1, len(keys)):
            left, right = keys[left_index], keys[right_index]
            left_target = target_arrays[left].astype(bool)
            right_target = target_arrays[right].astype(bool)
            intersection = int(np.logical_and(left_target, right_target).sum())
            union = int(np.logical_or(left_target, right_target).sum())
            independence[f"{left}_vs_{right}"] = {
                "intersection_points": intersection,
                "jaccard": intersection / union if union else 0.0,
                "left_is_subset_of_right": bool(np.all(~left_target | right_target)),
                "right_is_subset_of_left": bool(np.all(~right_target | left_target)),
            }
    payload = {
        "window_count": window_count,
        "selection": "rounded linspace across all usable official test windows",
        "same_windows_for_all_ratios": True,
        "masks_independently_drawn_and_not_designed_as_nested": True,
        "protocols": records,
        "pairwise_mask_checks": independence,
    }
    write_json(protocol_dir / "manifest.json", payload)
    return payload


def run_smoke(
    config: dict[str, Any], config_path: Path
) -> tuple[CSDI_Physio, dict[str, Any]]:
    train_store, test_store = load_stores(config, config_path)
    length = int(config["data"]["window_length"])
    stride = int(config["data"]["stride"])
    train_dataset = ESAWindowDataset(train_store, length, stride)
    result_dir = result_root(config, config_path) / "smoke"
    result_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(config["train"]["seed"]))
    cpu_model = make_model(config, torch.device("cpu"))
    cpu_loader = make_training_loader(train_dataset, 1, int(config["train"]["seed"]), 1)
    cpu_batch = next(iter(cpu_loader))
    cpu_started = time.perf_counter()
    cpu_loss = cpu_model(cpu_batch, is_train=1)
    cpu_loss.backward()
    cpu_duration = time.perf_counter() - cpu_started
    if not torch.isfinite(cpu_loss):
        raise FloatingPointError("CPU smoke loss is non-finite")
    del cpu_model, cpu_batch

    if not torch.cuda.is_available():
        raise RuntimeError("The formal experiment requires the configured CUDA GPU")
    device = torch.device(config["train"]["device"])
    torch.cuda.empty_cache()
    set_global_seed(int(config["train"]["seed"]))
    gpu_model = make_model(config, device)
    smoke_steps = int(config["smoke"]["train_steps"])
    smoke_train_config = copy.deepcopy(config["train"])
    smoke_train_config.update(
        {"steps": smoke_steps, "checkpoint_interval": smoke_steps, "lr_milestones": []}
    )
    gpu_loader = make_training_loader(
        train_dataset,
        int(config["train"]["batch_size"]),
        int(config["train"]["seed"]),
        smoke_steps,
    )
    gpu_summary = train_fixed_steps(
        gpu_model, gpu_loader, smoke_train_config, result_dir / "gpu_training"
    )
    payload = {
        "cpu_forward_backward": {
            "loss": float(cpu_loss.item()),
            "duration_seconds": cpu_duration,
        },
        "gpu_training": gpu_summary,
    }
    write_json(result_dir / "smoke_summary.json", payload)
    return gpu_model, payload


def benchmark_chunks(
    config: dict[str, Any],
    config_path: Path,
    model: CSDI_Physio | None = None,
) -> dict[str, Any]:
    _, test_store = load_stores(config, config_path)
    evaluation = config["evaluation"]
    smoke_windows = int(config["smoke"]["evaluation_windows"])
    protocol = create_protocols(config, config_path, test_store, smoke_windows)
    protocol_path = Path(protocol["protocols"]["missing_50"]["path"])
    dataset = ESAWindowDataset.from_protocol(
        test_store, protocol_path, int(config["data"]["window_length"])
    )
    loader = make_evaluation_loader(dataset, int(evaluation["batch_size"]))
    if model is None:
        device = torch.device(config["train"]["device"])
        model = make_model(config, device)
        final_checkpoint = resolve_path(config["train"]["output_dir"], config_path) / "checkpoint_step_032000.pth"
        if final_checkpoint.is_file():
            load_checkpoint(final_checkpoint, model)
    model.eval()
    results = []
    nsample = int(evaluation["nsample"])
    for chunk in sorted({int(value) for value in evaluation["chunk_candidates"]}, reverse=True):
        torch.cuda.empty_cache()
        set_global_seed(int(config["train"]["seed"]) + chunk)
        try:
            result = evaluate_model(
                model,
                loader,
                ratio=0.5,
                nsample=nsample,
                sample_chunk_size=chunk,
                means=test_store.means,
                scales=test_store.scales,
                channel_names=test_store.channel_names,
                channel_metadata=test_store.channel_metadata,
                quantiles=[float(value) for value in evaluation["quantiles"]],
                max_batches=1,
            )
            results.append(
                {
                    "chunk_size": chunk,
                    "success": True,
                    "duration_seconds_one_50_sample_batch": result["duration_seconds"],
                    "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            results.append({"chunk_size": chunk, "success": False, "error": str(exc)})
    safe = [
        record
        for record in results
        if record["success"]
        and record["peak_gpu_memory_bytes"] <= int(evaluation["max_peak_memory_bytes"])
    ]
    if not safe:
        raise RuntimeError("No sample chunk candidate stayed below the configured memory ceiling")
    chosen = max(safe, key=lambda value: value["chunk_size"])
    requested_windows = int(evaluation["requested_windows"])
    batches = math.ceil(requested_windows / int(evaluation["batch_size"]))
    projected_seconds = (
        chosen["duration_seconds_one_50_sample_batch"]
        * batches
        * len(evaluation["missing_ratios"])
    )
    fallback = projected_seconds > float(evaluation["max_projected_hours"]) * 3600
    selected_windows = (
        int(evaluation["fallback_windows"]) if fallback else requested_windows
    )
    payload = {
        "nsample": nsample,
        "batch_size": int(evaluation["batch_size"]),
        "memory_ceiling_bytes": int(evaluation["max_peak_memory_bytes"]),
        "results": results,
        "chosen_chunk_size": chosen["chunk_size"],
        "projected_seconds_three_ratios_requested_windows": projected_seconds,
        "projected_hours_three_ratios_requested_windows": projected_seconds / 3600,
        "fallback_triggered": fallback,
        "formal_window_count": selected_windows,
    }
    benchmark_path = resolve_path(evaluation["output_dir"], config_path) / "benchmark.json"
    write_json(benchmark_path, payload)
    create_protocols(config, config_path, test_store, selected_windows)
    return payload


def latest_checkpoint(training_dir: Path) -> Path | None:
    paths = sorted(training_dir.glob("checkpoint_step_*.pth"))
    return paths[-1] if paths else None


def run_training(config: dict[str, Any], config_path: Path, resume: bool) -> dict[str, Any]:
    train_store, _ = load_stores(config, config_path)
    dataset = ESAWindowDataset(
        train_store,
        int(config["data"]["window_length"]),
        int(config["data"]["stride"]),
    )
    training_dir = resolve_path(config["train"]["output_dir"], config_path)
    final_checkpoint = training_dir / f"checkpoint_step_{int(config['train']['steps']):06d}.pth"
    if final_checkpoint.is_file():
        summary_path = training_dir / "training_summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = latest_checkpoint(training_dir) if resume else None
    start_step = 0
    if checkpoint is not None:
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        start_step = int(metadata["step"])
        del metadata
    loader = make_training_loader(
        dataset,
        int(config["train"]["batch_size"]),
        int(config["train"]["seed"]),
        int(config["train"]["steps"]),
        start_step,
    )
    set_global_seed(int(config["train"]["seed"]))
    model = make_model(config, torch.device(config["train"]["device"]))
    return train_fixed_steps(
        model,
        loader,
        config["train"],
        training_dir,
        start_step=start_step,
        resume_checkpoint=checkpoint,
    )


def run_evaluation(
    config: dict[str, Any], config_path: Path, force: bool = False
) -> dict[str, Any]:
    _, test_store = load_stores(config, config_path)
    evaluation_dir = resolve_path(config["evaluation"]["output_dir"], config_path)
    benchmark_path = evaluation_dir / "benchmark.json"
    if not benchmark_path.is_file():
        benchmark_chunks(config, config_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    window_count = int(benchmark["formal_window_count"])
    protocol_manifest = create_protocols(config, config_path, test_store, window_count)
    training_dir = resolve_path(config["train"]["output_dir"], config_path)
    final_step = int(config["train"]["steps"])
    final_checkpoint = training_dir / f"checkpoint_step_{final_step:06d}.pth"
    if not final_checkpoint.is_file():
        raise FileNotFoundError(f"Fixed final checkpoint is missing: {final_checkpoint}")
    device = torch.device(config["train"]["device"])
    model = make_model(config, device)
    load_checkpoint(final_checkpoint, model)
    all_channel_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    ratio_summaries = []
    for ratio in [float(value) for value in config["evaluation"]["missing_ratios"]]:
        key = f"missing_{int(round(ratio * 100)):02d}"
        output_path = evaluation_dir / f"metrics_{key}.json"
        if output_path.is_file() and not force:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            dataset = ESAWindowDataset.from_protocol(
                test_store,
                Path(protocol_manifest["protocols"][key]["path"]),
                int(config["data"]["window_length"]),
            )
            loader = make_evaluation_loader(dataset, int(config["evaluation"]["batch_size"]))
            set_global_seed(int(config["train"]["seed"]) + int(ratio * 10_000))
            result = evaluate_model(
                model,
                loader,
                ratio=ratio,
                nsample=int(config["evaluation"]["nsample"]),
                sample_chunk_size=int(benchmark["chosen_chunk_size"]),
                means=test_store.means,
                scales=test_store.scales,
                channel_names=test_store.channel_names,
                channel_metadata=test_store.channel_metadata,
                quantiles=[float(value) for value in config["evaluation"]["quantiles"]],
            )
            write_json(output_path, result)
        all_channel_rows.extend(result["channel_rows"])
        all_group_rows.extend(result["group_rows"])
        ratio_summaries.append(
            {key: value for key, value in result.items() if key not in {"channel_rows", "group_rows"}}
        )
    write_csv(evaluation_dir / "channel_metrics.csv", all_channel_rows)
    write_csv(evaluation_dir / "group_metrics.csv", all_group_rows)
    payload = {
        "fixed_model_step": final_step,
        "fixed_checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_sha256": sha256(final_checkpoint),
        "validation_or_test_model_selection": False,
        "benchmark": benchmark,
        "protocol": protocol_manifest,
        "ratios": ratio_summaries,
    }
    write_json(evaluation_dir / "evaluation_summary.json", payload)
    return payload


def run_report(config: dict[str, Any], config_path: Path) -> None:
    from report_results import generate_report

    generate_report(config, config_path)


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    processed = resolve_path(config["data"]["processed_dir"], config_path)
    if args.mode == "full" and not (processed / "manifest.json").is_file():
        preprocess(
            config,
            config_path,
            force=args.force_preprocess,
            skip_hash=False,
        )
    if args.mode == "preflight":
        print(json.dumps(run_preflight(config, config_path), indent=2, ensure_ascii=False))
    elif args.mode == "smoke":
        _, summary = run_smoke(config, config_path)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.mode == "benchmark":
        print(json.dumps(benchmark_chunks(config, config_path), indent=2, ensure_ascii=False))
    elif args.mode == "train":
        print(json.dumps(run_training(config, config_path, args.resume), indent=2, ensure_ascii=False))
    elif args.mode == "evaluate":
        print(json.dumps(run_evaluation(config, config_path, args.force_evaluation), indent=2, ensure_ascii=False))
    elif args.mode == "report":
        run_report(config, config_path)
    elif args.mode == "full":
        preflight = run_preflight(config, config_path)
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        smoke_model, smoke = run_smoke(config, config_path)
        print(json.dumps(smoke, indent=2, ensure_ascii=False))
        benchmark = benchmark_chunks(config, config_path, smoke_model)
        print(json.dumps(benchmark, indent=2, ensure_ascii=False))
        del smoke_model
        torch.cuda.empty_cache()
        training = run_training(config, config_path, args.resume)
        print(json.dumps(training, indent=2, ensure_ascii=False))
        evaluation = run_evaluation(config, config_path, args.force_evaluation)
        print(json.dumps(evaluation, indent=2, ensure_ascii=False))
        run_report(config, config_path)


if __name__ == "__main__":
    main()
