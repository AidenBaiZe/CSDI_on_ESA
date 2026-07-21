from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from main_model import CSDI_Physio  # noqa: E402

from ESA_mission1.dataset_esa import (  # noqa: E402
    ESAWindowDataset,
    ESAWindowStore,
    make_evaluation_loader,
    make_training_loader,
)
from ESA_mission1.experiment import (  # noqa: E402
    evaluate_model,
    load_checkpoint,
    set_global_seed,
    train_fixed_steps,
    write_json,
)
from masking import (  # noqa: E402
    FAMILY_NAMES,
    gap_case_condition,
    nested_structured_conditions,
)
from model import StructuredOnlyCSDI  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESA Mission 1 structured masking experiment")
    parser.add_argument(
        "--mode",
        choices=(
            "preflight",
            "protocols",
            "smoke",
            "smoke-cpu",
            "smoke-train",
            "smoke-eval-chunk",
            "evaluate-random",
            "train",
            "evaluate-structured",
            "report",
            "full",
        ),
        default="full",
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--chunk-size", type=int)
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle), path


def resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def result_root(config: dict[str, Any], config_path: Path) -> Path:
    return resolve_path(config["evaluation"]["result_root"], config_path)


def write_status(
    config: dict[str, Any], config_path: Path, status: str, stage: str, message: str
) -> None:
    write_json(
        result_root(config, config_path) / "status.json",
        {
            "status": status,
            "stage": stage,
            "message": message,
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def record_sources(config: dict[str, Any], config_path: Path) -> None:
    output = result_root(config, config_path) / "provenance"
    output.mkdir(parents=True, exist_ok=True)
    files = [
        config_path,
        HERE / "masking.py",
        HERE / "model.py",
        HERE / "run_structured.py",
        HERE / "report_structured.py",
        ROOT / "main_model.py",
        ROOT / "diff_models.py",
        ROOT / "ESA_mission1" / "dataset_esa.py",
        ROOT / "ESA_mission1" / "experiment.py",
    ]
    records = {}
    for path in files:
        if path.is_file():
            records[str(path.resolve())] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
    shutil.copy2(config_path, output / "config_used.yaml")
    write_json(
        output / "source_manifest.json",
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "files": records,
        },
    )


def load_stores(
    config: dict[str, Any], config_path: Path
) -> tuple[ESAWindowStore, ESAWindowStore]:
    processed = resolve_path(config["data"]["processed_dir"], config_path)
    return ESAWindowStore.load(processed, "train"), ESAWindowStore.load(processed, "test")


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": copy.deepcopy(config["model"]),
        "diffusion": copy.deepcopy(config["diffusion"]),
    }


def make_model(
    config: dict[str, Any], device: torch.device, kind: str
) -> CSDI_Physio:
    dimensions = int(config["data"]["channel_count"])
    if kind == "random_csdi":
        return CSDI_Physio(model_config(config), device, target_dim=dimensions).to(device)
    if kind == "structured_only_csdi":
        return StructuredOnlyCSDI(
            model_config(config),
            device,
            target_dim=dimensions,
            structured_config=config["structured_mask"],
        ).to(device)
    raise ValueError(f"Unknown model kind: {kind}")


def base_protocol_path(
    config: dict[str, Any], config_path: Path, ratio: float
) -> Path:
    directory = resolve_path(config["models"]["random_protocol_dir"], config_path)
    return directory / f"missing_{int(round(ratio * 100)):02d}.npz"


def structured_protocol_dir(config: dict[str, Any], config_path: Path) -> Path:
    return resolve_path(config["evaluation"]["output_dir"], config_path) / "protocols"


def _load_reference_windows(
    config: dict[str, Any], config_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = base_protocol_path(config, config_path, 0.1)
    with np.load(reference) as loaded:
        starts = loaded["window_starts"].copy()
        indices = loaded["window_indices"].copy()
        timestamps = loaded["timestamp_ns"].copy()
    requested = int(config["evaluation"]["window_count"])
    if starts.size != requested:
        raise ValueError(f"Reference protocol has {starts.size} windows, expected {requested}")
    return starts, indices, timestamps


def _save_protocol(
    path: Path,
    condition_masks: np.ndarray,
    observed_masks: np.ndarray,
    starts: np.ndarray,
    indices: np.ndarray,
    timestamps: np.ndarray,
    metadata: dict[str, Any],
    protocol_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    target_counts = (observed_masks - condition_masks).sum(axis=(1, 2), dtype=np.int64)
    observed_counts = observed_masks.sum(axis=(1, 2), dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        window_starts=starts,
        window_indices=indices,
        timestamp_ns=timestamps,
        condition_masks=condition_masks,
        observed_counts=observed_counts,
        target_counts=target_counts,
        **(protocol_arrays or {}),
        **{
            key: np.asarray(value)
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool))
        },
    )
    total_observed = int(observed_counts.sum())
    total_target = int(target_counts.sum())
    return {
        **metadata,
        "path": str(path.resolve()),
        "windows": int(len(starts)),
        "target_points": total_target,
        "observed_points": total_observed,
        "actual_missing_ratio": total_target / total_observed,
    }


def _gap_metadata(
    config: dict[str, Any], config_path: Path, channel_names: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = pd.read_csv(resolve_path(config["data"]["gap_labels"], config_path))
    types = pd.read_csv(resolve_path(config["data"]["anomaly_types"], config_path))
    merged = labels.merge(types[["ID", "Category"]], on="ID", how="left")
    gaps = merged[merged["Category"].str.lower().eq("communication gap")].copy()
    gaps["start"] = pd.to_datetime(gaps["StartTime"], utc=True).dt.tz_convert(None)
    gaps["end"] = pd.to_datetime(gaps["EndTime"], utc=True).dt.tz_convert(None)
    split = pd.Timestamp("2007-01-01")
    train = gaps[gaps["start"] <= split].copy()
    affected_names = sorted(train["Channel"].unique())
    lookup = {name: index for index, name in enumerate(channel_names)}
    missing = sorted(set(affected_names) - set(lookup))
    if missing:
        raise ValueError(f"Gap annotations reference unknown channels: {missing}")
    affected = np.asarray([lookup[name] for name in affected_names], dtype=np.int64)
    durations = (
        train.drop_duplicates("ID").assign(
            duration_seconds=lambda frame: (frame["end"] - frame["start"]).dt.total_seconds()
        )["duration_seconds"]
    )
    metadata = {
        "training_gap_events": int(train["ID"].nunique()),
        "affected_channels": affected_names,
        "affected_channel_count": int(affected.size),
        "duration_seconds_min": float(durations.min()),
        "duration_seconds_median": float(durations.median()),
        "duration_seconds_max": float(durations.max()),
        "test_gap_events": int(gaps[gaps["start"] > split]["ID"].nunique()),
    }
    return affected, metadata


def create_protocols(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    _, test_store = load_stores(config, config_path)
    starts, indices, timestamps = _load_reference_windows(config, config_path)
    length = int(config["data"]["window_length"])
    observed_masks = np.empty(
        (len(starts), length, len(test_store.channel_names)), dtype=np.uint8
    )
    for row, start in enumerate(starts):
        observed_masks[row] = test_store.observed_mask[
            :, int(start) : int(start) + length
        ].T
    output = structured_protocol_dir(config, config_path)
    ratios = tuple(float(value) for value in config["evaluation"]["missing_ratios"])
    records: list[dict[str, Any]] = []
    nesting_checks: dict[str, Any] = {}

    geometry_summary: dict[str, dict[str, Any]] = {}
    for family in config["evaluation"]["families"]:
        seed = int(config["evaluation"]["family_seeds"][family])
        masks_by_ratio = {
            ratio: np.empty_like(observed_masks) for ratio in ratios
        }
        geometry_by_ratio = {
            ratio: {
                "time_start": [],
                "time_length": [],
                "selected_channel_count": [],
                "channel_order": [],
            }
            for ratio in ratios
        }
        for row, start in enumerate(starts):
            conditions, geometry = nested_structured_conditions(
                observed_masks[row],
                family,
                ratios,
                seed,
                int(start),
                return_metadata=True,
            )
            for ratio in ratios:
                masks_by_ratio[ratio][row] = conditions[ratio]
                geometry_by_ratio[ratio]["time_start"].append(
                    geometry["time_starts"][ratio]
                )
                geometry_by_ratio[ratio]["time_length"].append(
                    geometry["time_lengths"][ratio]
                )
                geometry_by_ratio[ratio]["selected_channel_count"].append(
                    geometry["channel_counts"][ratio]
                )
                geometry_by_ratio[ratio]["channel_order"].append(
                    geometry["channel_order"]
                )
        targets = {
            ratio: (observed_masks - masks_by_ratio[ratio]).astype(bool)
            for ratio in ratios
        }
        ordered = sorted(ratios)
        family_checks = {}
        for left, right in zip(ordered[:-1], ordered[1:]):
            is_subset = bool(np.all(~targets[left] | targets[right]))
            family_checks[f"{left:.1f}_subset_{right:.1f}"] = is_subset
            if not is_subset:
                raise AssertionError(f"{family} targets are not nested: {left} -> {right}")
        nesting_checks[family] = family_checks
        geometry_summary[family] = {}
        for ratio in ratios:
            key = f"{family}_missing_{int(round(ratio * 100)):02d}"
            protocol_arrays = {
                name: np.asarray(values, dtype=np.int16)
                for name, values in geometry_by_ratio[ratio].items()
            }
            geometry_summary[family][f"{ratio:.1f}"] = {
                "time_length": int(protocol_arrays["time_length"][0]),
                "selected_channel_count": int(
                    protocol_arrays["selected_channel_count"][0]
                ),
            }
            records.append(
                _save_protocol(
                    output / f"{key}.npz",
                    masks_by_ratio[ratio],
                    observed_masks,
                    starts,
                    indices,
                    timestamps,
                    {
                        "key": key,
                        "mask_family": family,
                        "severity": f"{int(round(ratio * 100))}%",
                        "requested_missing_ratio": ratio,
                        "seed": seed,
                        "nested": True,
                    },
                    protocol_arrays,
                )
            )

    affected, gap_metadata = _gap_metadata(
        config, config_path, test_store.channel_names
    )
    for case in ("gap_onset", "gap_sustained"):
        masks = np.empty_like(observed_masks)
        for row in range(len(starts)):
            masks[row] = gap_case_condition(observed_masks[row], affected, case)
        records.append(
            _save_protocol(
                output / f"{case}.npz",
                masks,
                observed_masks,
                starts,
                indices,
                timestamps,
                {
                    "key": case,
                    "mask_family": "real_gap_case",
                    "severity": case.removeprefix("gap_"),
                    "requested_missing_ratio": -1.0,
                    "seed": 0,
                    "nested": False,
                },
            )
        )

    random_records = []
    for ratio in ratios:
        path = base_protocol_path(config, config_path, ratio)
        with np.load(path) as loaded:
            if not np.array_equal(loaded["window_starts"], starts):
                raise ValueError(
                    "Random and structured protocols do not use the same windows"
                )
            observed_points = int(loaded["observed_counts"].sum())
            target_points = int(loaded["target_counts"].sum())
        random_records.append(
            {
                "key": f"random_missing_{int(round(ratio * 100)):02d}",
                "mask_family": "random",
                "severity": f"{int(round(ratio * 100))}%",
                "requested_missing_ratio": ratio,
                "actual_missing_ratio": target_points / observed_points,
                "path": str(path.resolve()),
                "windows": int(len(starts)),
                "target_points": target_points,
                "observed_points": observed_points,
                "seed": int(1001 + ratio * 1000),
                "nested": False,
                "evaluation_only": True,
            }
        )

    manifest = {
        "window_count": int(len(starts)),
        "training_random_point_masks": False,
        "same_windows_as_random_experiment": True,
        "controlled_protocols_are_nested": True,
        "reference_random_protocol": str(
            base_protocol_path(config, config_path, 0.1).resolve()
        ),
        "records": records + random_records,
        "nesting_checks": nesting_checks,
        "geometry": geometry_summary,
        "real_gap_source": gap_metadata,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def protocol_manifest(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    path = structured_protocol_dir(config, config_path) / "manifest.json"
    return create_protocols(config, config_path) if not path.is_file() else json.loads(path.read_text(encoding="utf-8"))


def run_preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    train_store, test_store = load_stores(config, config_path)
    random_checkpoint = resolve_path(config["models"]["random_checkpoint"], config_path)
    if not random_checkpoint.is_file():
        raise FileNotFoundError(f"Random checkpoint is missing: {random_checkpoint}")
    starts, _, _ = _load_reference_windows(config, config_path)
    configured_families = set(config["structured_mask"]["family_probabilities"])
    if configured_families != set(FAMILY_NAMES):
        raise ValueError(
            "Structured-only training config must contain exactly "
            f"{FAMILY_NAMES}; found {sorted(configured_families)}"
        )
    if "random" in configured_families:
        raise ValueError("Random point masking is forbidden in structured-only training")
    payload = {
        "train_timepoints": int(train_store.timestamps_ns.size),
        "test_timepoints": int(test_store.timestamps_ns.size),
        "channels": len(train_store.channel_names),
        "window_length": int(config["data"]["window_length"]),
        "formal_windows": int(starts.size),
        "random_checkpoint": str(random_checkpoint),
        "random_checkpoint_sha256": sha256(random_checkpoint),
        "structured_training_steps": int(config["train"]["steps"]),
        "structured_training_families": list(FAMILY_NAMES),
        "random_point_masks_used_for_training": False,
        "validation_used": False,
    }
    write_json(result_root(config, config_path) / "preflight.json", payload)
    record_sources(config, config_path)
    return payload


def smoke_root(config: dict[str, Any], config_path: Path) -> Path:
    return result_root(config, config_path) / "smoke"


def run_smoke_cpu(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    train_store, _ = load_stores(config, config_path)
    dataset = ESAWindowDataset(
        train_store,
        int(config["data"]["window_length"]),
        int(config["data"]["stride"]),
    )
    loader = make_training_loader(dataset, 1, int(config["train"]["seed"]), 1)
    set_global_seed(int(config["train"]["seed"]))
    model = make_model(config, torch.device("cpu"), "structured_only_csdi")
    batch = next(iter(loader))
    started = time.perf_counter()
    loss = model(batch, is_train=1)
    loss.backward()
    duration = time.perf_counter() - started
    if not torch.isfinite(loss):
        raise FloatingPointError("CPU structured-only smoke loss is non-finite")
    payload = {
        "loss": float(loss.item()),
        "seconds": duration,
        "mask_statistics": model.mask_statistics(),
    }
    write_json(smoke_root(config, config_path) / "cpu.json", payload)
    return payload


def run_smoke_train(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    train_store, _ = load_stores(config, config_path)
    steps = int(config["smoke"]["train_steps"])
    output = smoke_root(config, config_path) / "gpu_training"
    checkpoint = output / f"checkpoint_step_{steps:06d}.pth"
    summary_path = output / "training_summary.json"
    device = torch.device(config["train"]["device"])
    set_global_seed(int(config["train"]["seed"]))
    model = make_model(config, device, "structured_only_csdi")
    if checkpoint.is_file() and summary_path.is_file():
        load_checkpoint(checkpoint, model)
        training = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        dataset = ESAWindowDataset(
            train_store,
            int(config["data"]["window_length"]),
            int(config["data"]["stride"]),
        )
        loader = make_training_loader(
            dataset,
            int(config["train"]["batch_size"]),
            int(config["train"]["seed"]),
            steps,
        )
        smoke_config = copy.deepcopy(config["train"])
        smoke_config["steps"] = steps
        smoke_config["checkpoint_interval"] = steps
        training = train_fixed_steps(model, loader, smoke_config, output)
    payload = {
        "training": training,
        "mask_statistics": model.mask_statistics(),
        "checkpoint": str(checkpoint.resolve()),
    }
    write_json(smoke_root(config, config_path) / "gpu_training.json", payload)
    return payload


def run_smoke_eval_chunk(
    config: dict[str, Any], config_path: Path, chunk_size: int
) -> dict[str, Any]:
    if chunk_size not in {5, 10, 20}:
        raise ValueError("Smoke evaluation chunk size must be 5, 10, or 20")
    _, test_store = load_stores(config, config_path)
    steps = int(config["smoke"]["train_steps"])
    checkpoint = (
        smoke_root(config, config_path)
        / "gpu_training"
        / f"checkpoint_step_{steps:06d}.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Smoke checkpoint is missing: {checkpoint}")
    record = next(
        row
        for row in protocol_manifest(config, config_path)["records"]
        if row["key"] == "time_block_missing_50"
    )
    dataset = ESAWindowDataset.from_protocol(
        test_store,
        Path(record["path"]),
        int(config["data"]["window_length"]),
    )
    loader = make_evaluation_loader(dataset, int(config["evaluation"]["batch_size"]))
    device = torch.device(config["train"]["device"])
    model = make_model(config, device, "structured_only_csdi")
    load_checkpoint(checkpoint, model)
    result = evaluate_model(
        model,
        loader,
        ratio=0.5,
        nsample=int(config["evaluation"]["nsample"]),
        sample_chunk_size=chunk_size,
        means=test_store.means,
        scales=test_store.scales,
        channel_names=test_store.channel_names,
        channel_metadata=test_store.channel_metadata,
        quantiles=[float(value) for value in config["evaluation"]["quantiles"]],
        max_batches=1,
    )
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"channel_rows", "group_rows"}
    }
    payload["chunk_size"] = chunk_size
    payload["within_memory_limit"] = (
        int(payload["peak_gpu_memory_bytes"])
        <= int(config["evaluation"]["max_peak_memory_bytes"])
    )
    write_json(
        smoke_root(config, config_path) / f"evaluation_chunk_{chunk_size}.json",
        payload,
    )
    return payload


def _run_smoke_child(
    mode: str, config_path: Path, chunk_size: int | None = None
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(HERE / "run_structured.py"),
        "--mode",
        mode,
        "--config",
        str(config_path),
    ]
    if chunk_size is not None:
        command.extend(["--chunk-size", str(chunk_size)])
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def run_smoke_suite(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    output = smoke_root(config, config_path)
    output.mkdir(parents=True, exist_ok=True)
    child_records = []
    for mode, chunk_size in (
        ("smoke-cpu", None),
        ("smoke-train", None),
        ("smoke-eval-chunk", 5),
        ("smoke-eval-chunk", 10),
        ("smoke-eval-chunk", 20),
    ):
        if mode == "smoke-cpu" and (output / "cpu.json").is_file():
            child_records.append(
                {"mode": mode, "chunk_size": None, "returncode": 0, "reused": True}
            )
            continue
        if mode == "smoke-train" and (output / "gpu_training.json").is_file():
            child_records.append(
                {"mode": mode, "chunk_size": None, "returncode": 0, "reused": True}
            )
            continue
        if chunk_size is not None and (
            (output / f"evaluation_chunk_{chunk_size}.json").is_file()
            or (output / f"evaluation_chunk_{chunk_size}.failed.json").is_file()
        ):
            child_records.append(
                {
                    "mode": mode,
                    "chunk_size": chunk_size,
                    "returncode": (
                        0
                        if (output / f"evaluation_chunk_{chunk_size}.json").is_file()
                        else 1
                    ),
                    "reused": True,
                }
            )
            continue
        completed = _run_smoke_child(mode, config_path, chunk_size)
        suffix = f"_{chunk_size}" if chunk_size is not None else ""
        (output / f"{mode}{suffix}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (output / f"{mode}{suffix}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        child_records.append(
            {"mode": mode, "chunk_size": chunk_size, "returncode": completed.returncode}
        )
        if completed.returncode != 0 and chunk_size is not None:
            write_json(
                output / f"evaluation_chunk_{chunk_size}.failed.json",
                {
                    "chunk_size": chunk_size,
                    "returncode": completed.returncode,
                    "reason": "isolated smoke subprocess failed or exceeded practical runtime",
                },
            )
        if completed.returncode != 0 and chunk_size is None:
            raise RuntimeError(
                f"{mode} failed in its isolated process; see smoke logs"
            )
    successful = []
    for chunk_size in (5, 10, 20):
        path = output / f"evaluation_chunk_{chunk_size}.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("within_memory_limit"):
                successful.append(result)
    if not successful:
        raise RuntimeError("No 50-sample evaluation chunk passed the 6.5 GB limit")
    selected = max(successful, key=lambda row: int(row["chunk_size"]))
    failed_chunks = []
    for path in sorted(output.glob("evaluation_chunk_*.failed.json")):
        failed_chunks.append(json.loads(path.read_text(encoding="utf-8")))
    throughput_path = output / "throughput_vectorized.json"
    payload = {
        "process_isolation": True,
        "children": child_records,
        "cpu": json.loads((output / "cpu.json").read_text(encoding="utf-8")),
        "gpu_training": json.loads(
            (output / "gpu_training.json").read_text(encoding="utf-8")
        ),
        "successful_evaluation_chunks": successful,
        "failed_evaluation_chunks": failed_chunks,
        "selected_chunk_size": int(selected["chunk_size"]),
        "memory_limit_bytes": int(config["evaluation"]["max_peak_memory_bytes"]),
        "vectorized_training_throughput": (
            json.loads(throughput_path.read_text(encoding="utf-8"))
            if throughput_path.is_file()
            else None
        ),
    }
    write_json(output / "summary.json", payload)
    write_json(
        output / "selected_chunk.json",
        {"sample_chunk_size": int(selected["chunk_size"]), "source": "smoke_suite"},
    )
    return payload


def _latest_checkpoint(directory: Path) -> Path | None:
    paths = sorted(directory.glob("checkpoint_step_*.pth"))
    return paths[-1] if paths else None


def run_training(
    config: dict[str, Any], config_path: Path, resume: bool
) -> dict[str, Any]:
    train_store, _ = load_stores(config, config_path)
    output = resolve_path(config["models"]["structured_output_dir"], config_path)
    final_step = int(config["train"]["steps"])
    final_checkpoint = output / f"checkpoint_step_{final_step:06d}.pth"
    set_global_seed(int(config["train"]["seed"]))
    model = make_model(
        config, torch.device(config["train"]["device"]), "structured_only_csdi"
    )
    if final_checkpoint.is_file():
        load_checkpoint(final_checkpoint, model)
        summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
        write_json(output / "structured_mask_statistics.json", model.mask_statistics())
        return summary
    checkpoint = _latest_checkpoint(output) if resume else None
    start_step = 0
    if checkpoint is not None:
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        start_step = int(metadata["step"])
        del metadata
    dataset = ESAWindowDataset(
        train_store,
        int(config["data"]["window_length"]),
        int(config["data"]["stride"]),
    )
    loader = make_training_loader(
        dataset,
        int(config["train"]["batch_size"]),
        int(config["train"]["seed"]),
        final_step,
        start_step,
    )
    summary = train_fixed_steps(
        model,
        loader,
        config["train"],
        output,
        start_step=start_step,
        resume_checkpoint=checkpoint,
    )
    write_json(output / "structured_mask_statistics.json", model.mask_statistics())
    return summary


def _enrich_result(
    result: dict[str, Any], model_kind: str, record: dict[str, Any]
) -> dict[str, Any]:
    result["model_kind"] = model_kind
    result["protocol_key"] = record["key"]
    result["mask_family"] = record["mask_family"]
    result["severity"] = record["severity"]
    result["requested_missing_ratio"] = record["requested_missing_ratio"]
    result["actual_missing_ratio"] = record["actual_missing_ratio"]
    for collection in ("channel_rows", "group_rows"):
        for row in result[collection]:
            row["model_kind"] = model_kind
            row["protocol_key"] = record["key"]
            row["mask_family"] = record["mask_family"]
            row["severity"] = record["severity"]
            row["requested_missing_ratio"] = record["requested_missing_ratio"]
            row["actual_missing_ratio"] = record["actual_missing_ratio"]
    result["headline_normalized"].update(
        {
            "model_kind": model_kind,
            "protocol_key": record["key"],
            "mask_family": record["mask_family"],
            "severity": record["severity"],
            "requested_missing_ratio": record["requested_missing_ratio"],
            "actual_missing_ratio": record["actual_missing_ratio"],
        }
    )
    return result


def evaluate_protocols(
    config: dict[str, Any],
    config_path: Path,
    model_kind: str,
    force: bool = False,
) -> dict[str, Any]:
    _, test_store = load_stores(config, config_path)
    manifest = protocol_manifest(config, config_path)
    if model_kind == "random_csdi":
        checkpoint = resolve_path(config["models"]["random_checkpoint"], config_path)
        records = [record for record in manifest["records"] if record["mask_family"] != "random"]
    elif model_kind == "structured_only_csdi":
        output = resolve_path(config["models"]["structured_output_dir"], config_path)
        checkpoint = output / f"checkpoint_step_{int(config['train']['steps']):06d}.pth"
        records = list(manifest["records"])
    else:
        raise ValueError(model_kind)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint is missing: {checkpoint}")
    device = torch.device(config["train"]["device"])
    model = make_model(config, device, model_kind)
    load_checkpoint(checkpoint, model)
    model.eval()
    metrics_dir = resolve_path(config["evaluation"]["output_dir"], config_path) / "metrics" / model_kind
    if model_kind == "random_csdi":
        requested = set(config["evaluation"]["random_baseline_protocols"])
        records = [record for record in records if record["key"] in requested]
        found = {record["key"] for record in records}
        if found != requested:
            raise ValueError(
                f"Random baseline protocols missing from manifest: {sorted(requested - found)}"
            )
    selected_chunk_path = smoke_root(config, config_path) / "selected_chunk.json"
    if selected_chunk_path.is_file():
        sample_chunk_size = int(
            json.loads(selected_chunk_path.read_text(encoding="utf-8"))["sample_chunk_size"]
        )
    else:
        sample_chunk_size = int(config["evaluation"]["sample_chunk_size"])
    summaries = []
    for index, record in enumerate(records, start=1):
        output_path = metrics_dir / f"{record['key']}.json"
        if output_path.is_file() and not force:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            dataset = ESAWindowDataset.from_protocol(
                test_store,
                Path(record["path"]),
                int(config["data"]["window_length"]),
            )
            loader = make_evaluation_loader(
                dataset, int(config["evaluation"]["batch_size"])
            )
            seed = int(config["train"]["seed"]) + zlib.crc32(
                f"{model_kind}:{record['key']}".encode("utf-8")
            )
            set_global_seed(seed)
            ratio_for_rows = (
                float(record["requested_missing_ratio"])
                if float(record["requested_missing_ratio"]) > 0
                else float(record["actual_missing_ratio"])
            )
            result = evaluate_model(
                model,
                loader,
                ratio=ratio_for_rows,
                nsample=int(config["evaluation"]["nsample"]),
                sample_chunk_size=sample_chunk_size,
                means=test_store.means,
                scales=test_store.scales,
                channel_names=test_store.channel_names,
                channel_metadata=test_store.channel_metadata,
                quantiles=[float(value) for value in config["evaluation"]["quantiles"]],
            )
            result = _enrich_result(result, model_kind, record)
            write_json(output_path, result)
            del loader, dataset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        summaries.append(
            {
                "index": index,
                "total": len(records),
                "protocol_key": record["key"],
                "path": str(output_path.resolve()),
                "headline_normalized": result["headline_normalized"],
            }
        )
        write_json(
            metrics_dir / "progress.json",
            {
                "model_kind": model_kind,
                "completed": index,
                "total": len(records),
                "last_protocol": record["key"],
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )
    payload = {
        "model_kind": model_kind,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "sample_chunk_size": sample_chunk_size,
        "evaluations": summaries,
    }
    write_json(metrics_dir / "summary.json", payload)
    return payload


def run_report(config: dict[str, Any], config_path: Path) -> None:
    from report_structured import generate_report

    generate_report(config, config_path)


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    try:
        if args.mode == "preflight":
            print(json.dumps(run_preflight(config, config_path), indent=2, ensure_ascii=False))
        elif args.mode == "protocols":
            print(json.dumps(create_protocols(config, config_path), indent=2, ensure_ascii=False))
        elif args.mode == "smoke":
            print(json.dumps(run_smoke_suite(config, config_path), indent=2, ensure_ascii=False))
        elif args.mode == "smoke-cpu":
            print(json.dumps(run_smoke_cpu(config, config_path), indent=2, ensure_ascii=False))
        elif args.mode == "smoke-train":
            print(json.dumps(run_smoke_train(config, config_path), indent=2, ensure_ascii=False))
        elif args.mode == "smoke-eval-chunk":
            if args.chunk_size is None:
                raise ValueError("--chunk-size is required for smoke-eval-chunk")
            print(
                json.dumps(
                    run_smoke_eval_chunk(config, config_path, args.chunk_size),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif args.mode == "evaluate-random":
            print(json.dumps(evaluate_protocols(config, config_path, "random_csdi", args.force_evaluation), indent=2, ensure_ascii=False))
        elif args.mode == "train":
            print(json.dumps(run_training(config, config_path, args.resume), indent=2, ensure_ascii=False))
        elif args.mode == "evaluate-structured":
            print(json.dumps(evaluate_protocols(config, config_path, "structured_only_csdi", args.force_evaluation), indent=2, ensure_ascii=False))
        elif args.mode == "report":
            run_report(config, config_path)
        elif args.mode == "full":
            write_status(config, config_path, "running", "preflight", "Checking data, checkpoints, and source hashes")
            run_preflight(config, config_path)
            write_status(config, config_path, "running", "protocols", "Building fixed nested structured protocols")
            create_protocols(config, config_path)
            write_status(config, config_path, "running", "smoke", "Running CPU/GPU structured-mask smoke checks")
            run_smoke_suite(config, config_path)
            write_status(config, config_path, "running", "train_structured", "Training Structured-Only-CSDI to 32,000 steps")
            run_training(config, config_path, args.resume)
            write_status(config, config_path, "running", "evaluate_random", "Evaluating Random-CSDI on five core structured protocols")
            evaluate_protocols(config, config_path, "random_csdi", args.force_evaluation)
            write_status(config, config_path, "running", "evaluate_structured", "Evaluating Structured-Only-CSDI on 11 structured and 3 random protocols")
            evaluate_protocols(config, config_path, "structured_only_csdi", args.force_evaluation)
            write_status(config, config_path, "running", "report", "Generating the comparison report and figures")
            run_report(config, config_path)
            write_status(config, config_path, "complete", "complete", "All structured masking training, evaluations, and reporting are complete")
    except Exception as exc:
        write_status(config, config_path, "failed", "failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
