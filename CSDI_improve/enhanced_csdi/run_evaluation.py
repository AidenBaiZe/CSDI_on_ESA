from __future__ import annotations

import argparse
import gc
import json
import sys
import types
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

HERE = Path(__file__).resolve().parent
CSDI_ROOT = HERE.parents[1] / "CSDI"
for source_root in (CSDI_ROOT / "ESA_mission1_structured", CSDI_ROOT, HERE):
    source = str(source_root)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)

from ESA_mission1.dataset_esa import (  # noqa: E402
    ESAWindowDataset,
    ESAWindowStore,
    make_evaluation_loader,
)
from ESA_mission1.experiment import (  # noqa: E402
    evaluate_model,
    load_checkpoint,
    set_global_seed,
    write_json,
)

from run_enhanced import ensure_graph, load_config, make_model, resolve_from_config  # noqa: E402


PRIMARY_ORDER = (
    "channel_dropout_missing_50",
    "rectangle_missing_50",
    "time_block_missing_50",
    "channel_dropout_missing_10",
    "channel_dropout_missing_90",
    "rectangle_missing_10",
    "rectangle_missing_90",
    "time_block_missing_10",
    "time_block_missing_90",
    "gap_onset",
    "gap_sustained",
)

LIGHT_RANDOM_ORDER = (0.03, 0.05, 0.08, 0.10)


def _attach_evaluation_side_info_adapter(model: torch.nn.Module) -> None:
    """Adapt the unchanged ESA evaluator to Enhanced CSDI's observed-data side input."""
    original_process_data = model.process_data
    original_get_side_info = model.get_side_info

    def process_and_cache(self: torch.nn.Module, batch: dict[str, torch.Tensor]):
        result = original_process_data(batch)
        self._evaluation_observed_data = result[0]
        return result

    def compatible_get_side_info(
        self: torch.nn.Module,
        observed_tp: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self, "_evaluation_observed_data"):
            raise RuntimeError("process_data must run before get_side_info")
        return original_get_side_info(
            observed_tp, cond_mask, self._evaluation_observed_data
        )

    model.process_data = types.MethodType(process_and_cache, model)
    model.get_side_info = types.MethodType(compatible_get_side_info, model)


def _load_manifest(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    path = resolve_from_config(config_path, config["evaluation"]["protocol_manifest"])
    if not path.is_file():
        raise FileNotFoundError(f"protocol manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_records(manifest: dict[str, Any], include_random: bool) -> list[dict[str, Any]]:
    lookup = {record["key"]: record for record in manifest["records"]}
    missing = [key for key in PRIMARY_ORDER if key not in lookup]
    if missing:
        raise ValueError(f"protocols missing from manifest: {missing}")
    records = [lookup[key] for key in PRIMARY_ORDER]
    if include_random:
        records.extend(
            lookup[key]
            for key in ("random_missing_10", "random_missing_50", "random_missing_90")
        )
    return records


def _light_random_records(config: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    manifest_path = resolve_from_config(
        config_path, config["evaluation"]["light_random_protocol_manifest"]
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"light-random protocol manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for ratio in LIGHT_RANDOM_ORDER:
        percent = int(round(ratio * 100))
        source = manifest["protocols"][f"missing_{percent:02d}"]
        records.append(
            {
                **source,
                "key": f"random_missing_{percent:02d}",
                "mask_family": "random",
                "severity": f"{percent}%",
            }
        )
    return records


def _enrich(result: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "model_kind": "enhanced_csdi",
        "protocol_key": record["key"],
        "mask_family": record["mask_family"],
        "severity": record["severity"],
        "requested_missing_ratio": record["requested_missing_ratio"],
        "actual_missing_ratio": record["actual_missing_ratio"],
    }
    result.update(metadata)
    result["headline_normalized"].update(metadata)
    for collection in ("channel_rows", "group_rows"):
        for row in result[collection]:
            row.update(metadata)
    return result


def _build_runtime(
    config: dict[str, Any], config_path: Path
) -> tuple[torch.nn.Module, ESAWindowStore, Path]:
    graph_path, _ = ensure_graph(config, config_path)
    requested = str(config["train"]["device"])
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    model = make_model(config, graph_path, device)
    checkpoint = resolve_from_config(config_path, config["train"]["output_dir"]) / (
        f"checkpoint_step_{int(config['train']['steps']):06d}.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"enhanced checkpoint is missing: {checkpoint}")
    load_checkpoint(checkpoint, model)
    model.eval()
    _attach_evaluation_side_info_adapter(model)
    store = ESAWindowStore.load(
        resolve_from_config(config_path, config["data"]["processed_dir"]), "test"
    )
    return model, store, checkpoint


def smoke(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    model, store, checkpoint = _build_runtime(config, config_path)
    record = _ordered_records(_load_manifest(config, config_path), False)[0]
    dataset = ESAWindowDataset.from_protocol(
        store, Path(record["path"]), int(config["data"]["window_length"])
    )
    loader = make_evaluation_loader(dataset, 2)
    set_global_seed(20260719)
    result = evaluate_model(
        model,
        loader,
        ratio=float(record["requested_missing_ratio"]),
        nsample=2,
        sample_chunk_size=2,
        means=store.means,
        scales=store.scales,
        channel_names=store.channel_names,
        channel_metadata=store.channel_metadata,
        quantiles=[0.1, 0.5, 0.9],
        max_batches=1,
    )
    payload = {
        "status": "passed",
        "checkpoint": str(checkpoint),
        "protocol_key": record["key"],
        "windows": result["processed_windows"],
        "headline_normalized": result["headline_normalized"],
        "duration_seconds": result["duration_seconds"],
        "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
    }
    write_json(
        resolve_from_config(config_path, config["evaluation"]["output_dir"])
        / "smoke_summary.json",
        payload,
    )
    return payload


def evaluate(
    config: dict[str, Any],
    config_path: Path,
    include_random: bool,
    force: bool,
    light_random: bool = False,
) -> dict[str, Any]:
    model, store, checkpoint = _build_runtime(config, config_path)
    records = (
        _light_random_records(config, config_path)
        if light_random
        else _ordered_records(_load_manifest(config, config_path), include_random)
    )
    output = resolve_from_config(config_path, config["evaluation"]["output_dir"])
    metrics_dir = output / "metrics"
    if light_random:
        metrics_dir /= "light_random"
    metrics_dir /= "enhanced_csdi"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, record in enumerate(records, start=1):
        output_path = metrics_dir / f"{record['key']}.json"
        if output_path.is_file() and not force:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            gc.collect()
            if model.device.type == "cuda":
                torch.cuda.empty_cache()
            dataset = ESAWindowDataset.from_protocol(
                store, Path(record["path"]), int(config["data"]["window_length"])
            )
            loader = make_evaluation_loader(dataset, int(config["evaluation"]["batch_size"]))
            if light_random:
                # Match the original random-mask evaluator for paired diffusion noise.
                seed = int(config["train"]["seed"]) + int(
                    float(record["requested_missing_ratio"]) * 10_000
                )
            else:
                # Match the Structured-only evaluator's protocol seed for paired diffusion noise.
                seed = int(config["train"]["seed"]) + zlib.crc32(
                    f"structured_only_csdi:{record['key']}".encode("utf-8")
                )
            set_global_seed(seed)
            ratio = float(record["requested_missing_ratio"])
            if ratio <= 0:
                ratio = float(record["actual_missing_ratio"])
            result = evaluate_model(
                model,
                loader,
                ratio=ratio,
                nsample=int(config["evaluation"]["nsample"]),
                sample_chunk_size=int(config["evaluation"]["sample_chunk_size"]),
                means=store.means,
                scales=store.scales,
                channel_names=store.channel_names,
                channel_metadata=store.channel_metadata,
                quantiles=[float(value) for value in config["evaluation"]["quantiles"]],
            )
            result = _enrich(result, record)
            write_json(output_path, result)
            del loader, dataset
        summaries.append(
            {
                "index": index,
                "total": len(records),
                "protocol_key": record["key"],
                "headline_normalized": result["headline_normalized"],
                "path": str(output_path.resolve()),
            }
        )
        write_json(
            metrics_dir / "progress.json",
            {
                "model_kind": "enhanced_csdi",
                "completed": index,
                "total": len(records),
                "last_protocol": record["key"],
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )
    payload = {
        "model_kind": "enhanced_csdi",
        "checkpoint": str(checkpoint.resolve()),
        "protocol_count": len(records),
        "nsample": int(config["evaluation"]["nsample"]),
        "sample_chunk_size": int(config["evaluation"]["sample_chunk_size"]),
        "evaluations": summaries,
    }
    write_json(metrics_dir / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Enhanced CSDI on fixed ESA protocols")
    parser.add_argument("mode", choices=("smoke", "primary", "full", "light_random"))
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.mode == "smoke":
        result = smoke(config, config_path)
    else:
        result = evaluate(
            config,
            config_path,
            args.mode == "full",
            args.force,
            light_random=args.mode == "light_random",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
