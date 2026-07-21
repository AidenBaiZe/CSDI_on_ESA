from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

HERE = Path(__file__).resolve().parent
CSDI_ROOT = HERE.parents[1] / "CSDI"
STRUCTURED_ROOT = CSDI_ROOT / "ESA_mission1_structured"
for source_root in (CSDI_ROOT, STRUCTURED_ROOT):
    source = str(source_root)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)

from model import StructuredOnlyCSDI  # noqa: E402
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


RATIOS = (0.03, 0.05, 0.08, 0.10)


def _load_structured_config() -> dict[str, Any]:
    with (STRUCTURED_ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _records() -> list[dict[str, Any]]:
    manifest_path = (
        HERE
        / "results"
        / "mission1_light_random_baseline_seed1"
        / "evaluation"
        / "protocols"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for ratio in RATIOS:
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
        "model_kind": "structured_only_csdi",
        "protocol_key": record["key"],
        "mask_family": "random",
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


def _build() -> tuple[torch.nn.Module, ESAWindowStore, dict[str, Any], Path]:
    config = _load_structured_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StructuredOnlyCSDI(
        {"model": config["model"], "diffusion": config["diffusion"]},
        device,
        target_dim=int(config["data"]["channel_count"]),
        structured_config=config["structured_mask"],
    ).to(device)
    checkpoint = (
        STRUCTURED_ROOT
        / "results"
        / "mission1_structured_only_seed1"
        / "training"
        / "checkpoint_step_032000.pth"
    )
    load_checkpoint(checkpoint, model)
    model.eval()
    store = ESAWindowStore.load(CSDI_ROOT / "ESA_mission1" / "data" / "processed", "test")
    return model, store, config, checkpoint


def run(smoke: bool = False) -> dict[str, Any]:
    model, store, config, checkpoint = _build()
    records = _records()
    if smoke:
        records = records[:1]
    output = (
        HERE
        / "results"
        / "mission1_light_random_structured_baseline_seed1"
        / "evaluation"
        / "metrics"
        / "structured_only_csdi"
    )
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, record in enumerate(records, start=1):
        output_path = output / f"{record['key']}.json"
        if output_path.is_file() and not smoke:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            gc.collect()
            if model.device.type == "cuda":
                torch.cuda.empty_cache()
            dataset = ESAWindowDataset.from_protocol(
                store, Path(record["path"]), int(config["data"]["window_length"])
            )
            loader = make_evaluation_loader(dataset, 2 if smoke else 8)
            ratio = float(record["requested_missing_ratio"])
            # Identical diffusion-sampling seed to the Enhanced light-random run.
            set_global_seed(int(config["train"]["seed"]) + int(ratio * 10_000))
            result = evaluate_model(
                model,
                loader,
                ratio=ratio,
                nsample=2 if smoke else 50,
                sample_chunk_size=2 if smoke else 5,
                means=store.means,
                scales=store.scales,
                channel_names=store.channel_names,
                channel_metadata=store.channel_metadata,
                quantiles=[float(value) for value in config["evaluation"]["quantiles"]],
                max_batches=1 if smoke else None,
            )
            result = _enrich(result, record)
            if not smoke:
                write_json(output_path, result)
        summaries.append(
            {
                "index": index,
                "total": len(records),
                "protocol_key": record["key"],
                "headline_normalized": result["headline_normalized"],
            }
        )
        if not smoke:
            write_json(
                output / "progress.json",
                {
                    "model_kind": "structured_only_csdi",
                    "completed": index,
                    "total": len(records),
                    "last_protocol": record["key"],
                    "updated_at": datetime.now().astimezone().isoformat(),
                },
            )
    payload = {
        "status": "smoke_passed" if smoke else "completed",
        "checkpoint": str(checkpoint.resolve()),
        "evaluations": summaries,
    }
    if not smoke:
        write_json(output / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.smoke), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
