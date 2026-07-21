from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
CSDI_ROOT = HERE.parents[1] / "CSDI"
for source_root in (CSDI_ROOT / "ESA_mission1_structured", CSDI_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
else:
    sys.path.remove(str(HERE))
    sys.path.insert(0, str(HERE))

from ESA_mission1.dataset_esa import (  # noqa: E402
    ESAWindowDataset,
    ESAWindowStore,
    make_training_loader,
)
from ESA_mission1.experiment import set_global_seed, train_fixed_steps  # noqa: E402

from graph_utils import export_validated_difference_graph, load_graph  # noqa: E402
from model import EnhancedStructuredCSDI  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_from_config(config_path: Path, value: str) -> Path:
    return (config_path.resolve().parent / value).resolve()


def ensure_graph(config: dict[str, Any], config_path: Path) -> tuple[Path, dict[str, Any] | None]:
    graph_config = config["graph"]
    artifact = resolve_from_config(config_path, graph_config["artifact_path"])
    if artifact.is_file():
        return artifact, None
    metadata = export_validated_difference_graph(
        resolve_from_config(config_path, graph_config["candidate_path"]),
        artifact,
        int(graph_config["top_k"]),
    )
    return artifact, metadata


def select_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def make_model(
    config: dict[str, Any], graph_path: Path, device: torch.device
) -> EnhancedStructuredCSDI:
    channels = int(config["data"]["channel_count"])
    adjacency = torch.from_numpy(load_graph(graph_path, channels)).to(device)
    model_config = {
        "diffusion": copy.deepcopy(config["diffusion"]),
        "model": copy.deepcopy(config["model"]),
        "frequency": copy.deepcopy(config["frequency"]),
    }
    return EnhancedStructuredCSDI(
        model_config,
        device,
        target_dim=channels,
        structured_config=config["structured_mask"],
        graph_adjacency=adjacency,
    ).to(device)


def make_dataset(config: dict[str, Any], config_path: Path) -> ESAWindowDataset:
    data_config = config["data"]
    store = ESAWindowStore.load(
        resolve_from_config(config_path, data_config["processed_dir"]), "train"
    )
    return ESAWindowDataset(
        store,
        window_length=int(data_config["window_length"]),
        stride=int(data_config["stride"]),
    )


def smoke(config: dict[str, Any], config_path: Path, graph_path: Path) -> dict[str, Any]:
    set_global_seed(int(config["train"]["seed"]))
    device = select_device(config["train"]["device"])
    smoke_config = copy.deepcopy(config)
    smoke_config["diffusion"].update(
        {"layers": 1, "channels": 8, "nheads": 1, "diffusion_embedding_dim": 16, "num_steps": 2}
    )
    smoke_config["model"].update({"timeemb": 16, "featureemb": 4})
    smoke_config["frequency"].update({"output_dim": 8})
    model = make_model(smoke_config, graph_path, device)
    dataset = make_dataset(smoke_config, config_path)
    loader = make_training_loader(dataset, batch_size=2, seed=1, total_steps=2)
    batch = next(iter(loader))
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, is_train=1)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    graph_gate_grad = model.diffmodel.residual_layers[0].graph_gate.grad
    frequency_gate_grad = model.diffmodel.residual_layers[0].frequency_gate.grad
    model.eval()
    evaluation_batch = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    evaluation_batch["gt_mask"] = evaluation_batch["observed_mask"].clone()
    evaluation_batch["gt_mask"][:, 32:48, :] = 0
    samples, observed, target_mask, _, _ = model.evaluate(
        evaluation_batch, n_samples=1
    )

    # Construct and execute the configured 4x64 model once as a separate integration check.
    full_model = make_model(config, graph_path, device).eval()
    full_observed, _, full_tp, _, _, _ = full_model.process_data(batch)
    full_condition = evaluation_batch["gt_mask"].to(device).float().permute(0, 2, 1)
    with torch.no_grad():
        full_side = full_model.get_side_info(full_tp, full_condition, full_observed)
        full_input = full_model.set_input_to_diffmodel(
            torch.randn_like(full_observed), full_observed, full_condition
        )
        full_prediction = full_model.diffmodel(
            full_input, full_side, torch.zeros(full_observed.shape[0], dtype=torch.long, device=device)
        )
    result = {
        "status": "passed",
        "device": str(device),
        "losses": losses,
        "sample_shape": list(samples.shape),
        "observed_shape": list(observed.shape),
        "target_points": int(target_mask.sum().item()),
        "finite_samples": bool(torch.isfinite(samples).all().item()),
        "configured_model": {
            "parameter_count": int(sum(parameter.numel() for parameter in full_model.parameters())),
            "prediction_shape": list(full_prediction.shape),
            "finite_prediction": bool(torch.isfinite(full_prediction).all().item()),
        },
        "graph_gate_has_gradient": graph_gate_grad is not None
        and bool(torch.isfinite(graph_gate_grad).all().item()),
        "frequency_gate_has_gradient": frequency_gate_grad is not None
        and bool(torch.isfinite(frequency_gate_grad).all().item()),
        "diagnostics": model.architecture_diagnostics(),
    }
    if not result["finite_samples"] or not result["configured_model"]["finite_prediction"]:
        raise FloatingPointError("smoke sampling produced non-finite values")
    return result


def train(config: dict[str, Any], config_path: Path, graph_path: Path) -> dict[str, Any]:
    set_global_seed(int(config["train"]["seed"]))
    device = select_device(config["train"]["device"])
    model = make_model(config, graph_path, device)
    dataset = make_dataset(config, config_path)
    train_config = config["train"]
    loader = make_training_loader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        seed=int(train_config["seed"]),
        total_steps=int(train_config["steps"]),
    )
    output_dir = resolve_from_config(config_path, train_config["output_dir"])
    return train_fixed_steps(model, loader, train_config, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced CSDI for ESA Mission 1")
    parser.add_argument("mode", choices=("build-graph", "smoke", "train"))
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    graph_path, graph_metadata = ensure_graph(config, config_path)
    if args.mode == "build-graph":
        result: dict[str, Any] = graph_metadata or {
            "status": "already_exists",
            "path": str(graph_path),
        }
    elif args.mode == "smoke":
        result = smoke(config, config_path, graph_path)
        summary_path = HERE / "results" / "smoke_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        result["summary_path"] = str(summary_path.resolve())
    else:
        result = train(config, config_path, graph_path)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
