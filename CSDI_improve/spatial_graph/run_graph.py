from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
CSDI_ROOT = HERE.parents[1] / "CSDI"
STRUCTURED_ROOT = CSDI_ROOT / "ESA_mission1_structured"
for path in reversed((HERE, STRUCTURED_ROOT, CSDI_ROOT)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ESA_mission1.dataset_esa import ESAWindowStore  # noqa: E402
from graph import build_and_save_graph, load_graph  # noqa: E402
from graph_model import StructuredGraphCSDI  # noqa: E402


def resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: Path) -> tuple[dict, Path]:
    path = path.resolve()
    return yaml.safe_load(path.read_text(encoding="utf-8")), path


def build_graph(config: dict, config_path: Path) -> dict:
    store = ESAWindowStore.load(
        resolve_path(config["data"]["processed_dir"], config_path), "train"
    )
    graph_config = config["graph"]
    return build_and_save_graph(
        store.normalized_values,
        store.observed_mask,
        store.channel_names,
        resolve_path(config["models"]["graph_path"], config_path),
        top_k=int(graph_config["top_k"]),
        min_abs_correlation=float(graph_config["min_abs_correlation"]),
        chunk_size=int(graph_config["correlation_chunk_size"]),
    )


def make_model(
    config: dict, config_path: Path, device: torch.device
) -> StructuredGraphCSDI:
    target_dim = int(config["data"]["channel_count"])
    adjacency = torch.from_numpy(
        load_graph(
            resolve_path(config["models"]["graph_path"], config_path), target_dim
        )
    )
    model_config = {
        "model": copy.deepcopy(config["model"]),
        "diffusion": copy.deepcopy(config["diffusion"]),
    }
    return StructuredGraphCSDI(
        model_config,
        device,
        target_dim,
        config["structured_mask"],
        adjacency,
    ).to(device)


def smoke(config: dict, config_path: Path) -> dict:
    graph_path = resolve_path(config["models"]["graph_path"], config_path)
    if not graph_path.is_file():
        build_graph(config, config_path)
    smoke_config = copy.deepcopy(config)
    smoke_config["diffusion"].update(
        {
            "layers": 1,
            "channels": 8,
            "nheads": 1,
            "diffusion_embedding_dim": 16,
            "num_steps": 2,
        }
    )
    smoke_config["model"].update({"timeemb": 16, "featureemb": 4})
    model = make_model(smoke_config, config_path, torch.device("cpu"))
    batch = 2
    features = int(config["data"]["channel_count"])
    length = int(config["data"]["window_length"])
    x = torch.randn(batch, 2, features, length)
    cond = torch.randn(batch, 21, features, length)
    step = torch.zeros(batch, dtype=torch.long)
    with torch.no_grad():
        output = model.diffmodel(x, cond, step)
    return {
        "output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all()),
        "graph": model.graph_diagnostics(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESA Mission 1 graph-enhanced spatial CSDI")
    parser.add_argument("--mode", choices=("build-graph", "smoke"), default="smoke")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    result = (
        build_graph(config, config_path)
        if args.mode == "build-graph"
        else smoke(config, config_path)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
