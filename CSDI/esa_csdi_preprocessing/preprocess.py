from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_NAMES = {
    0: "nominal",
    1: "anomaly",
    2: "rare_event",
    3: "communication_gap",
}
LABEL_PRIORITY = {0: 0, 2: 1, 1: 2, 3: 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the ESA Mission 1 lightweight subset for CSDI."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing processed artifact.",
    )
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    root = path.parent
    required = {
        "archive_path",
        "mission",
        "channels",
        "start_time",
        "end_time",
        "resample_seconds",
        "split_ratios",
        "output_name",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config fields: {missing}")
    config["archive_path"] = str((root / config["archive_path"]).resolve())
    return config, root


def validate_config(config: dict[str, Any]) -> None:
    archive = Path(config["archive_path"])
    if not archive.is_file():
        raise FileNotFoundError(f"Mission archive not found: {archive}")
    channels = config["channels"]
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("channels must be a non-empty unique list")
    if any(not isinstance(channel, int) or channel <= 0 for channel in channels):
        raise ValueError("channel identifiers must be positive integers")
    seconds = int(config["resample_seconds"])
    if seconds <= 0:
        raise ValueError("resample_seconds must be positive")
    start = pd.Timestamp(config["start_time"])
    end = pd.Timestamp(config["end_time"])
    if start >= end:
        raise ValueError("start_time must be earlier than end_time")
    ratios = np.asarray(config["split_ratios"], dtype=float)
    if ratios.shape != (3,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("split_ratios must contain three positive values summing to 1")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_content_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def extract_selected_files(
    archive: Path,
    interim_root: Path,
    mission: str,
    channels: list[int],
) -> dict[str, Path]:
    entries = [
        f"{mission}/labels.csv",
        f"{mission}/anomaly_types.csv",
        *[f"{mission}/channels/channel_{channel}.zip" for channel in channels],
    ]
    expected = {entry: interim_root.joinpath(*entry.split("/")) for entry in entries}
    missing = [entry for entry, path in expected.items() if not path.is_file()]
    if missing:
        interim_root.mkdir(parents=True, exist_ok=True)
        command = ["tar", "-xf", str(archive), "-C", str(interim_root), *missing]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to selectively extract Mission 1 files.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
    absent = [entry for entry, path in expected.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Archive entries were not extracted: {absent}")
    return expected


def read_channel(path: Path, channel_name: str) -> pd.Series:
    frame = pd.read_pickle(path)
    if isinstance(frame, pd.Series):
        series = frame
    elif isinstance(frame, pd.DataFrame):
        if channel_name in frame.columns:
            series = frame[channel_name]
        elif frame.shape[1] == 1:
            series = frame.iloc[:, 0]
        else:
            raise ValueError(
                f"{path} has unexpected columns: {frame.columns.to_list()}"
            )
    else:
        raise TypeError(f"{path} contains {type(frame).__name__}, expected Series/DataFrame")

    series = series.copy()
    index = pd.to_datetime(series.index, utc=True).tz_convert(None)
    series.index = index
    series = pd.to_numeric(series, errors="coerce")
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise ValueError(f"{channel_name} is empty")
    return series.astype(np.float64)


def category_to_code(category: str) -> int:
    normalized = str(category).strip().lower()
    if normalized == "anomaly":
        return 1
    if normalized == "rare event":
        return 2
    return 3


def apply_labels(
    grid: pd.DatetimeIndex,
    channel_names: list[str],
    labels: pd.DataFrame,
    anomaly_types: pd.DataFrame,
    frequency: pd.Timedelta,
) -> np.ndarray:
    type_map = {
        str(row.ID): category_to_code(row.Category)
        for row in anomaly_types.itertuples(index=False)
    }
    label_codes = np.zeros((len(grid), len(channel_names)), dtype=np.uint8)
    channel_to_index = {name: index for index, name in enumerate(channel_names)}

    for row in labels.itertuples(index=False):
        channel = str(row.Channel)
        if channel not in channel_to_index:
            continue
        event_id = str(row.ID)
        if event_id not in type_map:
            raise ValueError(f"Missing anomaly type for event {event_id}")
        code = type_map[event_id]
        start = pd.Timestamp(row.StartTime).ceil(frequency)
        end = pd.Timestamp(row.EndTime).floor(frequency)
        if end < start:
            end = start
        left = int(grid.searchsorted(start, side="left"))
        right = int(grid.searchsorted(end, side="right"))
        if left >= len(grid) or right <= 0 or left >= right:
            continue
        left = max(left, 0)
        right = min(right, len(grid))
        column = channel_to_index[channel]
        current = label_codes[left:right, column]
        replace = np.fromiter(
            (LABEL_PRIORITY[code] > LABEL_PRIORITY[int(value)] for value in current),
            dtype=bool,
            count=len(current),
        )
        current[replace] = code
        label_codes[left:right, column] = current
    return label_codes


def split_codes(length: int, ratios: list[float]) -> tuple[np.ndarray, tuple[int, int]]:
    train_end = int(math.floor(length * ratios[0]))
    valid_end = train_end + int(math.floor(length * ratios[1]))
    if train_end <= 0 or valid_end <= train_end or valid_end >= length:
        raise ValueError("Dataset is too short for the requested split ratios")
    codes = np.full(length, 2, dtype=np.uint8)
    codes[:train_end] = 0
    codes[train_end:valid_end] = 1
    return codes, (train_end, valid_end)


def json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_scalar)
        handle.write("\n")


def preprocess(config: dict[str, Any], project_root: Path, force: bool) -> Path:
    archive = Path(config["archive_path"])
    mission = str(config["mission"])
    channels = [int(channel) for channel in config["channels"]]
    channel_names = [f"channel_{channel}" for channel in channels]
    interim_root = project_root / "data" / "interim"
    output_dir = project_root / "data" / "processed" / str(config["output_name"])
    artifact_path = output_dir / "aligned.npz"
    manifest_path = output_dir / "manifest.json"
    quality_path = output_dir / "quality_report.json"

    if any(path.exists() for path in (artifact_path, manifest_path, quality_path)) and not force:
        raise FileExistsError(
            f"Processed output already exists in {output_dir}. Use --force to replace it."
        )

    extracted = extract_selected_files(archive, interim_root, mission, channels)
    labels_path = extracted[f"{mission}/labels.csv"]
    anomaly_types_path = extracted[f"{mission}/anomaly_types.csv"]
    labels = pd.read_csv(labels_path, parse_dates=["StartTime", "EndTime"])
    anomaly_types = pd.read_csv(anomaly_types_path)
    for column in ("StartTime", "EndTime"):
        labels[column] = pd.to_datetime(labels[column], utc=True).dt.tz_convert(None)

    series_by_channel: dict[str, pd.Series] = {}
    for channel, channel_name in zip(channels, channel_names):
        path = extracted[f"{mission}/channels/channel_{channel}.zip"]
        series_by_channel[channel_name] = read_channel(path, channel_name)

    requested_start = pd.Timestamp(config["start_time"])
    requested_end = pd.Timestamp(config["end_time"])
    common_start = max(series.index.min() for series in series_by_channel.values())
    common_end = min(series.index.max() for series in series_by_channel.values())
    effective_start = max(requested_start, common_start)
    effective_end = min(requested_end, common_end)
    frequency = pd.Timedelta(seconds=int(config["resample_seconds"]))
    effective_start = effective_start.ceil(frequency)
    effective_end = effective_end.floor(frequency)
    if effective_start >= effective_end:
        raise ValueError(
            f"No common time range: requested {requested_start}..{requested_end}, "
            f"available {common_start}..{common_end}"
        )
    grid = pd.date_range(effective_start, effective_end, freq=frequency)

    raw_values = np.empty((len(grid), len(channel_names)), dtype=np.float32)
    update_mask = np.zeros(raw_values.shape, dtype=np.uint8)
    for column, channel_name in enumerate(channel_names):
        series = series_by_channel[channel_name]
        aligned = series.reindex(grid, method="ffill")
        raw_values[:, column] = aligned.to_numpy(dtype=np.float32)
        updates = series.loc[(series.index > grid[0] - frequency) & (series.index <= grid[-1])]
        update_times = updates.index.ceil(frequency)
        positions = grid.get_indexer(update_times)
        positions = positions[positions >= 0]
        update_mask[np.unique(positions), column] = 1

    label_codes = apply_labels(
        grid, channel_names, labels, anomaly_types, frequency
    )
    finite_mask = np.isfinite(raw_values)
    observed_mask = (finite_mask & (label_codes != 3)).astype(np.uint8)
    clean_mask = (finite_mask & (label_codes == 0)).astype(np.uint8)

    splits, (train_end, valid_end) = split_codes(
        len(grid), [float(value) for value in config["split_ratios"]]
    )
    means = np.empty(len(channel_names), dtype=np.float64)
    stds = np.empty(len(channel_names), dtype=np.float64)
    for column, channel_name in enumerate(channel_names):
        train_clean = (splits == 0) & (clean_mask[:, column] == 1)
        values = raw_values[train_clean, column].astype(np.float64)
        if values.size == 0:
            raise ValueError(f"No clean training observations for {channel_name}")
        means[column] = values.mean()
        stds[column] = values.std()
        if not np.isfinite(stds[column]) or stds[column] <= 1e-12:
            raise ValueError(f"Invalid training standard deviation for {channel_name}")

    normalized_values = ((raw_values - means) / stds).astype(np.float32)
    normalized_values[observed_mask == 0] = 0.0
    timestamps = grid.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    arrays = {
        "timestamps_ns": timestamps,
        "raw_values": raw_values,
        "normalized_values": normalized_values,
        "observed_mask": observed_mask,
        "clean_mask": clean_mask,
        "label_code": label_codes,
        "update_mask": update_mask,
        "split_code": splits,
    }
    content_hash = array_content_hash(arrays)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **arrays)

    extracted_hashes = {
        str(path.relative_to(interim_root)).replace("\\", "/"): sha256_file(path)
        for path in extracted.values()
    }
    split_names = {0: "train", 1: "validation", 2: "test"}
    split_summary = {}
    for code, name in split_names.items():
        indices = np.flatnonzero(splits == code)
        split_summary[name] = {
            "count": int(indices.size),
            "start_time": grid[indices[0]].isoformat(),
            "end_time": grid[indices[-1]].isoformat(),
        }

    manifest = {
        "schema_version": 1,
        "source": {
            "dataset": "ESA Anomaly Dataset",
            "doi": "10.5281/zenodo.12528696",
            "archive": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "selected_file_sha256": extracted_hashes,
        },
        "configuration": {
            **{key: value for key, value in config.items() if key != "archive_path"},
            "archive_path": str(archive),
        },
        "schema": {
            "channel_order": channel_names,
            "array_shape": [len(grid), len(channel_names)],
            "timestamp_unit": "nanoseconds_since_unix_epoch",
            "label_codes": {str(key): value for key, value in LABEL_NAMES.items()},
            "split_codes": {"0": "train", "1": "validation", "2": "test"},
        },
        "time_range": {
            "requested_start": requested_start.isoformat(),
            "requested_end": requested_end.isoformat(),
            "effective_start": grid[0].isoformat(),
            "effective_end": grid[-1].isoformat(),
            "resample_seconds": int(config["resample_seconds"]),
            "timepoint_count": len(grid),
        },
        "splits": split_summary,
        "split_indices": {
            "train": [0, train_end],
            "validation": [train_end, valid_end],
            "test": [valid_end, len(grid)],
        },
        "normalization": {
            "source": "clean training positions only",
            "mean": dict(zip(channel_names, means.tolist())),
            "std": dict(zip(channel_names, stds.tolist())),
        },
        "array_content_sha256": content_hash,
    }
    write_json(manifest_path, manifest)

    per_channel = {}
    for column, channel_name in enumerate(channel_names):
        counts = Counter(int(value) for value in label_codes[:, column])
        per_channel[channel_name] = {
            "raw_start": series_by_channel[channel_name].index.min().isoformat(),
            "raw_end": series_by_channel[channel_name].index.max().isoformat(),
            "observed_fraction": float(observed_mask[:, column].mean()),
            "clean_fraction": float(clean_mask[:, column].mean()),
            "update_fraction": float(update_mask[:, column].mean()),
            "label_counts": {
                LABEL_NAMES[code]: int(counts.get(code, 0)) for code in LABEL_NAMES
            },
            "raw_value_min": float(np.nanmin(raw_values[:, column])),
            "raw_value_max": float(np.nanmax(raw_values[:, column])),
        }
    quality_report = {
        "timepoint_count": len(grid),
        "channel_count": len(channel_names),
        "estimated_scalar_values": len(grid) * len(channel_names),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "per_channel": per_channel,
    }
    write_json(quality_path, quality_report)
    return output_dir


def main() -> None:
    args = parse_args()
    config, project_root = load_config(args.config)
    validate_config(config)
    output_dir = preprocess(config, project_root, args.force)
    print(f"Preprocessing complete: {output_dir}")
    print(f"Artifact: {output_dir / 'aligned.npz'}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Quality report: {output_dir / 'quality_report.json'}")


if __name__ == "__main__":
    main()
