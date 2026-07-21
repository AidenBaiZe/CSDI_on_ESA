from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dateutil.parser import parse as parse_date
from numpy.lib.format import open_memmap
from tqdm import tqdm


LABEL_NAMES = {
    0: "nominal",
    1: "anomaly",
    2: "rare_event",
    3: "communication_gap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess all ESA-ADB Mission 2 telemetry channels for CSDI."
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-archive-hash",
        action="store_true",
        help="Skip the expensive whole-archive SHA256 pass (intended only for tests).",
    )
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, path


def resolve_path(value: str, config_path: Path) -> Path:
    return (config_path.parent / value).resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_channel_entries(
    archive: Path, mission: str, expected_count: int
) -> tuple[list[int], dict[int, zipfile.ZipInfo]]:
    channel_re = re.compile(rf"^{re.escape(mission)}/channels/channel_(\d+)\.zip$")
    with zipfile.ZipFile(archive) as outer:
        entries: dict[int, zipfile.ZipInfo] = {}
        for info in outer.infolist():
            match = channel_re.match(info.filename)
            if match:
                entries[int(match.group(1))] = info
    channel_ids = sorted(entries)
    if len(channel_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {mission} telemetry channels, found "
            f"{len(channel_ids)}: {channel_ids}"
        )
    expected_ids = list(range(1, expected_count + 1))
    if channel_ids != expected_ids:
        raise ValueError(
            f"{mission} channel identifiers are not the expected archive-defined "
            f"sequence {expected_ids}; found {channel_ids}"
        )
    return channel_ids, entries


def extract_channel_archives(
    archive: Path,
    interim_dir: Path,
    channel_ids: list[int],
    entries: dict[int, zipfile.ZipInfo],
) -> dict[int, Path]:
    interim_dir.mkdir(parents=True, exist_ok=True)
    paths = {channel: interim_dir / f"channel_{channel}.zip" for channel in channel_ids}
    with zipfile.ZipFile(archive) as outer:
        for channel in tqdm(channel_ids, desc="extract channel archives"):
            target = paths[channel]
            info = entries[channel]
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            temporary = target.with_suffix(".zip.partial")
            if temporary.exists():
                temporary.unlink()
            try:
                with outer.open(info) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            except NotImplementedError as exc:
                raise RuntimeError(
                    f"The outer ZIP compression for {info.filename} is unsupported. "
                    "Channel entries are expected to be stored without compression."
                ) from exc
            if temporary.stat().st_size != info.file_size:
                raise IOError(
                    f"Extracted size mismatch for {info.filename}: "
                    f"{temporary.stat().st_size} != {info.file_size}"
                )
            temporary.replace(target)
    return paths


def _read_csv_from_archive_or_fallback(
    archive: Path, entry_name: str, fallback: Path
) -> tuple[pd.DataFrame, str]:
    try:
        with zipfile.ZipFile(archive) as outer:
            with outer.open(entry_name) as handle:
                return pd.read_csv(handle), f"archive:{entry_name}"
    except (NotImplementedError, RuntimeError, zipfile.BadZipFile):
        if not fallback.is_file():
            raise FileNotFoundError(
                f"Cannot read {entry_name} from the outer archive and fallback is missing: "
                f"{fallback}"
            )
        return pd.read_csv(fallback), str(fallback)


def load_annotation_tables(
    archive: Path, mission: str, labels_fallback: Path, anomaly_types_fallback: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    labels, labels_source = _read_csv_from_archive_or_fallback(
        archive, f"{mission}/labels.csv", labels_fallback
    )
    anomaly_types, types_source = _read_csv_from_archive_or_fallback(
        archive, f"{mission}/anomaly_types.csv", anomaly_types_fallback
    )
    required_labels = {"Channel", "ID", "StartTime", "EndTime"}
    required_types = {"ID", "Category"}
    if not required_labels.issubset(labels.columns):
        raise ValueError(f"labels.csv is missing {sorted(required_labels - set(labels.columns))}")
    if not required_types.issubset(anomaly_types.columns):
        raise ValueError(
            f"anomaly_types.csv is missing {sorted(required_types - set(anomaly_types.columns))}"
        )
    for column in ("StartTime", "EndTime"):
        labels[column] = labels[column].map(
            lambda value: pd.Timestamp(parse_date(str(value), ignoretz=True))
        )
    return labels, anomaly_types, {
        "labels": labels_source,
        "anomaly_types": types_source,
    }


def category_code(category: str) -> int:
    """Mission 2 annotations contain only anomalies and rare events.

    The official ESA-ADB Mission 2 preprocessing script has no communication-gap
    branch, so any other category would silently change semantics; fail loudly
    instead of guessing.
    """
    normalized = str(category).strip().lower()
    if normalized == "anomaly":
        return 1
    if normalized == "rare event":
        return 2
    raise ValueError(f"Unexpected Mission 2 anomaly category: {category!r}")


def read_channel(path: Path, channel_name: str) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    loaded = pd.read_pickle(path)
    if isinstance(loaded, pd.Series):
        values = loaded
    elif isinstance(loaded, pd.DataFrame):
        if channel_name in loaded.columns:
            values = loaded[channel_name]
        elif loaded.shape[1] == 1:
            values = loaded.iloc[:, 0]
        else:
            raise ValueError(f"{path} has unexpected columns: {loaded.columns.to_list()}")
    else:
        raise TypeError(f"{path} contains {type(loaded).__name__}, expected Series/DataFrame")

    index = pd.to_datetime(values.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    values = pd.Series(values.to_numpy(), index=index)
    values = values[~values.index.duplicated(keep="last")].sort_index()
    if values.empty:
        raise ValueError(f"{channel_name} is empty")

    factorization: dict[str, Any] | None = None
    if values.dtype == object:
        codes, uniques = pd.factorize(values.to_numpy())
        factorization = {
            "unique_count": int(len(uniques)),
            "categories": [str(value) for value in uniques[:64]],
            "categories_truncated": bool(len(uniques) > 64),
        }
        numeric = codes.astype(np.float64)
    else:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    frame = pd.DataFrame({"value": numeric}, index=values.index)
    return frame, factorization


def apply_derivative(frame: pd.DataFrame, channel_id: int, low: int, high: int) -> pd.DataFrame:
    result = frame.copy()
    if low <= channel_id <= high:
        values = result["value"].to_numpy(dtype=np.float64, copy=True)
        result["value"] = np.diff(values, append=values[-1])
    return result


def label_raw_samples(
    frame: pd.DataFrame,
    channel_name: str,
    labels: pd.DataFrame,
    type_by_id: dict[str, int],
) -> pd.DataFrame:
    result = frame.copy()
    raw_labels = np.zeros(len(result), dtype=np.uint8)
    channel_rows = labels[labels["Channel"].astype(str) == channel_name]
    index_ns = result.index.asi8
    for row in channel_rows.itertuples(index=False):
        event_id = str(row.ID)
        if event_id not in type_by_id:
            raise ValueError(f"Missing anomaly category for event {event_id}")
        left = int(np.searchsorted(index_ns, pd.Timestamp(row.StartTime).value, side="left"))
        right = int(np.searchsorted(index_ns, pd.Timestamp(row.EndTime).value, side="right"))
        if left < right:
            raw_labels[left:right] = type_by_id[event_id]
    result["label"] = raw_labels
    return result


def restoration_points(
    raw_index: pd.DatetimeIndex,
    raw_values: np.ndarray,
    raw_labels: np.ndarray,
    frequency: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of ESA-ADB's annotated-sample restoration loop."""
    if len(raw_index) < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.uint8),
        )
    freq_ns = int(frequency.value)
    index_ns = raw_index.asi8
    bin_ns = (index_ns // freq_ns) * freq_ns
    starts = np.r_[0, np.flatnonzero(np.diff(bin_ns)) + 1]
    ends = np.r_[starts[1:], len(bin_ns)]
    group_bins = bin_ns[starts]
    counts = ends - starts

    abnormal_positions = np.flatnonzero((raw_labels == 1) | (raw_labels == 2))
    if abnormal_positions.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.uint8),
        )
    abnormal_bins = bin_ns[abnormal_positions]
    _, reverse_first = np.unique(abnormal_bins[::-1], return_index=True)
    last_abnormal_positions = abnormal_positions[::-1][reverse_first]
    candidate_bins = bin_ns[last_abnormal_positions]
    order = np.argsort(candidate_bins)
    candidate_bins = candidate_bins[order]
    last_abnormal_positions = last_abnormal_positions[order]

    group_index = np.searchsorted(group_bins, candidate_bins)
    in_range = group_index < len(group_bins)
    group_index_safe = np.minimum(group_index, len(group_bins) - 1)
    valid = (
        in_range
        & (group_bins[group_index_safe] == candidate_bins)
        & (counts[group_index_safe] > 1)
        & (raw_labels[ends[group_index_safe] - 1] == 0)
    )
    positions = last_abnormal_positions[valid]
    return (
        candidate_bins[valid] + freq_ns,
        raw_values[positions].astype(np.float64, copy=False),
        raw_labels[positions].astype(np.uint8, copy=False),
    )


def resample_official(
    raw: pd.DataFrame,
    global_index: pd.DatetimeIndex,
    frequency: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if raw.empty:
        raise ValueError("Cannot resample an empty split")
    first = raw.index[0].floor(frequency)
    last = raw.index[-1].ceil(frequency)
    local_index = pd.date_range(first, last, freq=frequency)
    local = raw.reindex(local_index, method="ffill")
    local.iloc[0] = raw.iloc[0]

    restore_ns, restore_values, restore_labels = restoration_points(
        raw.index,
        raw["value"].to_numpy(dtype=np.float64, copy=False),
        raw["label"].to_numpy(dtype=np.uint8, copy=False),
        frequency,
    )
    if restore_ns.size:
        restore_index = pd.to_datetime(restore_ns)
        local = local.reindex(local.index.union(restore_index).sort_values())
        local.loc[restore_index, "value"] = restore_values
        local.loc[restore_index, "label"] = restore_labels

    global_frame = local.reindex(global_index).ffill().bfill()
    values = global_frame["value"].to_numpy(dtype=np.float64, copy=True)
    labels = global_frame["label"].to_numpy(dtype=np.uint8, copy=True)

    update = np.zeros(len(global_index), dtype=np.uint8)
    freq_ns = int(frequency.value)
    raw_ns = raw.index.asi8
    update_ns = ((raw_ns + freq_ns - 1) // freq_ns) * freq_ns
    all_update_ns = np.unique(np.concatenate([update_ns, restore_ns]))
    positions = np.searchsorted(global_index.asi8, all_update_ns)
    valid_positions = (positions >= 0) & (positions < len(global_index))
    positions = positions[valid_positions]
    matching = global_index.asi8[positions] == all_update_ns[valid_positions]
    update[positions[matching]] = 1

    leading_bfill = max(0, int((first - global_index[0]) / frequency))
    metadata = {
        "raw_start": raw.index[0].isoformat(),
        "raw_end": raw.index[-1].isoformat(),
        "local_resampled_start": first.isoformat(),
        "local_resampled_end": last.isoformat(),
        "restored_annotation_points": int(restore_ns.size),
        "leading_bfill_points": leading_bfill,
        "leading_bfill_start": global_index[0].isoformat() if leading_bfill else None,
        "leading_bfill_end": (
            global_index[leading_bfill - 1].isoformat() if leading_bfill else None
        ),
    }
    return values, labels, update, metadata


def longest_zero_run(mask: np.ndarray) -> int:
    nonzero = np.flatnonzero(mask)
    if nonzero.size == 0:
        return int(mask.size)
    boundaries = np.r_[-1, nonzero, mask.size]
    return int(np.max(np.diff(boundaries) - 1))


def compute_normalization(
    clean_values: np.ndarray, std_floor_threshold: float
) -> tuple[float, float, float, bool]:
    if clean_values.size == 0:
        raise ValueError("Cannot normalize without clean training values")
    mean = float(np.mean(clean_values, dtype=np.float64))
    raw_std = float(np.std(clean_values, dtype=np.float64))
    near_constant = (not math.isfinite(raw_std)) or raw_std <= std_floor_threshold
    scale = 1.0 if near_constant else raw_std
    return mean, raw_std, scale, near_constant


def classify_channel(
    clean_values: np.ndarray,
    channel_id: int,
    discrete_unique_threshold: int,
    differenced_low: int,
    differenced_high: int,
) -> tuple[int, bool, bool]:
    unique_count = int(np.unique(clean_values).size)
    discrete_like = unique_count <= discrete_unique_threshold
    differenced = differenced_low <= int(channel_id) <= differenced_high
    return unique_count, discrete_like, differenced


def create_split_arrays(split_dir: Path, channels: int, length: int) -> dict[str, np.memmap]:
    split_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "normalized_values": open_memmap(
            split_dir / "normalized_values.npy", mode="w+", dtype=np.float32, shape=(channels, length)
        ),
        "observed_mask": open_memmap(
            split_dir / "observed_mask.npy", mode="w+", dtype=np.uint8, shape=(channels, length)
        ),
        "update_mask": open_memmap(
            split_dir / "update_mask.npy", mode="w+", dtype=np.uint8, shape=(channels, length)
        ),
        "label_code": open_memmap(
            split_dir / "label_code.npy", mode="w+", dtype=np.uint8, shape=(channels, length)
        ),
        "observed_count": open_memmap(
            split_dir / "observed_count.npy", mode="w+", dtype=np.uint8, shape=(length,)
        ),
    }
    arrays["observed_count"][:] = 0
    return arrays


def flush_arrays(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def build_time_index(start: pd.Timestamp, end: pd.Timestamp, frequency: pd.Timedelta) -> pd.DatetimeIndex:
    index = pd.date_range(start, end, freq=frequency)
    if len(index) <= 0:
        raise ValueError(f"Empty official index {start}..{end}")
    return index


def preprocess(config: dict[str, Any], config_path: Path, force: bool, skip_hash: bool) -> Path:
    data = config["data"]
    mission = str(data["mission"])
    archive = resolve_path(data["archive"], config_path)
    interim_dir = resolve_path(data["interim_dir"], config_path)
    output_dir = resolve_path(data["processed_dir"], config_path)
    manifest_path = output_dir / "manifest.json"
    if not archive.is_file():
        raise FileNotFoundError(f"{mission} archive not found: {archive}")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Processed output already exists: {output_dir}; use --force to replace")
        shutil.rmtree(output_dir)

    expected_count = int(data["expected_channel_count"])
    differenced_low, differenced_high = (
        int(value) for value in data["differenced_channel_range"]
    )
    channel_ids, entries = discover_channel_entries(archive, mission, expected_count)
    channel_paths = extract_channel_archives(
        archive, interim_dir, channel_ids, entries
    )
    labels, anomaly_types, metadata_sources = load_annotation_tables(
        archive,
        mission,
        resolve_path(data["labels_fallback"], config_path),
        resolve_path(data["anomaly_types_fallback"], config_path),
    )
    type_by_id = {
        str(row.ID): category_code(row.Category)
        for row in anomaly_types.itertuples(index=False)
    }

    frequency = pd.Timedelta(seconds=int(data["resample_seconds"]))
    split_at = pd.Timestamp(data["split_at"])
    official_start = pd.Timestamp(data["official_start"])
    official_end = pd.Timestamp(data["official_end"])
    train_index = build_time_index(official_start, split_at, frequency)
    test_index = build_time_index(split_at, official_end, frequency)
    expected_points = {
        "train": int(data["expected_train_timepoints"]),
        "test": int(data["expected_test_timepoints"]),
    }
    for name, index in (("train", train_index), ("test", test_index)):
        if len(index) != expected_points[name]:
            raise ValueError(
                f"Official {name} index has {len(index)} points, expected {expected_points[name]}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_arrays = create_split_arrays(output_dir / "train", expected_count, len(train_index))
    test_arrays = create_split_arrays(output_dir / "test", expected_count, len(test_index))
    train_timestamps = open_memmap(
        output_dir / "train" / "timestamps_ns.npy",
        mode="w+",
        dtype=np.int64,
        shape=(len(train_index),),
    )
    test_timestamps = open_memmap(
        output_dir / "test" / "timestamps_ns.npy",
        mode="w+",
        dtype=np.int64,
        shape=(len(test_index),),
    )
    train_timestamps[:] = train_index.asi8
    test_timestamps[:] = test_index.asi8
    train_timestamps.flush()
    test_timestamps.flush()

    means = np.zeros(expected_count, dtype=np.float64)
    raw_stds = np.zeros(expected_count, dtype=np.float64)
    scales = np.zeros(expected_count, dtype=np.float64)
    channel_manifest: dict[str, Any] = {}
    start_time = time.perf_counter()

    for column, channel_id in enumerate(tqdm(channel_ids, desc="preprocess channels")):
        channel_name = f"channel_{channel_id}"
        raw, factorization = read_channel(channel_paths[channel_id], channel_name)
        raw = apply_derivative(raw, channel_id, differenced_low, differenced_high)
        raw = label_raw_samples(raw, channel_name, labels, type_by_id)
        train_raw = raw[raw.index <= split_at]
        test_raw = raw[raw.index > split_at]
        if train_raw.empty or test_raw.empty:
            raise ValueError(
                f"{channel_name} does not cover both official ESA-ADB partitions"
            )

        train_values, train_labels, train_update, train_meta = resample_official(
            train_raw, train_index, frequency
        )
        test_values, test_labels, test_update, test_meta = resample_official(
            test_raw, test_index, frequency
        )
        train_observed = (np.isfinite(train_values) & (train_labels != 3)).astype(np.uint8)
        test_observed = (np.isfinite(test_values) & (test_labels != 3)).astype(np.uint8)
        clean = np.isfinite(train_values) & (train_labels == 0)
        clean_values = train_values[clean]
        if clean_values.size == 0:
            raise ValueError(f"No clean training observations for {channel_name}")

        mean, raw_std, scale, near_constant = compute_normalization(
            clean_values, float(data["std_floor_threshold"])
        )
        unique_count, discrete_like, differenced = classify_channel(
            clean_values,
            channel_id,
            int(data["discrete_unique_threshold"]),
            differenced_low,
            differenced_high,
        )
        means[column] = mean
        raw_stds[column] = raw_std
        scales[column] = scale

        for split_arrays, values, split_labels, split_update, split_observed in (
            (train_arrays, train_values, train_labels, train_update, train_observed),
            (test_arrays, test_values, test_labels, test_update, test_observed),
        ):
            normalized = ((values - mean) / scale).astype(np.float32)
            normalized[split_observed == 0] = 0.0
            split_arrays["normalized_values"][column] = normalized
            split_arrays["observed_mask"][column] = split_observed
            split_arrays["update_mask"][column] = split_update
            split_arrays["label_code"][column] = split_labels
            split_arrays["observed_count"][:] = (
                split_arrays["observed_count"][:] + split_observed
            )

        channel_manifest[channel_name] = {
            "channel_id": channel_id,
            "differenced": differenced,
            "value_type": "discrete_like" if discrete_like else "continuous",
            "factorized": factorization is not None,
            "factorization": factorization,
            "clean_unique_count": unique_count,
            "clean_mean": mean,
            "clean_raw_std": raw_std,
            "normalization_scale": scale,
            "near_constant": near_constant,
            "train": {
                **train_meta,
                "observed_fraction": float(train_observed.mean()),
                "update_fraction": float(train_update.mean()),
                "longest_hold_points": longest_zero_run(train_update),
                "longest_hold_seconds": longest_zero_run(train_update)
                * int(data["resample_seconds"]),
                "label_counts": {
                    LABEL_NAMES[key]: int(value)
                    for key, value in Counter(train_labels.tolist()).items()
                },
            },
            "test": {
                **test_meta,
                "observed_fraction": float(test_observed.mean()),
                "update_fraction": float(test_update.mean()),
                "longest_hold_points": longest_zero_run(test_update),
                "longest_hold_seconds": longest_zero_run(test_update)
                * int(data["resample_seconds"]),
                "label_counts": {
                    LABEL_NAMES[key]: int(value)
                    for key, value in Counter(test_labels.tolist()).items()
                },
            },
        }
        del raw, train_raw, test_raw, clean_values
        del train_values, train_labels, train_update, train_observed
        del test_values, test_labels, test_update, test_observed

        if (column + 1) % 4 == 0:
            flush_arrays(train_arrays)
            flush_arrays(test_arrays)
            write_json(
                output_dir / "preprocess_progress.json",
                {
                    "completed_channels": column + 1,
                    "last_channel": channel_name,
                    "elapsed_seconds": time.perf_counter() - start_time,
                },
            )

    flush_arrays(train_arrays)
    flush_arrays(test_arrays)
    archive_sha256 = None if skip_hash else sha256_file(archive)
    channel_names = [f"channel_{value}" for value in channel_ids]
    factorized_channels = [
        name for name, metadata in channel_manifest.items() if metadata["factorized"]
    ]
    manifest = {
        "schema_version": 2,
        "source": {
            "dataset": f"ESA Anomaly Dataset / ESA-ADB {mission}",
            "archive": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": archive_sha256,
            "metadata_sources": metadata_sources,
            "channel_entries": {
                f"channel_{channel}": {
                    "archive_entry": entries[channel].filename,
                    "size_bytes": entries[channel].file_size,
                    "crc32": f"{entries[channel].CRC:08x}",
                }
                for channel in channel_ids
            },
        },
        "esa_adb_compatibility": {
            "official_input_dimensions": 104,
            "experiment_input_dimensions": expected_count,
            "telecommands_included": False,
            "split_at": split_at.isoformat(),
            "train_filter": "raw timestamp <= split_at",
            "test_filter": "raw timestamp > split_at",
            "resample": "18-second zero-order hold, global ffill().bfill()",
            "anomaly_sample_restoration": True,
            "communication_gap_labels": (
                "Mission 2 annotations contain only anomaly and rare-event categories; "
                "the official preprocessing has no communication-gap branch, so natural "
                "missingness can only come from non-finite values"
            ),
            "string_channels": "factorized to categorical integers like the official script",
            "differenced_channel_ids": list(range(differenced_low, differenced_high + 1)),
        },
        "schema": {
            "layout": "channel_first",
            "channel_order": channel_names,
            "channel_count": expected_count,
            "label_codes": {str(key): value for key, value in LABEL_NAMES.items()},
            "update_mask": (
                "1 for original raw updates and restored anomaly/rare samples; "
                "0 for zero-order-held or leading bfill values"
            ),
            "observed_mask": "finite value and label_code != communication_gap",
        },
        "partitions": {
            "train": {
                "path": "train",
                "start": train_index[0].isoformat(),
                "end": train_index[-1].isoformat(),
                "timepoints": len(train_index),
                "raw_filter": f"timestamp <= {split_at.isoformat()}",
            },
            "test": {
                "path": "test",
                "start": test_index[0].isoformat(),
                "end": test_index[-1].isoformat(),
                "timepoints": len(test_index),
                "raw_filter": f"timestamp > {split_at.isoformat()}",
            },
        },
        "normalization": {
            "source": "finite nominal points from the official training partition only",
            "formula": "(x - clean_mean) / normalization_scale",
            "near_constant_threshold": float(data["std_floor_threshold"]),
            "near_constant_scale": 1.0,
            "mean": dict(zip(channel_names, means.tolist())),
            "raw_std": dict(zip(channel_names, raw_stds.tolist())),
            "scale": dict(zip(channel_names, scales.tolist())),
        },
        "channel_classification": {
            "discrete_like_rule": (
                f"clean training exact unique count <= {int(data['discrete_unique_threshold'])}"
            ),
            "differenced_is_orthogonal": True,
            "factorized_channels": factorized_channels,
        },
        "channels": channel_manifest,
        "duration_seconds": time.perf_counter() - start_time,
    }
    write_json(manifest_path, manifest)
    progress = output_dir / "preprocess_progress.json"
    if progress.exists():
        progress.unlink()
    return output_dir


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    output_dir = preprocess(
        config, config_path, force=args.force, skip_hash=args.skip_archive_hash
    )
    print(f"Preprocessing complete: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
