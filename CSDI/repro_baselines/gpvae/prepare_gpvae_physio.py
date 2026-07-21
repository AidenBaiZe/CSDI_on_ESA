import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_physio import Physio_Dataset, attributes, get_idlist


def parse_args():
    parser = argparse.ArgumentParser(description="Export CSDI PhysioNet split as GP-VAE NPZ.")
    parser.add_argument("--missing-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nfold", type=int, default=0)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--outcomes", type=str, default="data/physio/Outcomes-a.txt")
    parser.add_argument(
        "--train-mask-mode",
        choices=["gt", "observed", "random"],
        default="gt",
        help="Use gt masks, natural observed masks, or CSDI-style random masks for train/valid GP-VAE inputs.",
    )
    return parser.parse_args()


def split_indices(num_records, seed, nfold):
    indlist = np.arange(num_records)
    np.random.seed(seed)
    np.random.shuffle(indlist)

    start = int(nfold * 0.2 * num_records)
    end = int((nfold + 1) * 0.2 * num_records)
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = int(num_records * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]
    return train_index, valid_index, test_index


def load_outcomes(path):
    outcomes = pd.read_csv(path).set_index("RecordID")
    return outcomes["In-hospital_death"].astype(np.float32).to_dict()


def has_selected_physio_signal(record_id):
    path = REPO_ROOT / "data" / "physio" / "set-a" / f"{record_id}.txt"
    data = pd.read_csv(path, usecols=["Parameter"])
    return data["Parameter"].isin(attributes).any()


def get_successful_idlist():
    ids = [str(x) for x in get_idlist()]
    return np.asarray([id_ for id_ in ids if has_selected_physio_signal(id_)])


def make_random_cond_mask(observed_mask, rng):
    cond_mask = observed_mask.copy()
    flat_observed = observed_mask.reshape(observed_mask.shape[0], -1)
    flat_cond = cond_mask.reshape(cond_mask.shape[0], -1)
    for i in range(flat_observed.shape[0]):
        obs_indices = np.flatnonzero(flat_observed[i] > 0)
        if len(obs_indices) == 0:
            continue
        sample_ratio = rng.rand()
        num_masked = int(round(len(obs_indices) * sample_ratio))
        if num_masked > 0:
            masked = rng.choice(obs_indices, size=num_masked, replace=False)
            flat_cond[i, masked] = 0.0
    return cond_mask


def build_arrays(dataset, indices, ids, labels, input_mask_mode="gt", rng=None):
    indices = np.asarray(indices, dtype=np.int64)
    observed = dataset.observed_values[indices].astype(np.float32)
    observed_mask = dataset.observed_masks[indices].astype(np.float32)
    gt_mask = dataset.gt_masks[indices].astype(np.float32)
    artificial_mask = (observed_mask - gt_mask).astype(np.float32)
    if input_mask_mode == "gt":
        input_mask = gt_mask
    elif input_mask_mode == "observed":
        input_mask = observed_mask
    elif input_mask_mode == "random":
        if rng is None:
            raise ValueError("random input_mask_mode requires rng")
        input_mask = make_random_cond_mask(observed_mask, rng)
    else:
        raise ValueError(f"Unknown input_mask_mode: {input_mask_mode}")

    x_full = observed.copy()
    x_miss = observed * input_mask
    missing_mask = (1.0 - input_mask).astype(np.float32)
    if input_mask_mode == "random":
        artificial_mask = (observed_mask - input_mask).astype(np.float32)
    y = np.asarray([labels[int(ids[i])] for i in indices], dtype=np.float32)
    record_ids = np.asarray([int(ids[i]) for i in indices], dtype=np.int64)

    return {
        "x_full": x_full,
        "x_miss": x_miss,
        "m_miss": missing_mask,
        "m_artificial": artificial_mask,
        "observed_mask": observed_mask,
        "gt_mask": gt_mask,
        "y": y,
        "record_ids": record_ids,
    }


def add_split_npz(payload, split_name, arrays, gpvae_name=None):
    name = gpvae_name or split_name
    payload[f"x_{name}_full"] = arrays["x_full"]
    payload[f"x_{name}_miss"] = arrays["x_miss"]
    payload[f"m_{name}_miss"] = arrays["m_miss"]
    payload[f"m_{name}_artificial"] = arrays["m_artificial"]
    payload[f"y_{name}"] = arrays["y"]
    payload[f"record_ids_{split_name}"] = arrays["record_ids"]
    payload[f"observed_mask_{split_name}"] = arrays["observed_mask"]
    payload[f"gt_mask_{split_name}"] = arrays["gt_mask"]


def main():
    args = parse_args()
    if args.outdir is None:
        ratio = str(args.missing_ratio)
        args.outdir = f"baseline_results/gpvae/physio_missing{ratio}_seed{args.seed}_fold{args.nfold}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outcomes_path = REPO_ROOT / args.outcomes
    if not outcomes_path.is_file():
        raise FileNotFoundError(f"Missing outcomes file: {outcomes_path}")

    raw_ids = get_idlist()
    ids = get_successful_idlist()
    dataset = Physio_Dataset(missing_ratio=args.missing_ratio, seed=args.seed)
    if len(ids) != len(dataset):
        raise ValueError(f"Successful id count {len(ids)} does not match dataset length {len(dataset)}")

    labels = load_outcomes(outcomes_path)
    missing_labels = [int(id_) for id_ in ids if int(id_) not in labels]
    if missing_labels:
        raise ValueError(f"Outcomes-a.txt is missing labels for {len(missing_labels)} records")

    train_index, valid_index, test_index = split_indices(len(dataset), args.seed, args.nfold)
    train_rng = np.random.RandomState(args.seed + 1000 * args.nfold + 17)
    valid_rng = np.random.RandomState(args.seed + 1000 * args.nfold + 53)
    train = build_arrays(dataset, train_index, ids, labels, args.train_mask_mode, train_rng)
    valid = build_arrays(dataset, valid_index, ids, labels, args.train_mask_mode, valid_rng)
    test = build_arrays(dataset, test_index, ids, labels, "gt")

    payload = {}
    add_split_npz(payload, "train", train)
    add_split_npz(payload, "valid", valid)
    add_split_npz(payload, "test", test)
    # Original GP-VAE train.py uses the arrays named val during training logs.
    # Keep that as CSDI's validation split; the runner evaluates x_test_* after
    # training so the test split is not used for training-time diagnostics.
    add_split_npz(payload, "valid", valid, gpvae_name="val")

    npz_path = outdir / "physionet_gpvae.npz"
    np.savez_compressed(npz_path, **payload)

    summary = {
        "missing_ratio": args.missing_ratio,
        "seed": args.seed,
        "nfold": args.nfold,
        "train_mask_mode": args.train_mask_mode,
        "npz_path": str(npz_path.resolve()),
        "num_records": int(len(dataset)),
        "raw_set_a_records": int(len(raw_ids)),
        "skipped_record_ids": [int(x) for x in raw_ids if str(x) not in set(ids.tolist())],
        "mask_semantics": {
            "m_*_miss": "1 means missing/hidden from GP-VAE reconstruction loss",
            "m_*_artificial": "1 means CSDI artificial evaluation target, observed_mask - gt_mask",
        },
        "splits": {
            "train": {
                "num_records": int(len(train_index)),
                "eval_points": int(train["m_artificial"].sum()),
                "first_record_id": int(train["record_ids"][0]),
            },
            "valid": {
                "num_records": int(len(valid_index)),
                "eval_points": int(valid["m_artificial"].sum()),
                "first_record_id": int(valid["record_ids"][0]),
            },
            "test": {
                "num_records": int(len(test_index)),
                "eval_points": int(test["m_artificial"].sum()),
                "first_record_id": int(test["record_ids"][0]),
            },
        },
    }
    with open(outdir / "data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
