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

from dataset_physio import Physio_Dataset, get_idlist


ATTRIBUTES = [
    "DiasABP", "HR", "Na", "Lactate", "NIDiasABP", "PaO2", "WBC", "pH",
    "Albumin", "ALT", "Glucose", "SaO2", "Temp", "AST", "Bilirubin",
    "HCO3", "BUN", "RespRate", "Mg", "HCT", "SysABP", "FiO2", "K",
    "GCS", "Cholesterol", "NISysABP", "TroponinT", "MAP", "TroponinI",
    "PaCO2", "Platelets", "Urine", "NIMAP", "Creatinine", "ALP",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export CSDI PhysioNet split as BRITS JSONL.")
    parser.add_argument("--missing-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nfold", type=int, default=0)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--outcomes", type=str, default="data/physio/Outcomes-a.txt")
    return parser.parse_args()


def parse_delta(masks, direction):
    if direction == "backward":
        masks = masks[::-1]

    deltas = []
    for h in range(48):
        if h == 0:
            deltas.append(np.ones(35, dtype=np.float32))
        else:
            deltas.append(np.ones(35, dtype=np.float32) + (1 - masks[h]) * deltas[-1])
    return np.asarray(deltas, dtype=np.float32)


def parse_rec(values, masks, evals, eval_masks, direction):
    deltas = parse_delta(masks, direction)
    forwards = pd.DataFrame(values).ffill().fillna(0.0).to_numpy(dtype=np.float32)

    return {
        "values": np.nan_to_num(values, nan=0.0).astype(np.float32).tolist(),
        "masks": masks.astype(np.int32).tolist(),
        "evals": np.nan_to_num(evals, nan=0.0).astype(np.float32).tolist(),
        "eval_masks": eval_masks.astype(np.int32).tolist(),
        "forwards": forwards.astype(np.float32).tolist(),
        "deltas": deltas.astype(np.float32).tolist(),
    }


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
    return outcomes["In-hospital_death"].astype(float).to_dict()


def has_selected_physio_signal(record_id):
    path = REPO_ROOT / "data" / "physio" / "set-a" / f"{record_id}.txt"
    data = pd.read_csv(path, usecols=["Parameter"])
    return data["Parameter"].isin(ATTRIBUTES).any()


def get_successful_idlist():
    ids = [str(x) for x in get_idlist()]
    return np.asarray([id_ for id_ in ids if has_selected_physio_signal(id_)])


def record_for_index(dataset, ids, labels, index):
    observed_values = dataset.observed_values[index].astype(np.float32)
    observed_masks = dataset.observed_masks[index].astype(np.float32)
    gt_masks = dataset.gt_masks[index].astype(np.float32)

    values = observed_values.copy()
    values[gt_masks == 0] = np.nan
    evals = observed_values.copy()
    masks = gt_masks.astype(bool)
    eval_masks = (observed_masks - gt_masks).astype(bool)

    record_id = int(ids[index])
    label = labels[record_id]

    return {
        "record_id": record_id,
        "label": label,
        "forward": parse_rec(values, masks, evals, eval_masks, "forward"),
        "backward": parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], "backward"),
    }


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def main():
    args = parse_args()
    outdir = args.outdir
    if outdir is None:
        ratio = str(args.missing_ratio)
        outdir = f"baseline_results/brits/physio_missing{ratio}_seed{args.seed}_fold{args.nfold}"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(args.outcomes):
        raise FileNotFoundError(f"Missing outcomes file: {args.outcomes}")

    raw_ids = get_idlist()
    ids = get_successful_idlist()
    dataset = Physio_Dataset(missing_ratio=args.missing_ratio, seed=args.seed)
    if len(ids) != len(dataset):
        raise ValueError(
            f"Successful id count {len(ids)} does not match CSDI dataset length {len(dataset)}"
        )
    labels = load_outcomes(args.outcomes)
    train_index, valid_index, test_index = split_indices(len(dataset), args.seed, args.nfold)

    missing_labels = [int(ids[i]) for i in range(len(ids)) if int(ids[i]) not in labels]
    if missing_labels:
        raise ValueError(f"Outcomes-a.txt is missing labels for {len(missing_labels)} records")

    split_to_indices = {
        "train": train_index,
        "valid": valid_index,
        "test": test_index,
    }

    summary = {
        "missing_ratio": args.missing_ratio,
        "seed": args.seed,
        "nfold": args.nfold,
        "num_records": int(len(dataset)),
        "raw_set_a_records": int(len(raw_ids)),
        "skipped_record_ids": [int(x) for x in raw_ids if str(x) not in set(ids.tolist())],
        "splits": {},
    }
    for split, indices in split_to_indices.items():
        records = [record_for_index(dataset, ids, labels, int(i)) for i in indices]
        write_jsonl(outdir / f"{split}.jsonl", records)
        eval_points = sum(
            np.asarray(rec["forward"]["eval_masks"], dtype=np.float32).sum()
            for rec in records
        )
        summary["splits"][split] = {
            "num_records": int(len(records)),
            "eval_points": int(eval_points),
            "first_record_id": int(records[0]["record_id"]) if records else None,
        }

    with open(outdir / "data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
