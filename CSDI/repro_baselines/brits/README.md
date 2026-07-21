# BRITS Healthcare Reproduction

This folder adapts the original BRITS implementation to the CSDI healthcare
imputation protocol without changing the BRITS model equations.

## Source

- Original BRITS implementation: `external/BRITS`
- Upstream commit used at clone time: record this with
  `git -C external/BRITS rev-parse HEAD`
- Official label file: `data/physio/Outcomes-a.txt`

The local changes under `external/BRITS` are Python 3 compatibility patches:
optional `ipdb`, `ujson` fallback, `map(...)` materialization, pandas
`.to_numpy()`, and Python 3 reverse indices. They do not alter the BRITS model
loss or forward pass.

## Prepare 10% Missing Data

```powershell
python repro_baselines\brits\prepare_brits_physio.py `
  --missing-ratio 0.1 `
  --seed 1 `
  --nfold 0
```

Output:

```text
baseline_results/brits/physio_missing0.1_seed1_fold0/
  train.jsonl
  valid.jsonl
  test.jsonl
  data_manifest.json
```

The raw PhysioNet `set-a` folder contains 4000 files, but CSDI's parser uses
3997 records because `140501`, `140936`, and `141264` contain no selected
time-varying signals from the 35-feature list. The manifest records this.

## Smoke Test

```powershell
python repro_baselines\brits\run_brits_physio.py `
  --data-dir baseline_results\brits\physio_missing0.1_seed1_fold0 `
  --output-dir baseline_results\brits\physio_missing0.1_seed1_fold0\smoke_epoch1 `
  --epochs 1 `
  --batch-size 64 `
  --hid-size 108 `
  --impute-weight 0.3 `
  --label-weight 1.0 `
  --seed 1 `
  --device cuda:0
```

The smoke test only checks the pipeline. It is not a reproduction result.

## Full 10% Run

```powershell
python repro_baselines\brits\run_brits_physio.py `
  --data-dir baseline_results\brits\physio_missing0.1_seed1_fold0 `
  --output-dir baseline_results\brits\physio_missing0.1_seed1_fold0\run_epoch1000 `
  --epochs 1000 `
  --batch-size 64 `
  --hid-size 108 `
  --impute-weight 0.3 `
  --label-weight 1.0 `
  --seed 1 `
  --device cuda:0
```

The target CSDI paper Table 3 value for BRITS at 10% missing is MAE `0.284`.
The runner writes `metrics.json`, `run_manifest.json`, and model checkpoints to
the output directory.

## Early-Stopping Grid

The complete early-stopping grid reuses the existing 10% fold0 result and runs
the remaining 14 jobs:

```powershell
python repro_baselines\brits\run_brits_grid.py `
  --max-epochs 300 `
  --patience 30 `
  --min-delta 0.0 `
  --seed 1 `
  --device cuda:0
```

Each new run writes to:

```text
baseline_results/brits/physio_missing{ratio}_seed1_fold{fold}/run_patience30/
```

Summarize completed folds at any time:

```powershell
python repro_baselines\brits\summarize_brits_results.py --seed 1
```

Summary CSV:

```text
baseline_results/brits/summary_brits_healthcare.csv
```
