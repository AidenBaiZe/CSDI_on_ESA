# ESA-CSDI baseline experiment

This directory connects the isolated ESA preprocessing artifact to the original
CSDI model without modifying the existing PhysioNet or PM2.5 reproduction.

## Protocol

- ESA Mission 1 channels 41-46;
- 96 timepoints (48 minutes) per window, stride 48;
- windows containing any anomaly, rare event or communication gap are excluded;
- training uses the original CSDI random 0%-100% target strategy;
- validation uses 50% missingness;
- test missing ratios are 10%, 50% and 90%;
- the original CSDI architecture is retained with `target_dim=6`.

## Validate the loader

```powershell
python -m unittest discover -s .\esa_csdi_experiment\tests -v
```

## Smoke run

```powershell
python .\esa_csdi_experiment\run_esa.py --mode smoke --seed 1 --device cuda:0
```

The smoke run uses 2 short epochs, 5 imputation samples and one evaluation batch.

## Formal seed-1 baseline

```powershell
python .\esa_csdi_experiment\run_esa.py --mode baseline --seed 1 --device cuda:0
```

The formal run uses 200 epochs and 100 imputation samples for each of 10%, 50% and
90% missingness. Results include model checkpoints, JSONL training logs, per-ratio
metrics and a combined summary. Generated samples are evaluated in streaming mode
in chunks of 10 and are not stored. Chunking parallelizes independent diffusion
trajectories without changing the model or the 50-step sampling process.

## Evaluate an existing checkpoint without retraining

```powershell
python .\esa_csdi_experiment\run_esa.py `
  --mode evaluate `
  --seed 1 `
  --device cuda:0 `
  --checkpoint .\esa_csdi_experiment\results\baseline_seed1_20260710_183452\model_best.pth
```

Evaluation mode recreates the deterministic 10%, 50% and 90% masks and loads the
saved model directly. Use `--missing-ratios 0.1` to run only one ratio.
