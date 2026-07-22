# GP-VAE and Latent ODE reproduction status

As of **2026-07-22 (Asia/Shanghai)**, this directory records the completed and
currently available PhysioNet reproduction results for the GP-VAE baseline in
CSDI Table 2 and the Latent ODE baseline in CSDI Table 4.

## Technical summary

- GP-VAE reproduces the 10% missingness result closely: **0.5769 +/- 0.0023**
  versus the paper's **0.574 +/- 0.003** CRPS.
- GP-VAE's current 50% and 90% results are substantially lower than the paper
  values. They are valid outputs of the current pipeline, but the discrepancy is
  too large to call an exact reproduction without a protocol audit.
- Latent ODE's completed 10% result is **0.6913 +/- 0.0031**, close to the paper's
  **0.700 +/- 0.002**. The 50% result is currently based on four folds and is
  **0.6915 +/- 0.0054**, compared with **0.676 +/- 0.003** in the paper.
- Latent ODE at 90% is still running. Fold 0 has reached epoch 6, with best
  validation CRPS 0.733041, but no test metric is available yet.

## Headline CRPS results

Lower is better. Reproduction values are the arithmetic mean across available
folds; uncertainty is the standard error across folds.

| Model | Missing ratio | Available folds | Reproduction CRPS | Paper CRPS | Difference |
|---|---:|---:|---:|---:|---:|
| GP-VAE | 10% | 5/5 | 0.5769 +/- 0.0023 | 0.574 +/- 0.003 | +0.0029 |
| GP-VAE | 50% | 4/5 | 0.6177 +/- 0.0014 | 0.774 +/- 0.004 | -0.1563 |
| GP-VAE | 90% | 4/5 | 0.7100 +/- 0.0006 | 0.998 +/- 0.001 | -0.2880 |
| Latent ODE | 10% | 5/5 | 0.6913 +/- 0.0031 | 0.700 +/- 0.002 | -0.0087 |
| Latent ODE | 50% | 4/5 | 0.6915 +/- 0.0054 | 0.676 +/- 0.003 | +0.0155 |
| Latent ODE | 90% | 0/5 | pending | 0.761 +/- 0.010 | pending |

## Scope and metric definition

- Dataset: PhysioNet Challenge 2012 set-a, using the CSDI preprocessing and
  fold split with seed 1.
- Evaluation: artificial targets only; natural missing values are excluded.
- Probabilistic evaluation: 100 generated samples per target.
- CRPS: normalized quantile loss averaged over quantiles 0.05, 0.10, ..., 0.95.
- Aggregation: unweighted mean and standard error across available folds.
- Paper references: CSDI Table 2 for GP-VAE and Table 4 for Latent ODE.

## Completion and interpretation limits

1. Rows marked 4/5 are interim summaries. They must be regenerated when the
   fifth fold finishes.
2. Latent ODE 90% has no completed test fold and therefore no reproduction
   estimate yet.
3. GP-VAE 50% and 90% use the current random-training-mask runs. Their large
   improvement over the published baseline is a protocol-comparability warning,
   not evidence that the published result was surpassed.
4. The 10%, 50%, and 90% runs do not all share identical completion status, so
   comparisons across missingness levels should remain descriptive.

## Files

- `gpvae_folds.csv`: selected GP-VAE per-fold metrics and aggregate rows.
- `latent_ode_folds.csv`: Latent ODE per-fold metrics and aggregate rows,
  including explicit pending rows.
- `run_status.json`: machine-readable completion state and active-run note.

Model checkpoints, generated samples, and prepared datasets are intentionally
excluded because they are large and can be regenerated from the experiment
code and manifests.

