# CSDI on ESA

This repository combines the two local codebases used for CSDI experiments on ESA data:

- `CSDI/`: the main CSDI experiment workspace.
- `CSDI_improve/`: enhanced CSDI and spatial-graph experiments.

## Included experiment artifacts

The repository keeps lightweight, reviewable outputs such as CSV/TSV metrics, JSON summaries, HTML reports, figures, configuration files, notebooks, and ordinary text logs.

## Excluded large artifacts

Raw and processed datasets, model checkpoints, serialized tensors, installers, archives, caches, per-sample prediction arrays, and large line-by-line training streams are intentionally excluded. These remain in the original local workspaces and can be regenerated from the included code and configuration.

