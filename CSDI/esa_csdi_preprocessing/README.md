# ESA Mission 1 preprocessing for CSDI

This directory is isolated from the existing CSDI reproduction. It prepares the
official ESA Mission 1 lightweight subset (channels 41-46) for a later CSDI data
loader without changing the original telemetry values or running anomaly
detection.

The implementation adapts the official ESA-ADB Mission 1 preprocessing choices:

- 30-second time grid;
- zero-order hold (last observation carried forward);
- ESA `labels.csv` and `anomaly_types.csv` annotations;
- chronological splits.

It deliberately differs from the full ESA-ADB preprocessing by selecting only six
channels, omitting telecommands and producing CSDI-oriented masks instead of the
full TimeEval CSV files.

## Run

From the repository root:

```powershell
python .\esa_csdi_preprocessing\preprocess.py --config .\esa_csdi_preprocessing\config.json
```

Use `--force` only when an existing processed artifact should be replaced.

## Outputs

Generated files are placed under
`data/processed/mission1_ch41_46_10m_30s/` inside this directory:

- `aligned.npz`: timestamps, raw and normalized values, masks, labels and split codes;
- `manifest.json`: provenance, schema, time boundaries, normalization statistics and hashes;
- `quality_report.json`: per-channel coverage and label counts.

Label codes are `0=nominal`, `1=anomaly`, `2=rare_event`, and
`3=communication_gap`. `observed_mask` is zero for communication gaps or invalid
values. `clean_mask` is one only for valid nominal values.

The artifact is not windowed and contains no artificial missingness. Window length,
stride and evaluation missing ratios belong to the later CSDI data loader.

## Validate

```powershell
python -m unittest discover -s .\esa_csdi_preprocessing\tests -v
```

Dataset source: ESA Anomaly Dataset, DOI 10.5281/zenodo.12528696.

