# Frequency analysis supporting notes

## Reporting job

- Question: Does ESA Mission 1 contain stable frequency structure that can justify a mask-aware frequency-conditioned CSDI branch?
- Audience: technical.
- Scope: Mission 1 training split only for spectral design evidence; fixed seed-1 test error is secondary hypothesis-generation evidence.
- Decision: choose a conditioning context and a first STFT configuration without changing the 96-step imputation target.

## Required structure mapping

1. Title -> report title block.
2. Technical summary -> `technical_summary`.
3. Key findings with visual evidence -> horizon, channel, mask and error sections.
4. Scope, data and definitions -> `scope`.
5. Methodology -> `method`.
6. Limitations and robustness -> `limitations`.
7. Recommended next steps -> parameter section and limitations.
8. Further questions -> `questions`.

## Selected configuration

- Conditioning context: 192 steps (96 minutes).
- Selection details: `{"horizon_steps": 192.0, "candidate_channels": 13.0, "strong_channels": 13.0, "median_score": 0.004229688201554319, "median_spectral_cosine": 0.8922820473547637, "dominant_peak_preservation_rate": 0.24456521739130435}`
- Fixed seed: 20260719
- Minimum nominal observed fraction: 99%

## Chart map

| Section | Question | Family / type | Fields | Claim | Palette |
|---|---|---|---|---|---|
| Horizon | How much context exposes usable periodicity? | comparison / grouped bar | context, candidate/strong counts | 48 minutes may be insufficient | single blue root + neutral |
| Channels | Which channels show the clearest stable frequency structure? | ranking / horizontal bar | channel, periodicity score | frequency value is channel-specific | single blue root |
| Mask proxy | Does a condition-only proxy preserve spectrum? | comparison / grouped bar | missing ratio, context-method series, cosine | linear proxy is safer than zero fill | hard two-root cap |
| Error association | Are periodic channels currently hard to impute? | relationship / scatter | periodicity score, MAE | association is hypothesis-generating only | single blue root |

## Evidence inventory

{
  "channel_metadata": 76,
  "channel_horizon_metrics": 380,
  "window_spectral_metrics": 35620,
  "horizon_overview": 5,
  "mask_robustness": 18,
  "mask_robustness_detail": 66180,
  "error_join": 76,
  "error_associations": 8,
  "recommended_parameters": 4
}

## Quantitative visual omission notes

- Exact candidate thresholds and parameter alternatives use narrative/table evidence because exact lookup matters more than another chart.
- Per-window spectra are preserved in CSV but omitted from the reader because tens of thousands of rows would not improve the decision.
