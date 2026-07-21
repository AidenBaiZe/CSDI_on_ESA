# Validation report

## Overall assessment: Share with caveats

The phase-1 result is methodologically suitable for choosing a graph candidate
for a small Graph-CSDI screening experiment. It does not establish physical
channel causality or final imputation improvement.

## Methodology review

- Only the Mission 1 training split is used.
- The first 80% is the chronological build partition and the last 20% is a
  continuous validation partition.
- Raw Pearson and first-difference Pearson use all eligible timestamps;
  Spearman uses a deterministic evenly spaced 160k sample.
- Graph stability uses four contiguous build blocks and all six block pairs.
- Reconstruction uses identical masks, windows and target points for the true,
  random and shuffled graph comparisons.

## Calculation spot-checks

- SQLite `pragma quick_check`: passed.
- Candidate correlation matrices: 76 x 76; neighbor matrices: 76 x 8.
- Aggregate reconstruction rows: 18; window-level rows: 2,304.
- Summary JSON values match the exported reconstruction CSV.
- Paired 10,000-draw bootstrap intervals for the difference graph versus random
  do not cross zero for channel-dropout or rectangle masks.

## Material caveats

- A linear Ridge proxy is not a substitute for Graph-CSDI training.
- First-difference graph edge weights are often small and their cross-period
  weight ranking is weak, despite stronger proxy reconstruction.
- Raw-level relationships may encode useful operating state as well as zero-order
  hold artifacts; labeling them as physical links would be unsupported.
- Portable HTML packaging passed structural verification only because a compatible
  Chromium headless-shell was unavailable.

## Decision

Use the first-difference Top-8 graph for a 3k-5k step Graph-only screening run,
with raw Pearson, random and label-shuffled graphs as controls. Do not start the
full 32k-step Graph-CSDI experiment until the true graph beats both negative
controls on fixed protocols.
