# ESA Mission 1 graph-enhanced spatial CSDI

This experiment lives entirely under `CSDI_improve` and reads the existing CSDI
code and processed Mission 1 arrays without modifying them.

It keeps the Structured-Only masking baseline and changes only feature/spatial
attention. A static weighted channel graph is estimated from the training split
with pairwise-complete Pearson correlation. Each channel retains its eight
strongest neighbors without an additional hard threshold, the graph is
symmetrized, and its absolute correlation weights become an additive attention
bias. Top-K already controls sparsity; omitting a hard threshold prevents isolated
channels while weak edges retain correspondingly weak weights.

The learnable graph strength in every residual layer starts at zero. Therefore,
the graph contributes no attention bias at initialization and its learned signed
strength is available through `model.graph_diagnostics()`.

```powershell
python spatial_graph/run_graph.py --mode build-graph
python spatial_graph/run_graph.py --mode smoke
pytest spatial_graph/tests -q
```

The first ablation should compare Structured-Only CSDI against this graph model
with the same seed, steps, masking distribution, protocols, and sample count.
Primary graph metrics are `channel_dropout_50` and `rectangle_50`; `time_block_50`
is a guardrail.

## Phase 1 validation result

`phase1_graph_validation.py` performs a chronological 80/20 training-only audit:
four-block graph stability, raw-level versus first-difference/Spearman checks, and
Ridge neighbor reconstruction against random and shuffled graphs. The result does
not support using the raw-value Pearson graph as the sole main graph. The
first-difference Top-8 graph is the preferred candidate for a 3k-5k step
Graph-CSDI screening run; the raw graph should remain an ablation.

Primary output: `phase1_results/phase1_report.html`.
