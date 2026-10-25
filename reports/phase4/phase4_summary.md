# Phase 4 evaluation and interpretation

FL=1 and NF=0. Confusion matrices are `[[TN, FP], [FN, TP]]`.
PR-AUC means average precision on continuous FL scores. Thresholds come from saved validation metrics.

## Status

Real-data evaluation is **incomplete**: no run passed saved-artifact, Phase 1 input, model-reload, and held-out prediction checks.


## Paired comparisons

No defensible paired comparison was available. Identical ordered test samples are required; overlapping windows prevent a routine McNemar p-value.

## Interpretability and precursor scope

SHAP and feature-pattern analysis were not run by this command; use `interpretability.py` on a verified run.
Phase 3 horizon names reuse the Phase 1 target. The existing runs do not establish 6/12/24/48-hour lead-time performance or a true pre-flare temporal pattern.

## Missing inputs and warnings

- `[WindowsPath('D:/Solar-Flare-Phase2/data/phase3'), WindowsPath('D:/Solar-Flare-Phase2/data/phase2')]`: No saved model runs found; restore Phase 3/Phase 2 run directories
- `D:\Solar-Flare-Phase2\data`: Phase 1 data directory is absent
