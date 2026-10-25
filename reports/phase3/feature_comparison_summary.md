# Phase 3 Feature Comparison Summary (baseline vs baseline_temporal)

Systematic comparison between raw magnetic moments (`baseline`, 24 features) and temporal active-region derivatives (`baseline_temporal`, 144 features) across 6h, 12h, 24h, and 48h horizons.

Generated from consolidated JSON metrics under `data/phase3/feature_comparison/`.

## Comparison Objective
To determine whether temporal evolution features (preflare magnetic field gradients, rate-of-change, and rolling statistical moments) provide statistically meaningful predictive advantage over static single-window magnetic measurements across increasing warning lead times.

## Evaluation Matrix (Random Forest Classifier)

### Region-Independent Split (`region_split`)

| Horizon | Feature Set | PR-AUC | FL Recall | FL Precision | FL F1 | TSS | HSS |
|---|---|---|---|---|---|---|---|
| 6h | baseline (24 feats) | — | — | — | — | — | — |
| 6h | baseline_temporal (144 feats) | — | — | — | — | — | — |
| 12h | baseline (24 feats) | — | — | — | — | — | — |
| 12h | baseline_temporal (144 feats) | — | — | — | — | — | — |
| 24h | baseline (24 feats) | — | — | — | — | — | — |
| 24h | baseline_temporal (144 feats) | — | — | — | — | — | — |
| 48h | baseline (24 feats) | — | — | — | — | — | — |
| 48h | baseline_temporal (144 feats) | — | — | — | — | — | — |

#### Temporal Advantage ($\Delta = \text{temporal} - \text{baseline}$, `region_split`)

| Horizon | $\Delta$ PR-AUC | $\Delta$ FL Recall | $\Delta$ TSS | $\Delta$ HSS | Direction |
|---|---|---|---|---|---|
| 6h | — | — | — | — | pending |
| 12h | — | — | — | — | pending |
| 24h | — | — | — | — | pending |
| 48h | — | — | — | — | pending |

---

### Chronological Split (`time_split`)

| Horizon | Feature Set | PR-AUC | FL Recall | FL Precision | FL F1 | TSS | HSS |
|---|---|---|---|---|---|---|---|
| 6h | baseline (24 feats) | — | — | — | — | — | — |
| 6h | baseline_temporal (144 feats) | — | — | — | — | — | — |
| 12h | baseline (24 feats) | — | — | — | — | — | — |
| 12h | baseline_temporal (144 feats) | — | — | — | — | — | — |
| 24h | baseline (24 feats) | — | — | — | — | — | — |
| 24h | baseline_temporal (144 feats) | — | — | — | — | — | — |
| 48h | baseline (24 feats) | — | — | — | — | — | — |
| 48h | baseline_temporal (144 feats) | — | — | — | — | — | — |

#### Temporal Advantage ($\Delta = \text{temporal} - \text{baseline}$, `time_split`)

| Horizon | $\Delta$ PR-AUC | $\Delta$ FL Recall | $\Delta$ TSS | $\Delta$ HSS | Direction |
|---|---|---|---|---|---|
| 6h | — | — | — | — | pending |
| 12h | — | — | — | — | pending |
| 24h | — | — | — | — | pending |
| 48h | — | — | — | — | pending |

---
*Phase 3 feature comparison. Consolidated results stored under gitignored `data/phase3/feature_comparison/`.*
