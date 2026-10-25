# Phase 3 Short-Horizon Forecast Results (6h, 12h)

Generated from saved `config.json`, `metrics.json`, `status.json`, and model artifacts under `data/phase3/short_horizon/`.
An unexecuted run has blank metrics (`—`). Reported terminal numbers are not substituted.

## Methodology and Metric Definitions
- **Warning Horizons**: 6 hours (`6h`) and 12 hours (`12h`) lead times.
- **Label Definition**: `FL` is M/X-class flare (positive class, 1); `NF` is non-flaring / B/C-class (negative class, 0).
- **Class Imbalance**: Documented dataset contains 331,185 total windows, 6,234 `FL` instances (1.88% prevalence).
- **Features**: `baseline` (24 raw magnetic features); `baseline_temporal` (144 features including 120 temporal derivatives and window moments).
- **Generalization Splits**:
  - `region_split`: active regions are partitioned independently across train/val/test to prevent spatial contamination.
  - `time_split`: chronological partitions 1–3 for train, partition 4 for validation, partition 5 for held-out test.
- **Decision Threshold**: Tuned on validation split to maximize $F_1$ score; test labels are strictly reserved for final held-out scoring.
- **Skill Scores**:
  - **TSS (True Skill Statistic)**: $\text{POD} - \text{POFD} = \frac{TP}{TP+FN} - \frac{FP}{FP+TN}$ (ranges $[-1.0, +1.0]$, unaffected by 1.88% class prevalence).
  - **HSS (Heidke Skill Score)**: $\frac{2(TP \cdot TN - FP \cdot FN)}{(TP+FN)(FN+TN) + (TP+FP)(FP+TN)}$ (evaluates accuracy improvement over random chance).

## Test-Set Evaluation Matrix

| Split | Horizon | Features | Model | PR-AUC | FL Recall | FL Precision | FL F1 | TSS | HSS | Run Identifier |
|---|---|---|---|---|---|---|---|---|---|---|
| region_split | 6h | baseline | logistic | — | — | — | — | — | — | pending |
| region_split | 6h | baseline | forest | — | — | — | — | — | — | pending |
| region_split | 6h | baseline_temporal | logistic | — | — | — | — | — | — | pending |
| region_split | 6h | baseline_temporal | forest | — | — | — | — | — | — | pending |
| region_split | 12h | baseline | logistic | — | — | — | — | — | — | pending |
| region_split | 12h | baseline | forest | — | — | — | — | — | — | pending |
| region_split | 12h | baseline_temporal | logistic | — | — | — | — | — | — | pending |
| region_split | 12h | baseline_temporal | forest | — | — | — | — | — | — | pending |
| time_split | 6h | baseline | logistic | — | — | — | — | — | — | pending |
| time_split | 6h | baseline | forest | — | — | — | — | — | — | pending |
| time_split | 6h | baseline_temporal | logistic | — | — | — | — | — | — | pending |
| time_split | 6h | baseline_temporal | forest | — | — | — | — | — | — | pending |
| time_split | 12h | baseline | logistic | — | — | — | — | — | — | pending |
| time_split | 12h | baseline | forest | — | — | — | — | — | — | pending |
| time_split | 12h | baseline_temporal | logistic | — | — | — | — | — | — | pending |
| time_split | 12h | baseline_temporal | forest | — | — | — | — | — | — | pending |

---
*Phase 3 short-horizon evaluation. Generated artifacts expected under gitignored `data/phase3/short_horizon/`.*
