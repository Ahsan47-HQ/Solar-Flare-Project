# Solar flare forecasting with SWAN-SF

This repository contains a completed local rebuild of Phases 2–4 on the extracted SWAN-SF data. Phase 1 preprocessing and its shared arrays were preserved. All three teammates should use the same `data/phase1/` arrays, `data/phase2/horizon_targets.npy`, and split assignments. Data and model binaries are ignored by Git and remain on this machine.

## Current results

The rebuilt experiments include Logistic Regression, Random Forest, XGBoost, and an ANN for the original 24-hour task, plus XGBoost for 6, 12, 24, and 48-hour minimum warning leads. All 33 saved runs passed model reload, source-data hash, held-out row, and metric verification. Read [the complete methods and results](REBUILD.md), the [Phase 2 table](reports/phase2/classical_model_results.md), [Phase 3 horizon tables](reports/phase3/short_horizon_results.md), and [Phase 4 summary](reports/phase4/phase4_summary.md).

The lead-time target is an **M/X flare peak** in `(forecast issue + H, forecast issue + H + 24 hours]`. H is the minimum warning time to the recorded peak. It is not a guarantee of H hours before flare onset. A window is eligible only when the full active-region series covers the target interval and at least 114 of its 120 expected 12-minute samples are present. The four Phase 3 leads share a common eligible cohort.

## Shared data

[DATASET.md](DATASET.md) documents the seven archives, extraction, Phase 1 arrays, feature order, cleaning, and split rules. The random split is used only as a leakage diagnostic; region and chronological splits support the generalization results. Phase 1 has 331,185 observation windows and 24 baseline plus 120 temporal features. The forecast-target report is at `data/phase2/horizon_targets_report.json`.

## Reproduce the completed rebuild

From `D:\Solar_Flare_Detection` in PowerShell, with the existing `.sfd` environment:

```powershell
.\.sfd\Scripts\python.exe forecast_labels.py
.\.sfd\Scripts\python.exe -u run_rebuild_matrix.py
.\.sfd\Scripts\python.exe -u rebuild_evaluate.py
.\.sfd\Scripts\python.exe publish_rebuild_reports.py
```

`run_rebuild_matrix.py` resumes completed configurations and trains one process at a time. It uses bounded CUDA XGBoost settings on the RTX 4050 and trains the ANN on CPU because this native Windows TensorFlow build does not expose CUDA. `phase2_sensitivity.py` records separate validation-only depth and architecture experiments. Rebuilding labels after training changes the input hash and requires rerunning the model matrix and evaluation. See [REBUILD.md](REBUILD.md) for the full resource settings and validation procedure.

## Project files

| Phase | Current entry points | Main outputs |
| --- | --- | --- |
| 1 | `extract_dataset.py`, `leakage_analysis.py`, `data_cleaning.py`, `temporal_features.py` | `data/phase1/`, `data/splits.csv` |
| 2 | `forecast_labels.py`, `run_rebuild_matrix.py`, `rebuild_train.py` | `data/phase2/horizon_targets.npy`, `data/rebuild/runs/phase2/`, `reports/phase2/` |
| 3 | `run_rebuild_matrix.py`, `rebuild_train.py` | `data/rebuild/runs/phase3/`, `reports/phase3/` |
| 4 | `rebuild_evaluate.py`, `publish_rebuild_reports.py` | `reports/rebuild/phase4/`, `reports/phase4/` |

Earlier scripts such as `classical_models.py`, `short_horizon.py`, `advanced_metrics.py`, and their dated report directories remain for history. Their old horizon labels and incomplete reports are superseded by the rebuild above. Do not combine their results with the verified rebuild tables.

The main test metrics are average precision (AP), ROC-AUC, precision, recall, F1, TSS, HSS, and confusion counts. Classification thresholds are selected on validation data only. The pre-flare plots and TreeSHAP ranks describe model associations and do not establish causal precursors.
