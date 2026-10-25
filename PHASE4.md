# Phase 4: evaluation and interpretation

Phase 4 reads completed Phase 2/3 runs. It does not fit models, change Phase 1 preprocessing, or choose a test-set threshold. `FL=1`, `NF=0`. It uses the saved validation threshold and the authoritative Phase 1 index, split, feature order, and normalization. A run is accepted only if its completion marker, model, saved preprocessing hashes, Phase 1 input hashes, held-out identities, recomputed confusion matrix, and stored metrics agree. Phase 3's saved `test_probabilities.npy` is checked against reloaded inference when present.

## Prerequisites

- Restore the original `data/phase1/` files and `data/splits.csv` used by the runs. Their SHA-256 hashes must match each run's `config.json`.
- Restore complete run directories under `data/phase3/` or `data/phase2/`, or pass explicit `--run-dir` paths. A run needs `config.json`, `metrics.json`, `status.json`, its model, and saved normalization JSON files.
- Use a compatible Python environment with NumPy, pandas, SciPy, scikit-learn, and joblib. XGBoost and TensorFlow are needed only for their corresponding saved model types. SHAP and matplotlib are optional for interpretability; the script records an explicit SHAP/plot blocker when unavailable. Inspect model-recorded versions before installing selective dependencies. The repository's `requirements.txt` is a broad snapshot, not a Phase 4 installation command.
- **Trust boundary:** joblib loads Python pickle data. Inspect the artifact source, then pass `--trust-model-files` only for files you trust. Native XGBoost JSON and Keras formats are loaded through their libraries.

## Commands

From the repository root, after restoring data and runs:

```bash
python classification_metrics.py --trust-model-files
python interpretability.py --trust-model-files --max-eval 40 --max-background 20 --max-pattern-per-class 1000
python advanced_metrics.py --trust-model-files --bootstrap-replicates 500 --seed 42 --interpretability-json PATH_TO_INTERPRETABILITY_RESULTS_JSON
python -B -m unittest discover -s tests -v
```

Replace `PATH_TO_INTERPRETABILITY_RESULTS_JSON` with the actual `interpretability_results.json` from the preceding command; omit that option if interpretability could not run. Each script supports repeatable `--run-dir PATH` or `--run-root PATH` arguments and `--data-dir PATH` for a different restored dataset location. The classification command writes a unique `reports/phase4/classification_*/` directory. The advanced command creates the four main `reports/phase4/phase4_*` files if absent; later invocations create a unique `reports/phase4/evaluation_*/` directory. Interpretability writes a unique `reports/phase4/interpretability_*/` directory. Existing reports are never overwritten. Prediction CSVs are saved under ignored `data/phase4/`, with sample ID, region ID, truth, FL probability and prediction. Large data and models remain under ignored `data/` or persistent external storage, never in Git.

Run a single trusted model when resources are limited:

```bash
python advanced_metrics.py --run-dir PATH_TO_SAVED_RUN --data-dir PATH_TO_DATA --trust-model-files --bootstrap-replicates 100
python interpretability.py --run-dir PATH_TO_SAVED_RUN --data-dir PATH_TO_DATA --trust-model-files --max-eval 20 --max-background 10
```

These commands are examples; replace the paths with actual restored locations. `--help` lists all options. Running with no data creates an explicit incomplete audit rather than fabricated scores.

## Metric and comparison policy

The main CSV records one row per verified run with test counts, FL prevalence, saved threshold, precision, recall, F1, confusion counts, average precision, ROC-AUC, TSS and HSS. The JSON retains negative-class and macro/weighted summaries, warnings, artifact hashes, and paired comparisons. Undefined values are `null` in JSON and blank in CSV. Average precision is scikit-learn's `average_precision_score` on continuous FL probabilities; it is not trapezoidal PR area.

For `[[TN, FP], [FN, TP]]`, TSS is `TP/(TP+FN) + TN/(TN+FP) - 1`, and HSS is `2*(TP*TN-FN*FP) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))`. Zero denominators yield `null`. Paired comparisons require identical ordered test IDs, labels, regions and source-index hashes. The bootstrap samples active regions with replacement so overlapping windows remain grouped. At least ten distinct test regions are required for an interval; smaller samples receive an explicit insufficient-data status. It provides percentile intervals for each model and their paired difference; it does not infer causation or universal superiority. Exact McNemar p-values are computed only when the held-out sample has one window per region and at least ten regions, then adjusted with Holm's method within each split/horizon family. With multiple windows per region, McNemar is marked not run. Random splits and smoke runs remain diagnostic and are excluded from formal pair comparisons.

SHAP uses TreeExplainer for forests/XGBoost, LinearExplainer for logistic regression, and a small KernelExplainer sample for ANN. The deterministic evaluation sample is approximately FL/NF balanced when both labels exist, so its SHAP ranking is **not** population-representative. The background sample comes only from training rows. Outputs record the explainer, package version, seed, actual rows, feature order and mean absolute SHAP importance. Model-native importance is saved separately with its source; where both are available, the report compares ranks and top-feature overlap. Explanations describe model association, not causal physical mechanisms. The exploratory FL-versus-NF feature table uses training-standardized held-out features. It is **not** a pre-flare trajectory or lead-time result.

## Scientific limit inherited from Phase 3

Phase 3's `short_horizon.py` and `long_horizon.py` label runs as 6, 12, 24 or 48 hours, but [`phase3_common.py`](phase3_common.py) loads the same Phase 1 pre-sliced target for every horizon and records in each config's `horizon_note` that horizon-specific datasets remain future work. Consequently Phase 4 can evaluate saved classifiers on the supplied binary task, but cannot claim measured differences in forecast lead time. A valid precursor study needs verified flare-onset times, a declared horizon definition, and rebuilt labels/windows from the full active-region series.

## Current checkout audit

At implementation time, this checkout contained no `data/` directory, saved model runs, or prediction arrays. The tracked Phase 3 CSV and Markdown files were templates with blank scores and `missing`/`not_run` status. This repository's initial Phase 4 main report is therefore an **incomplete input audit**. Run the commands above after restoring the original checked artifacts to populate a new run-specific result directory. Synthetic tests validate behavior but are not SWAN-SF performance results.

## Publishing without the local Phase 3 merge

The local `Prasiddhi` checkout contains a Phase 3 merge that is absent from `origin/Prasiddhi`. Committing only Phase 4 files locally does **not** keep that merge out of a direct push. After reviewing and committing the explicitly selected Phase 4 files, use a separate detached worktree based on a freshly fetched `origin/Prasiddhi`, cherry-pick the Phase 4 commit there, verify that its outgoing range contains only the intended new commit, and push **that detached HEAD** to `Prasiddhi`. Do not push the original local `Prasiddhi` HEAD. This procedure does not alter the original branch history; detailed commands belong in the handoff after the final file list is known.
