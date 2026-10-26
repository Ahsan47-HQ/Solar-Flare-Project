# SWAN-SF Phases 2–4 rebuild: method and results

## Data and forecast task

Phase 1 was reused without editing its arrays, normalization, or split assignments. It supplies 331,185 presliced 12-hour magnetic observation windows, each with 24 baseline values and 120 temporal features. The full SWAN-SF active-region series supply event annotations for new lead-time labels. `forecast_labels.py` writes `data/phase2/horizon_targets.npy` in exactly the Phase 1 index row order. The label report records 299,175 valid 0-hour windows, all of which agree with the supplied binary label (zero mismatches), and 210,341 windows valid at 48 hours.

The forecast is issued at the **end** of the observation window. A positive H-hour label means an M- or X-class flare **peak** in `(issue + H, issue + H + 24 hours]`. Thus 6 hours means at least 6 hours of warning to a recorded peak, with a fixed 24-hour forecast validity period. It does not mean a flare anywhere within the next 6 hours, nor 6 hours before flare onset. The original 24-hour benchmark is H=0. H=6, 12, 24, and 48 are distinct targets derived from the full series. Missing future coverage is marked unknown and excluded, requiring the source series through the interval end and at least 114 of 120 expected 12-minute samples. Phase 3 restricts all four leads to the same 210,341-window eligible cohort.

The `region_split` holds entire active regions together. The `time_split` trains on partitions 1–3, validates on 4, and tests on 5. The heavily overlapping `random_split` is included for one diagnostic comparison only. The Phase 1 training-only median imputation and z-score settings are applied unchanged to validation and test windows. No flare annotation, X-ray history, or quality column is used as a predictor. Temporal features use only readings within the observation interval.

## Training and evaluation

`run_rebuild_matrix.py` trains 33 complete configurations sequentially, saving each model, configuration, input hashes, validation and test row IDs, probabilities, metrics, and a completion marker in `data/rebuild/runs/`. Phase 2 trains Logistic Regression, Random Forest, XGBoost, and ANN across two generalization splits and two feature sets, plus the random diagnostic. Phase 3 trains XGBoost at four leads, two splits, and two feature sets. The model families share the same rows for each phase/split/lead, and baseline versus temporal comparisons are paired.

Logistic Regression uses balanced classes; Random Forest uses 80 depth-12 trees, balanced subsampling, and one worker. XGBoost uses CUDA histogram training, depth 3, at most 180 trees, 64 histogram bins, two workers, class-ratio weighting, and validation PR-AUC early stopping (20 rounds). The small ANN has 32 and 16 hidden units, class weights, batch size 2048, and at most 8 epochs with validation PR-AUC early stopping. Native Windows TensorFlow 2.21 ran the ANN on CPU; XGBoost used the RTX 4050. A bounded validation-only sensitivity study compared XGBoost depths 2/3/4 and ANN widths 16/8, 32/16, and 64/32 on baseline features in both splits. Those exploratory validation results are [saved separately](reports/phase2/validation_sensitivity.json); they did not alter the verified test matrix. This is not an exhaustive hyperparameter or architecture search.

In that sensitivity study, the highest validation AP was XGBoost depth 2 (0.448) and ANN 64/32 (0.367) for the region split; XGBoost depth 3 (0.550) and ANN 16/8 (0.594) for the time split. These are validation observations and should not be presented as held-out performance.

Each classification threshold maximizes F1 on its **validation** predictions. `rebuild_evaluate.py` verifies every saved model against current Phase 1 array/normalization and label hashes, authoritative validation/test IDs, recomputed metrics, and 64 held-out predictions after model reload. It generated average precision (AP), ROC-AUC, precision, recall, F1, TSS, HSS, and confusion counts. All 33 runs passed; zero were rejected. The original reports under the main `reports/phase2/`, `reports/phase3/`, and `reports/phase4/` paths were refreshed using `publish_rebuild_reports.py`. Historical dated report subdirectories remain as old snapshots and should not be used for current claims.

## Selected held-out results

| Task | Split | Features | Model | Test AP | Test TSS |
| --- | --- | --- | --- | ---: | ---: |
| H=0 | Region | Baseline | XGBoost | 0.421 | 0.343 |
| H=0 | Region | Baseline | ANN | 0.444 | 0.398 |
| H=0 | Time | Baseline | XGBoost | 0.455 | 0.519 |
| H=0 | Time | Baseline + temporal | XGBoost | 0.487 | 0.440 |
| H=0 | Random, leakage diagnostic | Baseline | XGBoost | 0.551 | 0.602 |
| H=6 | Region | Baseline | XGBoost | 0.413 | 0.333 |
| H=12 | Region | Baseline | XGBoost | 0.373 | 0.294 |
| H=24 | Region | Baseline | XGBoost | 0.321 | 0.344 |
| H=48 | Region | Baseline | XGBoost | 0.259 | 0.235 |

The full [Phase 2 results](reports/phase2/classical_model_metrics.csv), [Phase 3 results](reports/phase3/phase3_metrics.csv), and [Phase 4 combined results](reports/phase4/phase4_results.csv) include all configurations and metrics. AP is the primary ranking metric for this rare-event task; TSS reports thresholded skill. The higher random-split score is consistent with the measured window overlap and must not be used as independent-region performance. The same eligible rows are used across Phase 3 leads, although each lead has a different positive label and prevalence, so lead comparisons still change the task.

The paired temporal-minus-baseline AP differences were bootstrapped 500 times by active region on identical held-out windows. Every 95% interval includes zero, so these experiments do not establish a reliable AP gain from the extra temporal features. See the [feature comparison](reports/phase3/feature_comparison_summary.md). XGBoost TreeSHAP importance and representative observation-window trajectories are in [the Phase 4 reports](reports/rebuild/phase4/). They describe associations, not causes. For the chronological H=24 test set, 881 positive windows come from only 20 distinct positive active regions; interpretation therefore samples one window per region and remains uncertain.

## Reproduction

From `D:\Solar_Flare_Detection`, after setting up the existing `requirements.txt` environment and completing Phase 1:

```powershell
.\.sfd\Scripts\python.exe forecast_labels.py
.\.sfd\Scripts\python.exe -u run_rebuild_matrix.py
.\.sfd\Scripts\python.exe -u phase2_sensitivity.py
.\.sfd\Scripts\python.exe -u rebuild_evaluate.py
.\.sfd\Scripts\python.exe publish_rebuild_reports.py
.\.sfd\Scripts\python.exe -m unittest discover -s tests
```

`run_rebuild_matrix.py --phase phase2` or `--phase phase3` narrows the run. It skips completed configurations. Rebuild the targets before training; if the target file or Phase 1 arrays change, the evaluator intentionally rejects stale models until retrained. Model binaries and the extracted dataset live under gitignored `data/`, so another teammate needs the same shared local directory or a verified copy of its data and model artifacts to rerun verification. Markdown and CSV reports alone are insufficient to reload models.
