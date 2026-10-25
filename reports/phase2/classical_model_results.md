# Phase 2 model results

Generated from saved config.json, metrics.json, status.json and model files only.
An absent run has blank metrics. Reported terminal numbers are not substituted.

FL is M/X-class and positive (1); NF is negative (0). The documented full dataset has
331,185 windows, 6,234 FL (1.88%) and 324,951 NF; these counts are not reverified
unless Phase 1 files are available to the audit. Baseline has 24 magnetic features;
baseline_temporal has 144. Region split keeps regions separate; time split trains
on partitions 1–3, validates on 4 and tests on 5. Random split is diagnostic only.
Normalization is fitted by Phase 1 on each split's training rows. Validation selects
the F1-maximizing decision threshold; test labels do not select it. Average precision
(AP) summarizes the precision–recall curve and is not accuracy. Recall describes the
share of FL windows found; precision describes how many FL predictions are correct.
F1 combines precision and recall. ROC-AUC and confusion matrices are in the CSV.
No statistical significance or universal model ranking is inferred.

| Run | Model | Split | Features | Val AP | Val P/R/F1 | Threshold | Test AP | Test P/R/F1 | Test ROC-AUC | Reload |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| 20261024T174915Z_9fb61b61 | logistic | region_split | baseline | — | —/—/— | — | — | —/—/— | — | not_run |
| 20261024T175106Z_d3181de8 | forest | region_split | baseline | — | —/—/— | — | — | —/—/— | — | not_run |
| 20261024T175435Z_6f50f109 | logistic | region_split | baseline_temporal | — | —/—/— | — | — | —/—/— | — | not_run |
| 20261024T180048Z_429d9907 | forest | region_split | baseline_temporal | — | —/—/— | — | — | —/—/— | — | not_run |
| 20261024T180501Z_6260511c | logistic | time_split | baseline | — | —/—/— | — | — | —/—/— | — | not_run |
| 20261024T180647Z_8d604d59 | forest | time_split | baseline | — | —/—/— | — | — | —/—/— | — | not_run |
| 20261024T181101Z_26bf580e | logistic | time_split | baseline_temporal | — | —/—/— | — | — | —/—/— | — | not_run |
| 20261024T181645Z_f283cb01 | forest | time_split | baseline_temporal | — | —/—/— | — | — | —/—/— | — | not_run |

## Within-configuration comparison

Unavailable until at least two saved runs in one configuration are verified.

Metrics above are only final after checking the saved metrics files and
held-out inference. Model files are expected under gitignored data/phase2/;
Phase 1 arrays are expected under gitignored data/phase1/.
See experiment_manifest.json, artifact_inventory.json and model_reload_verification.md.
Reproduce with classical_models.py, xgboost_model.py or ann_model.py using the recorded
split, feature set, seed and parameters. Existing completed runs should be reused.
