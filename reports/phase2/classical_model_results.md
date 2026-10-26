# Phase 2: verified full-data model results

M/X flare peak in the next 24 hours; 12-hour magnetic observation windows. The random split is a leakage diagnostic only. The decision threshold was selected on validation F1.

| Split | Features | Model | Test N | FL | AP | Recall | Precision | TSS | HSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_split | baseline | xgboost | 44980 | 946 | 0.551 | 0.617 | 0.470 | 0.602 | 0.522 |
| region_split | baseline | ann | 45757 | 939 | 0.444 | 0.411 | 0.401 | 0.398 | 0.393 |
| region_split | baseline | forest | 45757 | 939 | 0.264 | 0.292 | 0.354 | 0.281 | 0.307 |
| region_split | baseline | logistic | 45757 | 939 | 0.393 | 0.490 | 0.332 | 0.469 | 0.381 |
| region_split | baseline | xgboost | 45757 | 939 | 0.421 | 0.354 | 0.409 | 0.343 | 0.367 |
| region_split | baseline_temporal | ann | 45757 | 939 | 0.393 | 0.420 | 0.432 | 0.408 | 0.414 |
| region_split | baseline_temporal | forest | 45757 | 939 | 0.276 | 0.249 | 0.328 | 0.239 | 0.270 |
| region_split | baseline_temporal | logistic | 45757 | 939 | 0.388 | 0.440 | 0.360 | 0.423 | 0.382 |
| region_split | baseline_temporal | xgboost | 45757 | 939 | 0.423 | 0.317 | 0.491 | 0.310 | 0.375 |
| time_split | baseline | ann | 67279 | 990 | 0.391 | 0.203 | 0.432 | 0.199 | 0.269 |
| time_split | baseline | forest | 67279 | 990 | 0.342 | 0.564 | 0.337 | 0.547 | 0.411 |
| time_split | baseline | logistic | 67279 | 990 | 0.458 | 0.425 | 0.517 | 0.419 | 0.460 |
| time_split | baseline | xgboost | 67279 | 990 | 0.455 | 0.530 | 0.410 | 0.519 | 0.453 |
| time_split | baseline_temporal | ann | 67279 | 990 | 0.428 | 0.232 | 0.565 | 0.230 | 0.323 |
| time_split | baseline_temporal | forest | 67279 | 990 | 0.370 | 0.654 | 0.325 | 0.633 | 0.423 |
| time_split | baseline_temporal | logistic | 67279 | 990 | 0.487 | 0.374 | 0.575 | 0.370 | 0.446 |
| time_split | baseline_temporal | xgboost | 67279 | 990 | 0.487 | 0.447 | 0.472 | 0.440 | 0.451 |

All 17 runs passed model reload, data-hash, held-out ID, and metric checks. Detailed configurations and probabilities are under `data/rebuild/runs/phase2/`.
The random-split AP must not be treated as independent-region generalization.
