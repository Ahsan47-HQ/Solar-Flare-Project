# Verified Phase 2--4 SWAN-SF results

Targets are M/X flare peaks in the 24-hour interval beginning after the stated minimum lead time.
All scores below come from held-out predictions and a decision threshold selected on validation data.

| Phase | Lead | Split | Features | Model | Test n / FL | AP | Recall | Precision | TSS | HSS |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| phase2 | 0h | random_split | baseline | xgboost | 44980 / 946 | 0.551 | 0.617 | 0.470 | 0.602 | 0.522 |
| phase2 | 0h | region_split | baseline | ann | 45757 / 939 | 0.444 | 0.411 | 0.401 | 0.398 | 0.393 |
| phase2 | 0h | region_split | baseline | forest | 45757 / 939 | 0.264 | 0.292 | 0.354 | 0.281 | 0.307 |
| phase2 | 0h | region_split | baseline | logistic | 45757 / 939 | 0.393 | 0.490 | 0.332 | 0.469 | 0.381 |
| phase2 | 0h | region_split | baseline | xgboost | 45757 / 939 | 0.421 | 0.354 | 0.409 | 0.343 | 0.367 |
| phase2 | 0h | region_split | baseline_temporal | ann | 45757 / 939 | 0.393 | 0.420 | 0.432 | 0.408 | 0.414 |
| phase2 | 0h | region_split | baseline_temporal | forest | 45757 / 939 | 0.276 | 0.249 | 0.328 | 0.239 | 0.270 |
| phase2 | 0h | region_split | baseline_temporal | logistic | 45757 / 939 | 0.388 | 0.440 | 0.360 | 0.423 | 0.382 |
| phase2 | 0h | region_split | baseline_temporal | xgboost | 45757 / 939 | 0.423 | 0.317 | 0.491 | 0.310 | 0.375 |
| phase2 | 0h | time_split | baseline | ann | 67279 / 990 | 0.391 | 0.203 | 0.432 | 0.199 | 0.269 |
| phase2 | 0h | time_split | baseline | forest | 67279 / 990 | 0.342 | 0.564 | 0.337 | 0.547 | 0.411 |
| phase2 | 0h | time_split | baseline | logistic | 67279 / 990 | 0.458 | 0.425 | 0.517 | 0.419 | 0.460 |
| phase2 | 0h | time_split | baseline | xgboost | 67279 / 990 | 0.455 | 0.530 | 0.410 | 0.519 | 0.453 |
| phase2 | 0h | time_split | baseline_temporal | ann | 67279 / 990 | 0.428 | 0.232 | 0.565 | 0.230 | 0.323 |
| phase2 | 0h | time_split | baseline_temporal | forest | 67279 / 990 | 0.370 | 0.654 | 0.325 | 0.633 | 0.423 |
| phase2 | 0h | time_split | baseline_temporal | logistic | 67279 / 990 | 0.487 | 0.374 | 0.575 | 0.370 | 0.446 |
| phase2 | 0h | time_split | baseline_temporal | xgboost | 67279 / 990 | 0.487 | 0.447 | 0.472 | 0.440 | 0.451 |
| phase3 | 12h | region_split | baseline | xgboost | 32144 / 792 | 0.373 | 0.304 | 0.425 | 0.294 | 0.341 |
| phase3 | 12h | region_split | baseline_temporal | xgboost | 32144 / 792 | 0.373 | 0.359 | 0.381 | 0.344 | 0.354 |
| phase3 | 12h | time_split | baseline | xgboost | 46395 / 828 | 0.367 | 0.327 | 0.405 | 0.319 | 0.352 |
| phase3 | 12h | time_split | baseline_temporal | xgboost | 46395 / 828 | 0.361 | 0.312 | 0.416 | 0.304 | 0.346 |
| phase3 | 24h | region_split | baseline | xgboost | 32144 / 728 | 0.321 | 0.360 | 0.343 | 0.344 | 0.336 |
| phase3 | 24h | region_split | baseline_temporal | xgboost | 32144 / 728 | 0.326 | 0.254 | 0.364 | 0.244 | 0.286 |
| phase3 | 24h | time_split | baseline | xgboost | 46395 / 881 | 0.347 | 0.453 | 0.331 | 0.435 | 0.369 |
| phase3 | 24h | time_split | baseline_temporal | xgboost | 46395 / 881 | 0.369 | 0.485 | 0.338 | 0.466 | 0.384 |
| phase3 | 48h | region_split | baseline | xgboost | 32144 / 688 | 0.259 | 0.244 | 0.373 | 0.235 | 0.283 |
| phase3 | 48h | region_split | baseline_temporal | xgboost | 32144 / 688 | 0.260 | 0.480 | 0.189 | 0.435 | 0.248 |
| phase3 | 48h | time_split | baseline | xgboost | 46395 / 954 | 0.359 | 0.460 | 0.268 | 0.434 | 0.321 |
| phase3 | 48h | time_split | baseline_temporal | xgboost | 46395 / 954 | 0.342 | 0.379 | 0.264 | 0.357 | 0.294 |
| phase3 | 6h | region_split | baseline | xgboost | 32144 / 800 | 0.413 | 0.343 | 0.490 | 0.333 | 0.391 |
| phase3 | 6h | region_split | baseline_temporal | xgboost | 32144 / 800 | 0.399 | 0.344 | 0.449 | 0.333 | 0.376 |
| phase3 | 6h | time_split | baseline | xgboost | 46395 / 810 | 0.409 | 0.438 | 0.419 | 0.427 | 0.418 |
| phase3 | 6h | time_split | baseline_temporal | xgboost | 46395 / 810 | 0.398 | 0.291 | 0.441 | 0.285 | 0.342 |

## Paired temporal-feature comparisons

Differences are temporal minus baseline AP on identical held-out windows. Intervals resample active regions to keep overlapping windows together.

- 6h region_split: AP difference -0.0139; 95% region-bootstrap interval [-0.04507611400727327, 0.007607360370159723].
- 6h time_split: AP difference -0.0104; 95% region-bootstrap interval [-0.05321111838595635, 0.01842944364485513].
- 12h region_split: AP difference +0.0000; 95% region-bootstrap interval [-0.06507778099847404, 0.0281582713689781].
- 12h time_split: AP difference -0.0070; 95% region-bootstrap interval [-0.08441766209683586, 0.04956317011718916].
- 24h region_split: AP difference +0.0043; 95% region-bootstrap interval [-0.01332307005147805, 0.02007280124748665].
- 24h time_split: AP difference +0.0225; 95% region-bootstrap interval [-0.005974803465905205, 0.04807857381414761].
- 48h region_split: AP difference +0.0006; 95% region-bootstrap interval [-0.015845368754231596, 0.014811367994955473].
- 48h time_split: AP difference -0.0174; 95% region-bootstrap interval [-0.07074239057525009, 0.03240100869273339].

## Interpretation

TreeSHAP: computed. Feature ranks are model associations, not causal precursors.

## Verification

33 saved models passed input-hash, split-ID, saved-metric, and reload checks.
Rejected runs: 0.
