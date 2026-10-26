# Phase 3: verified long lead-time results

A positive label is an M/X flare peak in `(issue+H, issue+H+24h]`. All four H values use the same eligible windows. XGBoost was trained with bounded CUDA resources.

| Lead | Split | Features | Test N | FL | FL regions | AP | Recall | TSS | HSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24 | region_split | baseline | 32144 | 728 | 21 | 0.321 | 0.360 | 0.344 | 0.336 |
| 24 | region_split | baseline_temporal | 32144 | 728 | 21 | 0.326 | 0.254 | 0.244 | 0.286 |
| 24 | time_split | baseline | 46395 | 881 | 20 | 0.347 | 0.453 | 0.435 | 0.369 |
| 24 | time_split | baseline_temporal | 46395 | 881 | 20 | 0.369 | 0.485 | 0.466 | 0.384 |
| 48 | region_split | baseline | 32144 | 688 | 18 | 0.259 | 0.244 | 0.235 | 0.283 |
| 48 | region_split | baseline_temporal | 32144 | 688 | 18 | 0.260 | 0.480 | 0.435 | 0.248 |
| 48 | time_split | baseline | 46395 | 954 | 20 | 0.359 | 0.460 | 0.434 | 0.321 |
| 48 | time_split | baseline_temporal | 46395 | 954 | 20 | 0.342 | 0.379 | 0.357 | 0.294 |

The target is peak time, so this does not directly measure lead to flare onset. The 48-hour cohort excludes windows without complete future coverage.
