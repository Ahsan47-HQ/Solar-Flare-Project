# Phase 3: verified short lead-time results

A positive label is an M/X flare peak in `(issue+H, issue+H+24h]`. All four H values use the same eligible windows. XGBoost was trained with bounded CUDA resources.

| Lead | Split | Features | Test N | FL | FL regions | AP | Recall | TSS | HSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | region_split | baseline | 32144 | 800 | 20 | 0.413 | 0.343 | 0.333 | 0.391 |
| 6 | region_split | baseline_temporal | 32144 | 800 | 20 | 0.399 | 0.344 | 0.333 | 0.376 |
| 6 | time_split | baseline | 46395 | 810 | 18 | 0.409 | 0.438 | 0.427 | 0.418 |
| 6 | time_split | baseline_temporal | 46395 | 810 | 18 | 0.398 | 0.291 | 0.285 | 0.342 |
| 12 | region_split | baseline | 32144 | 792 | 21 | 0.373 | 0.304 | 0.294 | 0.341 |
| 12 | region_split | baseline_temporal | 32144 | 792 | 21 | 0.373 | 0.359 | 0.344 | 0.354 |
| 12 | time_split | baseline | 46395 | 828 | 20 | 0.367 | 0.327 | 0.319 | 0.352 |
| 12 | time_split | baseline_temporal | 46395 | 828 | 20 | 0.361 | 0.312 | 0.304 | 0.346 |

The target is peak time, so this does not directly measure lead to flare onset. The 48-hour cohort excludes windows without complete future coverage.
