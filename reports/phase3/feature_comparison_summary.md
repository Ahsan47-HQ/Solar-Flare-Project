# Phase 3: baseline versus temporal features

Paired XGBoost AP differences on identical held-out windows; temporal minus baseline. 500 bootstrap replicates resample active regions, preserving window dependence.

| Lead | Split | AP difference | 95% region CI |
| --- | --- | --- | --- |
| 6h | region_split | -0.014 | [-0.045, 0.008] |
| 6h | time_split | -0.010 | [-0.053, 0.018] |
| 12h | region_split | 0.000 | [-0.065, 0.028] |
| 12h | time_split | -0.007 | [-0.084, 0.050] |
| 24h | region_split | 0.004 | [-0.013, 0.020] |
| 24h | time_split | 0.022 | [-0.006, 0.048] |
| 48h | region_split | 0.001 | [-0.016, 0.015] |
| 48h | time_split | -0.017 | [-0.071, 0.032] |

Every interval includes zero. These runs do not establish a reliable AP benefit from adding the 120 temporal features. See `reports/rebuild/phase3/` for AP/TSS plots.
