# Shared SWAN-SF data contract

Run `python extract_dataset.py` once from the repository root. It extracts all seven
Harvard Dataverse archives under `data/`. The directory is gitignored. The tarballs
remain in the repository root and are also gitignored.

| Path | Contents | Team use |
| --- | --- | --- |
| `data/SWAN/partition1` ... `partition5` | Full active-region series, one CSV per region | Build consistent 6/12/24/48-hour labels and features |
| `data/partition1/FL`, `data/partition1/NF` ... `partition5` | Pre-sliced labeled instances | Initial 24-hour benchmark and overlap study |
| `data/addenda/` | Supporting flare and X-ray records | Check labels and data quality when needed |
| `data/manifest.csv` | Original archive path, safe extracted path, class, region ID, start/end | Canonical sample index |
| `data/splits.csv` | Shared random, region, and time split assignments | Use the same train/validation/test rows across models |
| `data/leakage_summary.json` | Split sizes and overlap counts | Report leakage results |

After extraction, run `python leakage_analysis.py` to produce the last two files.
Every path in `manifest.csv` and `splits.csv` is relative to `data/`; for example,
`data / row["path"]` locates the CSV. Windows forbids colons in filenames, so the
extractor percent-encodes them as `%3A`. The `member` column retains the exact
original archive name. Do not use the encoded filename to infer the label or time;
use the manifest columns.

`FL` means M/X-class in the supplied binary task; `NF` includes flare-quiet and
B/C-class examples. The supplied instances use the dataset's existing slicing and
labeling setup. For other horizons, construct new labels from the full series.
The current Phase 2–4 rebuild defines H as a minimum lead to an M/X flare
**peak**, with a fixed 24-hour validity interval `(issue+H, issue+H+24h]`.
See [REBUILD.md](REBUILD.md) and `data/phase2/horizon_targets_report.json`
for coverage and validation.

For all models, select samples using `splits.csv` and fit imputation, scaling,
feature selection, resampling, and tuning on training data only. Use the random
split only to quantify leakage. Use the region or chronological split for the
generalization claim. The `time_split` column uses partitions 1-3 for training,
4 for validation, and 5 for testing. `region_split` keeps all windows from one
active-region ID together.

The full series include flare history, X-ray, and quality columns. Future flare
information belongs in labels, not predictive features. Agree on a feature list
with the data-cleaning teammate before training.

## Phase 1 measured results

All seven archives were extracted: 4,098 full active-region series, 331,185
pre-sliced instances, and 9 addenda files. There are 3,205 distinct region IDs
and 6,234 `FL` instances among the pre-sliced windows. Consecutive windows from
the same region have a median 91.5% observation overlap and a median 1-hour
start-time spacing.

With seed 42, 49,758 of 49,815 random-split test windows (99.89%) overlap an
observation interval in training. The region and chronological splits have zero
such test/train overlaps. These are structural leakage measurements; whether
they inflate model scores must be measured by training the same model under
each split.

## Phase 1 cleaning and temporal features

Run `.\.sfd\Scripts\python.exe data_cleaning.py` from the repository root. This
calls the feature definitions in `temporal_features.py` and scans each full
series once. `data/phase1/status.json` marks a completed build. The outputs are:

| Path | Meaning |
| --- | --- |
| `data/phase1/feature_index.csv` | One row per window, with path, label, region, partition, and all three split assignments |
| `data/phase1/baseline.npy` | Shape `(331185, 24)`; final observed value of each magnetic parameter |
| `data/phase1/temporal.npy` | Shape `(331185, 120)`; 1h and 3h lags, 3h rate of change, and final 3h rolling mean and standard deviation for each parameter |
| `data/phase1/feature_columns.json` | Column names in array order |
| `data/phase1/cleaning_report.json` | Missing-value, class-distribution, and training-only IQR outlier counts |
| `data/phase1/normalization_<split>_<set>.json` | Training-only median imputation and z-score parameters for each split and each array |

The row at index `i` in both NumPy arrays belongs to row `i` of
`feature_index.csv`. Load arrays with `np.load(path, mmap_mode="r")` to avoid
bringing all rows into memory. The `baseline` feature set uses `baseline.npy`;
the `baseline + temporal` set concatenates the two arrays. Use
`data_cleaning.transform_features(values, stats)` with the matching split's
normalization JSON for each array. Apply those same saved parameters to train,
validation, and test rows. Do not fit normalization on the full arrays.

The 24 inputs are magnetic measurements (`TOTUSJH` through `R_VALUE`). Flare
annotation, X-ray, and quality columns are excluded from predictors. Features
are computed only from each window's 12-hour observation interval. Missing
lags remain NaN until training-only imputation. Rolling summaries require at
least eight valid readings. The IQR outlier rule is descriptive; no row or
value was removed or clipped because extreme values may be relevant to flares.

Across 2,879,658 unique full-series observations, 6,082,002 of 69,111,792
magnetic cells (8.8%) are missing or nonfinite. The pre-sliced binary task has
6,234 `FL` windows out of 331,185 (1.88%).
