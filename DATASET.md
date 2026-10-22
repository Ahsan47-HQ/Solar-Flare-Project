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
labeling setup. For other horizons, construct new windows and labels from the full
series. Decide whether a horizon means *within H hours* or *at least H hours of
lead time* before creating those labels; these are different tasks.

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
