# Phase 2 execution and recovery

Phase 2 uses the Phase 1 `data/splits.csv` assignments and aligned arrays in
`data/phase1/`. FL is the positive class. `random_split` is a leakage diagnostic;
use `region_split` or `time_split` for generalization. All model scripts require
explicit execution. The full dataset and trained models belong under gitignored
`data/`, not in Git.

## Source and environment

Keep source work on `Prasiddhi`. The Colab checkout must contain the reviewed
source commit before running a new experiment. Confirm `git branch --show-current`
and `git rev-parse HEAD` in Colab. Run from `/content/Solar-Flare-Project`.
Do not overwrite the existing Colab `data/phase2/classical/` tree when updating
source. Confirm the Drive mount and inspect the actual Phase 1 backup location.
The expected training input is `/content/Solar-Flare-Project/data/` with complete
`phase1/status.json`, arrays, split index, and normalization files.

Check Python and installed `numpy`, `pandas`, `scikit-learn`, `joblib`,
`xgboost`, and `tensorflow` versions before installing anything. The repository's
`requirements.txt` is a full environment snapshot with many unrelated notebook
packages; do not install it blindly. The unverified pins include XGBoost 3.2.0
and TensorFlow 2.21.0. Confirm availability for the active Colab Python runtime.
XGBoost and TensorFlow are imported only by their respective training scripts.

## Preserve the completed classical runs first

Mount Google Drive explicitly in a Colab cell with
`from google.colab import drive; drive.mount('/content/drive')`. Verify that
the source directory contains
the eight run IDs listed in `phase2_artifacts.py`. Then run:

```bash
python phase2_artifacts.py backup \
  --source /content/Solar-Flare-Project/data/phase2/classical \
  --destination /content/drive/MyDrive/Solar-Flare-Project/processed/phase2/classical \
  --inventory /content/drive/MyDrive/Solar-Flare-Project/processed/phase2/classical_backup_inventory.json
```

The command copies missing files to temporary names, verifies SHA-256, and
checks every destination file. Existing files with different hashes are reported
as conflicts and retained. A nonzero exit means the backup is incomplete.
Preserve the inventory in Drive. Do not remove the Colab source until reload
and backup verification pass.

Then verify actual saved models and generate lightweight reports:

```bash
python phase2_artifacts.py audit \
  --phase2-root /content/Solar-Flare-Project/data/phase2 \
  --data-dir /content/Solar-Flare-Project/data \
  --report-dir /content/Solar-Flare-Project/reports/phase2 \
  --branch Prasiddhi --commit "$(git rev-parse HEAD)" \
  --backup-inventory /content/drive/MyDrive/Solar-Flare-Project/processed/phase2/classical_backup_inventory.json
```

The audit loads each trusted saved model, checks its config, normalization and
Phase 1 hashes, and runs inference on up to 32 held-out rows. It does not fit.
It writes a CSV, Markdown results, JSON manifest, reload report and file-hash
inventory. A nonzero exit indicates one or more required runs could not be
verified. The report generated on Windows before Drive access has blank scores
and must be refreshed from the actual Colab artifacts before it can be treated
as final.

## Remaining model families

Once the classical runs are preserved and the original experiment requirements
are confirmed, run XGBoost or ANN explicitly. Example commands for comparable
region and time evaluations are:

```bash
python xgboost_model.py --split region_split --features baseline_temporal
python xgboost_model.py --split time_split --features baseline_temporal
python ann_model.py --split region_split --features baseline_temporal
python ann_model.py --split time_split --features baseline_temporal
```

These are proposed configurations, not verified original requirements or
completed experiments. Each model writes its native model file, config,
normalization copies, metrics and completion marker under `data/phase2/`.
Use `--smoke-max-per-subset 100` for a diagnostic run before full training;
smoke results are never official results. After full runs, back up the complete
`xgboost/` and `ann/` trees using the same backup command with family-specific
source and destination paths, using a separate inventory file for each family.
Pass all three inventory files as repeated `--backup-inventory` arguments to
the audit command. It rehashes the current source and Drive copies before
summarizing backup status. The manifest keeps `complete: false` until the
original Phase 2 experiment matrix is available and reviewed.

## Recovery

After a runtime reset, remount Drive, restore the original repo branch/source
revision, and inspect the local and Drive inventories. Re-run `backup` to copy
only absent files and rehash every source/destination pair. Resolve any conflict
manually after comparing hashes. Re-run `audit` against the restored Phase 1
inputs before claiming reuse. A run without `status.json` marked complete is
unfinished; inspect its other files before considering retraining.

Run offline tests with `python -B -m unittest discover -s tests -v`. The tests
use synthetic data and do not substitute for full experiment or reload checks.
