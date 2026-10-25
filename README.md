# Solar Flare Project

This project studies how reliably machine-learning models can identify solar-flare activity from magnetic measurements of solar active regions. Its research aim, described in the team's *Investigating the Early-Warning Capability and Generalization of Machine Learning Models for Solar Flare Forecasting* synopsis, is to examine useful warning time, temporal precursors, and performance on new regions and later observations. The current repository implements a binary SWAN-SF window-classification pipeline and model code; experiments for distinct forecast horizons remain future work.

**Project status:** Phase 1 extraction, leakage analysis, cleaning, and feature generation are implemented and have documented measurements. Phase 2 has a manually triggered acquisition notebook and code for Logistic Regression, Random Forest, XGBoost, and an artificial neural network (ANN). The checked-in Phase 2 reports do not yet contain verified full-data model scores or successful model-reload results. This checkout does not contain the generated dataset or model binaries.

## Research questions and current task

Solar flares can affect space-based and ground-based systems, so a useful forecast must provide warning before an event and continue to work beyond familiar observations. The synopsis proposes comparing 6-, 12-, 24-, and 48-hour warning horizons, studying changes before flares, and testing whether apparent performance survives chronological and active-region separation. These are research objectives, **not completed horizon-specific results**.

The implemented task uses the supplied SWAN-SF pre-sliced windows. Each example contains a 12-hour observation interval sampled at 12-minute cadence. Its label is `FL` for the supplied M/X-class flare category and `NF` for the supplied non-flaring/B/C category; `FL` is the positive class. The repository describes these windows as an initial 24-hour benchmark, but the code does not reconstruct independent targets for the proposed 6/12/24/48-hour horizons. Establishing exact horizon semantics from the full series remains necessary before making early-warning claims. See [the data contract](DATASET.md).

## Project roadmap

| Stage | Purpose and inputs | Implemented deliverables | Evidence and status |
| --- | --- | --- | --- |
| Phase 1: data preparation | Seven original SWAN-SF archives | [Extraction](extract_dataset.py), [leakage analysis](leakage_analysis.py), [cleaning](data_cleaning.py), and [temporal features](temporal_features.py); manifests, split assignments, arrays, and normalization metadata under `data/` | Implemented; measurements are documented in [DATASET.md](DATASET.md). Generated files are absent from this checkout. |
| Phase 2A: acquisition | Official Harvard Dataverse metadata and archive downloads | [Acquisition notebook](notebooks/phase2a_dataset_acquisition.ipynb) with live metadata comparison, explicit download enablement, integrity checks, and persistent storage | Notebook implemented; no Colab run or local archives are evidenced here. |
| Phase 2: model development | Phase 1 aligned arrays and authoritative splits | [Classical baselines](classical_models.py), [XGBoost](xgboost_model.py), [ANN](ann_model.py), shared model utilities, synthetic tests, and artifact audit tools | Code and offline tests exist; full-data scores, Drive backups, and model reloads remain unverified in [Phase 2 reports](reports/phase2/classical_model_results.md). |
| Horizon and precursor study | Full active-region series and a defined forecast target for each horizon | Proposed comparisons of warning horizons, temporal patterns, and model generalization | Planned in the synopsis; no horizon-specific dataset or results are present in this branch. |

The stages connect as follows:

```text
Official SWAN-SF archives
  → extraction and sample manifest
  → leakage analysis and shared split assignments
  → aligned Phase 1 arrays, feature names, and training-fitted normalization
  → model training and validation-based threshold selection
  → held-out evaluation, saved artifacts, backup, and reload audit
```

## Dataset and scientific safeguards

The data source is [SWAN-SF, version 1.2, Harvard Dataverse, DOI:10.7910/DVN/EBCFKM](https://doi.org/10.7910/DVN/EBCFKM), credited to Angryk and colleagues. The official Dataverse metadata identifies its dataset license as **CC0 1.0**. This is the *dataset* license; this repository does not currently contain a code `LICENSE` file. The seven required archives are `SWAN.tar.gz`, `addenda.tar.gz`, and `partition1_instances.tar.gz` through `partition5_instances.tar.gz`. The addenda are extracted as supporting flare and X-ray records; X-ray and flare-annotation fields are excluded from model predictors.

Phase 1 reports 4,098 full active-region series, 331,185 pre-sliced instances, 3,205 distinct regions, and 6,234 `FL` instances (1.88%). Its baseline array has 24 magnetic features per window; its temporal array adds 120 lag, change, and rolling-summary features, giving 144 features for `baseline_temporal`. These are **documented Phase 1 measurements**, not remeasured in this checkout because `data/` is absent. Feature order is recorded in `data/phase1/feature_columns.json`, and array row *i* corresponds to row *i* of `data/phase1/feature_index.csv`. [DATASET.md](DATASET.md) describes the cleaning rules and outputs.

The [leakage analysis](leakage_analysis.py) assigns `random_split`, `region_split`, and `time_split` once for use across models. The random split is a leakage diagnostic: Phase 1 reports that 49,758 of 49,815 random-split test windows overlap a training observation interval. The region split keeps windows from each active region together. The time split uses partitions 1–3 for training, partition 4 for validation, and partition 5 for testing. Phase 1 reports zero test/train observation-interval overlap for the region and time splits. Generalization claims should use those two splits and compare models on equivalent assignments.

The Phase 1 feature builders use measurements from each observation window, excluding labels, region identifiers, timestamps, flare annotations, X-ray fields, and quality fields as predictors. Missing values are imputed and features scaled using the selected split's **training-only** statistics; the same stored parameters are applied to validation and test rows. The recorded IQR outlier counts are descriptive; the pipeline does not clip or remove outliers. Validation data selects decision thresholds, while test labels are reserved for final evaluation.

## Models, evaluation, and available results

| Model | Implementation | Imbalance and selection approach | Artifact family |
| --- | --- | --- | --- |
| Logistic Regression and Random Forest | [classical_models.py](classical_models.py) | Class weights by default; validation-based threshold | `data/phase2/classical/` |
| XGBoost classifier | [xgboost_model.py](xgboost_model.py) | Training-class ratio weight; validation-based early stopping and threshold | `data/phase2/xgboost/` |
| ANN classifier | [ann_model.py](ann_model.py) | Class weights; validation PR-AUC early stopping and threshold | `data/phase2/ann/` |

The supported feature sets are `baseline` and `baseline_temporal`. All model scripts read the shared Phase 1 split assignments and split-specific normalization files. Their evaluations record confusion matrix, precision, recall, F1, average precision (PR-AUC), ROC-AUC where defined, class counts, prevalence, and threshold, with validation and test results kept separate. Average precision and `FL` recall are especially relevant given the rare positive class. Random-split scores, if generated, are diagnostics rather than evidence of generalization. The synopsis also proposes TSS and HSS; the current model reports do not establish those measures.

**No model ranking or numerical performance claim is available yet.** The tracked [classical results table](reports/phase2/classical_model_results.md) has blank scores, and the [experiment manifest](reports/phase2/experiment_manifest.json) marks the study incomplete. The [reload report](reports/phase2/model_reload_verification.md) and [artifact inventory](reports/phase2/artifact_inventory.json) do not verify saved models. See [Phase 2 execution and recovery](PHASE2.md) for the pending backup and audit procedure. Synthetic unit-test success does not replace evaluation on SWAN-SF.

## Repository map

```text
DATASET.md                  Phase 1 data contract, splits, and measured data profile
PHASE2.md                   Colab execution, artifact backup, and recovery
notebooks/phase2a_dataset_acquisition.ipynb
                            Manual official-data acquisition and integrity checks
extract_dataset.py          Extract archives and write the sample manifest
leakage_analysis.py         Create shared split assignments and overlap report
data_cleaning.py            Build aligned features and normalization metadata
temporal_features.py        Magnetic and temporal feature definitions
classical_models.py         Logistic Regression and Random Forest
xgboost_model.py            XGBoost classifier
ann_model.py                Neural-network classifier
phase2_model_common.py      Shared Phase 2 artifact/evaluation helpers
phase2_artifacts.py         Backup, inventory, and model-reload audit
reports/phase2/             Checked-in report templates and audit outputs
tests/                      Small synthetic tests for data/model contracts
data/                       Generated data and model artifacts; ignored by Git
```

## Reproduce the implemented workflow

Run the commands below from the repository root on the `Prasiddhi` branch. They are entry points verified against the source; the full data and training commands have **not** been executed to prepare this README. [DATASET.md](DATASET.md) covers the Phase 1 contract, and [PHASE2.md](PHASE2.md) covers Colab setup and recovery in more detail.

1. **Prepare Python.** Use a compatible Python environment with NumPy, pandas, SciPy, scikit-learn, and joblib for Phase 1 and the classical models. XGBoost and TensorFlow are needed only for their respective model scripts. The pinned [requirements.txt](requirements.txt) is a broad environment snapshot, not a verified minimal Colab specification; inspect installed versions and install compatible packages selectively. On Windows, the project convention is a local `.sfd` virtual environment, which Git ignores.

2. **Acquire the seven archives.** Open the [Phase 2A notebook](notebooks/phase2a_dataset_acquisition.ipynb) in Colab, review its live official metadata and measured storage, mount Google Drive deliberately, and explicitly enable its download cell. It verifies the published size and MD5, calculates SHA-256, and writes an acquisition manifest to persistent Drive storage. It does **not** extract or preprocess. To use the extractor, make verified copies of the seven archives available at the repository root under the exact filenames above. The notebook's Drive `datasets/raw/` folder and the repository checkout are separate locations.

3. **Build Phase 1 outputs.** With all seven archives in place, run:

   ```bash
   python extract_dataset.py
   python leakage_analysis.py
   python data_cleaning.py
   ```

   These commands generate `data/manifest.csv`, `data/splits.csv`, `data/leakage_summary.json`, `data/phase1/feature_index.csv`, aligned `.npy` arrays, feature names, a cleaning report, normalization JSON files, and completion status. Check the status and actual dimensions before training; documented dimensions are not a substitute for local verification. The archives and `data/` are excluded from Git by [.gitignore](.gitignore).

4. **Run a model deliberately.** For example, use the region-independent split and the baseline features:

   ```bash
   python classical_models.py --split region_split --features baseline --models both
   python xgboost_model.py --split region_split --features baseline
   python ann_model.py --split region_split --features baseline
   ```

   Each command trains its model family and writes a run directory under `data/phase2/` with model, configuration, feature order, normalization metadata, metrics, and completion status. Use `--split time_split` for chronological evaluation and `--features baseline_temporal` for the combined features. The scripts also offer `--smoke-max-per-subset 100` for a small diagnostic run; its metrics must not be reported as full-data results. Inspect `--help` before changing seeds or model settings. Full-data training is better suited to a checked Colab runtime with persistent artifact backups.

5. **Audit artifacts and results.** Follow [PHASE2.md](PHASE2.md) to back up complete model run directories to persistent storage, verify SHA-256 inventories, reload models on held-out rows, and regenerate the lightweight reports. Treat a report as final only after its source model, metrics, preprocessing metadata, and backup have been checked. Neither the acquisition notebook nor a model command commits datasets or binaries to Git.

6. **Run offline tests.** The tests use small synthetic fixtures and do not require SWAN-SF:

   ```bash
   python -B -m unittest discover -s tests -v
   ```

## Limitations and next work

- The checked-in Phase 2 experiment matrix and reports are incomplete; full-data training, backup, and reload verification still need evidence from the actual execution environment.
- The supplied window labels support the current binary benchmark. Horizon-specific targets, lead-time definitions, and preflare-pattern analyses proposed by the synopsis still need to be designed and evaluated from the full active-region series.
- Overlapping windows make random splitting unsuitable for a generalization claim. Region and chronological evaluations answer different questions and should be reported separately.
- The dataset is highly imbalanced; precision, recall, PR-AUC, and class-specific errors matter more than accuracy alone. Any operational usefulness claim requires further validation beyond this repository's present evidence.

## Team and references

The team members named in the project synopsis are **Harshit Dharmesh Pandya**, **Prasiddhi Singhal**, and **Mohammed Ahsan Bagsaria**. Individual implementation roles are not documented here.

- Angryk, R. et al., *SWAN-SF*, Harvard Dataverse, version 1.2. [DOI:10.7910/DVN/EBCFKM](https://doi.org/10.7910/DVN/EBCFKM).
- Project team, *Investigating the Early-Warning Capability and Generalization of Machine Learning Models for Solar Flare Forecasting*, synopsis supplied for this project. Its proposed analyses are distinguished above from implemented work.
