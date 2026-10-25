# Phase 2 requirements audit — provisional

The repository at `60ba6b21dc2057dbfa5a62e7b284bc4a62bacf94` contains
`DATASET.md` and Phase 1 scripts, but no README, project synopsis, phasewise
plan, or detailed XGBoost/ANN experiment matrix. This audit uses the available
Phase 1 contract and the project owner's Phase 2 instructions. It must be
reconciled with the original project plan if that plan exists outside the repo.

| Requirement | Evidence | Status | Remaining action | Priority |
| --- | --- | --- | --- | --- |
| Shared leakage-aware splits | `DATASET.md`, `leakage_analysis.py`, `data_cleaning.py` | Defined in source; full files absent locally | Validate Colab `splits.csv` and Phase 1 hashes | Critical |
| Logistic Regression and Random Forest source | `classical_models.py`, 17 offline tests including new artifact checks | Implemented locally; full runs user-reported | Verify all eight saved runs, backup and inference | Critical |
| XGBoost classifier | Project owner's original Phase 2 deliverables; `xgboost_model.py` | Source implemented; mocked offline test passed | Run real XGBoost smoke and required full experiments in Colab | High |
| ANN classifier | Project owner's original Phase 2 deliverables; `ann_model.py` | Source implemented; mocked offline test passed | Run real TensorFlow smoke and required full experiments in Colab | High |
| Training-only preprocessing | Phase 1 normalization schema; `phase2_model_common.py` | Implemented and synthetic tested | Verify actual Colab normalization files and hashes | Critical |
| Validation-only threshold and selection | Model scripts and tests | Implemented and synthetic tested | Audit actual metrics and real inference | High |
| Persistent models and metadata | `phase2_artifacts.py`; Colab paths provided by owner | Backup code tested; Drive inaccessible in this workspace | Run backup in Colab, resolve conflicts, verify SHA-256 | Critical |
| Consolidated results | `phase2_artifacts.py`; `reports/phase2/` | Preliminary empty report generated because artifacts absent | Regenerate from actual saved metrics | Critical |
| Detailed original experiment matrix | Not found in tracked repo or local workspace | Unknown | Supply synopsis/phasewise plan or confirm model configurations | High |

The local Phase 2 reports have eight blank classical metric rows. The numbers
reported in terminal output are treated as unverified and are not substituted
for saved metrics. XGBoost/ANN real training and all Drive checks remain pending.

Before Phase 3: reconcile the original plan, verify all required Phase 2 runs,
hash-check persistent backups, reload and infer with every model, refresh the
reports from saved metrics, review Git changes, and obtain checkpoint approval.
