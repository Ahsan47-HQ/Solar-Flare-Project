"""Shared, training-only data preparation and artifact metadata for Phase 2."""

from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

import classical_models as cm


def prepare(data_dir: Path, split: str, features: str, seed: int, smoke_max: int):
    if smoke_max < 0 or smoke_max == 1:
        raise ValueError("smoke_max_per_subset must be 0 or at least 2")
    data = cm.load_phase1(data_dir, split, features)
    indices = cm.subset_indices(data, split, seed, smoke_max)
    labels = {part: data.labels[indices[part]].astype(np.int8) for part in cm.SUBSETS}
    if len(np.unique(labels["train"])) != 2:
        raise ValueError("Training subset must contain NF and FL")
    matrices = {part: cm.normalized_rows(data, indices[part]) for part in cm.SUBSETS}
    return data, indices, labels, matrices


def new_run_dir(output_dir: Path, split: str, features: str, smoke_max: int) -> Path:
    mode = "smoke" if smoke_max else "full"
    path = (output_dir / split / features / mode /
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_metadata(run_dir: Path, *, model_name: str, data: cm.Phase1Data,
                  indices: dict, labels: dict, probabilities: dict,
                  threshold: float, threshold_source: str, split: str,
                  features: str, seed: int, smoke_max: int,
                  parameters: dict, versions: dict, source_file: Path,
                  model_filename: str) -> None:
    """Write completion marker last, after the caller has saved the model."""
    if not (run_dir / model_filename).is_file():
        raise FileNotFoundError(run_dir / model_filename)
    normalization = {}
    for name, source in data.stats_paths.items():
        saved_name = f"normalization_{name}.json"
        shutil.copyfile(source, run_dir / saved_name)
        normalization[name] = {"saved_name": saved_name, "sha256": cm.sha256_file(source)}
    phase1 = data.data_dir / "phase1"
    inputs = {
        "feature_index.csv": phase1 / "feature_index.csv",
        "splits.csv": data.data_dir / "splits.csv",
        "feature_columns.json": phase1 / "feature_columns.json",
        "cleaning_report.json": phase1 / "cleaning_report.json",
        "status.json": phase1 / "status.json",
        "baseline.npy": phase1 / "baseline.npy",
    }
    if data.temporal is not None:
        inputs["temporal.npy"] = phase1 / "temporal.npy"
    interpretation = ("diagnostic_only" if smoke_max or split == "random_split" else
                      "region_independent" if split == "region_split" else "chronological")
    config = {
        "model": model_name, "model_file": model_filename, "split": split,
        "feature_set": features, "seed": seed, "smoke_max_per_subset": smoke_max,
        "model_parameters": parameters, "feature_columns": data.columns,
        "label_mapping": cm.LABELS,
        "preprocessing": {
            "method": "data_cleaning.transform_features; Phase 1 training-only median imputation and z-score",
            "normalization_files": normalization, "outlier_handling": "none",
        },
        "data_sha256": {name: cm.sha256_file(path) for name, path in inputs.items()},
        "source_sha256": {
            source_file.name: cm.sha256_file(source_file),
            "classical_models.py": cm.sha256_file(cm.ROOT / "classical_models.py"),
            "phase2_model_common.py": cm.sha256_file(Path(__file__)),
            "data_cleaning.py": cm.sha256_file(cm.ROOT / "data_cleaning.py"),
            "temporal_features.py": cm.sha256_file(cm.ROOT / "temporal_features.py"),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": cm.pd.__version__, "scipy": cm.scipy.__version__,
                     "scikit_learn": cm.sklearn.__version__, "joblib": cm.joblib.__version__,
                     **versions},
    }
    metrics = {
        "evaluation_interpretation": interpretation,
        "training": {"n": len(labels["train"]), "class_counts": {
            "NF": int(np.count_nonzero(labels["train"] == 0)),
            "FL": int(np.count_nonzero(labels["train"] == 1)),
        }},
        "full_split_counts": data.full_counts,
        "selected_row_counts": {part: len(indices[part]) for part in cm.SUBSETS},
        "threshold_selection": threshold_source,
        "validation": cm.evaluate(labels["val"], probabilities["val"], threshold),
        "test": cm.evaluate(labels["test"], probabilities["test"], threshold),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({
        "complete": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
