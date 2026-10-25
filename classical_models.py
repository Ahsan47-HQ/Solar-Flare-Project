"""Train leakage-aware Logistic Regression and Random Forest SWAN-SF baselines.

Example (from the repository root):
    python classical_models.py --split region_split --features baseline --models both

Outputs are written below data/phase2/classical by default (already gitignored).
The random split and every --smoke-max-per-subset run are diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from data_cleaning import INDEX_FIELDS, SPLITS, transform_features
from temporal_features import BASE_COLUMNS, TEMPORAL_COLUMNS


ROOT = Path(__file__).resolve().parent
LABELS = {"NF": 0, "FL": 1}
SUBSETS = ("train", "val", "test")
FEATURE_SETS = ("baseline", "baseline_temporal")
MODEL_NAMES = ("logistic", "forest")


@dataclass
class Phase1Data:
    baseline: np.ndarray
    temporal: np.ndarray | None
    labels: np.ndarray
    assignments: dict[str, np.ndarray]
    columns: list[str]
    stats: dict[str, dict]
    stats_paths: dict[str, Path]
    full_counts: dict[str, dict[str, int]]
    data_dir: Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def encode_label(label: str) -> int:
    if label not in LABELS:
        raise ValueError(f"Unexpected label {label!r}; expected NF or FL")
    return LABELS[label]


def validate_split_name(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"Invalid split {split!r}; choose from {SPLITS}")


def _read_assignments(data_dir: Path, n_rows: int, split: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Verify feature-index rows against the authoritative splits.csv by path."""
    split_path = data_dir / "splits.csv"
    index_path = data_dir / "phase1" / "feature_index.csv"
    for path in (split_path, index_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_by_path = {}
    with split_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != INDEX_FIELDS:
            raise ValueError("splits.csv columns do not match Phase 1's index schema")
        for row in reader:
            path = row["path"]
            if not path or path in source_by_path:
                raise ValueError(f"Empty or duplicate path in splits.csv: {path!r}")
            source_by_path[path] = tuple(row[field] for field in INDEX_FIELDS)
    if len(source_by_path) != n_rows:
        raise ValueError("splits.csv row count does not match feature arrays")
    labels = np.empty(n_rows, dtype=np.uint8)
    assignments = {name: np.empty(n_rows, dtype="U5") for name in SPLITS}
    counts = {part: Counter() for part in SUBSETS}
    region_parts: dict[str, str] = {}
    seen = set()
    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != INDEX_FIELDS:
            raise ValueError("feature_index.csv columns do not match Phase 1's index schema")
        for i, row in enumerate(reader):
            if i >= n_rows:
                raise ValueError("feature_index.csv has more rows than feature arrays")
            path = row["path"]
            if not path or path in seen:
                raise ValueError(f"Empty or duplicate path in feature_index.csv: {path!r}")
            seen.add(path)
            expected = source_by_path.get(path)
            if expected is None or tuple(row[field] for field in INDEX_FIELDS) != expected:
                raise ValueError(f"Feature index does not match splits.csv for {path!r}")
            labels[i] = encode_label(row["label"])
            for name in SPLITS:
                part = row[name]
                if part not in SUBSETS:
                    raise ValueError(f"Invalid {name} assignment {part!r} for {path!r}")
                assignments[name][i] = part
            region = row["region_id"]
            region_part = row["region_split"]
            if not region:
                raise ValueError(f"Missing region ID for {path!r}")
            if region in region_parts and region_parts[region] != region_part:
                raise ValueError(f"Region {region!r} crosses region_split subsets")
            region_parts[region] = region_part
            partition = row["partition"]
            expected_time = {
                "partition1": "train", "partition2": "train", "partition3": "train",
                "partition4": "val", "partition5": "test",
            }.get(partition)
            if expected_time is None or row["time_split"] != expected_time:
                raise ValueError(f"Invalid time_split for {path!r}: {partition!r}")
            counts[row[split]][row["label"]] += 1
    if len(seen) != n_rows:
        raise ValueError("feature_index.csv row count does not match feature arrays")
    for part in SUBSETS:
        if not counts[part]:
            raise ValueError(f"{split} has no {part} rows")
    return labels, assignments, {
        part: {"NF": counts[part]["NF"], "FL": counts[part]["FL"]}
        for part in SUBSETS
    }


def _validate_stats(stats: dict, columns: list[str], train_rows: int, path: Path) -> None:
    if stats.get("feature_columns") != columns or stats.get("training_rows") != train_rows:
        raise ValueError(f"Normalization metadata is incompatible with {path}")
    for key in ("impute_median", "mean_after_imputation", "scale_after_imputation"):
        values = stats.get(key)
        if not isinstance(values, list) or len(values) != len(columns):
            raise ValueError(f"Invalid normalization {key} in {path}")
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            raise ValueError(f"Nonfinite normalization {key} in {path}")
    if np.any(np.asarray(stats["scale_after_imputation"]) <= 0):
        raise ValueError(f"Nonpositive normalization scale in {path}")
    missing = stats.get("missing_training_values")
    if not isinstance(missing, dict) or list(missing) != columns:
        raise ValueError(f"Missing-value metadata is incompatible with {path}")


def load_phase1(data_dir: Path, split: str, feature_set: str) -> Phase1Data:
    validate_split_name(split)
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Invalid feature set {feature_set!r}")
    data_dir = Path(data_dir)
    phase1 = data_dir / "phase1"
    status = read_json(phase1 / "status.json")
    if status.get("complete") is not True:
        raise ValueError("Phase 1 status is not complete")
    baseline_path = phase1 / "baseline.npy"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline = np.load(baseline_path, mmap_mode="r", allow_pickle=False)
    if baseline.ndim != 2 or baseline.shape[1] != len(BASE_COLUMNS):
        raise ValueError("baseline.npy shape does not match Phase 1 feature definitions")
    n_rows = baseline.shape[0]
    if n_rows == 0 or status.get("windows") != n_rows:
        raise ValueError("Phase 1 status row count does not match baseline.npy")
    feature_columns = read_json(phase1 / "feature_columns.json")
    if feature_columns.get("baseline") != BASE_COLUMNS or feature_columns.get("temporal") != TEMPORAL_COLUMNS:
        raise ValueError("Feature names/order do not match Phase 1 feature definitions")
    temporal_path = phase1 / "temporal.npy"
    if not temporal_path.is_file():
        raise FileNotFoundError(temporal_path)
    temporal_array = np.load(temporal_path, mmap_mode="r", allow_pickle=False)
    if temporal_array.ndim != 2 or temporal_array.shape != (n_rows, len(TEMPORAL_COLUMNS)):
        raise ValueError("temporal.npy rows or columns do not align with baseline.npy")
    temporal = temporal_array if feature_set == "baseline_temporal" else None
    labels, assignments, counts = _read_assignments(data_dir, n_rows, split)
    report = read_json(phase1 / "cleaning_report.json")
    observed = {"NF": int(np.count_nonzero(labels == 0)), "FL": int(np.count_nonzero(labels == 1))}
    if report.get("windows") != n_rows or report.get("class_distribution") != observed:
        raise ValueError("Cleaning report counts do not match feature_index.csv")
    if report.get("class_distribution_by_split", {}).get(split) != counts:
        raise ValueError("Cleaning report split counts do not match feature_index.csv")
    train_rows = sum(counts["train"].values())
    stats = {}
    stats_paths = {}
    sets = ("baseline", "temporal") if temporal is not None else ("baseline",)
    for name in sets:
        path = phase1 / f"normalization_{split}_{name}.json"
        values = read_json(path)
        _validate_stats(values, BASE_COLUMNS if name == "baseline" else TEMPORAL_COLUMNS, train_rows, path)
        stats[name] = values
        stats_paths[name] = path
    return Phase1Data(
        baseline=baseline, temporal=temporal, labels=labels, assignments=assignments,
        columns=BASE_COLUMNS + (TEMPORAL_COLUMNS if temporal is not None else []),
        stats=stats, stats_paths=stats_paths, full_counts=counts, data_dir=data_dir,
    )


def subset_indices(data: Phase1Data, split: str, seed: int, smoke_max: int) -> dict[str, np.ndarray]:
    """Select within existing assignments; smoke mode is stratified and diagnostic."""
    result = {}
    for part in SUBSETS:
        indices = np.flatnonzero(data.assignments[split] == part)
        if smoke_max and len(indices) > smoke_max:
            rng = np.random.default_rng(seed + SUBSETS.index(part))
            positive = indices[data.labels[indices] == 1]
            negative = indices[data.labels[indices] == 0]
            if len(positive) and len(negative) and smoke_max >= 2:
                n_positive = max(1, min(len(positive), round(smoke_max * len(positive) / len(indices))))
                n_negative = smoke_max - n_positive
                if n_negative > len(negative):
                    n_negative = len(negative)
                    n_positive = smoke_max - n_negative
                indices = np.sort(np.concatenate((
                    rng.choice(positive, n_positive, replace=False),
                    rng.choice(negative, n_negative, replace=False),
                )))
            else:
                indices = np.sort(rng.choice(indices, smoke_max, replace=False))
        result[part] = indices
    return result


def normalized_rows(data: Phase1Data, indices: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Transform selected rows in batches using only saved training-fitted stats."""
    output = np.empty((len(indices), len(data.columns)), dtype=np.float32)
    base_width = len(BASE_COLUMNS)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        batch = indices[start:stop]
        output[start:stop, :base_width] = transform_features(data.baseline[batch], data.stats["baseline"])
        if data.temporal is not None:
            output[start:stop, base_width:] = transform_features(
                data.temporal[batch], data.stats["temporal"]
            )
    if not np.all(np.isfinite(output)):
        raise ValueError("Normalized features contain nonfinite values")
    return output


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, str]:
    if len(np.unique(y_true)) < 2:
        return 0.5, "fixed_0.5_validation_has_one_class"
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5, "fixed_0.5_no_thresholds"
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best = np.flatnonzero(f1 == np.max(f1))
    return float(thresholds[best[-1]]), "validation_max_f1"


def _class_metrics(tp: int, fp: int, fn: int) -> dict:
    support = tp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / support if support else None
    f1 = (2 * tp / (2 * tp + fp + fn)) if support else None
    return {"support": support, "precision": precision, "recall": recall, "f1": f1}


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("Evaluation labels and probabilities must have matching nonempty lengths")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    predicted = probabilities >= threshold
    truth = y_true == 1
    tn = int(np.count_nonzero(~truth & ~predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    tp = int(np.count_nonzero(truth & predicted))
    both_classes = tp + fn > 0 and tn + fp > 0
    return {
        "n": len(y_true),
        "class_counts": {"NF": tn + fp, "FL": tp + fn},
        "fl_prevalence": (tp + fn) / len(y_true),
        "threshold": threshold,
        "confusion_matrix_labels": ["NF", "FL"],
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "FL": _class_metrics(tp, fp, fn),
        "NF": _class_metrics(tn, fn, fp),
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if both_classes else None,
        "undefined_auc_reason": None if both_classes else "Only one class is present",
    }


def make_model(name: str, args: argparse.Namespace):
    if name == "logistic":
        return LogisticRegression(
            C=args.logistic_c, max_iter=args.logistic_max_iter, solver="liblinear",
            class_weight="balanced" if args.class_weight == "balanced" else None,
            random_state=args.seed,
        )
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=args.forest_trees, max_depth=args.forest_max_depth,
            min_samples_leaf=args.forest_min_leaf, n_jobs=args.n_jobs,
            class_weight="balanced_subsample" if args.class_weight == "balanced" else None,
            random_state=args.seed,
        )
    raise ValueError(f"Unknown model: {name}")


def run(args: argparse.Namespace) -> list[Path]:
    if args.smoke_max_per_subset < 0 or args.smoke_max_per_subset == 1:
        raise ValueError("--smoke-max-per-subset must be 0 or at least 2")
    if args.logistic_c <= 0 or args.logistic_max_iter <= 0:
        raise ValueError("Logistic settings must be positive")
    if args.forest_trees <= 0 or args.forest_min_leaf <= 0 or args.forest_max_depth <= 0:
        raise ValueError("Forest settings must be positive")
    data = load_phase1(args.data_dir, args.split, args.features)
    indices = subset_indices(data, args.split, args.seed, args.smoke_max_per_subset)
    y = {part: data.labels[indices[part]].astype(np.int8) for part in SUBSETS}
    if len(np.unique(y["train"])) < 2:
        raise ValueError("Training subset must contain both NF and FL")
    matrices = {part: normalized_rows(data, indices[part]) for part in SUBSETS}
    names = MODEL_NAMES if args.models == "both" else (args.models,)
    data_hashes = {
        "feature_index.csv": sha256_file(data.data_dir / "phase1" / "feature_index.csv"),
        "splits.csv": sha256_file(data.data_dir / "splits.csv"),
        "feature_columns.json": sha256_file(data.data_dir / "phase1" / "feature_columns.json"),
        "cleaning_report.json": sha256_file(data.data_dir / "phase1" / "cleaning_report.json"),
        "status.json": sha256_file(data.data_dir / "phase1" / "status.json"),
        "baseline.npy": sha256_file(data.data_dir / "phase1" / "baseline.npy"),
    }
    if data.temporal is not None:
        data_hashes["temporal.npy"] = sha256_file(data.data_dir / "phase1" / "temporal.npy")
    source_hashes = {
        name: sha256_file(ROOT / name)
        for name in ("classical_models.py", "data_cleaning.py", "temporal_features.py")
    }
    saved = []
    for name in names:
        model = make_model(name, args)
        model.fit(matrices["train"], y["train"])
        if list(model.classes_) != [0, 1]:
            raise ValueError("Model class order differs from NF=0, FL=1")
        val_prob = model.predict_proba(matrices["val"])[:, 1]
        threshold, threshold_source = choose_threshold(y["val"], val_prob)
        test_prob = model.predict_proba(matrices["test"])[:, 1]
        mode = "smoke" if args.smoke_max_per_subset else "full"
        interpretation = (
            "diagnostic_only"
            if mode == "smoke" or args.split == "random_split"
            else "region_independent" if args.split == "region_split" else "chronological"
        )
        run_dir = (
            args.output_dir / args.split / args.features / mode / name
            / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        for set_name, path in data.stats_paths.items():
            shutil.copyfile(path, run_dir / f"normalization_{set_name}.json")
        config = {
            "model": name, "split": args.split, "feature_set": args.features,
            "seed": args.seed, "class_weight_strategy": model.class_weight,
            "smoke_max_per_subset": args.smoke_max_per_subset,
            "model_parameters": model.get_params(),
            "feature_columns": data.columns,
            "label_mapping": LABELS,
            "preprocessing": {
                "method": "data_cleaning.transform_features; Phase 1 training-only median imputation and z-score",
                "normalization_files": {
                    set_name: {"saved_name": f"normalization_{set_name}.json", "sha256": sha256_file(path)}
                    for set_name, path in data.stats_paths.items()
                },
                "outlier_handling": "none",
            },
            "data_sha256": data_hashes,
            "source_sha256": source_hashes,
            "versions": {
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
        }
        results = {
            "evaluation_interpretation": interpretation,
            "training": {
                "n": len(y["train"]),
                "class_counts": {"NF": int(np.count_nonzero(y["train"] == 0)),
                                 "FL": int(np.count_nonzero(y["train"] == 1))},
            },
            "full_split_counts": data.full_counts,
            "selected_row_counts": {part: len(indices[part]) for part in SUBSETS},
            "threshold_selection": threshold_source,
            "validation": evaluate(y["val"], val_prob, threshold),
            "test": evaluate(y["test"], test_prob, threshold),
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        joblib.dump(model, run_dir / "model.joblib", compress=3)
        (run_dir / "status.json").write_text(
            json.dumps({"complete": True, "completed_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2),
            encoding="utf-8",
        )
        print(
            f"{name} {args.split} {args.features} {mode}: "
            f"validation AP={results['validation']['pr_auc_average_precision']} "
            f"FL recall={results['validation']['FL']['recall']}; "
            f"test AP={results['test']['pr_auc_average_precision']} "
            f"FL recall={results['test']['FL']['recall']} [{interpretation}]"
        )
        print(f"Saved: {run_dir}")
        saved.append(run_dir)
    return saved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "phase2" / "classical")
    parser.add_argument("--split", choices=SPLITS, default="region_split")
    parser.add_argument("--features", choices=FEATURE_SETS, default="baseline")
    parser.add_argument("--models", choices=("both", *MODEL_NAMES), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--logistic-max-iter", type=int, default=500)
    parser.add_argument("--forest-trees", type=int, default=100)
    parser.add_argument("--forest-max-depth", type=int, default=16)
    parser.add_argument("--forest-min-leaf", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--smoke-max-per-subset", type=int, default=0,
                        help="Use at most N rows from each existing subset; diagnostic only (0 means full).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
