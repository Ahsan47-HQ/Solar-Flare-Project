"""Shared data and metrics for the verified Phase 2--4 rebuild."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_cleaning import transform_features


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = DATA / "rebuild" / "runs"
REPORTS = ROOT / "reports" / "rebuild"
LEADS = (0, 6, 12, 24, 48)
SPLITS = ("region_split", "time_split", "random_split")
FEATURES = ("baseline", "baseline_temporal")
MODELS = ("logistic", "forest", "xgboost", "ann")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(data_dir: Path = DATA) -> dict:
    index = data_dir / "phase1" / "feature_index.csv"
    splits = {name: [] for name in SPLITS}
    paths, regions = [], []
    with index.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paths.append(row["path"])
            regions.append(row["region_id"])
            for split in SPLITS:
                splits[split].append(row[split])
    return {
        "paths": np.asarray(paths, dtype=object),
        "regions": np.asarray(regions, dtype=object),
        "splits": {name: np.asarray(values, dtype="U5") for name, values in splits.items()},
    }


def rows_for(index: dict, targets: np.ndarray, lead: int, split: str, *, common_cohort: bool) -> dict[str, np.ndarray]:
    if lead not in LEADS or split not in SPLITS:
        raise ValueError("Invalid lead or split")
    column = LEADS.index(lead)
    valid = targets[:, column] >= 0
    if common_cohort:
        valid &= np.all(targets[:, 1:] >= 0, axis=1)
    assignments = index["splits"][split]
    return {part: np.flatnonzero(valid & (assignments == part)) for part in ("train", "val", "test")}


def features_for(data_dir: Path, rows: np.ndarray, split: str, feature_set: str) -> np.ndarray:
    phase1 = data_dir / "phase1"
    base = np.load(phase1 / "baseline.npy", mmap_mode="r")
    base_stats = json.loads((phase1 / f"normalization_{split}_baseline.json").read_text(encoding="utf-8"))
    x = transform_features(base[rows], base_stats).astype(np.float32)
    if feature_set == "baseline_temporal":
        temporal = np.load(phase1 / "temporal.npy", mmap_mode="r")
        stats = json.loads((phase1 / f"normalization_{split}_temporal.json").read_text(encoding="utf-8"))
        z = transform_features(temporal[rows], stats).astype(np.float32)
        x = np.column_stack((x, z))
    elif feature_set != "baseline":
        raise ValueError(feature_set)
    return np.ascontiguousarray(x)


def feature_names(data_dir: Path, feature_set: str) -> list[str]:
    names = json.loads((data_dir / "phase1" / "feature_columns.json").read_text(encoding="utf-8"))
    return names["baseline"] + (names["temporal"] if feature_set == "baseline_temporal" else [])


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probability)
    score = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.nanargmax(score)])


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    predicted = probability >= threshold
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    tss = (tp / (tp + fn) - fp / (fp + tn)) if (tp + fn and fp + tn) else None
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fp * fn) / hss_den if hss_den else None
    return {
        "n": len(y), "positive": int(y.sum()), "prevalence": float(y.mean()),
        "threshold": threshold,
        "average_precision": float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "tss": tss, "hss": hss,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def run_path(phase: str, lead: int, split: str, feature_set: str, model: str) -> Path:
    return RUNS / phase / f"lead_{lead}h" / split / feature_set / model


def load_targets(data_dir: Path = DATA) -> np.ndarray:
    return np.load(data_dir / "phase2" / "horizon_targets.npy", mmap_mode="r")

