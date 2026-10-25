"""Read-only adapters for completed Phase 2/3 SWAN-SF model runs.

Evaluation never fits a model or preprocessing step. Joblib is a pickle format:
only pass --trust-model-files for artifacts from a source you trust.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

import classical_models as cm

ROOT = Path(__file__).resolve().parent
SPLITS = ("region_split", "time_split", "random_split")
MODELS = {"logistic": "model.joblib", "forest": "model.joblib",
          "xgboost": "model.xgb.json", "ann": "model.keras"}


@dataclass
class EvaluatedRun:
    run_dir: Path
    config: dict
    source_metrics: dict
    model_path: Path
    model_sha256: str
    threshold: float
    threshold_source: str
    labels: np.ndarray
    probabilities: np.ndarray
    sample_ids: list[str]
    region_ids: list[str]
    indices: np.ndarray
    data: cm.Phase1Data
    model: object

    @property
    def run_id(self) -> str:
        return self.run_dir.name


def new_directory(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(8):
        path = parent / f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            pass
    raise FileExistsError(f"Cannot allocate a new directory below {parent}")


def write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def discover_runs(roots: list[Path], explicit: list[Path]) -> list[Path]:
    found = {Path(p).resolve() for p in explicit}
    for root in roots:
        if root.is_dir():
            found.update(p.parent.resolve() for p in root.rglob("config.json"))
    return sorted(found)


def read_run_metadata(run_dir: Path) -> tuple[dict, dict, Path, float, str]:
    """Check identity, completion, threshold and saved preprocessing evidence."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory missing: {run_dir}")
    cfg = cm.read_json(run_dir / "config.json")
    met = cm.read_json(run_dir / "metrics.json")
    status = cm.read_json(run_dir / "status.json")
    if status.get("complete") is not True:
        raise ValueError("Run completion marker is not true")
    model_name = cfg.get("model")
    if model_name not in MODELS:
        raise ValueError(f"Unsupported model: {model_name!r}")
    if cfg.get("split") not in SPLITS or cfg.get("feature_set") not in cm.FEATURE_SETS:
        raise ValueError("Unknown split or feature set")
    if cfg.get("label_mapping") != cm.LABELS:
        raise ValueError("Label mapping is not NF=0, FL=1")
    if not isinstance(cfg.get("seed"), int) or not isinstance(cfg.get("smoke_max_per_subset"), int):
        raise ValueError("Missing seed or smoke selection metadata")
    if cfg["smoke_max_per_subset"] < 0:
        raise ValueError("Invalid smoke selection metadata")
    name = cfg.get("model_file", MODELS[model_name])
    if name != MODELS[model_name]:
        raise ValueError("Unexpected model filename")
    model_path = run_dir / name
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model missing: {model_path}")
    try:
        threshold = float(met["validation"]["threshold"])
        test_threshold = float(met["test"]["threshold"])
        source = met["threshold_selection"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Saved validation threshold is unavailable") from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1 or test_threshold != threshold:
        raise ValueError("Validation and test thresholds differ or are invalid")
    if not isinstance(source, str) or not source:
        raise ValueError("Threshold source is missing")
    expected_norm = ("baseline", "temporal") if cfg["feature_set"] == "baseline_temporal" else ("baseline",)
    norm = cfg.get("preprocessing", {}).get("normalization_files", {})
    if set(norm) != set(expected_norm):
        raise ValueError("Saved normalization set does not match feature set")
    for part, info in norm.items():
        filename = f"normalization_{part}.json"
        if info.get("saved_name") != filename:
            raise ValueError("Unsafe normalization filename")
        path = run_dir / filename
        if not path.is_file() or cm.sha256_file(path) != info.get("sha256"):
            raise ValueError(f"Saved normalization hash mismatch: {part}")
    return cfg, met, model_path, threshold, source


def _check_input_hashes(cfg: dict, data: cm.Phase1Data) -> None:
    expected = cfg.get("data_sha256")
    required = {"splits.csv", "feature_index.csv", "feature_columns.json",
                "cleaning_report.json", "status.json", "baseline.npy"}
    if data.temporal is not None:
        required.add("temporal.npy")
    if not isinstance(expected, dict) or not required.issubset(expected):
        raise ValueError("Saved Phase 1 input hashes are incomplete")
    phase1 = data.data_dir / "phase1"
    for name, digest in expected.items():
        if name not in required:
            raise ValueError(f"Unexpected Phase 1 input name: {name}")
        path = data.data_dir / name if name == "splits.csv" else phase1 / name
        if not path.is_file() or cm.sha256_file(path) != digest:
            raise ValueError(f"Phase 1 input hash mismatch: {name}")
    norm = cfg["preprocessing"]["normalization_files"]
    for part, info in norm.items():
        if cm.sha256_file(data.stats_paths[part]) != info["sha256"]:
            raise ValueError(f"Phase 1 normalization changed: {part}")


def _sample_metadata(data_dir: Path, indices: np.ndarray, n_rows: int) -> tuple[list[str], list[str]]:
    wanted = set(int(i) for i in indices)
    ids, regions = [], []
    with (data_dir / "phase1" / "feature_index.csv").open(newline="", encoding="utf-8") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if i in wanted:
                ids.append(row["path"])
                regions.append(row["region_id"])
        if i + 1 != n_rows:
            raise ValueError("Feature index row count changed during evaluation")
    if len(ids) != len(indices) or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate test sample identifiers")
    # The Phase 1 index and subset indices are both in ascending row order.
    if not np.array_equal(indices, np.sort(indices)):
        raise ValueError("Test sample indices are not ordered")
    return ids, regions


def _load_model(cfg: dict, path: Path, trust_joblib: bool) -> object:
    name = cfg["model"]
    if name in ("logistic", "forest"):
        if not trust_joblib:
            raise PermissionError("Joblib model loading requires --trust-model-files; inspect the source first")
        import joblib
        return joblib.load(path)
    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("Install a compatible xgboost version to reload this run") from exc
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        return model
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install a compatible tensorflow version to reload this ANN run") from exc
    return tf.keras.models.load_model(path, compile=False, safe_mode=True)


def _predict(model: object, cfg: dict, data: cm.Phase1Data, indices: np.ndarray,
             batch_size: int) -> np.ndarray:
    width = len(data.columns)
    if cfg["model"] == "ann":
        if model.input_shape[-1] != width:
            raise ValueError("Saved ANN feature count differs")
    else:
        if list(model.classes_) != [0, 1] or model.n_features_in_ != width:
            raise ValueError("Saved model class order or feature count differs")
    result = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        x = cm.normalized_rows(data, indices[start:stop])
        if cfg["model"] == "ann":
            probabilities = np.asarray(model.predict(x, verbose=0)).reshape(-1)
        else:
            scores = np.asarray(model.predict_proba(x))
            if scores.shape != (len(x), 2):
                raise ValueError("Model returned an invalid probability matrix")
            probabilities = scores[:, 1]
        result[start:stop] = probabilities
    validate_probabilities(result, len(indices))
    return result


def validate_probabilities(probabilities: np.ndarray, n: int) -> None:
    if probabilities.ndim != 1 or probabilities.shape != (n,) or n == 0:
        raise ValueError("Probability vector has the wrong shape or is empty")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")


def evaluate_run(run_dir: Path, data_dir: Path, *, trust_joblib: bool,
                 batch_size: int = 4096) -> EvaluatedRun:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cfg, met, model_path, threshold, source = read_run_metadata(run_dir)
    data = cm.load_phase1(data_dir, cfg["split"], cfg["feature_set"])
    if cfg.get("feature_columns") != data.columns:
        raise ValueError("Saved feature order differs from Phase 1")
    _check_input_hashes(cfg, data)
    indices_by_part = cm.subset_indices(data, cfg["split"], cfg["seed"], cfg["smoke_max_per_subset"])
    counts = {part: len(indices_by_part[part]) for part in cm.SUBSETS}
    if met.get("selected_row_counts") != counts:
        raise ValueError("Saved subset counts differ from current Phase 1 assignments")
    indices = indices_by_part["test"]
    labels = data.labels[indices].astype(np.int8)
    ids, regions = _sample_metadata(data_dir, indices, len(data.labels))
    model = _load_model(cfg, model_path, trust_joblib)
    probabilities = _predict(model, cfg, data, indices, batch_size)
    saved_predictions = run_dir / "test_probabilities.npy"
    if saved_predictions.is_file():
        stored = np.load(saved_predictions, mmap_mode="r", allow_pickle=False)
        validate_probabilities(stored, len(labels))
        if not np.allclose(stored, probabilities, atol=1e-5, rtol=1e-5):
            raise ValueError("Recomputed probabilities differ from saved Phase 3 vector")
    recomputed = cm.evaluate(labels, probabilities, threshold)
    original = met["test"]
    for key in ("n", "class_counts", "confusion_matrix"):
        if recomputed[key] != original.get(key):
            raise ValueError(f"Recomputed test {key} differs from saved metrics")
    for key in ("pr_auc_average_precision", "roc_auc"):
        a, b = recomputed[key], original.get(key)
        if (a is None) != (b is None) or (a is not None and not np.isclose(a, b, atol=1e-5)):
            raise ValueError(f"Recomputed test {key} differs from saved metrics")
    return EvaluatedRun(run_dir, cfg, met, model_path, cm.sha256_file(model_path),
                        threshold, source, labels, probabilities, ids, regions, indices, data, model)


def environment_versions() -> dict:
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__}


def git_revision() -> dict:
    import subprocess
    def call(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"branch": call("branch", "--show-current"), "commit": call("rev-parse", "HEAD")}
