"""Export and verify saved classifiers as ONNX, preserving the source models.

Usage: .sfd/Scripts/python.exe -m backend.convert_models_onnx [--limit N]
Outputs live in backend/models and are copied into the backend bundle after validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
OUTPUT = Path(__file__).resolve().parent / "models"
os.environ["HELION_DATA_DIR"] = str(DATA)
os.environ["HELION_SOURCE_ARTIFACTS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import onnx
import onnxruntime as ort

from backend.features import transform_features
from backend.server import catalog, read_json


def validation_input(item: dict, count: int = 64) -> np.ndarray:
    rows = np.load(item["_run_dir"] / "test_rows.npy", mmap_mode="r")
    labels = np.load(DATA / "phase2" / "horizon_targets.npy", mmap_mode="r")
    lead_column = (0, 6, 12, 24, 48).index(item["lead_hours"])
    chosen = []
    for label in (0, 1):
        candidates = rows[labels[rows, lead_column] == label]
        if len(candidates):
            chosen.extend(candidates[np.linspace(0, len(candidates) - 1,
                                                 min(count // 2, len(candidates)), dtype=int)])
    chosen = np.asarray(chosen, dtype=np.int64)
    split = item["split"]
    base = np.load(DATA / "phase1" / "baseline.npy", mmap_mode="r")[chosen]
    stats = read_json(DATA / "phase1" / f"normalization_{split}_baseline.json")
    features = transform_features(base, stats)
    if item["feature_set"] == "baseline_temporal":
        temporal = np.load(DATA / "phase1" / "temporal.npy", mmap_mode="r")[chosen]
        stats = read_json(DATA / "phase1" / f"normalization_{split}_temporal.json")
        features = np.column_stack((features, transform_features(temporal, stats)))
    return np.ascontiguousarray(features, dtype=np.float32)


def source_model(item: dict):
    path = item["_run_dir"] / item["_config"]["model_file"]
    family = item["model"]
    if family == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(str(path))
        model.set_params(device="cpu", n_jobs=2)
        return model
    if family == "ann":
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False)
    import joblib
    model = joblib.load(path)
    if family == "forest":
        model.n_jobs = 1
    return model


def export_model(item: dict, model, destination: Path) -> None:
    family = item["model"]
    width = item["feature_count"]
    if family == "xgboost":
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        # XGBoost predict_proba uses best_iteration when early stopping was used.
        # Export only that same number of trees.
        best = item["_config"].get("extra", {}).get("best_iteration")
        if best is not None:
            model._Booster = model.get_booster()[:int(best) + 1]
        converted = onnxmltools.convert_xgboost(
            model, initial_types=[("features", FloatTensorType([None, width]))],
            target_opset=15)
        onnx.save(converted, destination)
    elif family in {"logistic", "forest"}:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        converted = convert_sklearn(
            model, initial_types=[("features", FloatTensorType([None, width]))],
            options={id(model): {"zipmap": False}}, target_opset=17)
        onnx.save(converted, destination)
    elif family == "ann":
        model.export(str(destination), format="onnx", verbose=False, opset_version=17)
    else:
        raise ValueError(f"Unsupported model family: {family}")


def probabilities(session: ort.InferenceSession, features: np.ndarray) -> np.ndarray:
    outputs = session.run(None, {session.get_inputs()[0].name: features})
    for value in reversed(outputs):
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[0] == len(features):
            if value.shape[1] == 2:
                return value[:, 1].astype(np.float64)
            if value.shape[1] == 1:
                return value[:, 0].astype(np.float64)
    raise ValueError(f"No probability array found in ONNX outputs: {[v.shape for v in outputs]}")


def convert_one(item: dict) -> dict:
    source_file = item["_run_dir"] / item["_config"]["model_file"]
    source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    model = source_model(item)
    features = validation_input(item)
    family = item["model"]
    if family == "ann":
        expected = model.predict(features, verbose=0).reshape(-1).astype(np.float64)
    else:
        expected = model.predict_proba(features)[:, 1].astype(np.float64)
    destination = OUTPUT / (item["id"].replace(":", "_") + ".onnx")
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_model(item, model, destination)
    onnx.checker.check_model(str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(destination), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    actual = probabilities(session, features)
    difference = np.abs(actual - expected)
    max_error = float(np.max(difference))
    threshold = item["threshold"]
    disagreement = int(np.sum((actual >= threshold) != (expected >= threshold)))
    # Converted float32 graphs can differ slightly, especially for neural nets.
    verified = math.isfinite(max_error) and max_error <= 1e-4 and disagreement == 0
    if not verified:
        destination.unlink(missing_ok=True)
    return {"model_id": item["id"], "family": family, "verified": verified,
            "source_model_sha256": source_hash,
            "sample_count": len(features), "max_abs_probability_error": max_error,
            "threshold_disagreements": disagreement,
            "onnx_file": f"rebuild/onnx/{destination.name}" if verified else None,
            "onnx_sha256": hashlib.sha256(destination.read_bytes()).hexdigest() if verified else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    items = sorted(catalog().values(), key=lambda x: x["id"])
    if args.limit is not None:
        # Representative first examples of each family for a quick converter check.
        items = [next(item for item in items if item["model"] == family)
                 for family in ("xgboost", "logistic", "forest", "ann")][:args.limit]
    report = []
    for index, item in enumerate(items, 1):
        try:
            result = convert_one(item)
        except Exception as error:
            result = {"model_id": item["id"], "family": item["model"],
                      "verified": False, "error": f"{type(error).__name__}: {error}"}
        report.append(result)
        print(f"{index}/{len(items)} {item['id']}: {result}", flush=True)
    if args.limit is None:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if any(not item["verified"] for item in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
