"""FastAPI service for saved, retrospective solar-flare models."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import threading
from datetime import datetime, timedelta
from functools import lru_cache
from http import HTTPStatus
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from backend.features import (
    BASE_COLUMNS,
    MAGNETIC_COLUMNS,
    TEMPORAL_COLUMNS,
    compute_region_features,
    transform_features,
)

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("HELION_DATA_DIR", ROOT / "data")).resolve()
RESULTS = DATA / "reports" / "phase4" / "phase4_results.csv"
MAX_REQUEST_BYTES = 1_200_000
MODEL_LOCK = threading.RLock()
LOGGER = logging.getLogger("solar-web")
DEFAULT_ORIGINS = "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173"
ORIGINS = [origin.strip().rstrip("/") for origin in os.environ.get("HELION_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if origin.strip()]
app = FastAPI(title="Helion API", version="2.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    observations: list[dict]


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_ids: list[str]
    observations: list[dict]


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def onnx_path(model_id: str) -> Path:
    return DATA / "rebuild" / "onnx" / f"{model_id.replace(':', '_')}.onnx"


@lru_cache(maxsize=1)
def catalog() -> dict[str, dict]:
    items = {}
    run_root = DATA / "rebuild" / "runs"
    for config_path in run_root.glob("phase*/lead_*h/*/*/*/config.json"):
        run_dir = config_path.parent
        try:
            config = read_json(config_path)
            status = read_json(run_dir / "status.json")
            metrics = read_json(run_dir / "metrics.json")
            model_id = ":".join(str(config[key]) for key in
                                ("phase", "lead_hours", "split", "feature_set", "model"))
            has_onnx = onnx_path(model_id).is_file()
            source_only = os.environ.get("HELION_SOURCE_ARTIFACTS") == "1" and (run_dir / config["model_file"]).is_file()
            if not status.get("complete") or not (has_onnx or source_only):
                continue
        except (OSError, ValueError, KeyError, TypeError) as error:
            LOGGER.warning("Ignoring incomplete run %s: %s", run_dir, error)
            continue
        items[model_id] = {
            "id": model_id,
            "phase": config["phase"],
            "lead_hours": config["lead_hours"],
            "split": config["split"],
            "feature_set": config["feature_set"],
            "model": config["model"],
            "feature_count": len(config["feature_names"]),
            "target_window_hours": config["target_window_hours"],
            "threshold": metrics["validation"]["threshold"],
            "test_ap": metrics["test"]["average_precision"],
            "test_tss": metrics["test"]["tss"],
            "test_positive": metrics["test"]["positive"],
            "test_n": metrics["test"]["n"],
            "diagnostic_only": config["split"] == "random_split",
            "inference_format": "onnx" if has_onnx else "source-only",
            "_run_dir": run_dir,
            "_config": config,
        }
    if not items:
        raise APIError(HTTPStatus.SERVICE_UNAVAILABLE, "models_missing",
                       "No complete saved model runs were found in data/rebuild/runs.")
    return items


def public_model(model: dict) -> dict:
    return {key: value for key, value in model.items() if not key.startswith("_")}


def get_model(model_id: object) -> dict:
    if not isinstance(model_id, str):
        raise APIError(400, "model_required", "Choose a saved model.")
    model = catalog().get(model_id)
    if model is None:
        raise APIError(404, "model_unknown", "The selected model is unavailable.")
    if model["diagnostic_only"]:
        raise APIError(400, "diagnostic_model", "Random-split runs are analysis diagnostics, not forecast choices.")
    return model


@lru_cache(maxsize=12)
def load_saved_model(model_id: str):
    get_model(model_id)
    converted = onnx_path(model_id)
    if not converted.is_file():
        raise APIError(503, "onnx_missing", "The verified ONNX model is unavailable.")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(str(converted), sess_options=options,
                                providers=["CPUExecutionProvider"])


@lru_cache(maxsize=6)
def load_stats(split: str, kind: str) -> dict:
    if split not in {"region_split", "time_split"} or kind not in {"baseline", "temporal"}:
        raise APIError(400, "preprocessing", "Unsupported preprocessing selection.")
    return read_json(DATA / "phase1" / f"normalization_{split}_{kind}.json")


def parse_timestamp(raw: object) -> datetime:
    if not isinstance(raw, str):
        raise APIError(422, "timestamp", "Each row needs an ISO timestamp.")
    try:
        stamp = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise APIError(422, "timestamp", "A timestamp is not valid ISO date-time text.") from exc
    if stamp.tzinfo is not None:
        raise APIError(422, "timestamp_timezone", "Use the source timestamps without a timezone suffix.")
    return stamp


def parse_observations(raw: object) -> tuple[pd.DataFrame, datetime, float]:
    if not isinstance(raw, list) or len(raw) != 60:
        raise APIError(422, "window_length", "Provide exactly 60 readings at 12-minute spacing.")
    clean_rows = []
    times = []
    finite_cells = 0
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise APIError(422, "observation", f"Reading {i + 1} must be an object.")
        stamp = parse_timestamp(row.get("Timestamp", row.get("timestamp")))
        if times and stamp - times[-1] != timedelta(minutes=12):
            raise APIError(422, "cadence", "Readings must be chronological and exactly 12 minutes apart.")
        times.append(stamp)
        clean = {"Timestamp": stamp.isoformat(sep=" ")}
        for name in MAGNETIC_COLUMNS:
            value = row.get(name)
            if value is None or value == "":
                clean[name] = np.nan
                continue
            if isinstance(value, bool):
                raise APIError(422, "numeric_value", f"{name} in reading {i + 1} is not numeric.")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise APIError(422, "numeric_value", f"{name} in reading {i + 1} is not numeric.") from exc
            if not math.isfinite(numeric):
                clean[name] = np.nan
            elif abs(numeric) > 1e35:
                raise APIError(422, "numeric_range", f"{name} in reading {i + 1} is outside the accepted range.")
            else:
                clean[name] = numeric
                finite_cells += 1
        clean_rows.append(clean)
    completeness = finite_cells / (60 * len(MAGNETIC_COLUMNS))
    if completeness < 0.5:
        raise APIError(422, "insufficient_data", "Fewer than half of the required magnetic readings are present.")
    return pd.DataFrame(clean_rows), times[-1], completeness


def make_features(frame: pd.DataFrame, model: dict) -> np.ndarray:
    window = [{"start": frame.iloc[0]["Timestamp"],
               "end": frame.iloc[-1]["Timestamp"], "path": "submitted-window"}]
    base, temporal = compute_region_features(frame, window)
    x_base = transform_features(base.to_numpy(), load_stats(model["split"], "baseline"))
    if model["feature_set"] == "baseline_temporal":
        x_time = transform_features(temporal.to_numpy(), load_stats(model["split"], "temporal"))
        values = np.column_stack((x_base, x_time))
        names = BASE_COLUMNS + TEMPORAL_COLUMNS
    else:
        values = x_base
        names = BASE_COLUMNS
    if names != model["_config"]["feature_names"]:
        raise APIError(500, "feature_contract", "The saved model feature order does not match the current pipeline.")
    return np.ascontiguousarray(values, dtype=np.float32)


def predict(model_id: object, observations: object) -> dict:
    model = get_model(model_id)
    frame, issue, completeness = parse_observations(observations)
    features = make_features(frame, model)
    with MODEL_LOCK:
        predictor = load_saved_model(model["id"])
        outputs = predictor.run(None, {predictor.get_inputs()[0].name: features})
        scores = next((value for value in reversed(outputs)
                       if isinstance(value, np.ndarray) and value.ndim == 2
                       and value.shape[0] == 1 and value.shape[1] in (1, 2)), None)
        if scores is None:
            raise APIError(500, "onnx_output", "The converted model returned no score.")
        probability = float(scores[0, -1])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise APIError(500, "probability", "The saved model returned an invalid probability.")
    start = issue + timedelta(hours=model["lead_hours"])
    end = start + timedelta(hours=model["target_window_hours"])
    return {
        "model_id": model["id"], "probability": probability,
        "threshold": model["threshold"], "positive": probability >= model["threshold"],
        "issue_time": issue.isoformat(), "target_start": start.isoformat(),
        "target_end": end.isoformat(), "lead_hours": model["lead_hours"],
        "input_completeness": round(completeness, 4),
        "feature_count": model["feature_count"],
        "warning": "Many readings were missing and imputed from training data." if completeness < .8 else None,
        "target_note": "Recorded M/X flare peak in the 24-hour interval after the selected minimum lead.",
    }


@lru_cache(maxsize=1)
def feature_index() -> list[dict]:
    with (DATA / "phase1" / "feature_index.csv").open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@lru_cache(maxsize=1)
def target_array():
    return np.load(DATA / "phase2" / "horizon_targets.npy", mmap_mode="r")


def example(model_id: object, kind: str) -> dict:
    model = get_model(model_id)
    if kind not in {"flare", "quiet"}:
        raise APIError(400, "example_kind", "Choose flare or quiet example.")
    packed = DATA / "examples" / f"{model['id'].replace(':', '_')}_{kind}.json"
    if packed.is_file():
        return read_json(packed)
    test_rows = np.load(model["_run_dir"] / "test_rows.npy", mmap_mode="r")
    target_column = (0, 6, 12, 24, 48).index(model["lead_hours"])
    label = 1 if kind == "flare" else 0
    candidates = test_rows[target_array()[test_rows, target_column] == label]
    if not len(candidates):
        raise APIError(404, "example_missing", "No held-out example is available for this selection.")
    # A fixed offset avoids selecting a different example on each request.
    row_id = int(candidates[min(17, len(candidates) - 1)])
    meta = feature_index()[row_id]
    source = (DATA / meta["path"]).resolve()
    if not source.is_relative_to(DATA.resolve()) or not source.is_file():
        raise APIError(500, "example_source", "The saved example source is unavailable.")
    frame = pd.read_csv(source, sep="\t", usecols=["Timestamp", *MAGNETIC_COLUMNS])
    if len(frame) != 60:
        raise APIError(500, "example_source", "The saved example is not a 60-reading window.")
    observations = []
    for record in frame.to_dict(orient="records"):
        observations.append({key: (None if pd.isna(value) else value)
                             for key, value in record.items()})
    return {
        "kind": kind, "known_label": label, "region_id": meta["region_id"],
        "issue_time": meta["end"], "observations": observations,
        "note": "Held-out historical window. Its known label is provided for retrospective comparison.",
    }


@lru_cache(maxsize=1)
def overview() -> dict:
    labels = read_json(DATA / "phase2" / "horizon_targets_report.json")
    leakage = read_json(DATA / "leakage_summary.json")
    audit = read_json(DATA / "reports" / "phase4" / "phase4_results.json")
    with RESULTS.open(newline="", encoding="utf-8") as file:
        results = list(csv.DictReader(file))
    if len(results) != 33 or audit["verified_run_count"] != 33 or audit["warnings"]:
        raise APIError(503, "unverified_results", "The verified result manifest is incomplete.")
    numeric = ("lead_hours", "n_test", "positive", "prevalence", "threshold", "ap",
               "roc_auc", "precision", "recall", "f1", "tss", "hss", "tn", "fp", "fn", "tp")
    cleaned = []
    for row in results:
        entry = {key: value for key, value in row.items() if key not in {"run_dir", "reload_checked"}}
        for key in numeric:
            entry[key] = float(entry[key]) if key not in {"lead_hours", "n_test", "positive", "tn", "fp", "fn", "tp"} else int(entry[key])
        cleaned.append(entry)
    return {
        "dataset": {"windows": leakage["windows"], "regions": leakage["regions"],
                    "original_positive": labels["per_lead"]["0h"]["positive"],
                    "full_series": 4098, "features_baseline": 24, "features_temporal": 120},
        "verified_runs": audit["verified_run_count"],
        "leads": [{"hours": h, **labels["per_lead"][f"{h}h"]} for h in (0, 6, 12, 24, 48)],
        "splits": {key: {"counts": value["counts"],
                          "overlap_fraction": value["overlapping_test_fraction"],
                          "overlap_count": value["test_windows_overlapping_training_observations"],
                          "test_windows": value["test_windows"]}
                   for key, value in leakage["splits"].items()},
        "results": cleaned,
        "paired_comparisons": audit["paired_comparisons"],
        "top_features": audit["interpretation"]["top_features"][:8],
    }


def error_response(error: APIError) -> JSONResponse:
    return JSONResponse(status_code=error.status,
                        content={"error": {"code": error.code, "message": error.message}},
                        headers={"Cache-Control": "no-store"})


@app.exception_handler(APIError)
async def handle_api_error(_request: Request, error: APIError):
    return error_response(error)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request: Request, _error: RequestValidationError):
    return error_response(APIError(422, "invalid_request", "Check the request fields and try again."))


@app.exception_handler(Exception)
async def handle_unexpected_error(_request: Request, error: Exception):
    LOGGER.exception("Request failed: %s", error)
    if isinstance(error, (FileNotFoundError, OSError)):
        return error_response(APIError(503, "artifact_unavailable", "A required model artifact is unavailable."))
    return error_response(APIError(500, "internal_error", "The request could not be completed."))


@app.middleware("http")
async def limit_body(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/api/"):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_REQUEST_BYTES:
            return error_response(APIError(413, "request_too_large", "Observation data exceeds the 1.2 MB limit."))
        if "application/json" not in request.headers.get("content-type", ""):
            return error_response(APIError(415, "content_type", "Send application/json."))
        if len(await request.body()) > MAX_REQUEST_BYTES:
            return error_response(APIError(413, "request_too_large", "Observation data exceeds the 1.2 MB limit."))
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/api/health")
def health():
    return {"status": "ready", "models": len(catalog())}


@app.get("/api/catalog")
def get_catalog():
    models = sorted((public_model(item) for item in catalog().values()),
                    key=lambda item: (item["lead_hours"], item["split"],
                                      item["feature_set"], item["model"]))
    return {"models": models, "magnetic_columns": MAGNETIC_COLUMNS,
            "window_rows": 60, "cadence_minutes": 12}


@app.get("/api/overview")
def get_overview():
    return overview()


@app.get("/api/example")
def get_example(model_id: str, kind: str = "flare"):
    return example(model_id, kind)


@app.get("/api/models/{model_id}/metrics")
def get_metrics(model_id: str):
    item = get_model(model_id)
    metrics = read_json(item["_run_dir"] / "metrics.json")
    return {"model": public_model(item), "validation": metrics["validation"],
            "test": metrics["test"]}


@app.post("/api/predict")
def post_predict(body: PredictionRequest):
    return predict(body.model_id, body.observations)


@app.post("/api/compare")
def post_compare(body: ComparisonRequest):
    if not 2 <= len(body.model_ids) <= 5 or len(set(body.model_ids)) != len(body.model_ids):
        raise APIError(422, "comparison_models", "Choose 2 to 5 distinct saved models.")
    models = [get_model(model_id) for model_id in body.model_ids]
    if len({(m["split"], m["feature_set"], m["model"]) for m in models}) != 1:
        raise APIError(422, "comparison_scope", "Compare one model family, split, and feature set across leads.")
    return {"predictions": [predict(model_id, body.observations) for model_id in body.model_ids],
            "note": "Each score refers to its own 24-hour target interval; scores are not calibrated alert probabilities."}


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Helion FastAPI service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("backend.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()


