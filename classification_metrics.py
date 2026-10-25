"""Re-evaluate saved SWAN-SF runs without fitting; FL=1 is the positive class.

Use --trust-model-files only for locally trusted joblib artifacts. Outputs are
created in new directories; existing Phase 1-3 files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import classical_models as cm
import phase4_common as common


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray,
                           threshold: float) -> dict:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if labels.ndim != 1 or labels.size == 0 or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be a nonempty NF=0/FL=1 vector")
    common.validate_probabilities(probabilities, len(labels))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and within [0, 1]")
    base = cm.evaluate(labels, probabilities, threshold)
    supports = {name: base[name]["support"] for name in ("NF", "FL")}
    base["predicted_class_counts"] = {
        "NF": int(np.count_nonzero(probabilities < threshold)),
        "FL": int(np.count_nonzero(probabilities >= threshold)),
    }
    summaries = {}
    for metric in ("precision", "recall", "f1"):
        values = [base[name][metric] for name in ("NF", "FL")]
        # A class absent in the truth or an undefined precision is not a zero.
        summaries[metric] = {
            "macro": float(np.mean(values)) if all(v is not None for v in values) else None,
            "weighted": (float(sum(base[name][metric] * supports[name] for name in ("NF", "FL"))
                               / len(labels)) if all(v is not None for v in values) else None),
        }
    base["summaries"] = summaries
    base["undefined_class_reason"] = (
        "Test labels contain only one class" if min(supports.values()) == 0 else None
    )
    return base


def prediction_rows(run: common.EvaluatedRun) -> list[dict]:
    if len(run.sample_ids) != len(run.labels) or len(run.region_ids) != len(run.labels):
        raise ValueError("Sample metadata is not aligned with labels")
    if len(set(run.sample_ids)) != len(run.sample_ids):
        raise ValueError("Duplicate sample IDs")
    predicted = (run.probabilities >= run.threshold).astype(np.int8)
    return [{"sample_id": sid, "region_id": region, "label": int(label),
             "probability_FL": float(prob), "prediction": int(pred)}
            for sid, region, label, prob, pred in zip(
                run.sample_ids, run.region_ids, run.labels, run.probabilities, predicted)]


def save_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "region_id", "label",
                                                   "probability_FL", "prediction"))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_available(run_dirs: list[Path], data_dir: Path, *, trust_joblib: bool,
                       predictions_dir: Path | None = None,
                       batch_size: int = 4096) -> tuple[list[dict], list[dict], list[common.EvaluatedRun]]:
    results, warnings, evaluated = [], [], []
    for path in run_dirs:
        try:
            run = common.evaluate_run(path, data_dir, trust_joblib=trust_joblib,
                                      batch_size=batch_size)
            metrics = classification_metrics(run.labels, run.probabilities, run.threshold)
            record = {
                "run_id": run.run_id, "run_dir": str(path), "model": run.config["model"],
                "phase": run.config.get("phase", "phase2"),
                "horizon_label": run.config.get("horizon"), "split": run.config["split"],
                "feature_set": run.config["feature_set"], "seed": run.config["seed"],
                "mode": "smoke" if run.config["smoke_max_per_subset"] else "full",
                "interpretation": run.source_metrics.get("evaluation_interpretation"),
                "threshold_source": run.threshold_source, "threshold": run.threshold,
                "model_sha256": run.model_sha256,
                "config_sha256": cm.sha256_file(path / "config.json"),
                "metrics_sha256": cm.sha256_file(path / "metrics.json"),
                "test": metrics, "status": "verified",
                "scientific_warning": (
                    "Phase 3 horizon label reuses the same Phase 1 target; no independent lead-time label"
                    if run.config.get("horizon") else None),
            }
            if predictions_dir is not None:
                destination = predictions_dir / f"{run.run_id}_test_predictions.csv"
                if destination.exists():
                    raise FileExistsError(f"Prediction target already exists: {destination}")
                save_predictions(destination, prediction_rows(run))
                record["predictions_path"] = str(destination)
                record["predictions_sha256"] = cm.sha256_file(destination)
            results.append(record)
            evaluated.append(run)
        except (FileNotFoundError, PermissionError, RuntimeError, ValueError,
                KeyError, ImportError, OSError, json.JSONDecodeError) as exc:
            warnings.append({"run_dir": str(path), "status": "blocked",
                             "reason": f"{type(exc).__name__}: {exc}"})
    return results, warnings, evaluated


def write_classification_csv(path: Path, results: list[dict], warnings: list[dict]) -> None:
    fields = ("run_id", "run_dir", "phase", "horizon_label", "model", "split", "feature_set",
              "mode", "status", "n_test", "n_fl", "fl_prevalence", "threshold",
              "threshold_source", "fl_precision", "fl_recall", "fl_f1",
              "nf_precision", "nf_recall", "nf_f1", "macro_f1", "weighted_f1",
              "tn", "fp", "fn", "tp", "pr_auc_average_precision", "roc_auc",
              "predictions_path", "reason")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            m = item["test"]
            tn, fp = m["confusion_matrix"][0]
            fn, tp = m["confusion_matrix"][1]
            writer.writerow({"run_id": item["run_id"], "run_dir": item["run_dir"],
                             "phase": item["phase"], "horizon_label": item["horizon_label"],
                             "model": item["model"], "split": item["split"],
                             "feature_set": item["feature_set"], "mode": item["mode"],
                             "status": item["status"], "n_test": m["n"],
                             "n_fl": m["class_counts"]["FL"],
                             "fl_prevalence": m["fl_prevalence"],
                             "threshold": item["threshold"],
                             "threshold_source": item["threshold_source"],
                             "fl_precision": m["FL"]["precision"],
                             "fl_recall": m["FL"]["recall"], "fl_f1": m["FL"]["f1"],
                             "nf_precision": m["NF"]["precision"],
                             "nf_recall": m["NF"]["recall"], "nf_f1": m["NF"]["f1"],
                             "macro_f1": m["summaries"]["f1"]["macro"],
                             "weighted_f1": m["summaries"]["f1"]["weighted"],
                             "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                             "pr_auc_average_precision": m["pr_auc_average_precision"],
                             "roc_auc": m["roc_auc"],
                             "predictions_path": item.get("predictions_path")})
        for warning in warnings:
            writer.writerow({"run_dir": warning["run_dir"], "status": "blocked",
                             "reason": warning["reason"]})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=common.ROOT / "data")
    parser.add_argument("--run-root", action="append", type=Path, default=None,
                        help="Search this tree for saved runs; repeatable")
    parser.add_argument("--run-dir", action="append", type=Path, default=[],
                        help="Explicit saved run; repeatable")
    parser.add_argument("--output-root", type=Path, default=common.ROOT / "reports" / "phase4")
    parser.add_argument("--predictions-root", type=Path, default=common.ROOT / "data" / "phase4")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--trust-model-files", action="store_true",
                        help="Allow loading trusted joblib model files, which use pickle")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    roots = args.run_root if args.run_root is not None else [args.data_dir / "phase3", args.data_dir / "phase2"]
    dirs = common.discover_runs(roots, args.run_dir)
    destination = common.new_directory(args.output_root, "classification")
    predictions = args.predictions_root / destination.name
    results, warnings, _ = evaluate_available(dirs, args.data_dir,
                                                trust_joblib=args.trust_model_files,
                                                predictions_dir=predictions,
                                                batch_size=args.batch_size)
    if not dirs:
        warnings.append({"run_dir": ", ".join(str(root) for root in roots), "status": "blocked",
                         "reason": "No saved run directories found; restore Phase 3/Phase 2 artifacts"})
    if not args.data_dir.is_dir():
        warnings.append({"run_dir": str(args.data_dir), "status": "blocked",
                         "reason": "Phase 1 data directory is absent"})
    common.write_json_new(destination / "classification_results.json",
                          {"results": results, "warnings": warnings})
    write_classification_csv(destination / "classification_results.csv", results, warnings)
    print(f"Classification: {len(results)} verified, {len(warnings)} warning(s); {destination}")
    return destination


if __name__ == "__main__":
    run(parse_args())
