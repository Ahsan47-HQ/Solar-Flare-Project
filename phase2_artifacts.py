"""Verify, back up, and report Phase 2 artifacts without retraining.

Only load model files that you created or otherwise trust: joblib uses pickle.
Run in Colab after mounting Drive. Reports contain observed artifact data only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

import classical_models as cm


# Run IDs supplied by the project owner. Presence and contents are checked below.
EXPECTED_CLASSICAL = (
    ("region_split", "baseline", "logistic", "20261024T174915Z_9fb61b61"),
    ("region_split", "baseline", "forest", "20261024T175106Z_d3181de8"),
    ("region_split", "baseline_temporal", "logistic", "20261024T175435Z_6f50f109"),
    ("region_split", "baseline_temporal", "forest", "20261024T180048Z_429d9907"),
    ("time_split", "baseline", "logistic", "20261024T180501Z_6260511c"),
    ("time_split", "baseline", "forest", "20261024T180647Z_8d604d59"),
    ("time_split", "baseline_temporal", "logistic", "20261024T181101Z_26bf580e"),
    ("time_split", "baseline_temporal", "forest", "20261024T181645Z_f283cb01"),
)

# Terminal values provided by the owner. These are comparison targets, never
# substituted into results when saved metrics are absent.
REPORTED_CLASSICAL = {
    "20261024T174915Z_9fb61b61": (0.37036090420463236, 0.4864864864864865, 0.3940281055009417, 0.4845580404685836),
    "20261024T175106Z_d3181de8": (0.33136526646903824, 0.46756756756756757, 0.23359581854468525, 0.35782747603833864),
    "20261024T175435Z_6f50f109": (0.3902067923097733, 0.44594594594594594, 0.38784005140812755, 0.44195953141640043),
    "20261024T180048Z_429d9907": (0.3736002991892269, 0.43648648648648647, 0.25759563812557973, 0.2715654952076677),
    "20261024T180501Z_6260511c": (0.48142744793240516, 0.5330472103004292, 0.45467993244975935, 0.42727272727272725),
    "20261024T180647Z_8d604d59": (0.2727397109854959, 0.4111587982832618, 0.31977677925152254, 0.4797979797979798),
    "20261024T181101Z_26bf580e": (0.4961213282205448, 0.5253218884120172, 0.4853720613769681, 0.3606060606060606),
    "20261024T181645Z_f283cb01": (0.2933313843407104, 0.45236051502145924, 0.3724302321692834, 0.6101010101010101),
}


def expected_run_paths(root: Path) -> list[Path]:
    return [root / split / features / "full" / model / run_id
            for split, features, model, run_id in EXPECTED_CLASSICAL]


def file_inventory(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in artifact tree: {path}")
        if path.is_file():
            records.append({"path": path.relative_to(root).as_posix(),
                            "bytes": path.stat().st_size, "sha256": cm.sha256_file(path)})
    return records


def backup_tree(source: Path, destination: Path) -> dict:
    """Copy absent files, preserve conflicts, then hash-check the entire source tree."""
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source.is_symlink() or destination.is_symlink():
        raise ValueError("Artifact roots must not be symlinks")
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    source_files = file_inventory(source)
    if not source_files:
        raise ValueError("No files found at source")
    expected_missing = []
    expected_invalid = []
    if source.name == "classical":
        required_runs = expected_run_paths(source)
    else:
        required_runs = sorted(path.parent for path in source.rglob("config.json"))
        if not required_runs:
            expected_missing.append("No model run config.json found")
    for run in required_runs:
        for name in ("config.json", "metrics.json", "status.json", "normalization_baseline.json"):
            if not (run / name).is_file():
                expected_missing.append((run / name).relative_to(source).as_posix())
        if source.name == "classical" and not (run / "model.joblib").is_file():
            expected_missing.append((run / "model.joblib").relative_to(source).as_posix())
        if run.parts[-4] == "baseline_temporal" and not (run / "normalization_temporal.json").is_file():
            expected_missing.append((run / "normalization_temporal.json").relative_to(source).as_posix())
        if run.is_dir():
            checked = verify_run(run, Path("__phase1_unavailable__"))
            if checked["completion_status"] != "complete":
                expected_invalid.append({"run": run.relative_to(source).as_posix(),
                                         "reasons": checked["limitations"]})
    destination.mkdir(parents=True, exist_ok=True)
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in destination artifact tree: {path}")
    copied, matched, conflicts = [], [], []
    for item in source_files:
        src = source / item["path"]
        dst = destination / item["path"]
        if dst.is_symlink():
            conflicts.append(item["path"])
            continue
        if dst.exists():
            (matched if dst.is_file() and cm.sha256_file(dst) == item["sha256"]
             else conflicts).append(item["path"])
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        temp = dst.with_name(dst.name + f".partial.{uuid4().hex}")
        try:
            shutil.copyfile(src, temp)
            if cm.sha256_file(temp) != item["sha256"] or temp.stat().st_size != item["bytes"]:
                raise IOError(f"Copy failed verification: {src}")
            if dst.exists():
                conflicts.append(item["path"])
                continue
            os.rename(temp, dst)
            copied.append(item["path"])
        finally:
            if temp.exists():
                temp.unlink()
    verified = []
    for item in source_files:
        dst = destination / item["path"]
        if dst.is_file() and cm.sha256_file(dst) == item["sha256"]:
            verified.append(item["path"])
    missing = sorted(set(item["path"] for item in source_files) - set(verified) - set(conflicts))
    return {"source": str(source), "destination": str(destination),
            "source_files": len(source_files), "copied": copied, "already_matched": matched,
            "verified": verified, "conflicts": conflicts, "missing": missing,
            "expected_missing": expected_missing,
            "expected_invalid": expected_invalid,
            "complete": len(verified) == len(source_files) and not conflicts and not missing
            and not expected_missing and not expected_invalid,
            "inventory": source_files}


def verify_backup_inventory(result: dict) -> dict:
    """Rehash the current source and destination; distrust a stale inventory flag."""
    source, destination = Path(result["source"]), Path(result["destination"])
    listed = result.get("inventory", [])
    if not source.is_dir() or not destination.is_dir() or not listed:
        return {"family": source.name, "complete": False, "verified_files": 0,
                "source_files": len(listed), "reason": "Missing source, destination or inventory"}
    current = file_inventory(source)
    if current != listed:
        return {"family": source.name, "complete": False, "verified_files": 0,
                "source_files": len(listed), "reason": "Source differs from inventory"}
    verified = sum((destination / item["path"]).is_file() and
                   cm.sha256_file(destination / item["path"]) == item["sha256"]
                   for item in listed)
    return {"family": source.name,
            "complete": bool(result.get("complete")) and verified == len(listed),
            "verified_files": verified, "source_files": len(listed),
            "reason": None if verified == len(listed) else "Destination hash mismatch or missing file"}


def _model_path(run_dir: Path, config: dict) -> Path:
    name = config.get("model_file") or "model.joblib"
    if Path(name).name != name:
        raise ValueError("Unsafe model filename in config")
    return run_dir / name


def verify_run(run_dir: Path, data_dir: Path) -> dict:
    record = {"run_id": run_dir.name, "artifact_path": str(run_dir),
              "model": run_dir.parent.name, "split": run_dir.parents[3].name,
              "feature_set": run_dir.parents[2].name,
              "completion_status": "missing", "reload_status": "not_run",
              "inference_status": "not_run", "limitations": []}
    if not run_dir.is_dir():
        record["limitations"].append("Run directory absent")
        return record
    try:
        config = cm.read_json(run_dir / "config.json")
        metrics = cm.read_json(run_dir / "metrics.json")
        status = cm.read_json(run_dir / "status.json")
        if run_dir.parent.name in ("logistic", "forest"):
            expected_model = run_dir.parent.name
            expected_split = run_dir.parents[3].name
            expected_features = run_dir.parents[2].name
            expected_mode = run_dir.parents[1].name
        else:
            expected_model = config.get("model")
            if expected_model not in ("xgboost", "ann"):
                raise ValueError("Unsupported nonclassical model type")
            expected_split = run_dir.parents[2].name
            expected_features = run_dir.parents[1].name
            expected_mode = run_dir.parent.name
        if (config.get("model"), config.get("split"), config.get("feature_set")) != (
                expected_model, expected_split, expected_features):
            raise ValueError("Run directory and config identity differ")
        if ("smoke" if config.get("smoke_max_per_subset") else "full") != expected_mode:
            raise ValueError("Run mode and config differ")
        model_path = _model_path(run_dir, config)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        for name, info in config["preprocessing"]["normalization_files"].items():
            saved = run_dir / info["saved_name"]
            if not saved.is_file() or cm.sha256_file(saved) != info["sha256"]:
                raise ValueError(f"Saved {name} normalization hash mismatch")
        if config.get("label_mapping") != cm.LABELS:
            raise ValueError("Incorrect label mapping")
        if not status.get("complete"):
            raise ValueError("Completion marker is not true")
        threshold = metrics["validation"]["threshold"]
        if not 0 <= threshold <= 1 or metrics["test"]["threshold"] != threshold:
            raise ValueError("Validation/test threshold mismatch")
        record.update({"completion_status": "complete", "model": config["model"],
                       "split": config["split"], "feature_set": config["feature_set"],
                       "seed": config["seed"], "config": config, "metrics": metrics,
                       "model_path": str(model_path), "model_sha256": cm.sha256_file(model_path),
                       "threshold": threshold})
        if run_dir.name in REPORTED_CLASSICAL:
            observed = (metrics["validation"]["pr_auc_average_precision"],
                        metrics["validation"]["FL"]["recall"],
                        metrics["test"]["pr_auc_average_precision"],
                        metrics["test"]["FL"]["recall"])
            reported = REPORTED_CLASSICAL[run_dir.name]
            record["reported_metrics_match"] = all(
                value is not None and abs(value - expected) <= 1e-10
                for value, expected in zip(observed, reported)
            )
            if not record["reported_metrics_match"]:
                record["limitations"].append("Saved metrics differ from reported terminal values")
        if not data_dir.is_dir():
            record["limitations"].append("Phase 1 data unavailable; inference not run")
            return record
        data = cm.load_phase1(data_dir, config["split"], config["feature_set"])
        if config["feature_columns"] != data.columns:
            raise ValueError("Feature order mismatch")
        phase1 = data_dir / "phase1"
        input_paths = {"splits.csv": data_dir / "splits.csv"}
        input_paths.update({name: phase1 / name for name in config["data_sha256"]
                            if name != "splits.csv"})
        for name, expected in config["data_sha256"].items():
            if cm.sha256_file(input_paths[name]) != expected:
                raise ValueError(f"Phase 1 input hash mismatch: {name}")
        for name, info in config["preprocessing"]["normalization_files"].items():
            if cm.sha256_file(data.stats_paths[name]) != info["sha256"]:
                raise ValueError(f"Phase 1 normalization differs from saved {name}")
        selected = cm.subset_indices(data, config["split"], config["seed"],
                                     config["smoke_max_per_subset"])
        sample = selected["test"][:32]
        if not len(sample):
            raise ValueError("No held-out rows")
        x = cm.normalized_rows(data, sample)
        if config["model"] in ("logistic", "forest"):
            import joblib
            model = joblib.load(model_path)
            if list(model.classes_) != [0, 1]:
                raise ValueError("Saved model class order is not NF, FL")
            if model.n_features_in_ != len(data.columns):
                raise ValueError("Saved model feature count differs")
            proba = model.predict_proba(x)[:, 1]
        elif config["model"] == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            if list(model.classes_) != [0, 1] or model.n_features_in_ != len(data.columns):
                raise ValueError("XGBoost class order or feature count differs")
            proba = model.predict_proba(x)[:, 1]
        elif config["model"] == "ann":
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            if model.input_shape[-1] != len(data.columns):
                raise ValueError("ANN feature count differs")
            proba = model.predict(x, verbose=0).reshape(-1)
        else:
            raise ValueError("Unsupported model type")
        if proba.shape != (len(sample),) or not np.all(np.isfinite(proba)):
            raise ValueError("Inference returned invalid shape or nonfinite predictions")
        if np.any((proba < 0) | (proba > 1)):
            raise ValueError("Inference returned values outside [0, 1]")
        predicted = proba >= threshold
        if predicted.shape != (len(sample),):
            raise ValueError("Threshold application failed")
        record["reload_status"] = "passed"
        record["inference_status"] = "passed"
        record["sample_rows"] = len(sample)
        record["limitations"].append("No saved prediction vector; numerical equality to original in-memory predictions untested")
    except Exception as exc:
        record["limitations"].append(f"{type(exc).__name__}: {exc}")
        if record["completion_status"] == "complete":
            record["reload_status"] = "failed"
            record["inference_status"] = "failed"
        else:
            record["completion_status"] = "invalid"
    return record


def collect_runs(classical_root: Path, phase2_root: Path) -> list[Path]:
    paths = expected_run_paths(classical_root)
    known = set(paths)
    for family in ("xgboost", "ann"):
        root = phase2_root / family
        if root.is_dir():
            for config in sorted(root.rglob("config.json")):
                if config.parent not in known:
                    paths.append(config.parent)
                    known.add(config.parent)
    return paths


CSV_FIELDS = ("experiment_id", "run_id", "split", "feature_set", "model",
              "validation_ap", "validation_fl_precision", "validation_fl_recall",
              "validation_fl_f1", "validation_roc_auc", "validation_confusion_matrix",
              "validation_fl_prevalence", "selected_threshold", "test_ap",
              "test_fl_precision", "test_fl_recall", "test_fl_f1", "test_roc_auc",
              "test_confusion_matrix", "test_fl_prevalence", "seed", "model_parameters",
              "artifact_path", "reload_status", "completion_status")


def csv_row(record: dict) -> dict:
    metrics = record.get("metrics", {})
    config = record.get("config", {})
    val = metrics.get("validation", {})
    test = metrics.get("test", {})
    return {"experiment_id": f"{record.get('split', '')}:{record.get('feature_set', '')}",
            "run_id": record["run_id"], "split": record.get("split", ""),
            "feature_set": record.get("feature_set", ""), "model": record.get("model", ""),
            "validation_ap": val.get("pr_auc_average_precision"),
            "validation_fl_precision": val.get("FL", {}).get("precision"),
            "validation_fl_recall": val.get("FL", {}).get("recall"),
            "validation_fl_f1": val.get("FL", {}).get("f1"),
            "validation_roc_auc": val.get("roc_auc"),
            "validation_confusion_matrix": json.dumps(val["confusion_matrix"]) if "confusion_matrix" in val else "",
            "validation_fl_prevalence": val.get("fl_prevalence"),
            "selected_threshold": record.get("threshold"),
            "test_ap": test.get("pr_auc_average_precision"),
            "test_fl_precision": test.get("FL", {}).get("precision"),
            "test_fl_recall": test.get("FL", {}).get("recall"),
            "test_fl_f1": test.get("FL", {}).get("f1"),
            "test_roc_auc": test.get("roc_auc"),
            "test_confusion_matrix": json.dumps(test["confusion_matrix"]) if "confusion_matrix" in test else "",
            "test_fl_prevalence": test.get("fl_prevalence"), "seed": config.get("seed"),
            "model_parameters": json.dumps(config["model_parameters"], sort_keys=True)
            if "model_parameters" in config else "", "artifact_path": record["artifact_path"],
            "reload_status": record["reload_status"],
            "completion_status": record["completion_status"]}


def write_reports(records: list[dict], phase2_root: Path, report_dir: Path, branch: str,
                  commit: str, backup_results: list[dict] | None = None) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "classical_model_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_row(r) for r in records)
    inventory = []
    for family in ("classical", "xgboost", "ann"):
        for item in file_inventory(phase2_root / family):
            inventory.append({"family": family, **item})
    (report_dir / "artifact_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    families = {r.get("model") for r in records if r["completion_status"] == "complete"}
    runs_verified = all(r["completion_status"] == "complete" and
                        r["reload_status"] == "passed" and
                        r.get("reported_metrics_match", True) for r in records)
    backup_summaries = [verify_backup_inventory(result) for result in (backup_results or [])]
    backed_families = {item["family"] for item in backup_summaries if item["complete"]}
    manifest = {
        "project": "Solar-Flare-Project", "phase": "Phase 2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch, "git_commit": commit, "dataset_doi": "10.7910/DVN/EBCFKM",
        "dataset_checksums": next((r["config"]["data_sha256"] for r in records
                                    if "config" in r), {}),
        "source_sha256": {name: cm.sha256_file(cm.ROOT / name) for name in
                          ("classical_models.py", "xgboost_model.py", "ann_model.py",
                           "phase2_model_common.py", "phase2_artifacts.py",
                           "data_cleaning.py", "temporal_features.py")},
        "runs": records, "artifact_inventory": "artifact_inventory.json",
        "backup_summaries": backup_summaries,
        "artifact_checks_complete": runs_verified,
        "technical_checks_complete": runs_verified and
                    {"classical", "xgboost", "ann"}.issubset(backed_families) and
                    {"logistic", "forest", "xgboost", "ann"}.issubset(families),
        "complete": False,
        "completion_limitation": "Original Phase 2 experiment matrix is unavailable in this repository",
    }
    (report_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    table = ["| Run | Model | Split | Features | Val AP | Val P/R/F1 | Threshold | Test AP | Test P/R/F1 | Test ROC-AUC | Reload |",
             "| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | --- |"]
    for r in records:
        row = csv_row(r)
        def show(value):
            return "—" if value is None or value == "" else str(value)
        val_prf = "/".join(show(row[k]) for k in
                           ("validation_fl_precision", "validation_fl_recall", "validation_fl_f1"))
        test_prf = "/".join(show(row[k]) for k in
                            ("test_fl_precision", "test_fl_recall", "test_fl_f1"))
        table.append("| " + " | ".join([
            show(row["run_id"]), show(row["model"]), show(row["split"]),
            show(row["feature_set"]), show(row["validation_ap"]), val_prf,
            show(row["selected_threshold"]), show(row["test_ap"]), test_prf,
            show(row["test_roc_auc"]), show(row["reload_status"]),
        ]) + " |")
    comparisons = []
    grouped = defaultdict(list)
    for record in records:
        if record["completion_status"] == "complete":
            grouped[(record["split"], record["feature_set"])].append(record)
    for (split, features), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        ranked = [r for r in group if r["metrics"]["validation"]["pr_auc_average_precision"] is not None]
        if ranked:
            leader = max(ranked, key=lambda r: r["metrics"]["validation"]["pr_auc_average_precision"])
            comparisons.append(f"- `{split}` / `{features}`: highest observed validation AP is "
                               f"{leader['model']} ({leader['metrics']['validation']['pr_auc_average_precision']:.6g}). "
                               "See the table and CSV for its held-out test result; this is a descriptive comparison.")
    intro = ["# Phase 2 model results", "",
             "Generated from saved config.json, metrics.json, status.json and model files only.",
             "An absent run has blank metrics. Reported terminal numbers are not substituted.", "",
             "FL is M/X-class and positive (1); NF is negative (0). The documented full dataset has",
             "331,185 windows, 6,234 FL (1.88%) and 324,951 NF; these counts are not reverified",
             "unless Phase 1 files are available to the audit. Baseline has 24 magnetic features;",
             "baseline_temporal has 144. Region split keeps regions separate; time split trains",
             "on partitions 1–3, validates on 4 and tests on 5. Random split is diagnostic only.",
             "Normalization is fitted by Phase 1 on each split's training rows. Validation selects",
             "the F1-maximizing decision threshold; test labels do not select it. Average precision",
             "(AP) summarizes the precision–recall curve and is not accuracy. Recall describes the",
             "share of FL windows found; precision describes how many FL predictions are correct.",
             "F1 combines precision and recall. ROC-AUC and confusion matrices are in the CSV.",
             "No statistical significance or universal model ranking is inferred.", "",
             *table, "", "## Within-configuration comparison", "",
             *(comparisons or ["Unavailable until at least two saved runs in one configuration are verified."]),
             "", "Metrics above are only final after checking the saved metrics files and",
             "held-out inference. Model files are expected under gitignored data/phase2/;",
             "Phase 1 arrays are expected under gitignored data/phase1/.",
             "See experiment_manifest.json, artifact_inventory.json and model_reload_verification.md.",
             "Reproduce with classical_models.py, xgboost_model.py or ann_model.py using the recorded",
             "split, feature set, seed and parameters. Existing completed runs should be reused."]
    (report_dir / "classical_model_results.md").write_text("\n".join(intro) + "\n", encoding="utf-8")
    reload_lines = ["# Saved model reload verification", "",
                    "Inference uses up to 32 held-out rows from the recorded split, saved preprocessing,",
                    "and the recorded validation threshold. No model is refit.", ""]
    for r in records:
        reload_lines += [f"- `{r['run_id']}` ({r.get('model')}, {r.get('split')}, "
                         f"{r.get('feature_set')}): artifact=`{r['artifact_path']}`; "
                         f"completion={r['completion_status']}; "
                         f"reload={r['reload_status']}; inference={r['inference_status']}; "
                         f"sample_rows={r.get('sample_rows', 0)}; "
                         f"limitations={'; '.join(r['limitations']) or 'none'}"]
    (report_dir / "model_reload_verification.md").write_text("\n".join(reload_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    backup = sub.add_parser("backup")
    backup.add_argument("--source", type=Path, default=cm.ROOT / "data" / "phase2" / "classical")
    backup.add_argument("--destination", type=Path, required=True)
    backup.add_argument("--inventory", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--phase2-root", type=Path, default=cm.ROOT / "data" / "phase2")
    audit.add_argument("--data-dir", type=Path, default=cm.ROOT / "data")
    audit.add_argument("--report-dir", type=Path, default=cm.ROOT / "reports" / "phase2")
    audit.add_argument("--branch", default="Prasiddhi")
    audit.add_argument("--commit", required=True)
    audit.add_argument("--backup-inventory", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "backup":
        result = backup_tree(args.source, args.destination)
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Verified {len(result['verified'])}/{result['source_files']} files; "
              f"conflicts={len(result['conflicts'])}; missing={len(result['missing'])}")
        return 0 if result["complete"] else 1
    classical = args.phase2_root / "classical"
    records = [verify_run(path, args.data_dir) for path in collect_runs(classical, args.phase2_root)]
    backup_results = [cm.read_json(path) for path in args.backup_inventory]
    write_reports(records, args.phase2_root, args.report_dir, args.branch, args.commit,
                  backup_results=backup_results)
    passed = sum(r["reload_status"] == "passed" for r in records)
    print(f"Reload/inference passed: {passed}/{len(records)}")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
