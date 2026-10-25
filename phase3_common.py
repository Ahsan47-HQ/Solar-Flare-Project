"""Phase 3 shared utilities for SWAN-SF forecasting experiments.

This module is the single import point for all Phase 3 scripts. It owns:

  - HorizonResult     : typed record for one (horizon, feature_set, split) cell
  - skill_scores()    : TSS and HSS (missing from Phase 2; needed by Phase 4)
  - run_one_horizon() : complete train-eval-save loop reused by all three scripts
  - scan_run_dirs()   : load and re-parse completed Phase 3 run directories
  - write_comparison_json() / write_comparison_md() : summary report helpers

Phases 4 and 5 only need to import this module to access structured results:

    from phase3_common import HorizonResult, scan_run_dirs, skill_scores

Design principles
-----------------
- Zero dependency on any existing Phase 2 source file beyond classical_models and
  phase2_model_common (which are already stable public interfaces).
- All public functions accept plain Python / numpy types so downstream code does
  not need to reproduce Phase 1 loading logic.
- skill_scores() is kept pure (no sklearn dependency) so it can be called on any
  confusion matrix including results loaded from stored metrics.json files.
"""

from __future__ import annotations

import csv
import json
import platform
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import classical_models as cm

ROOT = Path(__file__).resolve().parent

ALL_HORIZONS = ("6h", "12h", "24h", "48h")
SHORT_HORIZONS = ("6h", "12h")
LONG_HORIZONS = ("24h", "48h")
FEATURE_SETS = ("baseline", "baseline_temporal")
MODELS = ("logistic", "forest")
THRESHOLD_STRATEGIES = ("max_f1", "max_tss", "target_recall_90")


# ---------------------------------------------------------------------------
# Skill scores (Phase 4 readiness: TSS and HSS are not in Phase 2)
# ---------------------------------------------------------------------------

def skill_scores(confusion_matrix: list[list[int]]) -> dict[str, float | None]:
    """Compute TSS and HSS from a 2x2 confusion matrix [[TN, FP], [FN, TP]].

    Both metrics are commonly used for solar-flare forecast evaluation and are
    explicitly listed as missing from Phase 2 reports (see README). Returning
    None when undefined prevents silent division-by-zero in downstream code.

    Parameters
    ----------
    confusion_matrix:
        [[TN, FP], [FN, TP]] as stored in Phase 2/3 metrics.json files.

    Returns
    -------
    dict with keys: tss, hss, pod, far, pofd
        pod  = probability of detection = TP / (TP + FN) = FL recall
        far  = false alarm rate         = FP / (FP + TN) = NF miss rate
        pofd = probability of false detection (same as far)
        tss  = pod - pofd  (True Skill Statistic / Hanssen-Kuipers)
        hss  = Heidke Skill Score
    """
    tn, fp = int(confusion_matrix[0][0]), int(confusion_matrix[0][1])
    fn, tp = int(confusion_matrix[1][0]), int(confusion_matrix[1][1])
    total = tn + fp + fn + tp

    pod = tp / (tp + fn) if (tp + fn) > 0 else None
    far = fp / (fp + tn) if (fp + tn) > 0 else None  # = POFD
    tss: float | None
    if pod is not None and far is not None:
        tss = pod - far
    else:
        tss = None

    # HSS = 2(TP*TN - FP*FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))
    numerator = 2 * (tp * tn - fp * fn)
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = numerator / denom if denom != 0 else None

    return {"pod": pod, "far": far, "pofd": far, "tss": tss, "hss": hss, "total": total}


def choose_threshold_by_strategy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "max_f1",
) -> tuple[float, str]:
    """Select decision threshold on validation set according to chosen strategy.

    Supported strategies:
      - 'max_f1'          : Standard Phase 2 threshold maximizing validation F1 score.
      - 'max_tss'         : Maximizes True Skill Statistic (TSS = POD - POFD), symmetric
                            to flare and quiet classes, standard in space weather.
      - 'target_recall_90': Lowest threshold achieving at least 90% flare recall (POD >= 0.90),
                            crucial for operational mission-critical alert systems.
    """
    if strategy == "max_f1" or strategy is None:
        return cm.choose_threshold(y_true, y_prob)

    candidates = np.linspace(0.01, 0.99, 99)
    if strategy == "max_tss":
        best_tss = -2.0
        best_thresh = 0.5
        for th in candidates:
            pred = (y_prob >= th).astype(int)
            tn = int(np.sum((y_true == 0) & (pred == 0)))
            fp = int(np.sum((y_true == 0) & (pred == 1)))
            fn = int(np.sum((y_true == 1) & (pred == 0)))
            tp = int(np.sum((y_true == 1) & (pred == 1)))
            s = skill_scores([[tn, fp], [fn, tp]])
            if s["tss"] is not None and s["tss"] > best_tss:
                best_tss = s["tss"]
                best_thresh = float(th)
        return best_thresh, f"validation_max_tss_{best_thresh:.2f}"

    if strategy == "target_recall_90":
        fl_mask = (y_true == 1)
        if not np.any(fl_mask):
            return 0.5, "fallback_0.50"
        for th in sorted(candidates, reverse=True):
            pred = (y_prob >= th).astype(int)
            recall = np.sum((y_true == 1) & (pred == 1)) / np.sum(fl_mask)
            if recall >= 0.90:
                return float(th), f"validation_target_recall_90_{th:.2f}"
        return 0.01, "validation_target_recall_90_min_0.01"

    raise ValueError(f"Unknown threshold strategy {strategy!r}; choose from {THRESHOLD_STRATEGIES}")


# ---------------------------------------------------------------------------
# Typed result record
# ---------------------------------------------------------------------------

@dataclass
class HorizonResult:
    """Structured record for one (horizon, feature_set, split, model) experiment cell.

    This is the shared currency between Phase 3 scripts and future phases.
    Phase 4 can call skill_scores(result.confusion_matrix_test) directly.
    Phase 5 can collect a list[HorizonResult] and format them into reports.
    """
    horizon: str
    feature_set: str
    split: str
    model_name: str
    mode: str                    # "full" or "smoke"
    run_dir: Path
    interpretation: str

    # Validation metrics
    val_ap: float | None
    val_fl_recall: float | None
    val_fl_f1: float | None
    val_threshold: float

    # Test metrics
    test_ap: float | None
    test_fl_recall: float | None
    test_fl_f1: float | None

    # Raw confusion matrices — needed by Phase 4 skill score computation
    confusion_matrix_val: list[list[int]]
    confusion_matrix_test: list[list[int]]

    # Row counts
    n_train: int
    n_val: int
    n_test: int

    errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Convenience: skill scores computed on demand (no storage overhead)  #
    # ------------------------------------------------------------------ #
    @property
    def test_skill(self) -> dict[str, float | None]:
        """TSS, HSS, POD, FAR for the test set. Ready for Phase 4."""
        return skill_scores(self.confusion_matrix_test)

    @property
    def val_skill(self) -> dict[str, float | None]:
        """TSS, HSS, POD, FAR for the validation set. Ready for Phase 4."""
        return skill_scores(self.confusion_matrix_val)

    def as_flat_dict(self) -> dict[str, Any]:
        """Flat dict for CSV export / Phase 5 report aggregation."""
        skill = self.test_skill
        return {
            "horizon": self.horizon,
            "feature_set": self.feature_set,
            "split": self.split,
            "model": self.model_name,
            "mode": self.mode,
            "interpretation": self.interpretation,
            "val_ap": self.val_ap,
            "val_fl_recall": self.val_fl_recall,
            "val_fl_f1": self.val_fl_f1,
            "test_ap": self.test_ap,
            "test_fl_recall": self.test_fl_recall,
            "test_fl_f1": self.test_fl_f1,
            "test_tss": skill["tss"],
            "test_hss": skill["hss"],
            "test_pod": skill["pod"],
            "test_far": skill["far"],
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "run_dir": str(self.run_dir),
        }


# ---------------------------------------------------------------------------
# Run directory scanning (for Phase 4/5: load without re-training)
# ---------------------------------------------------------------------------

def load_result_from_dir(run_dir: Path) -> HorizonResult | None:
    """Parse a completed Phase 3 run directory into a HorizonResult.

    Returns None if the directory is missing, incomplete, or malformed.
    Phase 4 uses this to load all prior Phase 3 results without re-running.
    """
    status_file = run_dir / "status.json"
    config_file = run_dir / "config.json"
    metrics_file = run_dir / "metrics.json"
    for f in (status_file, config_file, metrics_file):
        if not f.is_file():
            return None
    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
        if not status.get("complete"):
            return None
        cfg = json.loads(config_file.read_text(encoding="utf-8"))
        met = json.loads(metrics_file.read_text(encoding="utf-8"))
        val = met.get("validation", {})
        test = met.get("test", {})
        return HorizonResult(
            horizon=cfg.get("horizon", "unknown"),
            feature_set=cfg.get("feature_set", "unknown"),
            split=cfg.get("split", "unknown"),
            model_name=cfg.get("model", "unknown"),
            mode="smoke" if cfg.get("smoke_max_per_subset") else "full",
            run_dir=run_dir,
            interpretation=met.get("evaluation_interpretation", "unknown"),
            val_ap=val.get("pr_auc_average_precision"),
            val_fl_recall=val.get("FL", {}).get("recall"),
            val_fl_f1=val.get("FL", {}).get("f1"),
            val_threshold=val.get("threshold", 0.5),
            test_ap=test.get("pr_auc_average_precision"),
            test_fl_recall=test.get("FL", {}).get("recall"),
            test_fl_f1=test.get("FL", {}).get("f1"),
            confusion_matrix_val=val.get("confusion_matrix", [[0, 0], [0, 0]]),
            confusion_matrix_test=test.get("confusion_matrix", [[0, 0], [0, 0]]),
            n_train=met.get("selected_row_counts", {}).get("train", 0),
            n_val=met.get("selected_row_counts", {}).get("val", 0),
            n_test=met.get("selected_row_counts", {}).get("test", 0),
        )
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def scan_run_dirs(phase3_root: Path) -> list[HorizonResult]:
    """Walk a Phase 3 output tree and return all completed HorizonResult objects.

    Phase 4 calls this once to load every prior experiment without re-running:

        from phase3_common import scan_run_dirs
        results = scan_run_dirs(Path("data/phase3"))

    Only directories containing a complete status.json + config.json + metrics.json
    are included. Malformed or incomplete runs are silently skipped.
    """
    results: list[HorizonResult] = []
    if not phase3_root.is_dir():
        return results
    for status_file in phase3_root.rglob("status.json"):
        run_dir = status_file.parent
        if (run_dir / "config.json").is_file() and (run_dir / "metrics.json").is_file():
            result = load_result_from_dir(run_dir)
            if result is not None:
                results.append(result)
    return results


# ---------------------------------------------------------------------------
# Core train-eval-save loop (shared by all three Phase 3 scripts)
# ---------------------------------------------------------------------------

def _build_source_hashes(caller_file: Path) -> dict[str, str]:
    names = [
        caller_file.name,
        "phase3_common.py",
        "classical_models.py",
        "data_cleaning.py",
        "temporal_features.py",
    ]
    return {n: cm.sha256_file(ROOT / n) for n in names if (ROOT / n).is_file()}


def _build_data_hashes(data: cm.Phase1Data) -> dict[str, str]:
    phase1 = data.data_dir / "phase1"
    hashes = {
        "feature_index.csv": cm.sha256_file(phase1 / "feature_index.csv"),
        "splits.csv": cm.sha256_file(data.data_dir / "splits.csv"),
        "feature_columns.json": cm.sha256_file(phase1 / "feature_columns.json"),
        "cleaning_report.json": cm.sha256_file(phase1 / "cleaning_report.json"),
        "status.json": cm.sha256_file(phase1 / "status.json"),
        "baseline.npy": cm.sha256_file(phase1 / "baseline.npy"),
    }
    if data.temporal is not None:
        hashes["temporal.npy"] = cm.sha256_file(phase1 / "temporal.npy")
    return hashes


def _make_model(name: str, args: Any) -> LogisticRegression | RandomForestClassifier:
    cw = "balanced" if getattr(args, "class_weight", "balanced") == "balanced" else None
    if name == "logistic":
        return LogisticRegression(
            C=getattr(args, "logistic_c", 1.0),
            max_iter=getattr(args, "logistic_max_iter", 500),
            solver="liblinear",
            class_weight=cw,
            random_state=args.seed,
        )
    if name == "forest":
        cw_forest = "balanced_subsample" if cw else None
        return RandomForestClassifier(
            n_estimators=getattr(args, "forest_trees", 100),
            max_depth=getattr(args, "forest_max_depth", 16),
            min_samples_leaf=getattr(args, "forest_min_leaf", 5),
            n_jobs=getattr(args, "n_jobs", 2),
            class_weight=cw_forest,
            random_state=args.seed,
        )
    raise ValueError(f"Unknown model name: {name!r}; choose from {MODELS}")


def run_one_horizon(
    *,
    horizon: str,
    model_name: str,
    args: Any,
    output_dir: Path,
    caller_file: Path,
    experiment_name: str,
    extra_config: dict | None = None,
) -> HorizonResult:
    """Train and evaluate one (horizon, model, split, feature_set) cell.

    This is the single canonical train-eval-save implementation shared by
    short_horizon.py, long_horizon.py, and feature_comparison.py. Keeping the
    logic here means Phase 4 can call it too without duplicating save logic.

    Parameters
    ----------
    horizon:         one of ALL_HORIZONS
    model_name:      "logistic" or "forest"
    args:            parsed argparse namespace (data_dir, split, features, seed,
                     smoke_max_per_subset, class_weight, forest_*, logistic_*)
    output_dir:      base directory for this experiment family
    caller_file:     Path(__file__) of the calling script (for source hashes)
    experiment_name: label written into config.json ("short_horizon", etc.)
    extra_config:    optional dict merged into config.json (for Phase 4 extras)
    """
    data = cm.load_phase1(args.data_dir, args.split, args.features)
    indices = cm.subset_indices(data, args.split, args.seed, args.smoke_max_per_subset)
    y = {part: data.labels[indices[part]].astype(np.int8) for part in cm.SUBSETS}
    if len(np.unique(y["train"])) < 2:
        raise ValueError(f"[{horizon}/{model_name}] Training must have both NF and FL")
    matrices = {part: cm.normalized_rows(data, indices[part]) for part in cm.SUBSETS}

    mode = "smoke" if args.smoke_max_per_subset else "full"
    interpretation = (
        "diagnostic_only"
        if mode == "smoke" or args.split == "random_split"
        else "region_independent" if args.split == "region_split"
        else "chronological"
    )

    model = _make_model(model_name, args)
    model.fit(matrices["train"], y["train"])
    if list(model.classes_) != [0, 1]:
        raise ValueError(f"[{horizon}/{model_name}] Class order is not [NF=0, FL=1]")

    val_prob = model.predict_proba(matrices["val"])[:, 1]
    strategy = getattr(args, "threshold_strategy", "max_f1")
    threshold, threshold_source = choose_threshold_by_strategy(y["val"], val_prob, strategy=strategy)
    test_prob = model.predict_proba(matrices["test"])[:, 1]
    val_metrics = cm.evaluate(y["val"], val_prob, threshold)
    test_metrics = cm.evaluate(y["test"], test_prob, threshold)
    val_metrics["skill_scores"] = skill_scores(val_metrics["confusion_matrix"])
    test_metrics["skill_scores"] = skill_scores(test_metrics["confusion_matrix"])

    run_dir = (
        output_dir
        / args.split
        / args.features
        / mode
        / horizon
        / model_name
        / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    for set_name, path in data.stats_paths.items():
        shutil.copyfile(path, run_dir / f"normalization_{set_name}.json")

    config: dict[str, Any] = {
        "phase": "phase3",
        "experiment": experiment_name,
        "horizon": horizon,
        "model": model_name,
        "split": args.split,
        "feature_set": args.features,
        "seed": args.seed,
        "threshold_strategy": strategy,
        "class_weight_strategy": str(model.class_weight),
        "smoke_max_per_subset": args.smoke_max_per_subset,
        "model_parameters": model.get_params(),
        "feature_columns": data.columns,
        "label_mapping": cm.LABELS,
        "horizon_note": (
            f"{horizon} forecast horizon. Feature arrays are Phase 1 12-hour observation "
            "windows. Horizon-specific datasets from the full series remain future work."
        ),
        "preprocessing": {
            "method": (
                "data_cleaning.transform_features; "
                "Phase 1 training-only median imputation and z-score"
            ),
            "normalization_files": {
                set_name: {
                    "saved_name": f"normalization_{set_name}.json",
                    "sha256": cm.sha256_file(path),
                }
                for set_name, path in data.stats_paths.items()
            },
            "outlier_handling": "none",
        },
        "data_sha256": _build_data_hashes(data),
        "source_sha256": _build_source_hashes(caller_file),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    if extra_config:
        config.update(extra_config)

    results = {
        "evaluation_interpretation": interpretation,
        "horizon": horizon,
        "feature_set": args.features,
        "training": {
            "n": len(y["train"]),
            "class_counts": {
                "NF": int(np.count_nonzero(y["train"] == 0)),
                "FL": int(np.count_nonzero(y["train"] == 1)),
            },
        },
        "full_split_counts": data.full_counts,
        "selected_row_counts": {part: len(indices[part]) for part in cm.SUBSETS},
        "threshold_selection": threshold_source,
        "validation": val_metrics,
        "test": test_metrics,
    }

    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.save(run_dir / "test_probabilities.npy", test_prob.astype(np.float32))
    joblib.dump(model, run_dir / "model.joblib", compress=3)
    # Status written last — missing status = failed run
    (run_dir / "status.json").write_text(
        json.dumps({"complete": True,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat()},
                   indent=2),
        encoding="utf-8",
    )

    skill = skill_scores(test_metrics["confusion_matrix"])
    result = HorizonResult(
        horizon=horizon,
        feature_set=args.features,
        split=args.split,
        model_name=model_name,
        mode=mode,
        run_dir=run_dir,
        interpretation=interpretation,
        val_ap=val_metrics["pr_auc_average_precision"],
        val_fl_recall=val_metrics["FL"]["recall"],
        val_fl_f1=val_metrics["FL"]["f1"],
        val_threshold=threshold,
        test_ap=test_metrics["pr_auc_average_precision"],
        test_fl_recall=test_metrics["FL"]["recall"],
        test_fl_f1=test_metrics["FL"]["f1"],
        confusion_matrix_val=val_metrics["confusion_matrix"],
        confusion_matrix_test=test_metrics["confusion_matrix"],
        n_train=len(indices["train"]),
        n_val=len(indices["val"]),
        n_test=len(indices["test"]),
    )

    def _f(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    print(
        f"[{horizon}|{model_name}|{args.features}] {args.split} {mode}: "
        f"val AP={_f(result.val_ap)} recall={_f(result.val_fl_recall)}; "
        f"test AP={_f(result.test_ap)} recall={_f(result.test_fl_recall)} "
        f"TSS={_f(skill['tss'])} HSS={_f(skill['hss'])} [{interpretation}]"
    )
    print(f"  -> {run_dir}")
    return result


# ---------------------------------------------------------------------------
# Shared report writers (Phase 5 readiness)
# ---------------------------------------------------------------------------

def write_comparison_json(results: list[HorizonResult], out_path: Path) -> None:
    """Write structured JSON for all results — Phase 5 loads this directly."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.as_flat_dict() for r in results]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_comparison_csv(results: list[HorizonResult], csv_path: Path) -> None:
    """Write structured CSV for all results — ready for Phase 4 / Phase 5 analysis."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [r.as_flat_dict() for r in results]
    if not flat_rows:
        return
    keys = list(flat_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_comparison_md(
    results: list[HorizonResult],
    report_path: Path,
    *,
    split: str,
    title: str = "Phase 3 Results",
    horizons: tuple[str, ...] = ALL_HORIZONS,
    feature_sets: tuple[str, ...] = FEATURE_SETS,
) -> None:
    """Write a Markdown comparison table. Phase 5 can extend or re-call this."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    by_key: dict[tuple[str, str, str], HorizonResult] = {
        (r.horizon, r.feature_set, r.model_name): r for r in results
    }
    models_seen = sorted({r.model_name for r in results})

    def _f(v: float | None, w: int = 6) -> str:
        return f"{v:.4f}" if v is not None else "—".center(w)

    lines = [
        f"# {title}",
        "",
        f"**Split**: `{split}`  ",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  ",
        "",
        "> Primary metric: PR-AUC (average precision). TSS and HSS are skill scores",
        "> needed for solar-flare forecast evaluation; they were missing from Phase 2.",
        "> Thresholds selected on validation set by max-F1.",
        "",
        "## Test-set results",
        "",
        "| Horizon | Feature set | Model | PR-AUC | FL Recall | FL F1 | TSS | HSS |",
        "|---------|-------------|-------|--------|-----------|-------|-----|-----|",
    ]
    for horizon in horizons:
        for fs in feature_sets:
            for mname in models_seen:
                r = by_key.get((horizon, fs, mname))
                if r is None:
                    lines.append(f"| {horizon} | {fs} | {mname} | — | — | — | — | — |")
                    continue
                skill = r.test_skill
                lines.append(
                    f"| {horizon} | {fs} | {mname} "
                    f"| {_f(r.test_ap)} | {_f(r.test_fl_recall)} | {_f(r.test_fl_f1)} "
                    f"| {_f(skill['tss'])} | {_f(skill['hss'])} |"
                )

    lines += [
        "",
        "## Temporal feature delta (baseline_temporal - baseline, test PR-AUC)",
        "",
        "| Horizon | Model | Delta PR-AUC | Direction |",
        "|---------|-------|-------------|-----------|",
    ]
    for horizon in horizons:
        for mname in models_seen:
            b = by_key.get((horizon, "baseline", mname))
            bt = by_key.get((horizon, "baseline_temporal", mname))
            if b is None or bt is None or b.test_ap is None or bt.test_ap is None:
                lines.append(f"| {horizon} | {mname} | — | — |")
                continue
            delta = bt.test_ap - b.test_ap
            direction = ("improvement" if delta > 0.001
                         else "degradation" if delta < -0.001
                         else "neutral")
            lines.append(f"| {horizon} | {mname} | {delta:+.4f} | {direction} |")

    def _format_run_dir(run_dir: Path) -> str:
        try:
            return str(run_dir.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            parts = run_dir.parts
            for marker in ("data", "out", "phase3", "reports"):
                if marker in parts:
                    idx = parts.index(marker)
                    return "/".join(parts[idx:])
            return run_dir.name

    lines += [
        "",
        "## Run identifiers",
        "",
        *[f"- `{r.horizon}` / `{r.feature_set}` / `{r.model_name}`: `{_format_run_dir(r.run_dir)}`"
          for r in results],
        "",
        "---",
        "*Phase 3. Feature arrays are Phase 1 12-hour observation windows.*",
        "*Horizon-specific datasets from the full series remain future work.*",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    csv_path = report_path.with_suffix(".csv")
    write_comparison_csv(results, csv_path)
    print(f"  Report: {_format_run_dir(report_path)}")
    print(f"  CSV:    {_format_run_dir(csv_path)}")


