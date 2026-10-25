"""Phase 4 held-out metrics and paired comparisons for saved SWAN-SF runs.

No models or preprocessing are fitted. Paired uncertainty resamples active
regions, acknowledging dependence between overlapping windows. McNemar p-values
are reported only when each held-out region contributes one window.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

import classical_models as cm
import classification_metrics as basic
import phase4_common as common


METRIC_DEFINITIONS = {
    "pr_auc_average_precision": "sklearn average_precision_score on continuous FL probabilities; not trapezoidal PR area",
    "roc_auc": "sklearn roc_auc_score on continuous FL probabilities; null when one class is absent",
    "tss": "TP/(TP+FN) + TN/(TN+FP) - 1; null if either class is absent",
    "hss": "2*(TP*TN-FN*FP)/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)); null if denominator is zero",
    "confusion_matrix": "[[TN, FP], [FN, TP]] with rows true NF/FL and columns predicted NF/FL",
}


def skill_scores(matrix: list[list[int]]) -> dict:
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("Expected a 2x2 confusion matrix")
    values = np.asarray(matrix)
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError("Confusion counts must be nonnegative integers")
    tn, fp = [int(v) for v in values[0]]
    fn, tp = [int(v) for v in values[1]]
    positive, negative = tp + fn, tn + fp
    tss = tp / positive + tn / negative - 1 if positive and negative else None
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fn * fp) / denominator if denominator else None
    return {"tss": tss, "hss": hss, "counts": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "n": positive + negative}


def advanced_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    base = basic.classification_metrics(labels, probabilities, threshold)
    score = skill_scores(base["confusion_matrix"])
    return {"pr_auc_average_precision": base["pr_auc_average_precision"],
            "roc_auc": base["roc_auc"], "tss": score["tss"], "hss": score["hss"],
            "confusion_matrix": base["confusion_matrix"], "counts": score["counts"],
            "n": base["n"], "n_fl": base["class_counts"]["FL"],
            "fl_prevalence": base["fl_prevalence"]}


def validate_pair(a: common.EvaluatedRun, b: common.EvaluatedRun) -> None:
    if (a.config["split"] != b.config["split"] or
            a.config.get("horizon") != b.config.get("horizon") or
            bool(a.config["smoke_max_per_subset"]) != bool(b.config["smoke_max_per_subset"])):
        raise ValueError("Paired runs require the same split, horizon label, and mode")
    if a.sample_ids != b.sample_ids or a.region_ids != b.region_ids:
        raise ValueError("Paired runs require identical ordered test sample IDs and regions")
    if not np.array_equal(a.labels, b.labels):
        raise ValueError("Paired runs have different aligned labels")
    shared = ("feature_index.csv", "splits.csv", "status.json")
    if any(a.config["data_sha256"].get(key) != b.config["data_sha256"].get(key)
           for key in shared):
        raise ValueError("Paired runs refer to different Phase 1 input versions")


def _metric(labels: np.ndarray, probabilities: np.ndarray, threshold: float, name: str) -> float | None:
    if name == "average_precision":
        return float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    if name == "roc_auc":
        return float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    predicted = probabilities >= threshold
    truth = labels == 1
    tn = int(np.count_nonzero(~truth & ~predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    tp = int(np.count_nonzero(truth & predicted))
    if name in ("tss", "hss"):
        return skill_scores([[tn, fp], [fn, tp]])[name]
    if name == "fl_f1":
        return 2 * tp / (2 * tp + fp + fn) if tp + fn else None
    raise ValueError(f"Unknown paired metric: {name}")


def _percentile_ci(values: list[float], alpha: float) -> list[float] | None:
    if len(values) < 20:
        return None
    return [float(x) for x in np.quantile(values, [alpha / 2, 1 - alpha / 2])]


def paired_bootstrap(a: common.EvaluatedRun, b: common.EvaluatedRun, *,
                     metric: str = "average_precision", replicates: int = 500,
                     seed: int = 42, alpha: float = 0.05) -> dict:
    validate_pair(a, b)
    if replicates < 20 or not 0 < alpha < 1:
        raise ValueError("At least 20 replicates and 0<alpha<1 are required")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, region in enumerate(a.region_ids):
        groups[region].append(i)
    if len(groups) < 10:
        return {"status": "insufficient_independent_regions", "metric": metric,
                "n_regions": len(groups), "n_test": len(a.labels), "minimum_regions": 10,
                "difference": {"estimate": None, "ci": None}, "p_value": None}
    values_a, values_b, differences = [], [], []
    rng = np.random.default_rng(seed)
    group_arrays = [np.asarray(rows, dtype=np.int64) for rows in groups.values()]
    point_a = _metric(a.labels, a.probabilities, a.threshold, metric)
    point_b = _metric(b.labels, b.probabilities, b.threshold, metric)
    for _ in range(replicates):
        chosen = rng.integers(0, len(group_arrays), size=len(group_arrays))
        rows = np.concatenate([group_arrays[j] for j in chosen])
        y = a.labels[rows]
        value_a = _metric(y, a.probabilities[rows], a.threshold, metric)
        value_b = _metric(y, b.probabilities[rows], b.threshold, metric)
        if value_a is not None and value_b is not None:
            values_a.append(value_a)
            values_b.append(value_b)
            differences.append(value_a - value_b)
    return {
        "method": "paired_active_region_cluster_bootstrap_percentile",
        "metric": metric, "orientation": "first_minus_second", "n_test": len(a.labels),
        "n_regions": len(groups), "replicates_requested": replicates,
        "replicates_defined": len(differences), "seed": seed, "alpha": alpha,
        "first": {"estimate": point_a, "ci": _percentile_ci(values_a, alpha)},
        "second": {"estimate": point_b, "ci": _percentile_ci(values_b, alpha)},
        "difference": {"estimate": point_a - point_b if point_a is not None and point_b is not None else None,
                       "ci": _percentile_ci(differences, alpha)},
        "status": "ok" if _percentile_ci(differences, alpha) is not None else "insufficient_defined_replicates",
        "p_value": None,
    }


def mcnemar_if_independent(a: common.EvaluatedRun, b: common.EvaluatedRun) -> dict:
    validate_pair(a, b)
    if len(set(a.region_ids)) != len(a.region_ids):
        return {"test": "exact_mcnemar", "status": "not_run_overlapping_windows_within_regions",
                "p_value": None}
    if len(a.region_ids) < 10:
        return {"test": "exact_mcnemar", "status": "insufficient_independent_regions",
                "n_regions": len(a.region_ids), "minimum_regions": 10, "p_value": None}
    correct_a = (a.probabilities >= a.threshold) == a.labels
    correct_b = (b.probabilities >= b.threshold) == b.labels
    a_only = int(np.count_nonzero(correct_a & ~correct_b))
    b_only = int(np.count_nonzero(~correct_a & correct_b))
    discordant = a_only + b_only
    if discordant == 0:
        return {"test": "exact_mcnemar", "status": "no_discordant_predictions",
                "a_only_correct": a_only, "b_only_correct": b_only, "p_value": None}
    return {"test": "exact_mcnemar", "status": "ok", "a_only_correct": a_only,
            "b_only_correct": b_only, "discordant": discordant,
            "p_value": float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)}


def holm_correct(items: list[dict]) -> None:
    eligible = [(i, item["mcnemar"]["p_value"]) for i, item in enumerate(items)
                if item["mcnemar"].get("p_value") is not None]
    eligible.sort(key=lambda entry: entry[1])
    running = 0.0
    m = len(eligible)
    for rank, (index, p_value) in enumerate(eligible):
        running = max(running, min(1.0, (m - rank) * p_value))
        items[index]["mcnemar"]["p_value_holm"] = running
        items[index]["mcnemar"]["correction"] = "Holm, within this evaluation family"


def compare_runs(runs: list[common.EvaluatedRun], *, replicates: int, seed: int) -> tuple[list[dict], list[dict]]:
    comparisons, warnings = [], []
    groups: dict[tuple, list[common.EvaluatedRun]] = defaultdict(list)
    for run in runs:
        if run.config["split"] == "random_split" or run.config["smoke_max_per_subset"]:
            continue
        groups[(run.config["split"], run.config.get("horizon"))].append(run)
    metrics = ("average_precision", "roc_auc", "fl_f1", "tss", "hss")
    for (split, horizon), family in groups.items():
        family_results = []
        for i, a in enumerate(family):
            for b in family[i + 1:]:
                if a.config["model"] == b.config["model"] and a.config["feature_set"] == b.config["feature_set"]:
                    continue  # duplicate training configurations are not an independent comparison
                try:
                    validate_pair(a, b)
                    bootstrap = {name: paired_bootstrap(a, b, metric=name,
                                                         replicates=replicates, seed=seed)
                                 for name in metrics}
                    family_results.append({"first_run_id": a.run_id, "second_run_id": b.run_id,
                                           "split": split, "horizon_label": horizon,
                                           "n_test": len(a.labels), "bootstrap": bootstrap,
                                           "mcnemar": mcnemar_if_independent(a, b),
                                           "interpretation": "paired association; no causal or universal superiority claim"})
                except ValueError as exc:
                    warnings.append({"first_run_id": a.run_id, "second_run_id": b.run_id,
                                     "status": "comparison_blocked", "reason": str(exc)})
        holm_correct(family_results)
        comparisons.extend(family_results)
    return comparisons, warnings


def _false_error_summary(run: common.EvaluatedRun) -> dict:
    predicted = run.probabilities >= run.threshold
    false_positive = Counter(region for region, y, p in zip(run.region_ids, run.labels, predicted)
                             if y == 0 and p)
    false_negative = Counter(region for region, y, p in zip(run.region_ids, run.labels, predicted)
                             if y == 1 and not p)
    return {"false_positive_count": sum(false_positive.values()),
            "false_negative_count": sum(false_negative.values()),
            "regions_with_false_positives": len(false_positive),
            "regions_with_false_negatives": len(false_negative),
            "top_false_positive_region_counts": false_positive.most_common(5),
            "top_false_negative_region_counts": false_negative.most_common(5)}


def _results_csv(path: Path, results: list[dict], warnings: list[dict]) -> None:
    fields = ("run_id", "model", "split", "horizon_label", "feature_set", "mode", "status",
              "n_test", "n_fl", "fl_prevalence", "threshold", "threshold_source",
              "fl_precision", "fl_recall", "fl_f1", "nf_precision", "nf_recall", "nf_f1",
              "macro_f1", "weighted_f1", "tn", "fp", "fn", "tp", "pr_auc_average_precision",
              "roc_auc", "tss", "hss", "run_dir", "predictions_path", "reason")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            m, adv = item["test"], item["advanced"]
            writer.writerow({"run_id": item["run_id"], "model": item["model"],
                             "split": item["split"], "horizon_label": item["horizon_label"],
                             "feature_set": item["feature_set"], "mode": item["mode"],
                             "status": item["status"], "n_test": m["n"],
                             "n_fl": m["class_counts"]["FL"], "fl_prevalence": m["fl_prevalence"],
                             "threshold": item["threshold"], "threshold_source": item["threshold_source"],
                             "fl_precision": m["FL"]["precision"], "fl_recall": m["FL"]["recall"],
                             "fl_f1": m["FL"]["f1"], "nf_precision": m["NF"]["precision"],
                             "nf_recall": m["NF"]["recall"], "nf_f1": m["NF"]["f1"],
                             "macro_f1": m["summaries"]["f1"]["macro"],
                             "weighted_f1": m["summaries"]["f1"]["weighted"],
                             "tn": adv["counts"]["TN"], "fp": adv["counts"]["FP"],
                             "fn": adv["counts"]["FN"], "tp": adv["counts"]["TP"],
                             "pr_auc_average_precision": adv["pr_auc_average_precision"],
                             "roc_auc": adv["roc_auc"], "tss": adv["tss"], "hss": adv["hss"],
                             "run_dir": item["run_dir"], "predictions_path": item.get("predictions_path")})
        for warning in warnings:
            writer.writerow({"run_dir": warning.get("run_dir"), "status": warning["status"],
                             "reason": warning.get("reason")})


def _format(value: object) -> str:
    return "—" if value is None else f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)


def _summary(results: list[dict], comparisons: list[dict], warnings: list[dict],
             interpretation_path: Path | None) -> str:
    lines = ["# Phase 4 evaluation and interpretation", "",
             "FL=1 and NF=0. Confusion matrices are `[[TN, FP], [FN, TP]]`.",
             "PR-AUC means average precision on continuous FL scores. Thresholds come from saved validation metrics.", ""]
    if not results:
        lines += ["## Status", "", "Real-data evaluation is **incomplete**: no run passed saved-artifact, Phase 1 input, model-reload, and held-out prediction checks.", ""]
    else:
        lines += ["## Verified held-out runs", "", "| Run | Model | Split | Horizon label | Features | N / FL | AP | FL recall | TSS | HSS |",
                  "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for item in results:
            m, adv = item["test"], item["advanced"]
            lines.append(f"| {item['run_id']} | {item['model']} | {item['split']} | {item['horizon_label'] or '—'} | {item['feature_set']} | {m['n']} / {m['class_counts']['FL']} | {_format(adv['pr_auc_average_precision'])} | {_format(m['FL']['recall'])} | {_format(adv['tss'])} | {_format(adv['hss'])} |")
        lines += ["", "Scores from smoke runs and random splits are diagnostics only.", ""]
        for split in ("region_split", "time_split"):
            candidates = [r for r in results if r["split"] == split and r["mode"] == "full"
                          and r["advanced"]["pr_auc_average_precision"] is not None]
            if candidates:
                best = max(candidates, key=lambda r: r["advanced"]["pr_auc_average_precision"])
                lines.append(f"Highest observed {split} test average precision: {_format(best['advanced']['pr_auc_average_precision'])} ({best['run_id']}); descriptive, not proof of superiority.")
        lines += ["", "## Errors", ""]
        for item in results:
            errors = item["false_errors"]
            lines.append(f"- `{item['run_id']}`: {errors['false_positive_count']} false positives and {errors['false_negative_count']} false negatives across {errors['regions_with_false_positives']} and {errors['regions_with_false_negatives']} regions respectively; region counts are in JSON.")
    lines += ["", "## Paired comparisons", ""]
    if comparisons:
        for item in comparisons:
            ap = item["bootstrap"]["average_precision"]["difference"]
            lines.append(f"- `{item['first_run_id']}` minus `{item['second_run_id']}` ({item['split']}): AP difference {_format(ap['estimate'])}; 95% region-cluster bootstrap CI {ap['ci']}; McNemar: {item['mcnemar']['status']}.")
    else:
        lines.append("No defensible paired comparison was available. Identical ordered test samples are required; overlapping windows prevent a routine McNemar p-value.")
    lines += ["", "## Interpretability and precursor scope", ""]
    if interpretation_path and interpretation_path.is_file():
        interpreted = json.loads(interpretation_path.read_text(encoding="utf-8"))
        available = {item["run_id"]: item for item in interpreted.get("results", [])}
        for item in results:
            result = available.get(item["run_id"])
            if result is None:
                continue
            shap_result = result.get("shap", {})
            if shap_result.get("status") == "computed":
                importance_path = Path(shap_result["importance_path"])
                if importance_path.is_file() and cm.sha256_file(importance_path) == shap_result.get("importance_sha256"):
                    with importance_path.open(newline="", encoding="utf-8") as handle:
                        top = [row["feature"] for _, row in zip(range(3), csv.DictReader(handle))]
                    lines.append(f"- `{item['run_id']}`: highest mean-absolute-SHAP features in the sampled rows: {', '.join(top)}. This is model association, not causation.")
                else:
                    lines.append(f"- `{item['run_id']}`: SHAP file is missing or changed; finding withheld.")
            else:
                lines.append(f"- `{item['run_id']}`: SHAP {shap_result.get('status', 'unavailable')}.")
        if not available:
            lines.append("No verified SHAP results in the supplied interpretability file.")
        lines.append(f"Separate interpretability output: `{interpretation_path}`.")
    else:
        lines.append("SHAP and feature-pattern analysis were not run by this command; use `interpretability.py` on a verified run.")
    lines.append("Phase 3 horizon names reuse the Phase 1 target. The existing runs do not establish 6/12/24/48-hour lead-time performance or a true pre-flare temporal pattern.")
    lines += ["", "## Missing inputs and warnings", ""]
    if warnings:
        lines.extend(f"- `{w.get('run_dir', w.get('first_run_id', 'audit'))}`: {w.get('reason', w.get('status'))}" for w in warnings)
    else:
        lines.append("No blocked runs in this audit. Review scientific scope and source artifacts before external claims.")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=common.ROOT / "data")
    parser.add_argument("--run-root", action="append", type=Path, default=None)
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, default=common.ROOT / "reports" / "phase4")
    parser.add_argument("--predictions-root", type=Path, default=common.ROOT / "data" / "phase4")
    parser.add_argument("--trust-model-files", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interpretability-json", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if args.bootstrap_replicates < 20:
        raise ValueError("--bootstrap-replicates must be at least 20")
    roots = args.run_root if args.run_root is not None else [args.data_dir / "phase3", args.data_dir / "phase2"]
    dirs = common.discover_runs(roots, args.run_dir)
    names = ("phase4_results.csv", "phase4_results.json", "phase4_summary.md", "phase4_run_manifest.json")
    if args.output_root.exists() and any((args.output_root / name).exists() for name in names):
        destination = common.new_directory(args.output_root, "evaluation")
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        destination = args.output_root
    prediction_dir = args.predictions_root / f"evaluation_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{destination.name}"
    results, warnings, evaluated = basic.evaluate_available(dirs, args.data_dir,
                                                               trust_joblib=args.trust_model_files,
                                                               predictions_dir=prediction_dir,
                                                               batch_size=args.batch_size)
    if not dirs:
        warnings.append({"run_dir": ", ".join(str(root) for root in roots), "status": "blocked",
                         "reason": "No saved model runs found; restore Phase 3/Phase 2 run directories"})
    if not args.data_dir.is_dir():
        warnings.append({"run_dir": str(args.data_dir), "status": "blocked",
                         "reason": "Phase 1 data directory is absent"})
    for item, saved in zip(results, evaluated):
        item["advanced"] = advanced_metrics(saved.labels, saved.probabilities, saved.threshold)
        item["false_errors"] = _false_error_summary(saved)
    comparisons, comparison_warnings = compare_runs(evaluated, replicates=args.bootstrap_replicates,
                                                      seed=args.seed)
    warnings.extend(comparison_warnings)
    if args.interpretability_json and not args.interpretability_json.is_file():
        warnings.append({"run_dir": str(args.interpretability_json), "status": "blocked",
                         "reason": "Interpretability JSON does not exist"})
    payload = {"status": "verified_results" if results else "incomplete_no_verified_runs",
               "metric_definitions": METRIC_DEFINITIONS, "results": results,
               "comparisons": comparisons, "warnings": warnings}
    csv_path = destination / names[0]
    json_path = destination / names[1]
    summary_path = destination / names[2]
    manifest_path = destination / names[3]
    _results_csv(csv_path, results, warnings)
    common.write_json_new(json_path, payload)
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write(_summary(results, comparisons, warnings, args.interpretability_json))
    manifest = {"phase": "phase4", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "repository": common.git_revision(), "script_sha256": {
                    path.name: cm.sha256_file(path) for path in (
                        common.ROOT / "phase4_common.py", common.ROOT / "classification_metrics.py",
                        common.ROOT / "advanced_metrics.py")},
                "packages": common.environment_versions(), "random_seed": args.seed,
                "bootstrap_replicates": args.bootstrap_replicates,
                "data_dir": str(args.data_dir), "run_roots": [str(p) for p in roots],
                "run_ids": [item["run_id"] for item in results],
                "artifacts": [{"run_id": r["run_id"], "split": r["split"],
                               "feature_set": r["feature_set"], "threshold_source": r["threshold_source"],
                               "model_sha256": r["model_sha256"], "config_sha256": r["config_sha256"],
                               "metrics_sha256": r["metrics_sha256"],
                               "phase1_input_sha256": saved.config["data_sha256"],
                               "normalization_sha256": {part: info["sha256"] for part, info in
                                                         saved.config["preprocessing"]["normalization_files"].items()},
                               "predictions_sha256": r.get("predictions_sha256")}
                              for r, saved in zip(results, evaluated)],
                "outputs": {name: str(destination / name) for name in names},
                "test_status": "not_run_by_this_command", "warnings": warnings,
                "scientific_limitation": "Phase 3 nominal horizons do not have separate target labels"}
    common.write_json_new(manifest_path, manifest)
    print(f"Phase 4: {len(results)} verified run(s), {len(comparisons)} comparison(s), "
          f"{len(warnings)} warning(s); {destination}")
    return destination


if __name__ == "__main__":
    run(parse_args())
