"""SHAP, native importance, and exploratory FL/NF feature patterns for saved runs.

Phase 3's horizon labels reuse the same Phase 1 targets. Therefore the pattern
table is exploratory class association, not measured pre-flare lead time.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import classical_models as cm
import phase4_common as common


def deterministic_sample(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum < 1:
        raise ValueError("Sample limit must be positive")
    if len(indices) <= maximum:
        return np.asarray(indices, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(indices, size=maximum, replace=False))


def stratified_evaluation_indices(run: common.EvaluatedRun, maximum: int, seed: int) -> np.ndarray:
    """Select a documented FL/NF-balanced explanation sample when both exist."""
    if maximum < 2:
        raise ValueError("SHAP evaluation sample must allow at least two rows")
    positive = run.indices[run.labels == 1]
    negative = run.indices[run.labels == 0]
    if not len(positive) or not len(negative):
        return deterministic_sample(run.indices, maximum, seed)
    n_positive = min(len(positive), maximum // 2)
    n_negative = min(len(negative), maximum - n_positive)
    remaining = maximum - n_positive - n_negative
    if remaining:
        n_positive += min(remaining, len(positive) - n_positive)
    return np.sort(np.concatenate((deterministic_sample(positive, n_positive, seed),
                                   deterministic_sample(negative, n_negative, seed + 1))))


def native_importance(model: object, model_name: str, columns: list[str]) -> tuple[str, np.ndarray] | None:
    if model_name == "logistic":
        values = np.abs(np.asarray(model.coef_)[0])
        source = "absolute_standardized_logistic_coefficient"
    elif model_name in ("forest", "xgboost"):
        values = np.asarray(model.feature_importances_)
        source = "model_native_feature_importances"
    else:
        return None
    if values.shape != (len(columns),) or not np.all(np.isfinite(values)):
        raise ValueError("Native importance does not align with feature columns")
    return source, values.astype(float)


def shap_importance(run: common.EvaluatedRun, evaluation: np.ndarray,
                    background: np.ndarray, output: Path) -> dict:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP unavailable: install a compatible shap package in the evaluation environment") from exc
    model_name = run.config["model"]
    if model_name in ("forest", "xgboost"):
        explainer = shap.TreeExplainer(run.model)
        values = explainer.shap_values(evaluation)
        explainer_name = "TreeExplainer"
    elif model_name == "logistic":
        explainer = shap.LinearExplainer(run.model, background)
        values = explainer.shap_values(evaluation)
        explainer_name = "LinearExplainer"
    else:
        predict = lambda x: np.asarray(run.model.predict(np.asarray(x), verbose=0)).reshape(-1)
        explainer = shap.KernelExplainer(predict, background)
        values = explainer.shap_values(evaluation, nsamples=min(200, 2 * evaluation.shape[1] + 1))
        explainer_name = "KernelExplainer"
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[:, :, 1]
    if values.shape != evaluation.shape or not np.all(np.isfinite(values)):
        raise ValueError(f"SHAP values have unexpected shape {values.shape}; expected {evaluation.shape}")
    ranking = np.mean(np.abs(values), axis=0)
    path = output / "shap_importance.csv"
    _write_importance(path, run.data.columns, ranking, "mean_absolute_shap")
    figure = None
    plot_warning = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap.summary_plot(values, evaluation, feature_names=run.data.columns, show=False)
        figure = output / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(figure, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        plot_warning = f"SHAP plot unavailable: {type(exc).__name__}: {exc}"
        figure = None
    return {"status": "computed", "explainer": explainer_name, "shap_version": shap.__version__,
            "evaluation_rows": int(len(evaluation)), "background_rows": int(len(background)),
            "mean_absolute_values": [float(value) for value in ranking],
            "importance_path": str(path), "importance_sha256": cm.sha256_file(path),
            "summary_plot": str(figure) if figure else None,
            "plot_warning": plot_warning}


def _write_importance(path: Path, columns: list[str], values: np.ndarray, source: str) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("rank", "feature", "importance", "source"))
        writer.writeheader()
        for rank, i in enumerate(np.argsort(-values), 1):
            writer.writerow({"rank": rank, "feature": columns[int(i)],
                             "importance": float(values[i]), "source": source})


def exploratory_patterns(run: common.EvaluatedRun, output: Path, maximum_per_class: int,
                         seed: int) -> dict:
    """Compare held-out normalized features by existing FL/NF label, not lead time."""
    subsets = {}
    for label, name in ((0, "NF"), (1, "FL")):
        available = run.indices[run.labels == label]
        subsets[name] = deterministic_sample(available, maximum_per_class, seed + label)
    if not len(subsets["NF"]) or not len(subsets["FL"]):
        return {"status": "undefined_missing_class", "n_nf": len(subsets["NF"]),
                "n_fl": len(subsets["FL"]), "lead_time_analysis": "unavailable"}
    x_nf = cm.normalized_rows(run.data, subsets["NF"])
    x_fl = cm.normalized_rows(run.data, subsets["FL"])
    mean_nf, mean_fl = np.mean(x_nf, axis=0), np.mean(x_fl, axis=0)
    median_nf, median_fl = np.median(x_nf, axis=0), np.median(x_fl, axis=0)
    path = output / "exploratory_fl_vs_nf_features.csv"
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("feature", "n_nf", "n_fl", "nf_mean_standardized",
                                                   "fl_mean_standardized", "difference_fl_minus_nf",
                                                   "nf_median_standardized", "fl_median_standardized"))
        writer.writeheader()
        for j, name in enumerate(run.data.columns):
            writer.writerow({"feature": name, "n_nf": len(x_nf), "n_fl": len(x_fl),
                             "nf_mean_standardized": float(mean_nf[j]),
                             "fl_mean_standardized": float(mean_fl[j]),
                             "difference_fl_minus_nf": float(mean_fl[j] - mean_nf[j]),
                             "nf_median_standardized": float(median_nf[j]),
                             "fl_median_standardized": float(median_fl[j])})
    figure = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.argsort(-np.abs(mean_fl - mean_nf))[:min(15, len(run.data.columns))]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(np.arange(len(order)), (mean_fl - mean_nf)[order])
        ax.set_yticks(np.arange(len(order)), [run.data.columns[int(i)] for i in order])
        ax.invert_yaxis()
        ax.set_xlabel("FL minus NF mean, training-standardized units")
        ax.set_title("Exploratory held-out class association; no lead-time claim")
        figure = output / "exploratory_fl_vs_nf.png"
        fig.tight_layout()
        fig.savefig(figure, dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    return {"status": "exploratory_class_association_only", "n_nf": len(x_nf), "n_fl": len(x_fl),
            "feature_table": str(path), "feature_table_sha256": cm.sha256_file(path),
            "plot": str(figure) if figure else None, "lead_time_analysis": "unavailable",
            "limitation": "No independent flare-onset timestamp or horizon-specific target was validated"}


def interpret_run(run: common.EvaluatedRun, output: Path, *, seed: int,
                  max_eval: int, max_background: int, max_pattern_per_class: int) -> dict:
    evaluation_indices = stratified_evaluation_indices(run, max_eval, seed)
    train_indices = cm.subset_indices(run.data, run.config["split"], run.config["seed"],
                                      run.config["smoke_max_per_subset"])["train"]
    background_indices = deterministic_sample(train_indices, max_background, seed + 1)
    x_eval = cm.normalized_rows(run.data, evaluation_indices)
    x_background = cm.normalized_rows(run.data, background_indices)
    result = {"run_id": run.run_id, "model": run.config["model"],
              "split": run.config["split"], "feature_set": run.config["feature_set"],
              "horizon_label": run.config.get("horizon"), "seed": seed,
              "evaluation_row_indices": [int(i) for i in evaluation_indices],
              "evaluation_sampling": "approximately balanced FL/NF when both classes exist; not population-representative",
              "evaluation_class_counts": {"NF": int(np.count_nonzero(run.data.labels[evaluation_indices] == 0)),
                                          "FL": int(np.count_nonzero(run.data.labels[evaluation_indices] == 1))},
              "background_row_indices": [int(i) for i in background_indices],
              "feature_columns": run.data.columns, "model_sha256": run.model_sha256,
              "normalization_sha256": {name: cm.sha256_file(path) for name, path in run.data.stats_paths.items()},
              "model_native_importance": None, "importance_rank_comparison": None,
              "shap": None, "patterns": None}
    native = native_importance(run.model, run.config["model"], run.data.columns)
    if native:
        source, values = native
        native_path = output / "model_native_importance.csv"
        _write_importance(native_path, run.data.columns, values, source)
        result["model_native_importance"] = {"source": source, "path": str(native_path),
                                              "sha256": cm.sha256_file(native_path)}
    try:
        result["shap"] = shap_importance(run, x_eval, x_background, output)
    except Exception as exc:
        result["shap"] = {"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}
    if native and result["shap"]["status"] == "computed":
        shap_values = np.asarray(result["shap"]["mean_absolute_values"])
        native_values = native[1]
        rho = spearmanr(native_values, shap_values).statistic
        k = min(10, len(shap_values))
        native_top = set(np.argsort(-native_values)[:k])
        shap_top = set(np.argsort(-shap_values)[:k])
        result["importance_rank_comparison"] = {
            "method": "Spearman correlation of native and mean-absolute-SHAP feature scores",
            "spearman_rho": float(rho) if np.isfinite(rho) else None,
            "top_k": k, "top_k_overlap": len(native_top & shap_top),
            "caution": "Scores use different definitions; compare ranks only"}
    result["patterns"] = exploratory_patterns(run, output, max_pattern_per_class, seed)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=common.ROOT / "data")
    parser.add_argument("--run-root", action="append", type=Path, default=None)
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, default=common.ROOT / "reports" / "phase4")
    parser.add_argument("--trust-model-files", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-eval", type=int, default=40)
    parser.add_argument("--max-background", type=int, default=20)
    parser.add_argument("--max-pattern-per-class", type=int, default=1000)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if args.max_eval < 2 or min(args.max_background, args.max_pattern_per_class) < 1:
        raise ValueError("max-eval must be at least 2; other sample limits must be positive")
    roots = args.run_root if args.run_root is not None else [args.data_dir / "phase3", args.data_dir / "phase2"]
    dirs = common.discover_runs(roots, args.run_dir)
    destination = common.new_directory(args.output_root, "interpretability")
    results, warnings = [], []
    for path in dirs:
        try:
            saved = common.evaluate_run(path, args.data_dir, trust_joblib=args.trust_model_files)
            run_output = destination / saved.run_id
            run_output.mkdir(exist_ok=False)
            results.append(interpret_run(saved, run_output, seed=args.seed, max_eval=args.max_eval,
                                         max_background=args.max_background,
                                         max_pattern_per_class=args.max_pattern_per_class))
        except (FileNotFoundError, PermissionError, RuntimeError, ValueError, KeyError,
                ImportError, OSError, json.JSONDecodeError) as exc:
            warnings.append({"run_dir": str(path), "status": "blocked",
                             "reason": f"{type(exc).__name__}: {exc}"})
    if not dirs:
        warnings.append({"run_dir": ", ".join(str(root) for root in roots), "status": "blocked",
                         "reason": "No saved model runs found; restore Phase 3/Phase 2 artifacts"})
    if not args.data_dir.is_dir():
        warnings.append({"run_dir": str(args.data_dir), "status": "blocked",
                         "reason": "Phase 1 data directory is absent"})
    shap_complete = bool(results) and all(item["shap"]["status"] == "computed" for item in results)
    payload = {"status": ("complete_exploratory" if shap_complete else
                           "partial_shap_blocked" if results else "incomplete_no_verified_runs"),
               "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "repository": common.git_revision(), "seed": args.seed,
               "sample_limits": {"shap_evaluation": args.max_eval,
                                 "shap_background": args.max_background,
                                 "pattern_per_class": args.max_pattern_per_class},
               "results": results, "warnings": warnings,
               "scientific_limitation": "Phase 3 horizon labels reuse the Phase 1 binary target; class patterns are not lead-time precursors"}
    common.write_json_new(destination / "interpretability_results.json", payload)
    lines = ["# Phase 4 interpretability", "", "SHAP and model-native importance have different meanings. Neither establishes causation.",
             "The feature-pattern table contrasts FL/NF held-out windows; it is not a measured pre-flare lead-time analysis.", ""]
    for item in results:
        lines += [f"## {item['run_id']}", "",
                  f"Model: `{item['model']}`; split: `{item['split']}`; features: `{item['feature_set']}`.",
                  f"SHAP: {item['shap']['status']}; {item['shap'].get('reason', item['shap'].get('importance_path', ''))}.",
                  f"Pattern analysis: {item['patterns']['status']}.", ""]
    if warnings:
        lines += ["## Blockers", ""] + [f"- `{w['run_dir']}`: {w['reason']}" for w in warnings]
    with (destination / "interpretability_summary.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Interpretability: {len(results)} verified run(s), {len(warnings)} warning(s); {destination}")
    return destination


if __name__ == "__main__":
    run(parse_args())
