"""Verify saved Phase 2--3 runs and produce real-data Phase 4 results."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import rebuild_common as c


def reload_probability(config: dict, run_dir: Path, x: np.ndarray) -> np.ndarray:
    name = config["model"]
    path = run_dir / config["model_file"]
    if name in ("logistic", "forest"):
        model = joblib.load(path)  # Local artifacts produced by this rebuild.
        if name == "forest":
            model.n_jobs = 1
        return model.predict_proba(x)[:, 1]
    if name == "xgboost":
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(str(path))
        return model.predict_proba(x)[:, 1]
    if name == "ann":
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        model = tf.keras.models.load_model(path, compile=False)
        return model.predict(x, batch_size=2048, verbose=0).reshape(-1)
    raise ValueError(name)


def verify_one(run_dir: Path, index: dict, targets: np.ndarray, hashes: dict) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if not status.get("complete") or not (run_dir / config["model_file"]).is_file():
        raise ValueError("Run is incomplete or its model file is missing")
    for name, expected in config["training_data_hashes"].items():
        if name not in hashes:
            if name == "horizon_targets.npy":
                path = c.DATA / "phase2" / name
            elif name == "feature_index.csv":
                path = c.DATA / "phase1" / name
            else:
                path = c.DATA / "phase1" / name
            hashes[name] = c.file_hash(path)
        if hashes[name] != expected:
            raise ValueError(f"Source hash changed: {name}")
    lead, split = config["lead_hours"], config["split"]
    expected_rows = c.rows_for(index, targets, lead, split,
                               common_cohort=config["common_cohort_across_leads"])
    for part in ("val", "test"):
        actual = np.load(run_dir / f"{part}_rows.npy")
        if not np.array_equal(actual, expected_rows[part]):
            raise ValueError(f"{part} IDs differ from authoritative split")
    val_prob = np.load(run_dir / "val_probability.npy")
    test_prob = np.load(run_dir / "test_probability.npy")
    y_val = targets[expected_rows["val"], c.LEADS.index(lead)]
    y_test = targets[expected_rows["test"], c.LEADS.index(lead)]
    threshold = c.choose_threshold(y_val, val_prob)
    if not math.isclose(threshold, saved["validation"]["threshold"], abs_tol=1e-8):
        raise ValueError("Validation threshold mismatch")
    for part, y, prob in (("val", y_val, val_prob), ("test", y_test, test_prob)):
        recomputed = c.metrics(y, prob, threshold)
        for key, value in recomputed.items():
            earlier = saved["validation" if part == "val" else "test"][key]
            if value is None and earlier is None:
                continue
            if not math.isclose(float(value), float(earlier), rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(f"Saved {part} {key} does not match predictions")
    sample_positions = np.unique(np.linspace(0, len(expected_rows["test"]) - 1,
                                             min(64, len(expected_rows["test"])), dtype=int))
    sampled_rows = expected_rows["test"][sample_positions]
    x = c.features_for(c.DATA, sampled_rows, split, config["feature_set"])
    reloaded = reload_probability(config, run_dir, x)
    if not np.allclose(reloaded, test_prob[sample_positions], rtol=1e-4, atol=1e-5):
        raise ValueError("Reloaded model disagrees with saved held-out predictions")
    return {"run_dir": str(run_dir), "config": config, "metrics": saved,
            "n_reload_checked": len(sampled_rows), "status": "verified"}


def paired_bootstrap(a: dict, b: dict, index: dict, targets: np.ndarray, *, seed: int = 42,
                     repetitions: int = 500) -> dict:
    ca, cb = a["config"], b["config"]
    if ca["lead_hours"] != cb["lead_hours"] or ca["split"] != cb["split"]:
        raise ValueError("Paired runs require the same lead and split")
    aa, bb = Path(a["run_dir"]), Path(b["run_dir"])
    rows = np.load(aa / "test_rows.npy")
    if not np.array_equal(rows, np.load(bb / "test_rows.npy")):
        raise ValueError("Paired runs have different test IDs")
    y = targets[rows, c.LEADS.index(ca["lead_hours"])]
    pa, pb = np.load(aa / "test_probability.npy"), np.load(bb / "test_probability.npy")
    region = index["regions"][rows]
    groups: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(region):
        groups[str(item)].append(i)
    keys = np.asarray(list(groups), dtype=object)
    group_positions = [np.asarray(groups[key], dtype=np.int32) for key in keys]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        choice = rng.integers(0, len(keys), size=len(keys))
        positions = np.concatenate([group_positions[j] for j in choice])
        if len(np.unique(y[positions])) != 2:
            continue
        differences.append(float(average_precision_score(y[positions], pa[positions])
                                 - average_precision_score(y[positions], pb[positions])))
    point = a["metrics"]["test"]["average_precision"] - b["metrics"]["test"]["average_precision"]
    return {"first": a["run_dir"], "second": b["run_dir"], "difference_ap": point,
            "ci95_region_bootstrap": np.quantile(differences, [0.025, 0.975]).tolist() if differences else None,
            "bootstrap_repetitions_defined": len(differences), "test_regions": len(keys)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    return "--" if value is None else f"{value:.3f}"


def make_interpretation(verified: list[dict], index: dict, targets: np.ndarray) -> dict:
    chosen = next((r for r in verified if r["config"]["phase"] == "phase3"
                   and r["config"]["lead_hours"] == 24
                   and r["config"]["split"] == "time_split"
                   and r["config"]["feature_set"] == "baseline_temporal"
                   and r["config"]["model"] == "xgboost"), None)
    if chosen is None:
        return {"status": "no_verified_24h_time_split_temporal_xgboost"}
    import xgboost as xgb

    run_dir = Path(chosen["run_dir"])
    rows = np.load(run_dir / "test_rows.npy")
    y = targets[rows, c.LEADS.index(24)]
    rng = np.random.default_rng(42)
    def one_window_per_region(candidate_rows: np.ndarray) -> np.ndarray:
        by_region: dict[str, list[int]] = defaultdict(list)
        for row_id in candidate_rows:
            by_region[str(index["regions"][row_id])].append(int(row_id))
        return np.asarray([rng.choice(items) for _, items in sorted(by_region.items())], dtype=np.int64)

    positive_regions = one_window_per_region(rows[y == 1])
    negative_regions = one_window_per_region(rows[y == 0])
    positive_region_ids = set(index["regions"][positive_regions])
    negative_regions = np.asarray(
        [row_id for row_id in negative_regions if index["regions"][row_id] not in positive_region_ids],
        dtype=np.int64,
    )
    positive = positive_regions[:128]
    negative = rng.choice(negative_regions, size=min(len(positive), len(negative_regions)), replace=False)
    sample = np.concatenate((positive, negative))
    x = c.features_for(c.DATA, sample, "time_split", "baseline_temporal")
    model = xgb.XGBClassifier()
    model.load_model(str(run_dir / chosen["config"]["model_file"]))
    best = chosen["config"].get("extra", {}).get("best_iteration")
    iteration_range = (0, best + 1) if best is not None else (0, 0)
    contribution = model.get_booster().predict(
        xgb.DMatrix(x), pred_contribs=True, iteration_range=iteration_range
    )
    if contribution.shape[1] != x.shape[1] + 1:
        raise ValueError("Unexpected TreeSHAP contribution shape")
    importance = np.mean(np.abs(contribution[:, :-1]), axis=0)
    names = chosen["config"]["feature_names"]
    order = np.argsort(importance)[::-1]
    top = [{"feature": names[i], "mean_absolute_tree_shap": float(importance[i]),
            "positive_median_standardized": float(np.median(x[:len(positive), i])),
            "negative_median_standardized": float(np.median(x[len(positive):, i]))}
           for i in order[:20]]
    output = c.REPORTS / "phase4"
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    show = top[:15][::-1]
    ax.barh([x["feature"] for x in show], [x["mean_absolute_tree_shap"] for x in show])
    ax.set_xlabel("Mean absolute TreeSHAP contribution (log odds)")
    ax.set_title("24h minimum lead, held-out chronological test sample")
    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=160)
    plt.close(fig)
    # Compare the preceding 12-hour magnetic trajectories for held-out windows.
    # This is an association at the stated forecast lead, not a causal claim.
    magnetic = []
    for item in top:
        name = item["feature"].split("__", 1)[0]
        if name not in magnetic:
            magnetic.append(name)
        if len(magnetic) == 3:
            break
    baseline_names = c.feature_names(c.DATA, "baseline")
    stats = json.loads((c.DATA / "phase1" / "normalization_time_split_baseline.json").read_text(encoding="utf-8"))
    traced = {}
    for class_name, chosen_rows in (("FL", positive_regions), ("NF", negative_regions)):
        arrays = []
        for row_id in chosen_rows:
            path = c.DATA / index["paths"][row_id]
            frame = pd.read_csv(path, sep="\t", usecols=magnetic)
            if len(frame) != 60:
                continue
            values = frame[magnetic].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64).copy()
            for j, name in enumerate(magnetic):
                k = baseline_names.index(f"{name}__last")
                values[:, j] = (values[:, j] - stats["mean_after_imputation"][k]) / stats["scale_after_imputation"][k]
            arrays.append(values)
        stack = np.stack(arrays)
        traced[class_name] = {
            "n_windows": len(arrays),
            "median": np.nanmedian(stack, axis=0).tolist(),
            "q25": np.nanquantile(stack, 0.25, axis=0).tolist(),
            "q75": np.nanquantile(stack, 0.75, axis=0).tolist(),
        }
    hours = np.arange(-59, 1) * 0.2
    fig, axes = plt.subplots(len(magnetic), 1, figsize=(9, 3 * len(magnetic)), sharex=True)
    if len(magnetic) == 1:
        axes = [axes]
    for j, (name, ax) in enumerate(zip(magnetic, axes)):
        for label, color in (("FL", "#d97706"), ("NF", "#2563eb")):
            curve = traced[label]
            med = np.asarray(curve["median"])[:, j]
            q25 = np.asarray(curve["q25"])[:, j]
            q75 = np.asarray(curve["q75"])[:, j]
            ax.plot(hours, med, label=f"{label} median", color=color)
            ax.fill_between(hours, q25, q75, color=color, alpha=0.15)
        ax.set_ylabel(f"{name}\ntraining z-score")
        ax.legend(loc="best")
    axes[-1].set_xlabel("Hours before forecast issue")
    fig.suptitle("Held-out 24h lead windows: observed magnetic trajectories")
    fig.tight_layout()
    fig.savefig(output / "preflare_patterns.png", dpi=160)
    plt.close(fig)
    (output / "preflare_patterns.json").write_text(json.dumps({
        "features": magnetic, "hours_before_issue": hours.tolist(),
        "groups": traced,
        "scope": "Class-conditioned observation-window trajectories; no causal inference",
    }, indent=2), encoding="utf-8")
    return {"status": "computed", "run_dir": str(run_dir), "sample_positive": len(positive),
            "sample_negative": len(negative), "sampling": "one window per region; class-balanced TreeSHAP sample",
            "scope": "model association and class-conditioned 12h trajectories, not causation",
            "trajectory_positive_regions": len(positive_regions),
            "trajectory_negative_regions": len(negative_regions),
            "trajectory_features": magnetic, "top_features": top}


def main() -> None:
    if not c.RUNS.is_dir():
        raise SystemExit("No saved rebuilt runs")
    index = c.load_index(c.DATA)
    targets = c.load_targets(c.DATA)
    hashes = {}
    verified, warnings = [], []
    for marker in sorted(c.RUNS.rglob("status.json")):
        try:
            result = verify_one(marker.parent, index, targets, hashes)
            verified.append(result)
            print("Verified", marker.parent, flush=True)
        except Exception as exc:
            warnings.append({"run_dir": str(marker.parent), "reason": str(exc)})
            print("REJECTED", marker.parent, exc, flush=True)
    if not verified:
        raise SystemExit(f"No verified runs; first failures: {warnings[:3]}")
    rows = []
    for result in verified:
        cfg, m = result["config"], result["metrics"]["test"]
        test_rows = np.load(Path(result["run_dir"]) / "test_rows.npy")
        test_regions = index["regions"][test_rows]
        truth = targets[test_rows, c.LEADS.index(cfg["lead_hours"])]
        rows.append({"phase": cfg["phase"], "lead_hours": cfg["lead_hours"],
                     "split": cfg["split"], "features": cfg["feature_set"],
                     "model": cfg["model"], "n_test": m["n"], "positive": m["positive"],
                     "test_regions": len(set(test_regions)),
                     "positive_regions": len(set(test_regions[truth == 1])),
                     "prevalence": m["prevalence"], "threshold": m["threshold"],
                     "ap": m["average_precision"], "roc_auc": m["roc_auc"],
                     "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "tss": m["tss"], "hss": m["hss"],
                     "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
                     "reload_checked": result["n_reload_checked"], "run_dir": result["run_dir"]})
    for phase in ("phase2", "phase3"):
        write_csv(c.REPORTS / phase / "verified_results.csv", [r for r in rows if r["phase"] == phase])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, ylabel in (("ap", "Average precision"), ("tss", "True Skill Statistic")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, split in zip(axes, ("region_split", "time_split")):
            for feature_set, color in (("baseline", "#2563eb"),
                                       ("baseline_temporal", "#d97706")):
                series = [r for r in rows if r["phase"] == "phase3" and r["split"] == split
                          and r["model"] == "xgboost" and r["features"] == feature_set]
                series.sort(key=lambda item: item["lead_hours"])
                if series:
                    ax.plot([r["lead_hours"] for r in series], [r[metric] for r in series],
                            marker="o", label=feature_set, color=color)
            ax.set_title(split.replace("_", " "))
            ax.set_xlabel("Minimum lead time (hours)")
            ax.set_xticks((6, 12, 24, 48))
            ax.grid(alpha=0.2)
            ax.legend()
        axes[0].set_ylabel(ylabel)
        fig.tight_layout()
        destination = c.REPORTS / "phase3" / f"{metric}_by_lead.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=160)
        plt.close(fig)
    comparisons = []
    key = {(r["config"]["phase"], r["config"]["lead_hours"], r["config"]["split"],
            r["config"]["feature_set"], r["config"]["model"]): r for r in verified}
    for lead in (6, 12, 24, 48):
        for split in ("region_split", "time_split"):
            a = key.get(("phase3", lead, split, "baseline_temporal", "xgboost"))
            b = key.get(("phase3", lead, split, "baseline", "xgboost"))
            if a and b:
                item = paired_bootstrap(a, b, index, targets)
                item.update({"comparison": "temporal_minus_baseline", "lead_hours": lead, "split": split})
                comparisons.append(item)
    interpretation = make_interpretation(verified, index, targets)
    payload = {"verified_run_count": len(verified), "warnings": warnings,
               "results": rows, "paired_comparisons": comparisons,
               "interpretation": interpretation,
               "label_definition": "M/X flare peak in (issue+lead, issue+lead+24h]",
               "caution": "The balanced TreeSHAP sample is not population representative; association is not causation."}
    output = c.REPORTS / "phase4"
    output.mkdir(parents=True, exist_ok=True)
    (output / "verified_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output / "verified_results.csv", rows)
    lines = ["# Verified Phase 2--4 SWAN-SF results", "",
             "Targets are M/X flare peaks in the 24-hour interval beginning after the stated minimum lead time.",
             "All scores below come from held-out predictions and a decision threshold selected on validation data.",
             "", "| Phase | Lead | Split | Features | Model | Test n / FL | AP | Recall | Precision | TSS | HSS |",
             "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['phase']} | {row['lead_hours']}h | {row['split']} | {row['features']} | "
                     f"{row['model']} | {row['n_test']} / {row['positive']} | {fmt(row['ap'])} | "
                     f"{fmt(row['recall'])} | {fmt(row['precision'])} | {fmt(row['tss'])} | {fmt(row['hss'])} |")
    lines += ["", "## Paired temporal-feature comparisons", "",
              "Differences are temporal minus baseline AP on identical held-out windows. "
              "Intervals resample active regions to keep overlapping windows together.", ""]
    for item in comparisons:
        lines.append(f"- {item['lead_hours']}h {item['split']}: AP difference {item['difference_ap']:+.4f}; "
                     f"95% region-bootstrap interval {item['ci95_region_bootstrap']}.")
    lines += ["", "## Interpretation", "",
              f"TreeSHAP: {interpretation['status']}. Feature ranks are model associations, not causal precursors.", "",
              "## Verification", "",
              f"{len(verified)} saved models passed input-hash, split-ID, saved-metric, and reload checks.",
              f"Rejected runs: {len(warnings)}.", ""]
    (output / "verified_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Verified {len(verified)} runs; rejected {len(warnings)}", flush=True)


if __name__ == "__main__":
    main()
