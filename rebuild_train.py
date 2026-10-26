"""Train real SWAN-SF models for Phases 2 and 3 with bounded resources."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import rebuild_common as c


def make_model(name: str, y_train: np.ndarray, seed: int):
    if name == "logistic":
        return LogisticRegression(
            solver="liblinear", max_iter=200, class_weight="balanced", random_state=seed
        )
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=80, max_depth=12, min_samples_leaf=15,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=1, random_state=seed,
        )
    if name == "xgboost":
        import xgboost as xgb

        counts = np.bincount(y_train, minlength=2)
        return xgb.XGBClassifier(
            n_estimators=180, max_depth=3, learning_rate=0.07,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0,
            max_bin=64, tree_method="hist", device="cuda", n_jobs=2,
            scale_pos_weight=float(counts[0] / counts[1]),
            objective="binary:logistic", eval_metric="aucpr",
            early_stopping_rounds=20, random_state=seed,
        )
    raise ValueError(name)


def train_ann(x_train, y_train, x_val, y_val, seed: int):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    counts = np.bincount(y_train, minlength=2)
    weights = {0: float(len(y_train) / (2 * counts[0])), 1: float(len(y_train) / (2 * counts[1]))}
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=8, batch_size=2048, class_weight=weights, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=2, restore_best_weights=True
        )],
    )
    return model, {"epochs_completed": len(history.history["loss"]), "class_weights": weights,
                   "tensorflow_version": tf.__version__, "device": "cpu_native_windows"}


def run(phase: str, lead: int, split: str, feature_set: str, model_name: str, *, seed: int = 42) -> Path:
    if phase not in ("phase2", "phase3") or lead not in c.LEADS:
        raise ValueError("Invalid phase or lead")
    if split not in c.SPLITS or feature_set not in c.FEATURES or model_name not in c.MODELS:
        raise ValueError("Invalid split, feature set, or model")
    if phase == "phase3" and lead == 0:
        raise ValueError("Phase 3 requires a positive minimum lead time")
    destination = c.run_path(phase, lead, split, feature_set, model_name)
    if (destination / "status.json").is_file():
        status = json.loads((destination / "status.json").read_text(encoding="utf-8"))
        if status.get("complete"):
            print(f"Already complete: {destination}", flush=True)
            return destination
    destination.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    index = c.load_index(c.DATA)
    targets = c.load_targets(c.DATA)
    common_cohort = phase == "phase3"
    row_sets = c.rows_for(index, targets, lead, split, common_cohort=common_cohort)
    y_sets = {part: np.asarray(targets[rows, c.LEADS.index(lead)], dtype=np.int8)
              for part, rows in row_sets.items()}
    if any(len(np.unique(y)) != 2 for y in y_sets.values()):
        raise ValueError(f"Both classes required in every subset: {phase}/{lead}/{split}")
    print(f"Training {phase} {lead}h {split} {feature_set} {model_name}: "
          f"{[len(row_sets[p]) for p in ('train','val','test')]} rows", flush=True)
    x_train = c.features_for(c.DATA, row_sets["train"], split, feature_set)
    x_val = c.features_for(c.DATA, row_sets["val"], split, feature_set)
    x_test = c.features_for(c.DATA, row_sets["test"], split, feature_set)
    extra = {}
    if model_name == "ann":
        model, extra = train_ann(x_train, y_sets["train"], x_val, y_sets["val"], seed)
        predict = lambda x: model.predict(x, batch_size=2048, verbose=0).reshape(-1)
        model_file = "model.keras"
        model.save(destination / model_file)
    else:
        model = make_model(model_name, y_sets["train"], seed)
        if model_name == "xgboost":
            model.fit(x_train, y_sets["train"], eval_set=[(x_val, y_sets["val"])], verbose=False)
            config = json.loads(model.get_booster().save_config())
            extra = {"device": config["learner"]["generic_param"]["device"],
                     "best_iteration": int(model.best_iteration)}
            model_file = "model.xgb.json"
            model.save_model(str(destination / model_file))
        else:
            model.fit(x_train, y_sets["train"])
            model_file = "model.joblib"
            joblib.dump(model, destination / model_file, compress=3)
        predict = lambda x: model.predict_proba(x)[:, 1]
    val_probability = np.asarray(predict(x_val), dtype=np.float32)
    test_probability = np.asarray(predict(x_test), dtype=np.float32)
    threshold = c.choose_threshold(y_sets["val"], val_probability)
    metric_data = {
        "validation": c.metrics(y_sets["val"], val_probability, threshold),
        "test": c.metrics(y_sets["test"], test_probability, threshold),
        "threshold_source": "maximum validation F1",
    }
    for part in ("val", "test"):
        np.save(destination / f"{part}_rows.npy", row_sets[part])
        np.save(destination / f"{part}_probability.npy",
                val_probability if part == "val" else test_probability)
    hashes = {
        "feature_index.csv": c.file_hash(c.DATA / "phase1" / "feature_index.csv"),
        "horizon_targets.npy": c.file_hash(c.DATA / "phase2" / "horizon_targets.npy"),
        "baseline.npy": c.file_hash(c.DATA / "phase1" / "baseline.npy"),
        f"normalization_{split}_baseline.json": c.file_hash(
            c.DATA / "phase1" / f"normalization_{split}_baseline.json"
        ),
    }
    if feature_set == "baseline_temporal":
        hashes["temporal.npy"] = c.file_hash(c.DATA / "phase1" / "temporal.npy")
        hashes[f"normalization_{split}_temporal.json"] = c.file_hash(
            c.DATA / "phase1" / f"normalization_{split}_temporal.json"
        )
    config = {
        "phase": phase, "lead_hours": lead, "target_window_hours": 24,
        "label_definition": "M/X flare in (issue+lead, issue+lead+24h]",
        "common_cohort_across_leads": common_cohort,
        "split": split, "feature_set": feature_set, "model": model_name,
        "model_file": model_file, "seed": seed,
        "feature_names": c.feature_names(c.DATA, feature_set),
        "rows": {part: len(rows) for part, rows in row_sets.items()},
        "positives": {part: int(y.sum()) for part, y in y_sets.items()},
        "training_data_hashes": hashes,
        "source_sha256": c.file_hash(Path(__file__)),
        "python": platform.python_version(),
        "extra": extra,
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (destination / "metrics.json").write_text(json.dumps(metric_data, indent=2), encoding="utf-8")
    (destination / "status.json").write_text(json.dumps({
        "complete": True, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }, indent=2), encoding="utf-8")
    print(f"Saved {destination}: AP={metric_data['test']['average_precision']:.4f}, "
          f"TSS={metric_data['test']['tss']:.4f}, {time.monotonic()-start:.1f}s", flush=True)
    del model, x_train, x_val, x_test
    gc.collect()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase2", "phase3"), required=True)
    parser.add_argument("--lead", type=int, choices=c.LEADS, required=True)
    parser.add_argument("--split", choices=c.SPLITS, required=True)
    parser.add_argument("--features", choices=c.FEATURES, required=True)
    parser.add_argument("--model", choices=c.MODELS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.phase, args.lead, args.split, args.features, args.model, seed=args.seed)


if __name__ == "__main__":
    main()
