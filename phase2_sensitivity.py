"""Bounded validation-only XGBoost and ANN sensitivity study on Phase 2 baseline inputs."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

import rebuild_common as c


def run() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import xgboost as xgb

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    index = c.load_index()
    targets = c.load_targets()
    results = []
    for split in ("region_split", "time_split"):
        rows = c.rows_for(index, targets, 0, split, common_cohort=False)
        train, val = rows["train"], rows["val"]
        x_train = c.features_for(c.DATA, train, split, "baseline")
        x_val = c.features_for(c.DATA, val, split, "baseline")
        y_train = np.asarray(targets[train, 0], dtype=np.int8)
        y_val = np.asarray(targets[val, 0], dtype=np.int8)
        ratio = float(np.count_nonzero(y_train == 0) / np.count_nonzero(y_train == 1))

        for depth in (2, 3, 4):
            model = xgb.XGBClassifier(
                n_estimators=180, max_depth=depth, learning_rate=0.07,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0,
                max_bin=64, tree_method="hist", device="cuda", n_jobs=2,
                scale_pos_weight=ratio, objective="binary:logistic",
                eval_metric="aucpr", early_stopping_rounds=20, random_state=42,
            )
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
            score = float(average_precision_score(y_val, model.predict_proba(x_val)[:, 1]))
            results.append({"split": split, "model": "xgboost", "depth": depth,
                            "validation_ap": score, "best_iteration": int(model.best_iteration),
                            "train_n": len(train), "validation_n": len(val)})
            print(f"{split} XGBoost depth={depth}: validation AP={score:.4f}", flush=True)
            del model
            gc.collect()

        counts = np.bincount(y_train, minlength=2)
        weights = {0: float(len(y_train) / (2 * counts[0])),
                   1: float(len(y_train) / (2 * counts[1]))}
        for first, second in ((16, 8), (32, 16), (64, 32)):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(42)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(x_train.shape[1],)),
                tf.keras.layers.Dense(first, activation="relu"),
                tf.keras.layers.Dropout(0.15),
                tf.keras.layers.Dense(second, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss="binary_crossentropy",
                          metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
            history = model.fit(
                x_train, y_train, validation_data=(x_val, y_val),
                epochs=8, batch_size=2048, class_weight=weights, verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_pr_auc", mode="max", patience=2, restore_best_weights=True)],
            )
            probability = model.predict(x_val, batch_size=2048, verbose=0).reshape(-1)
            score = float(average_precision_score(y_val, probability))
            results.append({"split": split, "model": "ann", "units": [first, second],
                            "validation_ap": score, "epochs": len(history.history["loss"]),
                            "train_n": len(train), "validation_n": len(val)})
            print(f"{split} ANN {first}/{second}: validation AP={score:.4f}", flush=True)
            del model, history, probability
            tf.keras.backend.clear_session()
            gc.collect()
        del x_train, x_val, y_train, y_val
        gc.collect()

    dest = c.ROOT / "reports" / "phase2" / "validation_sensitivity.json"
    dest.write_text(json.dumps({"purpose": "validation-only bounded sensitivity; no test-set model selection",
                                "feature_set": "baseline", "seed": 42, "results": results}, indent=2),
                    encoding="utf-8")
    print(f"Saved {dest}", flush=True)


if __name__ == "__main__":
    run()
