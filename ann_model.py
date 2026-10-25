"""Train one small CPU-friendly ANN on the authoritative SWAN-SF split.

Example: python ann_model.py --split region_split --features baseline
Full training is only started when this script is run explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import classical_models as cm
import phase2_model_common as common


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=cm.ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=cm.ROOT / "data" / "phase2" / "ann")
    parser.add_argument("--split", choices=cm.SPLITS, default="region_split")
    parser.add_argument("--features", choices=cm.FEATURE_SETS, default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke-max-per-subset", type=int, default=0)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or args.patience < 0:
        raise ValueError("ANN epochs, batch size and learning rate must be positive; patience nonnegative")
    data, indices, labels, matrices = common.prepare(
        args.data_dir, args.split, args.features, args.seed, args.smoke_max_per_subset
    )
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to train this ANN") from exc
    tf.keras.utils.set_random_seed(args.seed)
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(data.columns),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    counts = np.bincount(labels["train"], minlength=2)
    total = int(counts.sum())
    weights = {0: total / (2 * int(counts[0])), 1: total / (2 * int(counts[1]))}
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=args.patience,
        restore_best_weights=True,
    )]
    history = model.fit(
        matrices["train"], labels["train"],
        validation_data=(matrices["val"], labels["val"]),
        epochs=args.epochs, batch_size=args.batch_size,
        class_weight=weights, callbacks=callbacks, verbose=2,
    )
    probabilities = {
        part: model.predict(matrices[part], batch_size=args.batch_size, verbose=0).reshape(-1)
        for part in ("val", "test")
    }
    threshold, source = cm.choose_threshold(labels["val"], probabilities["val"])
    run_dir = common.new_run_dir(args.output_dir, args.split, args.features,
                                 args.smoke_max_per_subset)
    model_file = "model.keras"
    model.save(run_dir / model_file)
    common.save_metadata(
        run_dir, model_name="ann", data=data, indices=indices, labels=labels,
        probabilities=probabilities, threshold=threshold, threshold_source=source,
        split=args.split, features=args.features, seed=args.seed,
        smoke_max=args.smoke_max_per_subset,
        parameters={"architecture": [64, 32, 1], "dropout": 0.2,
                    "epochs_requested": args.epochs, "epochs_completed": len(history.history["loss"]),
                    "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                    "patience": args.patience, "class_weights": weights,
                    "early_stopping_metric": "validation_pr_auc"},
        versions={"tensorflow": tf.__version__}, source_file=Path(__file__),
        model_filename=model_file,
    )
    print(f"Saved ANN run: {run_dir}")
    return run_dir


if __name__ == "__main__":
    run(parse_args())
