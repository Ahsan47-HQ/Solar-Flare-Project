"""Train one leakage-aware XGBoost SWAN-SF classifier.

Example: python xgboost_model.py --split region_split --features baseline
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
    parser.add_argument("--output-dir", type=Path, default=cm.ROOT / "data" / "phase2" / "xgboost")
    parser.add_argument("--split", choices=cm.SPLITS, default="region_split")
    parser.add_argument("--features", choices=cm.FEATURE_SETS, default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--smoke-max-per-subset", type=int, default=0)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if args.trees <= 0 or args.max_depth <= 0 or args.learning_rate <= 0:
        raise ValueError("XGBoost trees, depth, and learning rate must be positive")
    if args.early_stopping_rounds < 0 or args.n_jobs <= 0:
        raise ValueError("Early-stopping rounds must be nonnegative and n-jobs positive")
    data, indices, labels, matrices = common.prepare(
        args.data_dir, args.split, args.features, args.seed, args.smoke_max_per_subset
    )
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("XGBoost is required to train this model") from exc
    counts = np.bincount(labels["train"], minlength=2)
    scale_pos_weight = float(counts[0] / counts[1])
    params = {
        "n_estimators": args.trees, "max_depth": args.max_depth,
        "learning_rate": args.learning_rate, "objective": "binary:logistic",
        "eval_metric": "aucpr", "tree_method": "hist", "subsample": 0.8,
        "colsample_bytree": 0.8, "scale_pos_weight": scale_pos_weight,
        "random_state": args.seed, "n_jobs": args.n_jobs,
    }
    if args.early_stopping_rounds:
        params["early_stopping_rounds"] = args.early_stopping_rounds
    model = xgb.XGBClassifier(**params)
    model.fit(matrices["train"], labels["train"],
              eval_set=[(matrices["val"], labels["val"])], verbose=False)
    if list(model.classes_) != [0, 1]:
        raise ValueError("XGBoost class order differs from NF=0, FL=1")
    probabilities = {
        part: model.predict_proba(matrices[part])[:, 1] for part in ("val", "test")
    }
    threshold, source = cm.choose_threshold(labels["val"], probabilities["val"])
    run_dir = common.new_run_dir(args.output_dir, args.split, args.features,
                                 args.smoke_max_per_subset)
    model_file = "model.xgb.json"
    model.save_model(str(run_dir / model_file))
    common.save_metadata(
        run_dir, model_name="xgboost", data=data, indices=indices, labels=labels,
        probabilities=probabilities, threshold=threshold, threshold_source=source,
        split=args.split, features=args.features, seed=args.seed,
        smoke_max=args.smoke_max_per_subset,
        parameters={**params, "best_iteration": getattr(model, "best_iteration", None)},
        versions={"xgboost": xgb.__version__}, source_file=Path(__file__),
        model_filename=model_file,
    )
    print(f"Saved XGBoost run: {run_dir}")
    return run_dir


if __name__ == "__main__":
    run(parse_args())
