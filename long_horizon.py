"""Phase 3 long-horizon forecasting experiments for SWAN-SF.

Runs Logistic Regression and Random Forest classifiers under two forecast-horizon
configurations: 24-hour and 48-hour warning times. Longer horizons represent
earlier warnings and are generally harder to predict. All training logic lives in
phase3_common.run_one_horizon(); this script is orchestration only.

Example (from the repository root):
    python long_horizon.py --split region_split --features baseline
    python long_horizon.py --split time_split --features baseline_temporal
    python long_horizon.py --smoke-max-per-subset 100   # diagnostic only

Outputs go to data/phase3/long_horizon (gitignored).
The random split and every --smoke-max-per-subset run are diagnostics only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import classical_models as cm
import phase3_common as p3

ROOT = Path(__file__).resolve().parent
PHASE3_DEFAULT = ROOT / "data" / "phase3" / "long_horizon"


def run(args: argparse.Namespace) -> list[p3.HorizonResult]:
    if args.smoke_max_per_subset < 0 or args.smoke_max_per_subset == 1:
        raise ValueError("--smoke-max-per-subset must be 0 (full) or >= 2")
    if args.logistic_c <= 0 or args.logistic_max_iter <= 0:
        raise ValueError("Logistic settings must be positive")
    if args.forest_trees <= 0 or args.forest_min_leaf <= 0 or args.forest_max_depth <= 0:
        raise ValueError("Forest settings must be positive")

    horizons = args.horizons or list(p3.LONG_HORIZONS)
    models = ("logistic", "forest") if args.models == "both" else (args.models,)
    all_results: list[p3.HorizonResult] = []
    total = len(horizons) * len(models)
    done = 0
    for horizon in horizons:
        if horizon not in p3.LONG_HORIZONS:
            raise ValueError(f"Invalid horizon {horizon!r}; long_horizon accepts {p3.LONG_HORIZONS}")
        for model_name in models:
            done += 1
            print(f"\n--- [{done}/{total}] horizon={horizon} model={model_name} ---")
            result = p3.run_one_horizon(
                horizon=horizon,
                model_name=model_name,
                args=args,
                output_dir=args.output_dir,
                caller_file=Path(__file__),
                experiment_name="long_horizon",
            )
            all_results.append(result)

    mode = "smoke" if args.smoke_max_per_subset else "full"
    report_dir = getattr(args, "report_dir", None) or (ROOT / "reports" / "phase3")
    p3.write_comparison_md(
        all_results,
        report_dir / f"long_horizon_{args.split}_{mode}.md",
        split=args.split,
        title="Phase 3 Long-Horizon Results (24h, 48h)",
        horizons=tuple(horizons),
    )
    print(f"\nLong-horizon complete. {len(all_results)} run(s) saved.")
    return all_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PHASE3_DEFAULT)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "phase3")
    parser.add_argument("--split", choices=cm.SPLITS, default="region_split")
    parser.add_argument("--features", choices=cm.FEATURE_SETS, default="baseline")
    parser.add_argument("--models", choices=("both", "logistic", "forest"), default="both")
    parser.add_argument("--horizons", nargs="+", choices=list(p3.LONG_HORIZONS),
                        help="Subset of long horizons (default: 24h and 48h)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--logistic-max-iter", type=int, default=500)
    parser.add_argument("--forest-trees", type=int, default=100)
    parser.add_argument("--forest-max-depth", type=int, default=16)
    parser.add_argument("--forest-min-leaf", type=int, default=5)
    parser.add_argument("--threshold-strategy", choices=p3.THRESHOLD_STRATEGIES, default="max_f1",
                        help="Validation threshold tuning method (max_f1, max_tss, target_recall_90)")
    parser.add_argument("--smoke-max-per-subset", type=int, default=0,
                        help="Use at most N rows per subset for a fast diagnostic (0=full).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
