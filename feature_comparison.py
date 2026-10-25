"""Phase 3 feature-set comparison across all forecast horizons for SWAN-SF.

Runs a Random Forest classifier for every combination of:
  - Horizon  : 6h, 12h, 24h, 48h
  - Features : baseline (24 features) vs baseline_temporal (144 features)

All training logic is in phase3_common.run_one_horizon(). This script loops
the comparison matrix, collects HorizonResult objects, and writes a unified
Markdown + JSON report. Phase 4 can load those directly via
phase3_common.scan_run_dirs() or load_result_from_dir().

Example (from the repository root):
    python feature_comparison.py --split region_split
    python feature_comparison.py --split time_split
    python feature_comparison.py --smoke-max-per-subset 100   # diagnostic only

Outputs go to data/phase3/feature_comparison (gitignored).
Reports go to reports/phase3/feature_comparison_summary_<split>.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import classical_models as cm
import phase3_common as p3

ROOT = Path(__file__).resolve().parent
PHASE3_DEFAULT = ROOT / "data" / "phase3" / "feature_comparison"
REPORTS_DIR = ROOT / "reports" / "phase3"


def run(args: argparse.Namespace) -> list[p3.HorizonResult]:
    if args.smoke_max_per_subset < 0 or args.smoke_max_per_subset == 1:
        raise ValueError("--smoke-max-per-subset must be 0 (full) or >= 2")
    if args.forest_trees <= 0 or args.forest_min_leaf <= 0 or args.forest_max_depth <= 0:
        raise ValueError("Forest settings must be positive")

    horizons = args.horizons or list(p3.ALL_HORIZONS)
    features = args.features or list(p3.FEATURE_SETS)
    all_results: list[p3.HorizonResult] = []
    total = len(horizons) * len(features)
    done = 0
    for horizon in horizons:
        for feature_set in features:
            done += 1
            print(f"\n--- [{done}/{total}] horizon={horizon} features={feature_set} ---")
            # Override args.features for this cell
            cell_args = argparse.Namespace(**vars(args))
            cell_args.features = feature_set
            result = p3.run_one_horizon(
                horizon=horizon,
                model_name="forest",
                args=cell_args,
                output_dir=args.output_dir,
                caller_file=Path(__file__),
                experiment_name="feature_comparison",
                extra_config={
                    "comparison_note": (
                        "Systematic baseline vs baseline_temporal comparison "
                        "across all four horizons and both generalisation splits."
                    )
                },
            )
            all_results.append(result)

    # Write consolidated artefacts
    mode = "smoke" if args.smoke_max_per_subset else "full"
    comparison_dir = (
        args.output_dir
        / args.split
        / mode
        / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    )
    comparison_dir.mkdir(parents=True, exist_ok=True)
    p3.write_comparison_json(all_results, comparison_dir / "comparison_results.json")
    (comparison_dir / "status.json").write_text(
        json.dumps({"complete": True,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "cells": len(all_results)},
                   indent=2),
        encoding="utf-8",
    )

    # Markdown summary
    report_dir = getattr(args, "report_dir", None) or REPORTS_DIR
    p3.write_comparison_md(
        all_results,
        report_dir / f"feature_comparison_summary_{args.split}_{mode}.md",
        split=args.split,
        title="Phase 3 Feature Comparison (baseline vs baseline_temporal, all horizons)",
        horizons=tuple(horizons),
        feature_sets=tuple(features),
    )
    print(f"\nFeature comparison complete. {len(all_results)} cell(s). Artefacts: {comparison_dir}")
    return all_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PHASE3_DEFAULT)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--split", choices=cm.SPLITS, default="region_split")
    parser.add_argument("--horizons", nargs="+", choices=list(p3.ALL_HORIZONS),
                        help="Subset of horizons (default: all four)")
    parser.add_argument("--features", nargs="+", choices=list(p3.FEATURE_SETS),
                        help="Subset of feature sets (default: both)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced")
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
