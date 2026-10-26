"""Run the complete bounded Phase 2--3 experiment matrix, one process at a time."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import rebuild_common as c


def experiments() -> list[tuple[str, int, str, str, str]]:
    plan = []
    for split in ("region_split", "time_split"):
        for features in c.FEATURES:
            for model in c.MODELS:
                plan.append(("phase2", 0, split, features, model))
    plan.append(("phase2", 0, "random_split", "baseline", "xgboost"))
    for lead in (6, 12, 24, 48):
        for split in ("region_split", "time_split"):
            for features in c.FEATURES:
                plan.append(("phase3", lead, split, features, "xgboost"))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase2", "phase3", "all"), default="all")
    args = parser.parse_args()
    plan = [item for item in experiments() if args.phase == "all" or item[0] == args.phase]
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "TF_CPP_MIN_LOG_LEVEL": "2"})
    for i, (phase, lead, split, features, model) in enumerate(plan, 1):
        dest = c.run_path(phase, lead, split, features, model)
        marker = dest / "status.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("complete"):
            print(f"[{i}/{len(plan)}] Reusing {dest}", flush=True)
            continue
        print(f"[{i}/{len(plan)}] Starting {phase} {lead}h {split} {features} {model}", flush=True)
        command = [sys.executable, "-u", str(Path(__file__).with_name("rebuild_train.py")),
                   "--phase", phase, "--lead", str(lead), "--split", split,
                   "--features", features, "--model", model]
        subprocess.run(command, check=True, env=env, cwd=c.ROOT)
    print(f"Complete: {len(plan)} planned runs", flush=True)


if __name__ == "__main__":
    main()
