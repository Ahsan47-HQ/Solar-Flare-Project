"""Build actual lead-time targets from the full SWAN-SF flare annotations.

For a forecast issued at the end of a 12-hour observation window, H-hour lead
means an M/X flare in (issue + H, issue + H + 24 hours]. This fixed 24-hour
validity period makes H the minimum warning lead. Targets are unknown when the
full active-region series does not cover the complete prediction interval.
Only MFLARE/XFLARE annotations are used as targets, never as model predictors.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LEADS_HOURS = (0, 6, 12, 24, 48)
TARGET_HOURS = 24


def build(data_dir: Path = DATA) -> Path:
    data_dir = Path(data_dir)
    index_file = data_dir / "phase1" / "feature_index.csv"
    if not index_file.is_file():
        raise FileNotFoundError(index_file)
    groups: dict[tuple[str, str], list[tuple[int, str, str]]] = defaultdict(list)
    with index_file.open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            groups[(row["partition"], row["region_id"])].append((i, row["end"], row["label"]))
    total = sum(map(len, groups.values()))
    labels = np.full((total, len(LEADS_HOURS)), -1, dtype=np.int8)
    counts = Counter()
    canonical = Counter()
    for k, ((partition, region_id), windows) in enumerate(groups.items(), 1):
        source = data_dir / "SWAN" / partition / f"{region_id}.csv"
        frame = pd.read_csv(source, sep="\t", usecols=["Timestamp", "MFLARE", "XFLARE"])
        times = pd.to_datetime(frame["Timestamp"]).to_numpy(dtype="datetime64[ns]")
        if len(times) == 0 or np.any(times[1:] <= times[:-1]):
            raise ValueError(f"Nonincreasing timestamps: {source}")
        mx = (pd.to_numeric(frame["MFLARE"], errors="coerce").fillna(0).to_numpy() > 0) | (
            pd.to_numeric(frame["XFLARE"], errors="coerce").fillna(0).to_numpy() > 0
        )
        prefix = np.concatenate(([0], np.cumsum(mx, dtype=np.int32)))
        row_ids = np.fromiter((item[0] for item in windows), dtype=np.int64)
        issue = np.array([np.datetime64(item[1]) for item in windows], dtype="datetime64[ns]")
        for j, lead in enumerate(LEADS_HOURS):
            start = issue + np.timedelta64(lead, "h")
            stop = start + np.timedelta64(TARGET_HOURS, "h")
            first = np.searchsorted(times, start, side="right")
            last = np.searchsorted(times, stop, side="right")
            covered = (stop <= times[-1]) & (first < last)
            # Require at least 95% of the 12-minute slots in the target span.
            covered &= (last - first >= 114)
            target = (prefix[last] - prefix[first] > 0).astype(np.int8)
            labels[row_ids[covered], j] = target[covered]
            counts[(lead, "valid")] += int(covered.sum())
            counts[(lead, "positive")] += int(target[covered].sum())
            if lead == 0:
                original = np.fromiter((item[2] == "FL" for item in windows), dtype=bool)
                canonical["matched"] += int(np.count_nonzero(covered & (target == original)))
                canonical["mismatched"] += int(np.count_nonzero(covered & (target != original)))
        if k % 500 == 0:
            print(f"Labeled {k:,}/{len(groups):,} active regions", flush=True)
    output = data_dir / "phase2"
    output.mkdir(exist_ok=True)
    path = output / "horizon_targets.npy"
    np.save(path, labels)
    report = {
        "label_definition": "M/X peak in (issue + lead, issue + lead + 24h]",
        "lead_hours": list(LEADS_HOURS),
        "target_hours": TARGET_HOURS,
        "minimum_target_coverage": "114 of 120 12-minute slots and series through interval end",
        "row_order": "data/phase1/feature_index.csv",
        "per_lead": {
            f"{h}h": {"valid": counts[(h, "valid")], "positive": counts[(h, "positive")],
                       "censored": total - counts[(h, "valid")]}
            for h in LEADS_HOURS
        },
        "supplied_24h_label_comparison_on_valid_rows": dict(canonical),
    }
    (output / "horizon_targets_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return path


if __name__ == "__main__":
    build()
