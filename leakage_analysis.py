"""Create shared SWAN-SF splits and measure observation-window leakage.

Run after extraction: python leakage_analysis.py
Outputs data/splits.csv and data/leakage_summary.json for all team members.
The random split is diagnostic; the time and region splits are leakage controls.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIELDS = [
    "path", "region_id", "partition", "label", "start", "end",
    "random_split", "region_split", "time_split",
]


def stable_split(key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "val"
    return "test"


def time_split(partition: str) -> str:
    number = int(partition.removeprefix("partition"))
    return "train" if number <= 3 else "val" if number == 4 else "test"


def load_instances(manifest: Path, seed: int) -> list[dict[str, str]]:
    rows = []
    with manifest.open(newline="", encoding="utf-8") as f:
        for source in csv.DictReader(f):
            if source["kind"] != "instance":
                continue
            row = {field: source[field] for field in FIELDS[:6]}
            row["random_split"] = stable_split(source["member"], seed)
            row["region_split"] = stable_split(source["region_id"], seed)
            row["time_split"] = time_split(source["partition"])
            rows.append(row)
    return rows


def overlap_counts(rows: list[dict[str, str]], split_column: str) -> dict[str, int]:
    """Count test windows whose observation interval overlaps any train window."""
    groups: dict[str, dict[str, list[tuple[datetime, datetime]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        part = row[split_column]
        if part in ("train", "test"):
            groups[row["region_id"]][part].append(
                (datetime.fromisoformat(row["start"]), datetime.fromisoformat(row["end"]))
            )
    test_count = overlapping_test = shared_region_test = 0
    shared_regions = 0
    for partitions in groups.values():
        train = sorted(partitions["train"])
        test = partitions["test"]
        test_count += len(test)
        if not train or not test:
            continue
        shared_regions += 1
        shared_region_test += len(test)
        starts = [start for start, _ in train]
        max_ends = []
        latest = datetime.min
        for _, end in train:
            latest = max(latest, end)
            max_ends.append(latest)
        for start, end in test:
            last_possible = bisect.bisect_right(starts, end) - 1
            if last_possible >= 0 and max_ends[last_possible] >= start:
                overlapping_test += 1
    return {
        "test_windows": test_count,
        "test_windows_from_training_regions": shared_region_test,
        "regions_shared_by_train_and_test": shared_regions,
        "test_windows_overlapping_training_observations": overlapping_test,
        "overlapping_test_fraction": overlapping_test / test_count if test_count else 0.0,
    }


def adjacent_window_overlap(rows: list[dict[str, str]]) -> dict[str, float | int]:
    """Summarize overlap between consecutive windows from each active region."""
    groups: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    for row in rows:
        groups[row["region_id"]].append(
            (datetime.fromisoformat(row["start"]), datetime.fromisoformat(row["end"]))
        )
    fractions = []
    spacings = []
    for windows in groups.values():
        windows.sort()
        for (start1, end1), (start2, end2) in zip(windows, windows[1:]):
            duration = min((end1 - start1).total_seconds(), (end2 - start2).total_seconds())
            if duration <= 0:
                continue
            overlap = max(0.0, (min(end1, end2) - max(start1, start2)).total_seconds())
            fractions.append(overlap / duration)
            spacings.append((start2 - start1).total_seconds() / 3600)
    return {
        "adjacent_pairs": len(fractions),
        "pairs_with_overlap": sum(value > 0 for value in fractions),
        "median_overlap_fraction": median(fractions) if fractions else 0.0,
        "median_start_spacing_hours": median(spacings) if spacings else 0.0,
    }


def summarize(rows: list[dict[str, str]], seed: int) -> dict:
    result = {
        "seed": seed,
        "windows": len(rows),
        "regions": len({row["region_id"] for row in rows}),
        "labels": dict(Counter(row["label"] for row in rows)),
        "partitions": dict(Counter(row["partition"] for row in rows)),
        "adjacent_windows": adjacent_window_overlap(rows),
        "splits": {},
    }
    for column in ("random_split", "region_split", "time_split"):
        result["splits"][column] = {
            "counts": dict(Counter(row[column] for row in rows)),
            "labels_by_split": {
                split: dict(Counter(row["label"] for row in rows if row[column] == split))
                for split in ("train", "val", "test")
            },
            **overlap_counts(rows, column),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    status = args.data_dir / "extraction_status.json"
    if not status.exists() or not json.loads(status.read_text(encoding="utf-8")).get("complete"):
        raise SystemExit("Dataset extraction is incomplete; run python extract_dataset.py first")
    rows = load_instances(args.data_dir / "manifest.csv", args.seed)
    if not rows:
        raise SystemExit("No instance rows found in manifest.csv")
    destination = args.data_dir / "splits.csv"
    with destination.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args.seed)
    (args.data_dir / "leakage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
