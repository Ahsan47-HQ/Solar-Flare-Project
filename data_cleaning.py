"""Phase 1 cleaning audit and shared feature preparation for SWAN-SF.

Run with the project virtual environment: .sfd/Scripts/python.exe data_cleaning.py
The script scans each full active-region series once, writes baseline and temporal
feature arrays aligned to `feature_index.csv`, and fits median-imputation plus
z-score normalization parameters on the training rows of each shared split.
Outliers are reported, not removed: extreme magnetic values may be real signals.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_features import (
    BASE_COLUMNS,
    MAGNETIC_COLUMNS,
    TEMPORAL_COLUMNS,
    compute_region_features,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INDEX_FIELDS = [
    "path", "region_id", "partition", "label", "start", "end",
    "random_split", "region_split", "time_split",
]
SPLITS = ("random_split", "region_split", "time_split")


def load_inputs(data_dir: Path) -> tuple[dict, dict, int]:
    status_path = data_dir / "extraction_status.json"
    if not status_path.exists() or not json.loads(status_path.read_text(encoding="utf-8")).get("complete"):
        raise ValueError("Extract the complete dataset before running data_cleaning.py")
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    n = 0
    with (data_dir / "splits.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[(row["partition"], row["region_id"])].append(row)
            n += 1
    series = {}
    with (data_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["kind"] == "series":
                series[(row["partition"], row["region_id"])] = data_dir / row["path"]
    if not groups or not series:
        raise ValueError("The manifest or splits file is empty")
    unknown = groups.keys() - series.keys()
    if unknown:
        raise ValueError(f"Instance region has no full series: {next(iter(unknown))}")
    return groups, series, n


def fit_column_stats(
    matrix: np.ndarray, indices: np.ndarray, columns: list[str], *, outliers: bool
) -> tuple[dict, dict]:
    """Fit one column at a time to keep memory use low."""
    medians, means, scales, missing = [], [], [], []
    outlier_report = {}
    for j, name in enumerate(columns):
        values = np.asarray(matrix[indices, j], dtype=np.float64)
        finite = np.isfinite(values)
        clean = values[finite]
        median = float(np.median(clean)) if len(clean) else 0.0
        missing.append(int((~finite).sum()))
        if outliers and len(clean):
            q1, q3 = np.quantile(clean, [0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_report[name] = {
                "lower": float(lower), "upper": float(upper),
                "training_values_outside": int(((clean < lower) | (clean > upper)).sum()),
            }
        values[~finite] = median
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        medians.append(median)
        means.append(mean)
        scales.append(scale if scale > 0 else 1.0)
    return {
        "feature_columns": columns,
        "training_rows": len(indices),
        "impute_median": medians,
        "mean_after_imputation": means,
        "scale_after_imputation": scales,
        "missing_training_values": dict(zip(columns, missing)),
    }, outlier_report


def transform_features(values: np.ndarray, stats: dict) -> np.ndarray:
    """Apply saved training-only imputation and z-score normalization."""
    result = np.asarray(values, dtype=np.float64).copy()
    medians = np.asarray(stats["impute_median"], dtype=np.float64)
    means = np.asarray(stats["mean_after_imputation"], dtype=np.float64)
    scales = np.asarray(stats["scale_after_imputation"], dtype=np.float64)
    if result.shape[-1] != len(medians):
        raise ValueError("Feature count does not match the normalization file")
    invalid = ~np.isfinite(result)
    if invalid.any():
        result[invalid] = np.broadcast_to(medians, result.shape)[invalid]
    return (result - means) / scales


def build_phase1(data_dir: Path = DATA) -> None:
    data_dir = Path(data_dir)
    groups, series_paths, n = load_inputs(data_dir)
    output = data_dir / "phase1"
    output.mkdir(exist_ok=True)
    status_file = output / "status.json"
    status_file.unlink(missing_ok=True)
    baseline = np.lib.format.open_memmap(
        output / "baseline.npy", mode="w+", dtype=np.float64,
        shape=(n, len(BASE_COLUMNS)),
    )
    temporal = np.lib.format.open_memmap(
        output / "temporal.npy", mode="w+", dtype=np.float64,
        shape=(n, len(TEMPORAL_COLUMNS)),
    )
    (output / "feature_columns.json").write_text(
        json.dumps({"baseline": BASE_COLUMNS, "temporal": TEMPORAL_COLUMNS}, indent=2),
        encoding="utf-8",
    )
    train_indices = {split: [] for split in SPLITS}
    class_counts = Counter()
    partition_classes = Counter()
    split_classes = Counter()
    raw_missing = Counter()
    raw_rows = 0
    position = 0
    with (output / "feature_index.csv").open("w", newline="", encoding="utf-8") as index_file:
        writer = csv.DictWriter(index_file, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for series_number, ((partition, region_id), series_path) in enumerate(series_paths.items(), 1):
            frame = pd.read_csv(
                series_path, sep="\t", usecols=["Timestamp", *MAGNETIC_COLUMNS],
                low_memory=False,
            )
            numeric = frame[MAGNETIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
            values = numeric.to_numpy(dtype=np.float64)
            invalid = ~np.isfinite(values)
            raw_rows += len(frame)
            raw_missing.update(dict(zip(MAGNETIC_COLUMNS, invalid.sum(axis=0).tolist())))
            frame[MAGNETIC_COLUMNS] = numeric.replace([np.inf, -np.inf], np.nan)
            windows = groups.get((partition, region_id))
            if windows:
                base_frame, temporal_frame = compute_region_features(frame, windows)
                count = len(windows)
                baseline[position:position + count] = base_frame.to_numpy(dtype=np.float64)
                temporal[position:position + count] = temporal_frame.to_numpy(dtype=np.float64)
                for offset, row in enumerate(windows):
                    writer.writerow({field: row[field] for field in INDEX_FIELDS})
                    index = position + offset
                    class_counts[row["label"]] += 1
                    partition_classes[(row["partition"], row["label"])] += 1
                    for split in SPLITS:
                        split_classes[(split, row[split], row["label"])] += 1
                        if row[split] == "train":
                            train_indices[split].append(index)
                position += count
            if series_number % 500 == 0:
                baseline.flush()
                temporal.flush()
                print(f"Scanned {series_number:,}/{len(series_paths):,} series; {position:,}/{n:,} windows", flush=True)
    if position != n:
        raise ValueError(f"Expected {n} windows; built {position}")
    baseline.flush()
    temporal.flush()

    report = {
        "full_series": len(series_paths),
        "unique_raw_observation_rows": raw_rows,
        "raw_missing_or_nonfinite_by_magnetic_column": dict(raw_missing),
        "windows": n,
        "class_distribution": dict(class_counts),
        "class_distribution_by_partition": {
            partition: {label: partition_classes[(partition, label)] for label in ("FL", "NF")}
            for partition in sorted({key[0] for key in partition_classes})
        },
        "class_distribution_by_split": {
            split: {
                part: {label: split_classes[(split, part, label)] for label in ("FL", "NF")}
                for part in ("train", "val", "test")
            }
            for split in SPLITS
        },
        "outlier_rule": "Training baseline values outside Q1 - 1.5*IQR or Q3 + 1.5*IQR; reported, not removed",
        "outliers_by_split": {},
    }
    for split in SPLITS:
        indices = np.asarray(train_indices[split], dtype=np.int64)
        base_stats, outliers = fit_column_stats(baseline, indices, BASE_COLUMNS, outliers=True)
        temp_stats, _ = fit_column_stats(temporal, indices, TEMPORAL_COLUMNS, outliers=False)
        (output / f"normalization_{split}_baseline.json").write_text(
            json.dumps(base_stats, indent=2), encoding="utf-8"
        )
        (output / f"normalization_{split}_temporal.json").write_text(
            json.dumps(temp_stats, indent=2), encoding="utf-8"
        )
        report["outliers_by_split"][split] = outliers
    (output / "cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    status_file.write_text(
        json.dumps({"complete": True, "windows": n, "full_series": len(series_paths)}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {n:,} aligned feature rows to {output}", flush=True)
    print(f"Raw missing/nonfinite cells: {sum(raw_missing.values()):,} of {raw_rows * len(MAGNETIC_COLUMNS):,}", flush=True)
    print(f"Class distribution: {dict(class_counts)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    build_phase1(args.data_dir)


if __name__ == "__main__":
    main()
