"""Small, offline contract tests for the classical SWAN-SF baselines."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

import classical_models as cm
from data_cleaning import INDEX_FIELDS, SPLITS, fit_column_stats, transform_features
from temporal_features import BASE_COLUMNS, TEMPORAL_COLUMNS


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root: Path) -> Path:
    data = root / "data"
    phase1 = data / "phase1"
    phase1.mkdir(parents=True)
    rows = []
    for i in range(10):
        partition = f"partition{i // 2 + 1}"
        time_part = "train" if i < 6 else "val" if i < 8 else "test"
        rows.append({
            "path": f"{partition}/FL-or-NF/window{i}.csv",
            "region_id": f"region{i}",
            "partition": partition,
            "label": "FL" if i % 2 else "NF",
            "start": f"2020-01-{i + 1:02d}T00:00:00",
            "end": f"2020-01-{i + 1:02d}T12:00:00",
            "random_split": ("train", "val", "test")[i % 3],
            "region_split": time_part,
            "time_split": time_part,
        })
    for path, ordered_rows in (
        (data / "splits.csv", list(reversed(rows))),
        (phase1 / "feature_index.csv", rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
            writer.writeheader()
            writer.writerows(ordered_rows)
    baseline = np.arange(10 * 24, dtype=np.float64).reshape(10, 24) / 10
    temporal = np.arange(10 * 120, dtype=np.float64).reshape(10, 120) / 100
    baseline[7, 0] = np.nan  # Validation-only missing value.
    temporal[8, 2] = np.inf  # Test-only nonfinite value.
    np.save(phase1 / "baseline.npy", baseline)
    np.save(phase1 / "temporal.npy", temporal)
    write_json(phase1 / "status.json", {"complete": True, "windows": 10, "full_series": 10})
    write_json(phase1 / "feature_columns.json", {
        "baseline": BASE_COLUMNS, "temporal": TEMPORAL_COLUMNS,
    })
    by_split = {}
    for split in SPLITS:
        by_split[split] = {}
        for part in ("train", "val", "test"):
            labels = Counter(row["label"] for row in rows if row[split] == part)
            by_split[split][part] = {"FL": labels["FL"], "NF": labels["NF"]}
        train = np.asarray([i for i, row in enumerate(rows) if row[split] == "train"])
        for name, matrix, columns in (
            ("baseline", baseline, BASE_COLUMNS),
            ("temporal", temporal, TEMPORAL_COLUMNS),
        ):
            stats, _ = fit_column_stats(matrix, train, columns, outliers=False)
            write_json(phase1 / f"normalization_{split}_{name}.json", stats)
    write_json(phase1 / "cleaning_report.json", {
        "windows": 10,
        "class_distribution": {"FL": 5, "NF": 5},
        "class_distribution_by_split": by_split,
    })
    return data


class ClassicalModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = fixture(self.root)

    def test_label_mapping_and_positive_class(self):
        self.assertEqual(cm.encode_label("NF"), 0)
        self.assertEqual(cm.encode_label("FL"), 1)
        with self.assertRaises(ValueError):
            cm.encode_label("C")

    def test_alignment_and_feature_order(self):
        loaded = cm.load_phase1(self.data, "region_split", "baseline_temporal")
        self.assertIsInstance(loaded.baseline, np.memmap)
        self.assertIsInstance(loaded.temporal, np.memmap)
        self.assertEqual(len(loaded.labels), 10)
        self.assertEqual(len(loaded.columns), 144)
        self.assertEqual(loaded.columns, BASE_COLUMNS + TEMPORAL_COLUMNS)
        self.assertEqual(loaded.labels.tolist(), [0, 1] * 5)
        self.assertEqual(cm.subset_indices(loaded, "region_split", 42, 0)["test"].tolist(), [8, 9])

    def test_subsets_are_disjoint_and_time_policy_is_preserved(self):
        loaded = cm.load_phase1(self.data, "time_split", "baseline")
        selected = cm.subset_indices(loaded, "time_split", 42, 0)
        self.assertEqual(selected["train"].tolist(), list(range(6)))
        self.assertEqual(selected["val"].tolist(), [6, 7])
        self.assertEqual(selected["test"].tolist(), [8, 9])
        self.assertEqual(
            set(selected["train"]) | set(selected["val"]) | set(selected["test"]),
            set(range(10)),
        )
        self.assertFalse(set(selected["train"]) & set(selected["test"]))

    def test_missing_inputs_invalid_split_and_bad_alignment(self):
        with self.assertRaises(ValueError):
            cm.load_phase1(self.data, "made_up", "baseline")
        (self.data / "phase1" / "normalization_region_split_baseline.json").unlink()
        with self.assertRaises(FileNotFoundError):
            cm.load_phase1(self.data, "region_split", "baseline")
        self.data = fixture_fresh(self.root / "second")
        np.save(self.data / "phase1" / "baseline.npy", np.zeros((9, 24)))
        with self.assertRaises(ValueError):
            cm.load_phase1(self.data, "region_split", "baseline")

    def test_index_must_match_authoritative_splits(self):
        path = self.data / "phase1" / "feature_index.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["region_split"] = "test"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "does not match splits"):
            cm.load_phase1(self.data, "region_split", "baseline")

    def test_saved_split_normalization_only(self):
        path = self.data / "phase1" / "normalization_region_split_baseline.json"
        stats = json.loads(path.read_text(encoding="utf-8"))
        stats["mean_after_imputation"][0] += 1000
        write_json(path, stats)
        loaded = cm.load_phase1(self.data, "region_split", "baseline")
        val_index = cm.subset_indices(loaded, "region_split", 42, 0)["val"]
        with patch("data_cleaning.fit_column_stats", side_effect=AssertionError("refit attempted")):
            actual = cm.normalized_rows(loaded, val_index)
        expected = transform_features(loaded.baseline[val_index], stats).astype(np.float32)
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertEqual(loaded.stats["baseline"]["training_rows"], 6)

    def test_missing_or_wrong_normalization_is_rejected(self):
        path = self.data / "phase1" / "normalization_region_split_baseline.json"
        stats = json.loads(path.read_text(encoding="utf-8"))
        stats["training_rows"] = 10
        write_json(path, stats)
        with self.assertRaisesRegex(ValueError, "Normalization metadata"):
            cm.load_phase1(self.data, "region_split", "baseline")

    def test_metrics_imbalance_and_undefined_auc(self):
        metrics = cm.evaluate(np.array([0] * 9 + [1]), np.array([0.1] * 9 + [0.9]), 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[9, 0], [0, 1]])
        self.assertEqual(metrics["FL"]["recall"], 1.0)
        self.assertAlmostEqual(metrics["fl_prevalence"], 0.1)
        self.assertEqual(metrics["pr_auc_average_precision"], 1.0)
        single = cm.evaluate(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4]), 0.5)
        self.assertIsNone(single["pr_auc_average_precision"])
        self.assertIsNone(single["roc_auc"])
        self.assertIsNone(single["FL"]["recall"])
        self.assertEqual(single["undefined_auc_reason"], "Only one class is present")

    def test_determinism_artifacts_and_random_diagnostic(self):
        output = self.root / "outputs"
        args = cm.parse_args([
            "--data-dir", str(self.data), "--output-dir", str(output),
            "--split", "random_split", "--features", "baseline_temporal",
            "--models", "both", "--seed", "7", "--forest-trees", "4",
            "--forest-max-depth", "3", "--forest-min-leaf", "1",
            "--smoke-max-per-subset", "2",
        ])
        first = cm.run(args)
        second = cm.run(args)
        self.assertEqual(len(first), 2)
        for left, right in zip(first, second):
            a = json.loads((left / "metrics.json").read_text(encoding="utf-8"))
            b = json.loads((right / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(a, b)
            self.assertEqual(a["evaluation_interpretation"], "diagnostic_only")
            self.assertTrue((left / "model.joblib").is_file())
            self.assertTrue(json.loads((left / "status.json").read_text(encoding="utf-8"))["complete"])
            self.assertTrue((left / "normalization_baseline.json").is_file())
            self.assertTrue((left / "normalization_temporal.json").is_file())
            config = json.loads((left / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["label_mapping"], {"NF": 0, "FL": 1})
            self.assertEqual(len(config["feature_columns"]), 144)
            self.assertIn("baseline.npy", config["data_sha256"])
            self.assertIn("temporal.npy", config["data_sha256"])
            self.assertEqual(a["selected_row_counts"], {"train": 2, "val": 2, "test": 2})


def fixture_fresh(root: Path) -> Path:
    root.mkdir()
    return fixture(root)


if __name__ == "__main__":
    unittest.main()
