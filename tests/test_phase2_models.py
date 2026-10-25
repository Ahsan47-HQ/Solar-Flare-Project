"""Offline tests for model families, persistence and safe artifact handling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ann_model
import classical_models as cm
import phase2_artifacts as artifacts
import phase2_model_common as common
import xgboost_model
from test_classical_models import fixture


class Phase2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = fixture(self.root)

    def test_common_uses_authoritative_split_and_saved_stats(self):
        with patch("data_cleaning.fit_column_stats", side_effect=AssertionError("refit")):
            data, selected, labels, matrices = common.prepare(
                self.data, "time_split", "baseline_temporal", 42, 0
            )
        self.assertEqual(selected["train"].tolist(), list(range(6)))
        self.assertEqual(selected["val"].tolist(), [6, 7])
        self.assertEqual(selected["test"].tolist(), [8, 9])
        self.assertEqual(labels["train"].tolist(), [0, 1] * 3)
        self.assertEqual(matrices["test"].shape, (2, 144))
        self.assertEqual(data.columns, cm.BASE_COLUMNS + cm.TEMPORAL_COLUMNS)

    def test_backup_preserves_conflict_and_marks_missing_runs(self):
        source = self.root / "source"
        destination = self.root / "drive"
        source.mkdir()
        destination.mkdir()
        (source / "same.txt").write_text("abc")
        (source / "conflict.txt").write_text("new")
        (destination / "conflict.txt").write_text("old")
        result = artifacts.backup_tree(source, destination)
        self.assertIn("same.txt", result["verified"])
        self.assertIn("conflict.txt", result["conflicts"])
        self.assertEqual((destination / "conflict.txt").read_text(), "old")
        self.assertFalse(result["complete"])
        self.assertTrue(result["expected_missing"])

    def test_complete_classical_backup_rehashes_all_eight_runs(self):
        source = self.root / "classical"
        destination = self.root / "drive" / "classical"
        for run in artifacts.expected_run_paths(source):
            run.mkdir(parents=True)
            (run / "model.joblib").write_bytes(b"trusted-test-placeholder")
            (run / "normalization_baseline.json").write_text("{}")
            norms = {"baseline": {"saved_name": "normalization_baseline.json",
                                  "sha256": cm.sha256_file(run / "normalization_baseline.json")}}
            if run.parents[2].name == "baseline_temporal":
                (run / "normalization_temporal.json").write_text("{}")
                norms["temporal"] = {"saved_name": "normalization_temporal.json",
                                     "sha256": cm.sha256_file(run / "normalization_temporal.json")}
            (run / "config.json").write_text(json.dumps({
                "model": run.parent.name, "split": run.parents[3].name,
                "feature_set": run.parents[2].name, "smoke_max_per_subset": 0,
                "seed": 42,
                "label_mapping": {"NF": 0, "FL": 1},
                "preprocessing": {"normalization_files": norms},
            }))
            val_ap, val_recall, test_ap, test_recall = artifacts.REPORTED_CLASSICAL[run.name]
            (run / "metrics.json").write_text(json.dumps({
                "validation": {"threshold": 0.4, "pr_auc_average_precision": val_ap,
                               "FL": {"recall": val_recall}},
                "test": {"threshold": 0.4, "pr_auc_average_precision": test_ap,
                         "FL": {"recall": test_recall}},
            }))
            (run / "status.json").write_text('{"complete":true}')
        result = artifacts.backup_tree(source, destination)
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["verified"]), result["source_files"])
        self.assertTrue(artifacts.verify_backup_inventory(result)["complete"])
        first = destination / artifacts.expected_run_paths(source)[0].relative_to(source) / "model.joblib"
        first.write_bytes(b"tampered")
        self.assertFalse(artifacts.verify_backup_inventory(result)["complete"])

    def test_classical_reload_and_inference_on_held_out_rows(self):
        args = cm.parse_args(["--data-dir", str(self.data), "--output-dir", str(self.root / "runs"),
                              "--split", "region_split", "--features", "baseline",
                              "--models", "both", "--forest-trees", "4", "--forest-min-leaf", "1"])
        for run_dir in cm.run(args):
            result = artifacts.verify_run(run_dir, self.data)
            self.assertEqual(result["completion_status"], "complete")
            self.assertEqual(result["reload_status"], "passed")
            self.assertEqual(result["inference_status"], "passed")
            self.assertEqual(result["sample_rows"], 2)
            self.assertEqual(result["config"]["label_mapping"], {"NF": 0, "FL": 1})

    def test_reload_rejects_changed_phase1_normalization(self):
        args = cm.parse_args(["--data-dir", str(self.data), "--output-dir", str(self.root / "runs"),
                              "--models", "logistic"])
        run_dir = cm.run(args)[0]
        stats_path = self.data / "phase1" / "normalization_region_split_baseline.json"
        stats = json.loads(stats_path.read_text())
        stats["mean_after_imputation"][0] += 1
        stats_path.write_text(json.dumps(stats))
        result = artifacts.verify_run(run_dir, self.data)
        self.assertEqual(result["reload_status"], "failed")
        self.assertIn("normalization differs", " ".join(result["limitations"]))

    def test_missing_artifacts_do_not_turn_reported_scores_into_results(self):
        paths = artifacts.expected_run_paths(self.root / "empty" / "classical")
        records = [artifacts.verify_run(path, self.root / "missing_data") for path in paths]
        output = self.root / "reports"
        artifacts.write_reports(records, self.root / "empty", output, "Prasiddhi", "abc")
        with (output / "classical_model_metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["test_ap"] == "" for row in rows))
        manifest = json.loads((output / "experiment_manifest.json").read_text())
        self.assertFalse(manifest["complete"])
        self.assertTrue(all(row["completion_status"] == "missing" for row in rows))

    def test_xgboost_uses_validation_for_early_stopping_and_saves_metadata(self):
        capture = {}

        class FakeXGB:
            def __init__(self, **params):
                self.params = params
                self.classes_ = np.array([0, 1])
                self.best_iteration = 1
                self.n_features_in_ = 24

            def fit(self, x, y, eval_set, verbose):
                capture["train_rows"] = len(y)
                capture["validation_rows"] = len(eval_set[0][1])
                capture["validation_labels"] = eval_set[0][1].tolist()
                return self

            def predict_proba(self, x):
                p = np.full(len(x), 0.4)
                return np.column_stack((1 - p, p))

            def save_model(self, path):
                Path(path).write_text("{}")

            def load_model(self, path):
                if not Path(path).is_file():
                    raise FileNotFoundError(path)

        fake = types.SimpleNamespace(XGBClassifier=FakeXGB, __version__="test")
        args = xgboost_model.parse_args(["--data-dir", str(self.data),
                                         "--output-dir", str(self.root / "xgb"),
                                         "--trees", "3", "--split", "time_split"])
        with patch.dict(sys.modules, {"xgboost": fake}):
            run = xgboost_model.run(args)
            reload_result = artifacts.verify_run(run, self.data)
        self.assertEqual(reload_result["reload_status"], "passed", reload_result)
        self.assertEqual(reload_result["inference_status"], "passed")
        self.assertEqual(capture["train_rows"], 6)
        self.assertEqual(capture["validation_rows"], 2)
        self.assertEqual(capture["validation_labels"], [0, 1])
        self.assertTrue((run / "model.xgb.json").is_file())
        config = json.loads((run / "config.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        self.assertEqual(config["model"], "xgboost")
        self.assertEqual(config["model_parameters"]["scale_pos_weight"], 1.0)
        self.assertEqual(metrics["validation"]["threshold"], metrics["test"]["threshold"])

    def test_ann_uses_validation_and_persists_model(self):
        capture = {}

        class FakeModel:
            input_shape = (None, 24)

            def compile(self, **kwargs):
                pass

            def fit(self, x, y, validation_data, class_weight, **kwargs):
                capture["train_rows"] = len(y)
                capture["validation_rows"] = len(validation_data[1])
                capture["class_weight"] = class_weight
                return types.SimpleNamespace(history={"loss": [1.0, 0.8]})

            def predict(self, x, **kwargs):
                return np.full((len(x), 1), 0.4)

            def save(self, path):
                Path(path).write_text("fake")

        fake_keras = types.SimpleNamespace(
            utils=types.SimpleNamespace(set_random_seed=lambda seed: None),
            layers=types.SimpleNamespace(Input=lambda **kw: None, Dense=lambda *a, **kw: None,
                                         Dropout=lambda *a, **kw: None),
            Sequential=lambda layers: FakeModel(),
            optimizers=types.SimpleNamespace(Adam=lambda **kw: None),
            metrics=types.SimpleNamespace(AUC=lambda **kw: None),
            callbacks=types.SimpleNamespace(EarlyStopping=lambda **kw: None),
            models=types.SimpleNamespace(load_model=lambda path: FakeModel()),
        )
        fake_tf = types.SimpleNamespace(
            keras=fake_keras, __version__="test",
            config=types.SimpleNamespace(experimental=types.SimpleNamespace(
                enable_op_determinism=lambda: None)),
        )
        args = ann_model.parse_args(["--data-dir", str(self.data),
                                     "--output-dir", str(self.root / "ann"),
                                     "--epochs", "2"])
        with patch.dict(sys.modules, {"tensorflow": fake_tf}):
            run = ann_model.run(args)
            reload_result = artifacts.verify_run(run, self.data)
        self.assertEqual(reload_result["reload_status"], "passed")
        self.assertEqual(reload_result["inference_status"], "passed")
        self.assertEqual(capture["train_rows"], 6)
        self.assertEqual(capture["validation_rows"], 2)
        self.assertEqual(capture["class_weight"], {0: 1.0, 1: 1.0})
        self.assertTrue((run / "model.keras").is_file())
        config = json.loads((run / "config.json").read_text())
        self.assertEqual(config["model_parameters"]["epochs_completed"], 2)
        self.assertEqual(config["label_mapping"], {"NF": 0, "FL": 1})


if __name__ == "__main__":
    unittest.main()
