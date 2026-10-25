"""Offline Phase 4 tests; all model data and saved runs are synthetic."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

import advanced_metrics as advanced
import classification_metrics as basic
import classical_models as cm
import interpretability
import phase4_common as common
from test_classical_models import fixture


class TestPhase4PureMetrics(unittest.TestCase):
    def test_precision_recall_f1_and_confusion_order(self):
        scores = basic.classification_metrics(np.array([0, 0, 1, 1]),
                                               np.array([0.1, 0.8, 0.7, 0.2]), 0.5)
        self.assertEqual(scores["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(scores["FL"]["precision"], 0.5)
        self.assertEqual(scores["FL"]["recall"], 0.5)
        self.assertEqual(scores["FL"]["f1"], 0.5)
        self.assertEqual(scores["NF"]["f1"], 0.5)
        self.assertEqual(scores["summaries"]["f1"]["macro"], 0.5)

    def test_saved_threshold_is_applied_without_tuning(self):
        y = np.array([0, 1])
        p = np.array([0.4, 0.6])
        self.assertEqual(basic.classification_metrics(y, p, 0.7)["confusion_matrix"], [[1, 0], [1, 0]])
        self.assertEqual(basic.classification_metrics(y, p, 0.5)["confusion_matrix"], [[1, 0], [0, 1]])

    def test_missing_classes_and_zero_predicted_positive(self):
        scores = basic.classification_metrics(np.array([0, 0]), np.array([0.1, 0.2]), 0.5)
        self.assertIsNone(scores["FL"]["recall"])
        self.assertIsNone(scores["pr_auc_average_precision"])
        self.assertIsNone(scores["roc_auc"])
        self.assertIsNone(scores["summaries"]["f1"]["macro"])
        scores = basic.classification_metrics(np.array([0, 1]), np.array([0.1, 0.2]), 0.5)
        self.assertIsNone(scores["FL"]["precision"])
        self.assertEqual(scores["FL"]["f1"], 0)

    def test_probability_validation(self):
        for values in (np.array([[0.3], [0.5]]), np.array([0.2, np.nan]),
                       np.array([-0.1, 0.5]), np.array([0.2, 1.1])):
            with self.subTest(values=str(values)), self.assertRaises(ValueError):
                common.validate_probabilities(values, 2)
        with self.assertRaises(ValueError):
            basic.classification_metrics(np.array([0, 2]), np.array([0.2, 0.8]), 0.5)

    def test_tss_hss_and_zero_denominators(self):
        skill = advanced.skill_scores([[90, 10], [5, 5]])
        self.assertAlmostEqual(skill["tss"], 0.4)
        self.assertAlmostEqual(skill["hss"], 2 * (5 * 90 - 5 * 10) / ((5 + 5) * (5 + 90) + (5 + 10) * (10 + 90)))
        self.assertIsNone(advanced.skill_scores([[0, 0], [0, 0]])["hss"])
        self.assertIsNone(advanced.skill_scores([[10, 0], [0, 0]])["tss"])
        with self.assertRaises(ValueError):
            advanced.skill_scores([[1, -1], [0, 0]])

    def test_average_precision_and_roc_use_probabilities(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.05, 0.7, 0.8, 0.9])
        score = advanced.advanced_metrics(y, p, 0.5)
        self.assertEqual(score["pr_auc_average_precision"], 1.0)
        self.assertEqual(score["roc_auc"], 1.0)
        self.assertEqual(score["counts"], {"TN": 1, "FP": 1, "FN": 0, "TP": 2})


class TestPhase4RunContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(gc.collect)  # release Windows memory maps before removing fixture files
        self.root = Path(temporary.name)
        self.data_dir = fixture(self.root)
        args = cm.parse_args(["--data-dir", str(self.data_dir), "--output-dir", str(self.root / "runs"),
                              "--split", "region_split", "--features", "baseline", "--models", "logistic",
                              "--logistic-max-iter", "50"])
        self.run_dir = cm.run(args)[0]

    def test_reloads_aligned_rows_and_saved_threshold(self):
        run = common.evaluate_run(self.run_dir, self.data_dir, trust_joblib=True)
        self.assertEqual(run.sample_ids, ["partition5/FL-or-NF/window8.csv", "partition5/FL-or-NF/window9.csv"])
        self.assertEqual(run.labels.tolist(), [0, 1])
        self.assertEqual(run.threshold, run.source_metrics["validation"]["threshold"])
        self.assertEqual(run.threshold, run.source_metrics["test"]["threshold"])
        self.assertEqual(run.data.columns, run.config["feature_columns"])

    def test_untrusted_joblib_and_missing_model_are_blocked(self):
        with self.assertRaises(PermissionError):
            common.evaluate_run(self.run_dir, self.data_dir, trust_joblib=False)
        (self.run_dir / "model.joblib").rename(self.run_dir / "moved.joblib")
        with self.assertRaises(FileNotFoundError):
            common.evaluate_run(self.run_dir, self.data_dir, trust_joblib=True)

    def test_missing_data_and_hash_mismatch_are_blocked(self):
        with self.assertRaises(FileNotFoundError):
            common.evaluate_run(self.run_dir, self.root / "absent", trust_joblib=True)
        config = json.loads((self.run_dir / "config.json").read_text())
        config["data_sha256"]["baseline.npy"] = "0" * 64
        (self.run_dir / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            common.evaluate_run(self.run_dir, self.data_dir, trust_joblib=True)

    def test_feature_order_and_threshold_mismatch_are_blocked(self):
        config_path = self.run_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["feature_columns"] = list(reversed(config["feature_columns"]))
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "feature order"):
            common.evaluate_run(self.run_dir, self.data_dir, trust_joblib=True)
        config["feature_columns"] = list(reversed(config["feature_columns"]))
        config_path.write_text(json.dumps(config))
        metrics_path = self.run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["test"]["threshold"] = 0.99 if metrics["validation"]["threshold"] != 0.99 else 0.01
        metrics_path.write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "threshold"):
            common.evaluate_run(self.run_dir, self.data_dir, trust_joblib=True)

    def test_paired_alignment_and_cluster_bootstrap(self):
        run = common.evaluate_run(self.run_dir, self.data_dir, trust_joblib=True)
        altered = replace(run, sample_ids=list(reversed(run.sample_ids)))
        with self.assertRaisesRegex(ValueError, "identical ordered"):
            advanced.validate_pair(run, altered)
        changed_split = replace(run, config={**run.config, "split": "time_split"})
        with self.assertRaisesRegex(ValueError, "same split"):
            advanced.validate_pair(run, changed_split)
        small = advanced.paired_bootstrap(run, run, replicates=100, seed=9)
        self.assertEqual(small["status"], "insufficient_independent_regions")
        paired = replace(run, labels=np.tile([0, 1], 10),
                         probabilities=np.tile([0.1, 0.9], 10),
                         sample_ids=[f"sample{i}" for i in range(20)],
                         region_ids=[f"region{i}" for i in range(20)])
        comparison = advanced.paired_bootstrap(paired, paired, replicates=100, seed=9)
        self.assertEqual(comparison["n_regions"], 20)
        self.assertEqual(comparison["difference"]["estimate"], 0)
        self.assertEqual(comparison["difference"]["ci"], [0.0, 0.0])

    def test_mcnemar_independence_guard(self):
        run = common.evaluate_run(self.run_dir, self.data_dir, trust_joblib=True)
        repeated = replace(run, region_ids=["same", "same"])
        self.assertEqual(advanced.mcnemar_if_independent(repeated, repeated)["status"],
                         "not_run_overlapping_windows_within_regions")
        self.assertEqual(advanced.mcnemar_if_independent(run, run)["status"],
                         "insufficient_independent_regions")
        paired = replace(run, labels=np.tile([0, 1], 10),
                         probabilities=np.tile([0.1, 0.9], 10),
                         sample_ids=[f"sample{i}" for i in range(20)],
                         region_ids=[f"region{i}" for i in range(20)])
        worse = replace(paired, probabilities=np.full(20, 0.5), threshold=0.5)
        exact = advanced.mcnemar_if_independent(paired, worse)
        self.assertEqual(exact["status"], "ok")
        self.assertEqual(exact["discordant"], 10)
        self.assertIsNotNone(exact["p_value"])

    def test_prediction_serialization_and_no_overwrite(self):
        run = common.evaluate_run(self.run_dir, self.data_dir, trust_joblib=True)
        path = self.root / "predictions.csv"
        basic.save_predictions(path, basic.prediction_rows(run))
        self.assertEqual(len(path.read_text().splitlines()), 3)
        with self.assertRaises(FileExistsError):
            basic.save_predictions(path, basic.prediction_rows(run))
        path_json = self.root / "result.json"
        common.write_json_new(path_json, {"threshold": run.threshold})
        self.assertEqual(json.loads(path_json.read_text())["threshold"], run.threshold)
        with self.assertRaises(FileExistsError):
            common.write_json_new(path_json, {})

    def test_exploratory_pattern_and_feature_mapping(self):
        run = common.evaluate_run(self.run_dir, self.data_dir, trust_joblib=True)
        self.assertEqual(interpretability.stratified_evaluation_indices(run, 2, 7).tolist(), [8, 9])
        result = interpretability.exploratory_patterns(run, self.root, maximum_per_class=1, seed=7)
        self.assertEqual(result["status"], "exploratory_class_association_only")
        self.assertEqual(len((self.root / "exploratory_fl_vs_nf_features.csv").read_text().splitlines()), 25)
        native = interpretability.native_importance(run.model, "logistic", run.data.columns)
        self.assertEqual(native[0], "absolute_standardized_logistic_coefficient")
        self.assertEqual(native[1].shape, (24,))

    def test_empty_audit_creates_incomplete_manifest(self):
        args = advanced.parse_args(["--data-dir", str(self.root / "missing"),
                                    "--run-root", str(self.root / "missing_runs"),
                                    "--output-root", str(self.root / "phase4_reports"),
                                    "--predictions-root", str(self.root / "phase4_data"),
                                    "--bootstrap-replicates", "20"])
        output = advanced.run(args)
        manifest = json.loads((output / "phase4_run_manifest.json").read_text())
        results = json.loads((output / "phase4_results.json").read_text())
        self.assertEqual(manifest["run_ids"], [])
        self.assertEqual(results["status"], "incomplete_no_verified_runs")
        self.assertTrue((output / "phase4_results.csv").is_file())
        self.assertTrue((output / "phase4_summary.md").is_file())
        self.assertNotIn("data/phase4", [p.as_posix() for p in self.root.rglob("*")])

    def test_full_synthetic_report_and_artifact_manifest(self):
        args = advanced.parse_args(["--data-dir", str(self.data_dir), "--run-dir", str(self.run_dir),
                                    "--run-root", str(self.root / "no_other_runs"),
                                    "--output-root", str(self.root / "phase4_report"),
                                    "--predictions-root", str(self.root / "predictions"),
                                    "--trust-model-files", "--bootstrap-replicates", "20"])
        output = advanced.run(args)
        saved = json.loads((output / "phase4_results.json").read_text())
        manifest = json.loads((output / "phase4_run_manifest.json").read_text())
        self.assertEqual(saved["status"], "verified_results")
        self.assertEqual(saved["results"][0]["test"]["n"], 2)
        self.assertEqual(saved["results"][0]["split"], "region_split")
        self.assertEqual(len(saved["comparisons"]), 0)
        self.assertEqual(manifest["run_ids"], [self.run_dir.name])
        self.assertIn("feature_index.csv", manifest["artifacts"][0]["phase1_input_sha256"])
        self.assertTrue(Path(saved["results"][0]["predictions_path"]).is_file())
        self.assertEqual(len((output / "phase4_results.csv").read_text().splitlines()), 2)
        second = advanced.run(args)
        self.assertNotEqual(output, second)
        self.assertTrue((second / "phase4_results.json").is_file())

    def test_interpretability_reports_optional_shap_blocker(self):
        args = interpretability.parse_args(["--data-dir", str(self.data_dir),
                                            "--run-dir", str(self.run_dir),
                                            "--run-root", str(self.root / "no_other_runs"),
                                            "--output-root", str(self.root / "interpretation"),
                                            "--trust-model-files", "--max-eval", "2",
                                            "--max-background", "2", "--max-pattern-per-class", "1"])
        with patch("interpretability.shap_importance", side_effect=RuntimeError("shap unavailable in fixture")):
            output = interpretability.run(args)
        saved = json.loads((output / "interpretability_results.json").read_text())
        self.assertEqual(saved["results"][0]["shap"]["status"], "blocked")
        self.assertIn("shap", saved["results"][0]["shap"]["reason"].lower())
        self.assertEqual(saved["results"][0]["patterns"]["status"],
                         "exploratory_class_association_only")
        self.assertTrue((output / self.run_dir.name / "model_native_importance.csv").is_file())


if __name__ == "__main__":
    unittest.main()
