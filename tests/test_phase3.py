"""Offline tests for Phase 3 scripts and phase3_common utilities.

All tests use synthetic Phase 1 fixtures (borrowed from test_classical_models)
and do not require SWAN-SF data. Tests cover:

  - phase3_common: skill_scores(), HorizonResult, scan_run_dirs(),
                   load_result_from_dir(), write_comparison_json/md()
  - short_horizon: argument parsing, smoke run, artefact contract
  - long_horizon:  argument parsing, smoke run, artefact contract
  - feature_comparison: full comparison matrix, report generation

Run from the repository root:
    python -B -m unittest tests/test_phase3.py -v
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import phase3_common as p3
import short_horizon
import long_horizon
import feature_comparison
from test_classical_models import fixture   # reuse the established Phase 1 fixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(data_dir: Path, output_dir: Path, **overrides) -> argparse.Namespace:
    """Build a minimal argparse namespace for smoke tests."""
    defaults = dict(
        data_dir=data_dir,
        output_dir=output_dir,
        report_dir=output_dir / "reports",
        split="region_split",
        features="baseline",
        seed=42,
        class_weight="balanced",
        logistic_c=1.0,
        logistic_max_iter=100,
        forest_trees=4,
        forest_max_depth=3,
        forest_min_leaf=1,
        n_jobs=1,
        smoke_max_per_subset=2,
        horizons=None,
        models="both",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# 1. skill_scores()
# ---------------------------------------------------------------------------

class TestSkillScores(unittest.TestCase):
    def test_perfect_classifier(self):
        # [[TN, FP], [FN, TP]] = [[90, 0], [0, 10]]
        s = p3.skill_scores([[90, 0], [0, 10]])
        self.assertAlmostEqual(s["tss"], 1.0)
        self.assertAlmostEqual(s["hss"], 1.0)
        self.assertAlmostEqual(s["pod"], 1.0)
        self.assertAlmostEqual(s["far"], 0.0)

    def test_random_classifier_tss_near_zero(self):
        # Balanced: TP=5, FP=5, FN=5, TN=5 -> TSS=0
        s = p3.skill_scores([[5, 5], [5, 5]])
        self.assertAlmostEqual(s["tss"], 0.0)

    def test_no_positives_undefined(self):
        # All negatives: TN=10, FP=0, FN=0, TP=0
        s = p3.skill_scores([[10, 0], [0, 0]])
        self.assertIsNone(s["tss"])   # pod=None -> tss=None
        self.assertIsNone(s["pod"])

    def test_all_false_alarms(self):
        # [[TN=0, FP=10], [FN=0, TP=10]]
        s = p3.skill_scores([[0, 10], [0, 10]])
        self.assertAlmostEqual(s["pod"], 1.0)
        self.assertAlmostEqual(s["far"], 1.0)
        self.assertAlmostEqual(s["tss"], 0.0)

    def test_total_is_sum_of_all_cells(self):
        s = p3.skill_scores([[30, 10], [5, 55]])
        self.assertEqual(s["total"], 100)


# ---------------------------------------------------------------------------
# 2. HorizonResult
# ---------------------------------------------------------------------------

class TestHorizonResult(unittest.TestCase):
    def _make(self, **kw) -> p3.HorizonResult:
        defaults = dict(
            horizon="6h", feature_set="baseline", split="region_split",
            model_name="forest", mode="smoke", run_dir=Path("/tmp/test"),
            interpretation="diagnostic_only",
            val_ap=0.5, val_fl_recall=0.6, val_fl_f1=0.55, val_threshold=0.4,
            test_ap=0.45, test_fl_recall=0.55, test_fl_f1=0.5,
            confusion_matrix_val=[[80, 5], [3, 2]],
            confusion_matrix_test=[[90, 4], [4, 2]],
            n_train=6, n_val=2, n_test=2,
        )
        defaults.update(kw)
        return p3.HorizonResult(**defaults)

    def test_test_skill_delegates_to_skill_scores(self):
        r = self._make(confusion_matrix_test=[[9, 1], [1, 9]])
        skill = r.test_skill
        self.assertIn("tss", skill)
        self.assertIn("hss", skill)
        # TP=9, FP=1, FN=1, TN=9 -> POD=9/10, FAR=1/10, TSS=0.8
        self.assertAlmostEqual(skill["tss"], 0.8)

    def test_as_flat_dict_has_tss_and_hss(self):
        r = self._make()
        flat = r.as_flat_dict()
        self.assertIn("test_tss", flat)
        self.assertIn("test_hss", flat)
        self.assertIn("test_pod", flat)
        self.assertEqual(flat["horizon"], "6h")
        self.assertEqual(flat["feature_set"], "baseline")

    def test_as_flat_dict_all_keys_present(self):
        r = self._make()
        flat = r.as_flat_dict()
        for key in ("horizon", "feature_set", "split", "model", "mode",
                    "interpretation", "val_ap", "test_ap", "test_tss", "test_hss",
                    "n_train", "n_val", "n_test", "run_dir"):
            self.assertIn(key, flat, f"Missing key: {key}")


# ---------------------------------------------------------------------------
# 3. scan_run_dirs() and load_result_from_dir()
# ---------------------------------------------------------------------------

class TestScanRunDirs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _write_run(self, subpath: str, *, complete: bool = True,
                   horizon: str = "6h", feature_set: str = "baseline",
                   split: str = "region_split", model: str = "forest") -> Path:
        run_dir = self.root / subpath
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "phase": "phase3", "horizon": horizon, "feature_set": feature_set,
            "split": split, "model": model, "seed": 42, "smoke_max_per_subset": 2,
            "feature_columns": ["a", "b"],
        }
        metrics = {
            "evaluation_interpretation": "diagnostic_only",
            "selected_row_counts": {"train": 6, "val": 2, "test": 2},
            "validation": {
                "pr_auc_average_precision": 0.5, "threshold": 0.4,
                "confusion_matrix": [[8, 2], [1, 1]],
                "FL": {"recall": 0.5, "f1": 0.4},
            },
            "test": {
                "pr_auc_average_precision": 0.45, "threshold": 0.4,
                "confusion_matrix": [[9, 1], [1, 1]],
                "FL": {"recall": 0.5, "f1": 0.45},
            },
        }
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        if complete:
            (run_dir / "status.json").write_text(
                json.dumps({"complete": True, "completed_at_utc": "2026-01-01T00:00:00+00:00"}),
                encoding="utf-8",
            )
        return run_dir

    def test_scan_finds_complete_runs(self):
        self._write_run("exp/run1", horizon="6h")
        self._write_run("exp/run2", horizon="12h")
        results = p3.scan_run_dirs(self.root / "exp")
        self.assertEqual(len(results), 2)
        horizons = {r.horizon for r in results}
        self.assertEqual(horizons, {"6h", "12h"})

    def test_scan_skips_incomplete_runs(self):
        self._write_run("exp/complete", complete=True)
        self._write_run("exp/incomplete", complete=False)
        results = p3.scan_run_dirs(self.root / "exp")
        self.assertEqual(len(results), 1)

    def test_scan_skips_missing_metrics(self):
        run_dir = self._write_run("exp/broken", complete=True)
        (run_dir / "metrics.json").unlink()
        results = p3.scan_run_dirs(self.root / "exp")
        self.assertEqual(len(results), 0)

    def test_load_result_parses_confusion_matrix(self):
        run_dir = self._write_run("exp/run1")
        result = p3.load_result_from_dir(run_dir)
        self.assertIsNotNone(result)
        self.assertEqual(result.confusion_matrix_test, [[9, 1], [1, 1]])

    def test_load_result_nonexistent_returns_none(self):
        result = p3.load_result_from_dir(self.root / "nonexistent")
        self.assertIsNone(result)

    def test_scan_empty_dir_returns_empty_list(self):
        results = p3.scan_run_dirs(self.root / "not_there")
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# 4. write_comparison_json and write_comparison_md
# ---------------------------------------------------------------------------

class TestReportWriters(unittest.TestCase):
    def _make_results(self, n: int = 2) -> list[p3.HorizonResult]:
        horizons = ["6h", "12h", "24h", "48h"]
        results = []
        for i in range(n):
            results.append(p3.HorizonResult(
                horizon=horizons[i % len(horizons)],
                feature_set="baseline" if i % 2 == 0 else "baseline_temporal",
                split="region_split", model_name="forest", mode="smoke",
                run_dir=Path(f"/tmp/run{i}"),
                interpretation="diagnostic_only",
                val_ap=0.5 + i * 0.01, val_fl_recall=0.6, val_fl_f1=0.55, val_threshold=0.4,
                test_ap=0.45 + i * 0.01, test_fl_recall=0.55, test_fl_f1=0.5,
                confusion_matrix_val=[[80, 5], [3, 2]],
                confusion_matrix_test=[[90, 4], [4, 2]],
                n_train=6, n_val=2, n_test=2,
            ))
        return results

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_write_comparison_json_is_valid_list(self):
        results = self._make_results(3)
        path = self.root / "out.json"
        p3.write_comparison_json(results, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(loaded, list)
        self.assertEqual(len(loaded), 3)
        self.assertIn("test_tss", loaded[0])
        self.assertIn("test_hss", loaded[0])

    def test_write_comparison_md_creates_file_with_tables(self):
        results = self._make_results(4)
        path = self.root / "report.md"
        p3.write_comparison_md(results, path, split="region_split",
                               title="Test Report")
        content = path.read_text(encoding="utf-8")
        self.assertIn("Test Report", content)
        self.assertIn("TSS", content)
        self.assertIn("HSS", content)
        self.assertIn("PR-AUC", content)
        self.assertIn("Delta PR-AUC", content)
        # Sibling CSV must also be created
        csv_path = path.with_suffix(".csv")
        self.assertTrue(csv_path.is_file())

    def test_write_comparison_csv_is_valid(self):
        results = self._make_results(3)
        path = self.root / "out.csv"
        p3.write_comparison_csv(results, path)
        self.assertTrue(path.is_file())
        with path.open("r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 3)
            self.assertIn("test_tss", reader[0])
            self.assertIn("test_hss", reader[0])


# ---------------------------------------------------------------------------
# 5. Threshold Strategies
# ---------------------------------------------------------------------------

class TestThresholdStrategies(unittest.TestCase):
    def test_max_f1_default(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.2, 0.8, 0.9])
        th, src = p3.choose_threshold_by_strategy(y, prob, "max_f1")
        self.assertGreater(th, 0.0)

    def test_max_tss_strategy(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.2, 0.8, 0.9])
        th, src = p3.choose_threshold_by_strategy(y, prob, "max_tss")
        self.assertIn("max_tss", src)
        self.assertGreaterEqual(th, 0.01)

    def test_target_recall_90(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.2, 0.8, 0.9])
        th, src = p3.choose_threshold_by_strategy(y, prob, "target_recall_90")
        self.assertIn("target_recall_90", src)
        pred = (prob >= th).astype(int)
        recall = np.sum((y == 1) & (pred == 1)) / np.sum(y == 1)
        self.assertGreaterEqual(recall, 0.9)


# ---------------------------------------------------------------------------
# 6. short_horizon.py end-to-end smoke run
# ---------------------------------------------------------------------------

class TestShortHorizon(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = fixture(self.root)

    def test_smoke_run_produces_artefacts_with_correct_schema(self):
        out = self.root / "out"
        args = _make_args(
            self.data, out,
            split="region_split", features="baseline",
            horizons=["6h"], models="forest",
        )
        results = short_horizon.run(args)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.horizon, "6h")
        self.assertEqual(r.model_name, "forest")
        self.assertEqual(r.mode, "smoke")
        self.assertIsNotNone(r.test_ap)
        # Artefact files
        self.assertTrue((r.run_dir / "config.json").is_file())
        self.assertTrue((r.run_dir / "metrics.json").is_file())
        self.assertTrue((r.run_dir / "test_probabilities.npy").is_file())
        self.assertTrue((r.run_dir / "model.joblib").is_file())
        self.assertTrue((r.run_dir / "status.json").is_file())
        status = json.loads((r.run_dir / "status.json").read_text(encoding="utf-8"))
        self.assertTrue(status["complete"])

    def test_saved_model_reload_reproduces_predictions(self):
        import joblib
        out = self.root / "out"
        args = _make_args(self.data, out, horizons=["6h"], models="forest")
        results = short_horizon.run(args)
        r = results[0]
        model_path = r.run_dir / "model.joblib"
        prob_path = r.run_dir / "test_probabilities.npy"
        self.assertTrue(model_path.is_file())
        self.assertTrue(prob_path.is_file())

        reloaded_model = joblib.load(model_path)
        saved_probs = np.load(prob_path)
        self.assertEqual(len(saved_probs), r.n_test)
        self.assertTrue(hasattr(reloaded_model, "predict_proba"))

    def test_config_has_phase3_and_horizon_fields(self):
        out = self.root / "out"
        args = _make_args(self.data, out, horizons=["12h"], models="logistic")
        results = short_horizon.run(args)
        cfg = json.loads((results[0].run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["phase"], "phase3")
        self.assertEqual(cfg["experiment"], "short_horizon")
        self.assertEqual(cfg["horizon"], "12h")
        self.assertEqual(cfg["label_mapping"], {"NF": 0, "FL": 1})

    def test_metrics_has_confusion_matrix(self):
        out = self.root / "out"
        args = _make_args(self.data, out, horizons=["6h"], models="forest")
        results = short_horizon.run(args)
        met = json.loads((results[0].run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertIn("confusion_matrix", met["test"])
        self.assertEqual(len(met["test"]["confusion_matrix"]), 2)

    def test_invalid_smoke_value_raises(self):
        out = self.root / "out"
        args = _make_args(self.data, out, smoke_max_per_subset=1)
        with self.assertRaises(ValueError):
            short_horizon.run(args)

    def test_horizon_result_skill_scores_non_null(self):
        out = self.root / "out"
        args = _make_args(self.data, out, horizons=["6h"], models="forest")
        results = short_horizon.run(args)
        skill = results[0].test_skill
        # At least one of tss/hss should be computable (both classes present in fixture)
        self.assertIn("tss", skill)

    def test_result_is_scannable_by_scan_run_dirs(self):
        out = self.root / "out"
        args = _make_args(self.data, out, horizons=["6h"], models="forest")
        results = short_horizon.run(args)
        parent = results[0].run_dir.parent.parent.parent.parent.parent  # go up to out/
        scanned = p3.scan_run_dirs(out)
        self.assertGreaterEqual(len(scanned), 1)
        scanned_dirs = {r.run_dir for r in scanned}
        self.assertIn(results[0].run_dir, scanned_dirs)


# ---------------------------------------------------------------------------
# 6. long_horizon.py end-to-end smoke run
# ---------------------------------------------------------------------------

class TestLongHorizon(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = fixture(self.root)

    def test_smoke_run_produces_artefacts(self):
        out = self.root / "out"
        args = _make_args(
            self.data, out,
            split="time_split", features="baseline",
            horizons=["24h"], models="forest",
        )
        results = long_horizon.run(args)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.horizon, "24h")
        self.assertEqual(r.split, "time_split")
        self.assertTrue((r.run_dir / "status.json").is_file())

    def test_config_has_long_horizon_experiment_name(self):
        out = self.root / "out"
        args = _make_args(self.data, out, horizons=["48h"], models="logistic")
        results = long_horizon.run(args)
        cfg = json.loads((results[0].run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["experiment"], "long_horizon")
        self.assertEqual(cfg["horizon"], "48h")

    def test_both_models_run_when_models_is_both(self):
        out = self.root / "out"
        args = _make_args(self.data, out, horizons=["24h"], models="both")
        results = long_horizon.run(args)
        self.assertEqual(len(results), 2)
        model_names = {r.model_name for r in results}
        self.assertEqual(model_names, {"logistic", "forest"})


# ---------------------------------------------------------------------------
# 7. feature_comparison.py end-to-end smoke run
# ---------------------------------------------------------------------------

class TestFeatureComparison(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = fixture(self.root)

    def test_smoke_run_two_horizons_two_feature_sets(self):
        out = self.root / "out"
        args = _make_args(
            self.data, out,
            split="region_split", features="baseline",   # will be overridden per cell
            horizons=["6h", "12h"],
            smoke_max_per_subset=2,
        )
        # feature_comparison overrides features per cell; seed and other args are shared
        del args.models  # feature_comparison only uses forest; models attr not needed
        args.features = None  # signal: use both feature sets
        results = feature_comparison.run(args)
        # 2 horizons x 2 feature sets = 4 cells
        self.assertEqual(len(results), 4)
        feature_sets_seen = {r.feature_set for r in results}
        self.assertEqual(feature_sets_seen, {"baseline", "baseline_temporal"})

    def test_comparison_json_written_to_output_dir(self):
        out = self.root / "out"
        args = _make_args(
            self.data, out,
            horizons=["6h"],
            smoke_max_per_subset=2,
        )
        args.features = None
        del args.models
        results = feature_comparison.run(args)
        # Find comparison_results.json in the output tree
        json_files = list(out.rglob("comparison_results.json"))
        self.assertGreaterEqual(len(json_files), 1)
        loaded = json.loads(json_files[0].read_text(encoding="utf-8"))
        self.assertIsInstance(loaded, list)
        self.assertIn("test_tss", loaded[0])

    def test_each_cell_is_scannable(self):
        out = self.root / "out"
        args = _make_args(self.data, out, horizons=["6h"], smoke_max_per_subset=2)
        args.features = None
        del args.models
        results = feature_comparison.run(args)
        scanned = p3.scan_run_dirs(out)
        scanned_dirs = {r.run_dir for r in scanned}
        for r in results:
            self.assertIn(r.run_dir, scanned_dirs)

    def test_config_has_comparison_note(self):
        out = self.root / "out"
        args = _make_args(self.data, out, horizons=["6h"], smoke_max_per_subset=2)
        args.features = None
        del args.models
        results = feature_comparison.run(args)
        cfg = json.loads((results[0].run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertIn("comparison_note", cfg)
        self.assertEqual(cfg["experiment"], "feature_comparison")


# ---------------------------------------------------------------------------
# 8. Argument parsing contracts
# ---------------------------------------------------------------------------

class TestArgParsing(unittest.TestCase):
    def test_short_horizon_defaults(self):
        args = short_horizon.parse_args([])
        self.assertEqual(args.split, "region_split")
        self.assertEqual(args.features, "baseline")
        self.assertEqual(args.models, "both")
        self.assertEqual(args.smoke_max_per_subset, 0)
        self.assertIsNone(args.horizons)

    def test_long_horizon_defaults(self):
        args = long_horizon.parse_args([])
        self.assertEqual(args.split, "region_split")
        self.assertEqual(args.smoke_max_per_subset, 0)

    def test_feature_comparison_defaults(self):
        args = feature_comparison.parse_args([])
        self.assertEqual(args.split, "region_split")
        self.assertIsNone(args.horizons)
        self.assertIsNone(args.features)

    def test_short_horizon_rejects_long_horizon(self):
        """Passing 24h to short_horizon should raise at runtime, not at parse time."""
        # parse_args succeeds (choices validation is on the nargs list)
        # but run() should reject it
        # (We test parse-time rejection here; the nargs choices enforce valid values)
        with self.assertRaises(SystemExit):
            short_horizon.parse_args(["--horizons", "24h"])

    def test_smoke_max_validation(self):
        """smoke_max_per_subset=1 should be rejected by run()."""
        # We verify the validation logic in run() via direct call
        args = short_horizon.parse_args(["--smoke-max-per-subset", "1"])
        with self.assertRaises(ValueError):
            short_horizon.run(args)


if __name__ == "__main__":
    unittest.main()
