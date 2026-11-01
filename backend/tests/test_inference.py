"""High-value checks for the local inference path and its input contract."""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.server import APIError, example, predict
from backend.features import compute_region_features, transform_features
from data_cleaning import transform_features as training_transform_features
from temporal_features import compute_region_features as training_compute_region_features


MODEL_ID = "phase3:24:region_split:baseline:xgboost"
SOURCE = Path(__file__).resolve().parents[2] / "data"
SOURCE_RUN = SOURCE / "rebuild" / "runs" / "phase3" / "lead_24h" / "region_split" / "baseline" / "xgboost"


class WebInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = example(MODEL_ID, "flare")

    def test_recomputed_prediction_matches_saved_held_out_probability(self):
        rows = np.load(SOURCE_RUN / "test_rows.npy", mmap_mode="r")
        probabilities = np.load(SOURCE_RUN / "test_probability.npy", mmap_mode="r")
        targets = np.load(SOURCE / "phase2" / "horizon_targets.npy", mmap_mode="r")
        candidates = rows[targets[rows, 3] == 1]
        chosen = int(candidates[17])
        saved = float(probabilities[int(np.searchsorted(rows, chosen))])
        result = predict(MODEL_ID, self.sample["observations"])
        self.assertAlmostEqual(result["probability"], saved, places=6)
        self.assertEqual(result["lead_hours"], 24)
        self.assertEqual(result["feature_count"], 24)

    def test_inference_features_match_training_definitions(self):
        frame = pd.DataFrame(self.sample["observations"])
        window = [{"start": frame.iloc[0]["Timestamp"],
                   "end": frame.iloc[-1]["Timestamp"], "path": "test-window"}]
        actual_base, actual_temporal = compute_region_features(frame, window)
        source_base, source_temporal = training_compute_region_features(frame, window)
        np.testing.assert_allclose(actual_base, source_base, equal_nan=True)
        np.testing.assert_allclose(actual_temporal, source_temporal, equal_nan=True)
        stats = json.loads((SOURCE / "phase1" / "normalization_region_split_baseline.json").read_text())
        np.testing.assert_allclose(transform_features(actual_base.to_numpy(), stats),
                                   training_transform_features(source_base.to_numpy(), stats), equal_nan=True)

    def test_flare_annotation_cannot_change_features(self):
        clean = self.sample["observations"]
        with_annotation = [dict(row, MFLARE=999, XR_MAX=999) for row in clean]
        self.assertEqual(predict(MODEL_ID, clean)["probability"],
                         predict(MODEL_ID, with_annotation)["probability"])

    def test_rejects_wrong_window_length(self):
        with self.assertRaises(APIError) as raised:
            predict(MODEL_ID, self.sample["observations"][:-1])
        self.assertEqual(raised.exception.status, 422)


if __name__ == "__main__":
    unittest.main()
