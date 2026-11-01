"""HTTP contract checks for the separated FastAPI service."""

import unittest

from fastapi.testclient import TestClient

from backend.server import app, example


MODEL_0 = "phase2:0:region_split:baseline:xgboost"
MODEL_24 = "phase3:24:region_split:baseline:xgboost"


class APIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.observations = example(MODEL_24, "flare")["observations"]

    def test_catalog_metrics_and_cors(self):
        catalog = self.client.get("/api/catalog", headers={"Origin": "http://localhost:8080"})
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(catalog.headers["access-control-allow-origin"], "http://localhost:8080")
        self.assertEqual(len(catalog.json()["models"]), 33)
        metrics = self.client.get(f"/api/models/{MODEL_24}/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("average_precision", metrics.json()["test"])
        self.assertNotIn("_run_dir", str(metrics.json()))

    def test_prediction_and_cross_lead_comparison(self):
        body = {"model_id": MODEL_24, "observations": self.observations}
        single = self.client.post("/api/predict", json=body)
        self.assertEqual(single.status_code, 200)
        comparison = self.client.post("/api/compare", json={"model_ids": [MODEL_0, MODEL_24],
                                                              "observations": self.observations})
        self.assertEqual(comparison.status_code, 200)
        self.assertEqual(len(comparison.json()["predictions"]), 2)
        self.assertAlmostEqual(comparison.json()["predictions"][1]["probability"],
                               single.json()["probability"], places=7)

    def test_errors_keep_a_stable_shape(self):
        invalid = self.client.post("/api/predict", json={"model_id": MODEL_24,
                                                          "observations": self.observations[:-1]})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["error"]["code"], "window_length")
        duplicate = self.client.post("/api/compare", json={"model_ids": [MODEL_24, MODEL_24],
                                                            "observations": self.observations})
        self.assertEqual(duplicate.status_code, 422)
        self.assertEqual(duplicate.json()["error"]["code"], "comparison_models")
        oversized = self.client.post("/api/predict", content="x" * 1_200_001,
                                     headers={"Content-Type": "application/json"})
        self.assertEqual(oversized.status_code, 413)


if __name__ == "__main__":
    unittest.main()
