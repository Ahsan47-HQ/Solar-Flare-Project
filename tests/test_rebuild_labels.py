"""Boundary and censoring checks for the rebuilt lead-time targets."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_labels import build


class ForecastLabelsTests(unittest.TestCase):
    def test_minimum_lead_and_missing_future_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "phase1").mkdir()
            series = root / "SWAN" / "partition1"
            series.mkdir(parents=True)
            times = pd.date_range("2020-01-01", periods=451, freq="12min")
            frame = pd.DataFrame({
                "Timestamp": times.astype(str),
                "MFLARE": np.zeros(len(times), dtype=np.int8),
                "XFLARE": np.zeros(len(times), dtype=np.int8),
            })
            # Relative to the first issue at +12h: peaks at exactly +6h
            # and at +40h. The former is outside the open 6h boundary.
            frame.loc[90, "MFLARE"] = 1
            frame.loc[260, "XFLARE"] = 1
            frame.to_csv(series / "123.csv", sep="\t", index=False)
            with (root / "phase1" / "feature_index.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["partition", "region_id", "end", "label"])
                writer.writeheader()
                writer.writerow({"partition": "partition1", "region_id": "123",
                                 "end": str(times[60]), "label": "FL"})
                writer.writerow({"partition": "partition1", "region_id": "123",
                                 "end": str(times[400]), "label": "NF"})
            labels = np.load(build(root))
            np.testing.assert_array_equal(labels[0], [1, 0, 0, 1, 0])
            np.testing.assert_array_equal(labels[1], [-1, -1, -1, -1, -1])
            report = json.loads((root / "phase2" / "horizon_targets_report.json").read_text())
            self.assertEqual(report["supplied_24h_label_comparison_on_valid_rows"]["mismatched"], 0)


if __name__ == "__main__":
    unittest.main()
