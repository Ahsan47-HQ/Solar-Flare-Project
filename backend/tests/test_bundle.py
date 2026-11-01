"""The deployable artifact archive reproduces a source-data prediction."""

import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from backend.server import example, predict


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "backend" / "assets" / "bundle.zip"
MODEL = "phase3:24:region_split:baseline:xgboost"


class BundleTests(unittest.TestCase):
    def test_inference_without_source_dataset(self):
        original = predict(MODEL, example(MODEL, "flare")["observations"])["probability"]
        with tempfile.TemporaryDirectory() as temporary:
            with zipfile.ZipFile(BUNDLE) as archive:
                archive.extractall(temporary)
            environment = dict(os.environ, HELION_DATA_DIR=temporary)
            code = (
                "from backend.server import catalog,example,predict,overview; "
                f"m={MODEL!r}; expected={original!r}; "
                "assert len(catalog())==33; "
                "assert all(item['inference_format']=='onnx' for item in catalog().values()); "
                "assert len(overview()['results'])==33; "
                "actual=predict(m,example(m,'flare')['observations'])['probability']; "
                "assert abs(actual-expected)<1e-6, (actual,expected)"
            )
            subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                           check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
