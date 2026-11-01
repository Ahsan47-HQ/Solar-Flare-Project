"""The deployable backend runs without training files or the repository root."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]


class IsolatedBackendTests(unittest.TestCase):
    def test_api_runs_with_only_backend_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary) / "backend"
            isolated.mkdir()
            for name in ("__init__.py", "features.py", "server.py"):
                shutil.copy2(BACKEND / name, isolated / name)
            with zipfile.ZipFile(BACKEND / "assets" / "bundle.zip") as archive:
                archive.extractall(isolated / "data")
            script = (
                "from fastapi.testclient import TestClient; "
                "from backend.server import app; "
                "client=TestClient(app); "
                "assert client.get('/api/health').json()['models']==33; "
                "assert len(client.get('/api/overview').json()['results'])==33; "
                "m='phase3:24:region_split:baseline:xgboost'; "
                "sample=client.get('/api/example',params={'model_id':m}).json(); "
                "answer=client.post('/api/predict',json={'model_id':m,'observations':sample['observations']}); "
                "assert answer.status_code==200, answer.text"
            )
            environment = dict(os.environ, PYTHONPATH=temporary)
            subprocess.run([sys.executable, "-c", script], cwd=temporary, env=environment,
                           check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
