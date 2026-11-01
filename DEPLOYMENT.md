# Helion web app

Helion has two independent services: a static HTML/CSS/JavaScript frontend and a FastAPI backend. Neither service needs Node.js. The backend loads saved, verified models; it never retrains them. Scores describe an M/X flare **peak** in `(issue + H, issue + H + 24 hours]`, not flare onset, and are not calibrated operational alerts.

## Layout

| Path | Purpose |
| --- | --- |
| `frontend/` | Static site, local development server, and Nginx image |
| `backend/` | FastAPI service, inference feature code, ONNX models, tests, and Python image |
| `vercel.json` | Routes one Vercel project to the two container services |

Each Vercel service uses its own directory as its Docker build context. The backend bundle includes its verified result files; its image has no runtime dependency on the training scripts or raw dataset.

## Start with Docker

From the repository root, start Docker Desktop, then run:

```powershell
docker compose up --build
```

Open <http://localhost:8080>. The API is at <http://localhost:8000/api/health> and its OpenAPI page is at <http://localhost:8000/api/docs>. `compose.yaml` builds distinct frontend and backend images. The frontend calls the backend URL set by `HELION_API_BASE`. The backend allows only the configured `HELION_CORS_ORIGINS`; local Compose supplies the two localhost origins. Both containers accept `PORT` at runtime.

The artifact command creates `backend/assets/bundle.zip` from the local phase 1–4 outputs. The bundle contains 33 verified ONNX models, normalization statistics, result metadata, and 64 fixed held-out examples. The individual converted models are in `backend/models/`. The bundle does not contain the original `.keras`, `.joblib`, or `.xgb.json` files, the raw SWAN-SF archive, or all 331,185 windows. Regenerate the ONNX exports and bundle after any model or preprocessing change:

```powershell
.\.sfd\Scripts\python.exe -m backend.convert_models_onnx
.\.sfd\Scripts\python.exe -m backend.package_artifacts
```

The ZIP is a Docker build input and must be included in a Git deployment. The original trained model files remain under `data/rebuild/runs` locally.

## Run without Docker

From the repository root, in one terminal:

```powershell
.\.sfd\Scripts\python.exe -m pip install fastapi==0.120.4 uvicorn==0.38.0
.\.sfd\Scripts\python.exe -c "import zipfile; zipfile.ZipFile('backend/assets/bundle.zip').extractall('backend/data')"
.\.sfd\Scripts\python.exe -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
.\.sfd\Scripts\python.exe frontend\dev_server.py
```

Open <http://127.0.0.1:8080>. The frontend server writes its API address into `/config.js` at request time; the backend alone can also be used and tested without it.

## API

Errors use `{"error":{"code":"...","message":"..."}}`. Requests with observation arrays are limited to 1.2 MB. The OpenAPI specification is available at `/api/openapi.json`.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Check that saved model runs are available |
| `GET /api/catalog` | List saved runs and the 24 magnetic input columns |
| `GET /api/overview` | Verified experiment and split summaries |
| `GET /api/example?model_id=...&kind=flare` | Get a fixed held-out window; `kind=quiet` also works |
| `GET /api/models/{model_id}/metrics` | Validation and test metrics for a run |
| `POST /api/predict` | Recreate features and score one 60-reading window |
| `POST /api/compare` | Score the same input with two to five comparable lead models |

`POST /api/predict` accepts `{"model_id":"phase3:24:region_split:baseline:xgboost","observations":[...]}`. `POST /api/compare` uses `model_ids` instead of `model_id`; all IDs must share a model family, split, and feature set. The 60 observations must be 12 minutes apart, have `Timestamp` and the magnetic columns, and contain at least half the 24 × 60 magnetic cells. Extra flare and X-ray fields are ignored. The threshold comes from saved validation metrics.

## Vercel preparation

`vercel.json` describes two container services and routes `/api/*` to FastAPI and all other paths to the frontend. The frontend defaults to same-origin `/api` there, so a browser does not need cross-origin access. Set the Vercel project framework to **Services** when deploying. Docker Compose is only for local development; Vercel builds the two Dockerfiles separately. The backend build must receive `bundle.zip` from the repository. Its runtime uses ONNX Runtime rather than TensorFlow, XGBoost, or scikit-learn. No deployment has been performed.

## Checks

```powershell
.\.sfd\Scripts\python.exe -m unittest backend.tests.test_inference backend.tests.test_api backend.tests.test_bundle backend.tests.test_isolated
node --check frontend/static/app.js
docker compose config --quiet
```
