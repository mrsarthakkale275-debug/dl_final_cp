# AdaptiveInspect AI — Multimodal PCB Defect Inspection Backend

This repository contains the Python backend for **AdaptiveInspect AI**. The backend
provides REST APIs for RGB, X-ray (review-required), and adaptive inspection
workflows; inspection history; dashboard summaries; reports; dataset status; and
model status.

## Architecture

```text
backend/
  app/
    main.py                     FastAPI application and startup hooks
    core/
      config.py                 Environment-variable configuration
      database.py               SQLAlchemy engine, sessions, initialization
      logging_config.py         Centralized logging
      exceptions.py             Common exception definitions/handlers
    api/                        FastAPI routers, one module per concern
    schemas/                    Pydantic request/response models
    models/                     SQLAlchemy ORM models
    repositories/                Database access layer
    services/                   Business logic and inference orchestration
    ml/                         RGB anomaly model engine and training code
    utils/                      Upload, path, and security helpers
  storage/
    inspection_images/          Controlled image storage
    reports/                    Generated reports
  requirements.txt
  tests/
```

The request/data flow is:

```text
Upload -> image_service validation -> service layer -> model registry -> inference
       -> repository persistence -> adaptive decision -> report/dashboard
```

RGB and X-ray inference are deliberately separate. The adapter classes expose a
replaceable `predict_rgb()` / `predict_xray()` interface.

## Datasets

### VisA RGB dataset

- Location: `data/raw/visa_parquet/data/*.parquet`
- Images and masks are embedded in Parquet (`image.bytes`, `mask.bytes`, `label`).
- Training split: 3,614 normal images.
- Test split: 802 images (402 normal, 400 defective), with masks.
- No RGB model is trained as a classic classifier. The implemented RGB model is
  an anomaly detector trained on normal images only.

### X-ray dataset

- Source: `data/raw/xray/hf_data/` (from `dezoe/pcb-defect-multi-modal-dataset`)
- Prepared images: `data/processed/xray/images/` (602 PNG files)
- Manifest: `data/processed/xray/xray_manifest.csv`
- X-ray record count: 602
- Prepared image count: 602
- Label distribution:
  - `void`: 601
  - `solder_ball`: 1
- Rights distribution:
  - `source_license_conditions_apply`: 600
  - `permission_not_established`: 2

The two `permission_not_established` records are reported by dataset status and
must remain excluded from model training and distribution until verified.

**Important limitations**

- RGB and X-ray images are not paired observations of the same physical PCB.
- AdaptiveInspect never performs true multimodal feature fusion.
- The available X-ray labels do not support a reliable defect classifier.
- The X-ray endpoint uses a review-required workflow instead of fabricated predictions.

## Environment

From the project root:

```bash
cd /mnt/c/Users/Lenovo/Downloads/dl_final
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
```

Key environment variables:

- `DATABASE_URL` — SQLAlchemy database URL. Default: SQLite under `backend/`.
- `CORS_ORIGINS` — comma-separated allowed origins.
- `CORS_ALLOW_ALL` — set `true` to allow broad development CORS.
- `MAX_UPLOAD_BYTES` — upload limit in bytes. Default: 10 MiB.
- `DATA_DIR`, `MODELS_DIR`, `RGB_MODELS_DIR`, `XRAY_MODELS_DIR`
- `IMAGE_STORAGE_DIR`, `REPORT_STORAGE_DIR`
- `APP_DEBUG` — enables debug logging.
- `RGB_CONFIDENCE_THRESHOLD`, `RGB_UNCERTAINTY_THRESHOLD`, `ANOMALY_THRESHOLD`

## Starting the backend

From the project root:

```bash
source .venv/bin/activate
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

OpenAPI schema:

```text
http://127.0.0.1:8000/openapi.json
```

The API starts before RGB or X-ray model weights are available. Health and
model-status endpoints distinguish API availability from model readiness.

## RGB model pipeline

- Method: per-pixel multivariate Gaussian / Mahalanobis anomaly detection.
- Training: 3,614 normal VisA PCB training images.
- Evaluation: 802 held-out VisA test images and masks.
- Output: image-level anomaly score, normal/defective classification against a
  normal-only threshold, and a coarse thresholded pixel-distance localization map.
- The model does not provide calibrated confidence.

Prepare compact thumbnails (original Parquet data is not modified):

```bash
python prepare_rgb_dataset.py
```

Train the RGB model:

```bash
python backend/app/ml/rgb_train.py --output-dir models/rgb
```

This writes:

- `models/rgb/anomaly_model.npz`
- `models/rgb/anomaly_model_metadata.json`
- `models/rgb/anomaly_evaluation.json`

Measured evaluation is stored in `anomaly_evaluation.json` and is a real result
from the VisA test split:

- AUROC: 0.6066
- Image-level accuracy: 0.5075
- Precision: 0.7273
- Recall: 0.0200
- F1: 0.0389
- Mean localization IoU: 0.0277
- Global localization IoU: 0.0201

The RGB model is loaded automatically at startup if `models/rgb/anomaly_model.npz`
exists. If loading fails, the backend remains available and reports the model as
unavailable.

## X-ray pipeline

No X-ray model is trained. The X-ray dataset has 601 `void` labels and 1
`solder_ball` label, with only 2 rights-restricted records, and no reliable
negative/normal X-ray examples.

Therefore:

- `/api/inspect/xray` validates and stores the uploaded image.
- The model status remains `untrained`/`model_unavailable`.
- No RGB model is used for X-ray images.
- The inspection record is still saved.
- Adaptive decisions route X-ray-dependent outcomes to `human_review_required`.

## Model registry and inference flow

- `ModelRegistry` scans `models/rgb/` and `models/xray/` for supported weight
  extensions, including `.npz`.
- Engines are registered at startup only if they load successfully.
- RGB engine:
  - `model_id`: `rgb-pixel-mahalanobis-anomaly`
  - implements `predict_rgb(image)`
- X-ray engine:
  - no trained engine is registered
- `/api/model-status` reports separate RGB and X-ray readiness.

## API endpoints

General:

- `GET /api/health`
- `GET /api/model-status`
- `GET /api/datasets/status`

Inspection:

- `POST /api/inspect/rgb`
- `POST /api/inspect/xray`
- `POST /api/inspect/adaptive`

History:

- `GET /api/inspections`
- `GET /api/inspections/{inspection_id}`
- `DELETE /api/inspections/{inspection_id}`

Dashboard:

- `GET /api/dashboard/summary`

Reports:

- `GET /api/inspections/{inspection_id}/report`
- `GET /api/inspections/{inspection_id}/report?format=markdown`

Adaptive decisions include explicit recommendations, reasons, explanations,
model availability flags, and the thresholds used.

## Reports

Every completed inspection can generate a report from stored results. Reports
include inspection ID, timestamp, modality, model ID/version, anomaly score or
prediction when available, adaptive recommendation, and limitations.

- Default format: JSON
- Optional format: Markdown via `?format=markdown`
- Report files are written to `backend/storage/reports/`
- Values are never invented when a model is unavailable.

## Running tests

From the project root:

```bash
source .venv/bin/activate
python -m pytest backend/tests -q
```

The focused suite covers health/status endpoints, dataset status, RGB integration
with the trained model, X-ray unavailable behavior, invalid/corrupt/oversized
uploads, adaptive policy edge cases, inspection history and deletion, dashboard
totals, reports, and startup resilience with missing weights. Tests use temporary
databases and temporary storage paths.

## Known limitations and future improvements

- The RGB anomaly baseline is simple and has low recall on small/localized
  defects.
- The RGB model does not output calibrated confidence.
- The X-ray dataset does not yet support a defensible trained model.
- X-ray requests currently require human review.
- Localization is a thresholded distance map rather than a trained segmentation model.
- RGB and X-ray results are combined only at the decision-policy level.
- SQLite is used for initial development; larger deployments may need PostgreSQL.
- Pydantic/SQLAlchemy/Evaluation metrics should be recalculated if training code changes.

## Supabase PostgreSQL configuration

The backend now uses SQLAlchemy with Supabase PostgreSQL for normal runtime
traffic. Do not commit database credentials.

### Environment variables

Create or edit `backend/.env` locally:

```env
# Required for API traffic. Use the Supabase transaction pooler.
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:6543/postgres?sslmode=require

# Required for schema initialization and migrations.
DIRECT_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/postgres?sslmode=require

# Optional local fallback only:
# DATABASE_URL=sqlite:///adaptive_inspect.db
```

Use percent-encoding for special characters in passwords, for example
`@` becomes `%40`.

The application also accepts `DATABASE_URL` and `DIRECT_URL` from shell
environment variables. If these are not set, the backend falls back to local
SQLite for development and tests.

### Setup steps

1. Create a Supabase project.
2. Obtain the Postgres connection information from the Supabase dashboard.
3. Copy `backend/.env.example` to `backend/.env`.
4. Replace the placeholder credentials with percent-encoded values.
5. Install dependencies:

```bash
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
```

6. Start the backend:

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Startup calls `init_db()` and uses `Base.metadata.create_all()` on the direct
connection. Existing tables are not dropped.

7. Run focused tests:

```bash
python -m pytest backend/tests -q
```

### Safe SQLite-to-Supabase migration

A separate helper is provided:

```bash
source .venv/bin/activate
python backend/scripts/migrate_sqlite_to_postgres.py --dry-run
python backend/scripts/migrate_sqlite_to_postgres.py
```

The script reads the SQLite database in read-only fashion and inserts rows using
`ON CONFLICT (id) DO NOTHING`. It does not migrate image bytes; image files
remain in controlled storage and only their references are migrated. Never run
this against a database you do not own, and always validate counts first with
`--dry-run`.

## Frontend

The frontend is an empty-folder Vite React application under `frontend/`.

Install and run:

```bash
cd frontend
npm install
npm run dev
```

Production build:

```bash
cd frontend
npm run build
```

Set the API URL when needed:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000/api
```

The landing page includes a dynamically loaded React Three Fiber canvas with a 3D PCB/circuit scene. All dashboard, history, reports, model status, and inspection results use the FastAPI backend.
