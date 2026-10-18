"""FastAPI application entry point.

AdaptiveInspect AI — Adaptive Multimodal Deep Learning for
Industrial Anomaly Detection and Cost-Aware Quality Control.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import settings
from backend.app.core.database import init_db
from backend.app.core.exceptions import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from backend.app.core.logging_config import configure_logging, get_logger
from backend.app.api.dashboard import router as dashboard_router
from backend.app.api.datasets import router as datasets_router
from backend.app.api.health import router as health_router
from backend.app.api.inspections import router as inspections_router
from backend.app.api.models import router as models_router
from backend.app.api.reports import router as reports_router

configure_logging("DEBUG" if settings.debug else "INFO")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AdaptiveInspect AI backend.")
    init_db()
    logger.info("Database initialized (create_all, no data deletion).")

    # --- Register legacy RGB pixel-mahalanobis engine (if available) ---
    try:
        from backend.app.services.model_registry import registry as model_registry
        from backend.app.ml.rgb_engine import load_rgb_engine

        rgb_engine = load_rgb_engine()
        if rgb_engine is not None:
            model_registry.register_rgb_engine(rgb_engine)
            logger.info("RGB anomaly engine registered as ready.")
        else:
            logger.info("RGB anomaly engine unavailable; model remains untrained or unloaded.")

        try:
            from backend.app.ml.xray_engine import load_xray_engine

            xray_engine = load_xray_engine()
            if xray_engine is not None:
                model_registry.register_xray_engine(xray_engine)
                logger.info("X-ray engine registered as ready.")
            else:
                logger.info("X-ray engine unavailable; requests use review-required workflow.")
        except Exception as exc:
            logger.warning("X-ray engine registration failed: %s", exc)
    except Exception as exc:
        logger.warning("RGB engine registration failed: %s", exc)

    # --- Warm up MultimodalInspector (PyTorch CNN) ---
    try:
        from backend.app.ml.multimodal_cnn import create_inspector
        inspector = create_inspector(
            rgb_weights_path=str(settings.rgb_models_dir / "rgb_cnn_weights.pt")
            if (settings.rgb_models_dir / "rgb_cnn_weights.pt").exists() else None,
            xray_weights_path=str(settings.xray_models_dir / "xray_cnn_weights.pt")
            if (settings.xray_models_dir / "xray_cnn_weights.pt").exists() else None,
            patch_core_path=str(settings.rgb_models_dir / "patchcore_bank.pt")
            if (settings.rgb_models_dir / "patchcore_bank.pt").exists() else None,
        )
        # Register with the adaptive inspection service
        from backend.app.api.inspections import adaptive_service
        adaptive_service.inspector = inspector
        logger.info(
            "MultimodalInspector initialized (mock=%s, device=%s).",
            inspector.mock_mode, inspector.device,
        )
    except Exception as exc:
        logger.warning(
            "MultimodalInspector warmup failed (will lazy-load on first request): %s", exc
        )

    yield
    logger.info("Shutting down AdaptiveInspect AI backend.")


app = FastAPI(
    title="AdaptiveInspect AI Backend",
    description=(
        "Adaptive Multimodal Deep Learning for Industrial Anomaly Detection "
        "and Cost-Aware Quality Control. Two-stage inspection pipeline with "
        "RGB CNN (WideResNet50), X-ray CNN (ResNet34), entropy-based adaptive "
        "gating, cross-modal fusion, dynamic cost estimation, and real-time "
        "WebSocket streaming."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
if settings.cors_allow_all:
    allow_origins = ["*"]
else:
    allow_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=not settings.cors_allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Routers — legacy under /api, v1 also under /api
app.include_router(health_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(inspections_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(reports_router, prefix="/api")

from fastapi.staticfiles import StaticFiles
if settings.image_storage_dir.exists():
    app.mount("/api/storage/inspection_images", StaticFiles(directory=str(settings.image_storage_dir)), name="inspection_images")



@app.get("/", summary="Root")
def root() -> dict:
    return {
        "name": "AdaptiveInspect AI Backend",
        "version": "2.0.0",
        "description": "Adaptive Multimodal Deep Learning for Industrial Anomaly Detection",
        "docs": "/docs",
        "health": "/api/health",
        "v1_upload": "/api/v1/inspect/upload",
        "v1_stream": "/api/v1/inspect/stream (WebSocket)",
    }
