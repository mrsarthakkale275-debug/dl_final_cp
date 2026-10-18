"""Health and readiness endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.database import check_database, get_db
from backend.app.core.logging_config import get_logger
from backend.app.services.model_registry import registry

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=None, summary="API health and readiness")
def health(db: Session = Depends(get_db)) -> dict:
    db_ok = check_database()
    model_statuses = {"rgb": registry.rgb.status, "xray": registry.xray.status}
    api_status = "available"
    readiness = "ready" if db_ok else "degraded"
    overall = "healthy" if db_ok else "degraded"
    return {
        "api": api_status,
        "status": overall,
        "readiness": readiness,
        "database": "available" if db_ok else "unavailable",
        "models": model_statuses,
        "timestamp": datetime.now(timezone.utc),
    }


@router.get("/ready", summary="Readiness probe")
def ready(db: Session = Depends(get_db)) -> dict:
    return {"ready": check_database()}
