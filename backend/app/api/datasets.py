"""Dataset status endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas.models import DatasetStatusResponse
from backend.app.services.dataset_service import DatasetService

router = APIRouter(tags=["datasets"])
dataset_service = DatasetService()


@router.get("/datasets/status", response_model=DatasetStatusResponse, summary="Dataset and provenance status")
def datasets_status() -> DatasetStatusResponse:
    return dataset_service.status()
