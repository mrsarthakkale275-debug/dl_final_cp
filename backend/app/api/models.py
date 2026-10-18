"""Model status endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas.models import ModelStatusDetail, ModelStatusResponse
from backend.app.services.model_registry import registry

router = APIRouter(tags=["models"])


def _to_detail(handle) -> ModelStatusDetail:
    return ModelStatusDetail(
        name=handle.name,
        kind=handle.kind,
        available=len(handle.weights) > 0,
        ready=handle.ready,
        status=handle.status,
        model_id=handle.model_id,
        version=handle.version,
        weights=handle.weights,
        message=handle.message,
    )


@router.get("/model-status", response_model=ModelStatusResponse, summary="RGB and X-ray model availability")
def model_status() -> ModelStatusResponse:
    rgb_detail = _to_detail(registry.rgb)
    xray_detail = _to_detail(registry.xray)
    return ModelStatusResponse(
        rgb=rgb_detail,
        xray=xray_detail,
        overall_ready=registry.rgb.ready and registry.xray.ready,
    )
