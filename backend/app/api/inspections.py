"""Inspection endpoints — v1 API with upload, stream, and legacy compatibility.

Endpoints:
  POST /api/v1/inspect/upload   — Upload RGB (+ optional X-ray) image for full pipeline
  WS   /api/v1/inspect/stream   — WebSocket real-time webcam inspection
  POST /api/inspect/rgb          — Legacy RGB-only inspection
  POST /api/inspect/xray         — Legacy X-ray-only inspection
  POST /api/inspect/adaptive     — Legacy adaptive inspection
  GET  /api/inspections          — List inspection history
  GET  /api/inspections/{id}     — Get inspection by ID
  DELETE /api/inspections/{id}   — Delete inspection
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.core.logging_config import get_logger
from backend.app.repositories.inspection_repository import InspectionRepository
from backend.app.schemas.inspection import (
    AdaptiveDecision,
    InspectionListResponse,
    InspectionResponse,
    RGBInspectionResult,
    XRayInspectionResult,
)
from backend.app.services.adaptive_inspection import AdaptiveInspectionService
from backend.app.services.rgb_inspection import RGBInspectionService
from backend.app.services.xray_inspection import XRayInspectionService
from backend.app.services.image_service import ImageValidationError

logger = get_logger(__name__)
router = APIRouter(tags=["inspections"])

# Service singletons
adaptive_service = AdaptiveInspectionService()
rgb_service = RGBInspectionService()
xray_service = XRayInspectionService()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _inspection_to_response(inspection) -> InspectionResponse:
    results = [{"result_kind": r.result_kind, "payload": r.payload} for r in inspection.results]

    rgb_schema = None
    xray_schema = None
    adaptive_schema = None
    financial_schema = None
    for r in inspection.results:
        if r.result_kind == "rgb" and isinstance(r.payload, dict):
            p = r.payload
            from backend.app.schemas.inspection import BoundingBox
            boxes = []
            for b in p.get("bounding_boxes", []):
                if isinstance(b, dict):
                    try:
                        boxes.append(BoundingBox(**b))
                    except Exception:
                        pass
            rgb_schema = RGBInspectionResult(
                predicted_class=p.get("predicted_class"),
                anomaly_score=p.get("anomaly_score"),
                confidence=p.get("confidence"),
                entropy=p.get("entropy"),
                bounding_boxes=boxes,
                localization=p.get("localization"),
                class_scores=p.get("class_scores"),
                status=p.get("status", inspection.status),
                model_id=p.get("model_id"),
                model_version=p.get("model_version"),
                message=p.get("message"),
                eligibility=p.get("eligibility"),
                decision_outcome=p.get("decision_outcome"),
                decision_reason=p.get("decision_reason"),
                detected_object=p.get("detected_object"),
                condition_label=p.get("condition_label"),
                is_defective=p.get("is_defective"),
            )
        elif r.result_kind == "xray" and isinstance(r.payload, dict):
            p = r.payload
            xray_schema = XRayInspectionResult(
                predicted_class=p.get("predicted_class"),
                score=p.get("score", p.get("anomaly_score")),
                confidence=p.get("confidence"),
                class_scores=p.get("class_scores"),
                localization=p.get("localization"),
                internal_defects_found=p.get("internal_defects_found", 0),
                status=p.get("status", inspection.status),
                message=p.get("message"),
            )
        elif r.result_kind == "adaptive" and isinstance(r.payload, dict):
            try:
                adaptive_schema = AdaptiveDecision(**r.payload)
            except Exception:
                pass
        elif r.result_kind == "financial" and isinstance(r.payload, dict):
            try:
                from backend.app.schemas.inspection import FinancialImpact
                financial_schema = FinancialImpact(**r.payload)
            except Exception:
                pass

    detected_obj = rgb_schema.detected_object if rgb_schema else None
    cond_lbl = rgb_schema.condition_label if rgb_schema else None
    is_def = rgb_schema.is_defective if rgb_schema else None
    thresh = None
    for r in inspection.results:
        if isinstance(r.payload, dict):
            if thresh is None and "threshold" in r.payload:
                thresh = r.payload.get("threshold")
            if thresh is None and isinstance(r.payload.get("localization"), dict):
                thresh = r.payload.get("localization").get("image_threshold")
    if thresh is None and rgb_schema and isinstance(rgb_schema.localization, dict):
        thresh = rgb_schema.localization.get("image_threshold")

    return InspectionResponse(
        id=inspection.id,
        inspection_type=inspection.inspection_type,
        original_filename=inspection.original_filename,
        stored_image_reference=inspection.stored_image_path,
        predicted_class=inspection.predicted_class,
        score=inspection.score,
        confidence=inspection.confidence,
        model_id=inspection.model_id,
        model_version=inspection.model_version,
        status=inspection.status,
        error_details=inspection.error_details,
        processing_duration_ms=inspection.processing_duration_ms,
        created_at=inspection.created_at,
        results=results,
        rgb=rgb_schema,
        xray=xray_schema,
        adaptive=adaptive_schema,
        financial_impact=financial_schema,
        detected_object=detected_obj,
        condition_label=cond_lbl,
        is_defective=is_def,
        threshold=thresh,
    )


# ===========================================================================
# v1 Endpoints
# ===========================================================================

@router.post(
    "/v1/inspect/upload",
    response_model=InspectionResponse,
    summary="Full adaptive inspection (upload RGB + optional X-ray)",
)
async def v1_inspect_upload(
    file: UploadFile = File(..., description="RGB image file"),
    xray_file: Optional[UploadFile] = File(None, description="Optional paired X-ray image"),
    enable_grad_cam: bool = Form(False, description="Enable Grad-CAM heatmap"),
    object_category: Optional[str] = Form(None, description="Object category for safety flagging"),
    db: Session = Depends(get_db),
) -> InspectionResponse:
    """Run the full two-stage adaptive inspection pipeline.

    1. RGB CNN inspection with anomaly scoring & localization
    2. Adaptive gating with entropy-based routing
    3. Conditional X-ray CNN inspection (if triggered + image provided)
    4. Cross-modal fusion (if both modalities available)
    5. Dynamic cost estimation
    6. Visual overlay with HUD
    7. ISO-9001 QC report generation
    """
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_bytes} bytes.")

    xray_bytes = None
    xray_fn = None
    xray_ct = None
    if xray_file is not None:
        xray_bytes = await xray_file.read()
        if len(xray_bytes) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="X-ray file exceeds size limit.")
        xray_fn = xray_file.filename
        xray_ct = xray_file.content_type

    try:
        return adaptive_service.run(
            db, content, file.filename, file.content_type,
            xray_bytes, xray_fn, xray_ct,
            enable_grad_cam=enable_grad_cam,
            object_category=object_category,
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("v1 inspect/upload failed")
        raise HTTPException(status_code=500, detail="Inspection failed.")


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------

@router.websocket("/v1/inspect/stream")
async def v1_inspect_stream(websocket: WebSocket):
    """Real-time webcam inspection via WebSocket.

    Client sends: base64-encoded JPEG frames (or raw binary)
    Server responds: JSON with annotated_frame_base64 + telemetry

    Protocol:
    1. Client connects
    2. Client sends frames as base64 text or binary
    3. Server responds with JSON telemetry + annotated frame
    4. Client can send {"command": "toggle_gradcam"} to toggle Grad-CAM
    """
    await websocket.accept()
    logger.info("WebSocket stream connection accepted")

    frame_id = 0
    enable_grad_cam = False

    try:
        while True:
            # Receive frame data
            try:
                data = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break

            if "text" in data and data["text"]:
                text = data["text"]

                # Check for control commands
                try:
                    cmd = json.loads(text)
                    if isinstance(cmd, dict):
                        if cmd.get("command") == "toggle_gradcam":
                            enable_grad_cam = not enable_grad_cam
                            await websocket.send_json({
                                "type": "control",
                                "grad_cam_enabled": enable_grad_cam,
                            })
                            continue
                        elif cmd.get("command") == "ping":
                            await websocket.send_json({"type": "pong"})
                            continue
                except (json.JSONDecodeError, TypeError):
                    pass

                # Treat as base64 frame
                try:
                    frame_bytes = base64.b64decode(text)
                except Exception:
                    await websocket.send_json({"error": "Invalid base64 data"})
                    continue

            elif "bytes" in data and data["bytes"]:
                frame_bytes = data["bytes"]
            else:
                continue

            # Process frame
            try:
                result = adaptive_service.inspect_stream_frame(
                    frame_bytes=frame_bytes,
                    frame_id=frame_id,
                    enable_grad_cam=enable_grad_cam,
                )
                frame_id += 1
                await websocket.send_json(result)
            except Exception as e:
                logger.exception("Stream frame processing error")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket stream disconnected")
    except Exception as e:
        logger.exception("WebSocket stream error: %s", e)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ===========================================================================
# Legacy endpoints (backward compatible)
# ===========================================================================

@router.post("/inspect/rgb", response_model=InspectionResponse, summary="Run RGB inspection")
async def inspect_rgb(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> InspectionResponse:
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_bytes} bytes.")
    try:
        return rgb_service.run(db, content, file.filename, file.content_type)
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("RGB inspection failed")
        raise HTTPException(status_code=500, detail="RGB inspection failed.")


@router.post("/inspect/xray", response_model=InspectionResponse, summary="Run X-ray inspection")
async def inspect_xray(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> InspectionResponse:
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_bytes} bytes.")
    try:
        return xray_service.run(db, content, file.filename, file.content_type)
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("X-ray inspection failed")
        raise HTTPException(status_code=500, detail="X-ray inspection failed.")


@router.post("/inspect/adaptive", response_model=InspectionResponse, summary="Run adaptive inspection")
async def inspect_adaptive(
    file: UploadFile = File(...),
    xray_file: Optional[UploadFile] = None,
    db: Session = Depends(get_db),
) -> InspectionResponse:
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_bytes} bytes.")

    xray_bytes = None
    xray_fn = None
    xray_ct = None
    if xray_file is not None:
        xray_bytes = await xray_file.read()
        if len(xray_bytes) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="X-ray file exceeds limit.")
        xray_fn = xray_file.filename
        xray_ct = xray_file.content_type

    try:
        return adaptive_service.run(
            db, content, file.filename, file.content_type,
            xray_bytes, xray_fn, xray_ct,
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Adaptive inspection failed")
        raise HTTPException(status_code=500, detail="Adaptive inspection failed.")


# ===========================================================================
# CRUD endpoints
# ===========================================================================

@router.get("/inspections", response_model=InspectionListResponse, summary="List inspections")
def list_inspections(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    inspection_type: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> InspectionListResponse:
    repo = InspectionRepository(db)
    inspections = repo.list(skip=skip, limit=limit, inspection_type=inspection_type)
    total = repo.count(inspection_type=inspection_type)
    return InspectionListResponse(total=total, items=[_inspection_to_response(i) for i in inspections])


@router.get("/inspections/{inspection_id}", response_model=InspectionResponse, summary="Get inspection")
def get_inspection(inspection_id: str, db: Session = Depends(get_db)) -> InspectionResponse:
    repo = InspectionRepository(db)
    inspection = repo.get(inspection_id)
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    return _inspection_to_response(inspection)


@router.delete("/inspections/{inspection_id}", status_code=204, summary="Delete inspection")
def delete_inspection(inspection_id: str, db: Session = Depends(get_db)) -> None:
    repo = InspectionRepository(db)
    deleted = repo.delete(inspection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Inspection not found.")
