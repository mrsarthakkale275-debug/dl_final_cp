"""X-ray PCB inspection service."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from backend.app.core.logging_config import get_logger
from backend.app.repositories.inspection_repository import InspectionRepository
from backend.app.schemas.inspection import InspectionResponse, XRayInspectionResult
from backend.app.services.image_service import validate_upload, store_image

logger = get_logger(__name__)


class XRayInspectionService:
    def __init__(self, registry: Optional[Any] = None):
        if registry is None:
            from backend.app.services.model_registry import registry as default_registry

            registry = default_registry
        self.registry = registry

    def analyze(
        self,
        image_bytes: bytes,
        filename: Optional[str],
        content_type: Optional[str],
    ) -> Dict[str, Any]:
        validated = validate_upload(image_bytes, filename, content_type)
        stored_path = store_image(validated, image_bytes)

        engine = self.registry.get_xray_engine()
        start = time.perf_counter() * 1000.0
        if engine is None:
            outcome = {
                "status": "model_unavailable",
                "predicted_class": None,
                "score": None,
                "confidence": None,
                "class_scores": None,
                "localization": None,
                "model_id": getattr(self.registry.xray, "model_id", None),
                "model_version": getattr(self.registry.xray, "version", None),
                "message": getattr(self.registry.xray, "message", "X-ray model unavailable."),
            }
        else:
            try:
                raw = engine.predict_xray(validated.image.copy())
                outcome = {
                    "status": "completed",
                    "predicted_class": raw.get("predicted_class"),
                    "score": raw.get("score", raw.get("anomaly_score")),
                    "confidence": raw.get("confidence"),
                    "class_scores": raw.get("class_scores"),
                    "localization": raw.get("localization"),
                    "model_id": getattr(engine, "model_id", self.registry.xray.model_id),
                    "model_version": getattr(engine, "model_version", self.registry.xray.version),
                    "message": raw.get("message"),
                }
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("X-ray inference failed")
                outcome = {
                    "status": "error",
                    "predicted_class": None,
                    "score": None,
                    "confidence": None,
                    "class_scores": None,
                    "localization": None,
                    "model_id": getattr(self.registry.xray, "model_id", None),
                    "model_version": getattr(self.registry.xray, "version", None),
                    "message": f"X-ray inference failed: {exc}",
                    "error": str(exc),
                }
        duration = time.perf_counter() * 1000.0 - start
        outcome["processing_duration_ms"] = duration

        return {
            "validated": validated,
            "stored_image_path": stored_path,
            "image_bytes": image_bytes,
            "result": outcome,
        }

    def run(
        self,
        db,
        image_bytes: bytes,
        filename: Optional[str],
        content_type: Optional[str],
    ) -> InspectionResponse:
        analyzed = self.analyze(image_bytes, filename, content_type)
        validated = analyzed["validated"]
        outcome = analyzed["result"]
        repo = InspectionRepository(db)
        inspection = repo.create(
            inspection_type="xray",
            original_filename=validated.original_filename,
            stored_image_path=analyzed["stored_image_path"],
            image_sha256=validated.sha256,
            status=outcome["status"],
            error_details=outcome.get("error") or outcome.get("message") if outcome["status"] != "completed" else None,
            processing_duration_ms=outcome["processing_duration_ms"],
            predicted_class=outcome.get("predicted_class"),
            score=outcome.get("score"),
            confidence=outcome.get("confidence"),
            model_id=outcome.get("model_id"),
            model_version=outcome.get("model_version"),
        )
        repo.add_result(inspection.id, "xray", {
            "predicted_class": outcome.get("predicted_class"),
            "score": outcome.get("score"),
            "confidence": outcome.get("confidence"),
            "class_scores": outcome.get("class_scores"),
            "localization": outcome.get("localization"),
            "status": outcome.get("status"),
            "message": outcome.get("message"),
        })

        xray_schema = XRayInspectionResult(
            predicted_class=outcome.get("predicted_class"),
            score=outcome.get("score"),
            confidence=outcome.get("confidence"),
            class_scores=outcome.get("class_scores"),
            localization=outcome.get("localization"),
            status=outcome.get("status", "completed"),
            model_id=outcome.get("model_id"),
            model_version=outcome.get("model_version"),
            processing_duration_ms=outcome.get("processing_duration_ms"),
            message=outcome.get("message"),
        )

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
            results=[{"result_kind": "xray", "payload": {
                "predicted_class": outcome.get("predicted_class"),
                "score": outcome.get("score"),
                "confidence": outcome.get("confidence"),
                "class_scores": outcome.get("class_scores"),
                "localization": outcome.get("localization"),
                "status": outcome.get("status"),
                "message": outcome.get("message"),
            }}],
            xray=xray_schema,
        )
