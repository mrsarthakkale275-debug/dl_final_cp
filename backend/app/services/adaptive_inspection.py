"""Adaptive inspection service orchestrating the full two-stage pipeline.

Connects: RGB CNN → Adaptive Gating → X-ray CNN → Fusion → Cost Engine → Visualizer → QC Report
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from backend.app.core.config import settings
from backend.app.core.logging_config import get_logger
from backend.app.repositories.inspection_repository import InspectionRepository
from backend.app.schemas.inspection import (
    AdaptiveDecision,
    BoundingBox,
    FinancialImpact,
    FusionResult,
    InspectionResponse,
    ProductStatus,
    QCReport,
    RGBInspectionResult,
    RouteDecision,
    SeverityLevel,
    XRayInspectionResult,
)
from backend.app.services.adaptive_decision import AdaptiveGatingRouter
from backend.app.services.cost_engine import CostEngine, QualitySummaryGenerator, cost_engine, summary_generator
from backend.app.services.image_service import validate_upload, store_image
from backend.app.services.visualizer import FrameProcessor, encode_frame_base64, frame_processor

logger = get_logger(__name__)


def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR numpy array."""
    import cv2
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _dict_boxes_to_schema(boxes: list) -> list[BoundingBox]:
    """Convert list of dicts to BoundingBox schema objects."""
    result = []
    for b in boxes:
        if isinstance(b, BoundingBox):
            result.append(b)
        elif isinstance(b, dict):
            result.append(BoundingBox(
                x1=b.get("x1", 0), y1=b.get("y1", 0),
                x2=b.get("x2", 0), y2=b.get("y2", 0),
                label=b.get("label", "defect"),
                confidence=b.get("confidence", 0.0),
                area_pct=b.get("area_pct", 0.0),
                severity=b.get("severity"),
                defect_type=b.get("defect_type"),
            ))
    return result


class AdaptiveInspectionService:
    """Full two-stage adaptive inspection pipeline."""

    def __init__(
        self,
        inspector=None,
        gating_router: Optional[AdaptiveGatingRouter] = None,
        cost_eng: Optional[CostEngine] = None,
        visualizer: Optional[FrameProcessor] = None,
    ) -> None:
        self.inspector = inspector
        self.gating_router = gating_router or AdaptiveGatingRouter()
        self.cost_eng = cost_eng or cost_engine
        self.visualizer = visualizer or frame_processor

    def _get_inspector(self):
        """Lazy-load the multimodal inspector."""
        if self.inspector is None:
            try:
                from backend.app.ml.multimodal_cnn import create_inspector
                self.inspector = create_inspector()
                logger.info("MultimodalInspector lazy-loaded successfully.")
            except Exception as e:
                logger.warning("Could not load MultimodalInspector: %s", e)
                return None
        return self.inspector

    def analyze(
        self,
        image_bytes: bytes,
        filename: Optional[str],
        content_type: Optional[str],
        xray_image_bytes: Optional[bytes] = None,
        xray_filename: Optional[str] = None,
        xray_content_type: Optional[str] = None,
        enable_grad_cam: bool = False,
        object_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full adaptive inspection pipeline."""
        start_time = time.perf_counter()

        # Validate and store RGB image
        validated = validate_upload(image_bytes, filename, content_type)
        stored_path = store_image(validated, image_bytes)

        inspector = self._get_inspector()
        rgb_result: Dict[str, Any] = {}
        xray_result: Optional[Dict[str, Any]] = None
        fusion_result: Optional[Dict[str, Any]] = None
        annotated_frame: Optional[np.ndarray] = None

        # Convert PIL to numpy for CNN
        cv2_image = _pil_to_cv2(validated.image)

        # ==== Domain Guardrail Check ====
        from backend.app.ml.domain_detector import detect_object_and_condition
        anomalous_ratio = 0.0
        try:
            from backend.app.services.model_registry import registry as model_registry
            rgb_eng = model_registry.get_rgb_engine()
            if rgb_eng is not None:
                raw_rgb = rgb_eng.predict_rgb(validated.image.copy())
                loc = raw_rgb.get("localization") or {}
                anomalous_ratio = float(loc.get("anomalous_pixel_ratio", 0.0) or 0.0)
        except Exception as e:
            logger.debug("Domain guardrail anomalous ratio calculation skipped: %s", e)

        obj_info = detect_object_and_condition(
            validated.image,
            anomalous_ratio=anomalous_ratio,
        )
        is_pcb = obj_info["is_pcb"]
        object_label = obj_info["object_label"]
        condition_label = obj_info["condition_label"]
        is_defective_obj = obj_info["is_defective"]
        domain_message = obj_info["explanation"]

        if not is_pcb:
            unsupp_boxes = []
            img_h, img_w = cv2_image.shape[:2]
            for b in obj_info.get("defect_boxes", []):
                bx, by, bw, bh = b[:4]
                unsupp_boxes.append({
                    "x1": float(bx),
                    "y1": float(by),
                    "x2": float(bx + bw),
                    "y2": float(by + bh),
                    "label": "DEFECT / BREAKOUT",
                    "confidence": float(round(obj_info.get("confidence", 0.9), 3)),
                    "area_pct": float(round((bw * bh) / max(1, img_w * img_h) * 100.0, 2)),
                })

            rgb_result = {
                "status": "unsupported_object",
                "predicted_class": None,
                "anomaly_score": 0.0,
                "confidence": obj_info.get("confidence", 0.0),
                "entropy": 0.0,
                "bounding_boxes": unsupp_boxes,
                "class_scores": {
                    "normal": 0.985 if not is_defective_obj else 0.15,
                    "defective": 0.015 if not is_defective_obj else 0.85,
                },
                "detected_object": object_label,
                "condition_label": condition_label,
                "is_defective": is_defective_obj,
                "eligibility": "unsupported_object",
                "decision_outcome": "review_required",
                "decision_reason": domain_message,
                "message": domain_message,
            }
            decision = AdaptiveDecision(
                route=RouteDecision.TRIGGER_XRAY_INSPECTION,
                recommendation="human_review_required",
                reason=f"Rejected: {object_label}. Only Printed Circuit Boards are supported.",
                explanations=[domain_message],
                rgb_available=False,
                xray_available=False,
                xray_triggered=False,
                requires_human_review=True,
                policy=self.gating_router._policy_dict(),
            )
        else:
            # ==== Stage 1: RGB CNN Inspection ====
            if inspector is not None:
                try:
                    rgb_result = inspector.inspect_rgb(cv2_image, enable_grad_cam=enable_grad_cam)
                    rgb_result["status"] = "completed"
                    rgb_result["model_id"] = inspector.model_id
                    rgb_result["model_version"] = inspector.version
                except Exception as e:
                    logger.exception("RGB CNN inspection failed")
                    rgb_result = {
                        "status": "error",
                        "anomaly_score": 0.5,
                        "confidence": 0.0,
                        "entropy": 1.0,
                        "predicted_class": None,
                        "bounding_boxes": [],
                        "message": f"RGB inspection error: {e}",
                    }
            else:
                # Mock fallback
                rgb_result = {
                    "status": "model_unavailable",
                    "anomaly_score": 0.0,
                    "confidence": 0.0,
                    "entropy": 0.0,
                    "predicted_class": None,
                    "bounding_boxes": [],
                    "message": "CNN model unavailable; using fallback.",
                }

            rgb_result["detected_object"] = object_label
            pred_cls = rgb_result.get("predicted_class")
            rgb_result["condition_label"] = "Defective (Breakout Detected)" if pred_cls == "defective" else "Normal (PASS)"
            rgb_result["is_defective"] = (pred_cls == "defective")

            # ==== Adaptive Gating Router ====
            decision = self.gating_router.evaluate(
                rgb_result=rgb_result,
                xray_result=None,
                object_category=object_category,
            )

        # ==== Stage 2: X-ray Inspection (conditional) ====
        if decision.xray_triggered and xray_image_bytes is not None:
            xray_validated = validate_upload(xray_image_bytes, xray_filename, xray_content_type)
            store_image(xray_validated, xray_image_bytes)
            xray_cv2 = _pil_to_cv2(xray_validated.image)

            if inspector is not None:
                try:
                    xray_result = inspector.inspect_xray(xray_cv2)
                    xray_result["status"] = "completed"
                    xray_result["model_id"] = inspector.model_id

                    # Cross-modal fusion
                    if "features" in rgb_result and "features" in xray_result:
                        fusion_result = inspector.fuse(
                            rgb_result["features"],
                            xray_result["features"],
                        )
                except Exception as e:
                    logger.exception("X-ray CNN inspection failed")
                    xray_result = {
                        "status": "error",
                        "message": f"X-ray error: {e}",
                    }

            # Re-evaluate with X-ray results
            decision = self.gating_router.evaluate(
                rgb_result=rgb_result,
                xray_result=xray_result,
                fusion_result=fusion_result,
                object_category=object_category,
            )

        elif decision.xray_triggered and xray_image_bytes is None:
            decision.explanations.append(
                "X-ray scan was triggered but no X-ray image was provided. "
                "In production, the system would automatically request an X-ray."
            )

        # ==== Determine product status ====
        product_status = self._determine_product_status(decision, rgb_result)

        # ==== Cost Estimation ====
        boxes = _dict_boxes_to_schema(rgb_result.get("bounding_boxes", []))
        xray_boxes = _dict_boxes_to_schema(
            xray_result.get("bounding_boxes", []) if xray_result else []
        )

        financial_impact = self.cost_eng.estimate_repair_cost(
            bounding_boxes=boxes + xray_boxes,
            image_width=validated.width,
            image_height=validated.height,
            xray_triggered=decision.xray_triggered,
        )

        # ==== Visual Overlay ====
        try:
            if not is_pcb:
                from backend.app.services.rgb_inspection import _generate_annotated_image
                annotated_b64 = _generate_annotated_image(
                    validated.image,
                    predicted_class=None,
                    is_supported=False,
                    object_label=object_label,
                    boxes=obj_info.get("defect_boxes", []),
                    is_defective=is_defective_obj,
                    condition_label=condition_label,
                )
            else:
                processing_ms = (time.perf_counter() - start_time) * 1000
                annotated_frame = self.visualizer.process(
                    frame=cv2_image,
                    anomaly_score=rgb_result.get("anomaly_score", 0.0),
                    confidence=rgb_result.get("confidence", 0.0),
                    entropy=rgb_result.get("entropy", 0.0),
                    predicted_class=rgb_result.get("predicted_class") or "unknown",
                    bounding_boxes=boxes,
                    xray_boxes=xray_boxes,
                    product_status=product_status,
                    route=decision.route,
                    processing_ms=processing_ms,
                    model_id=rgb_result.get("model_id"),
                    financial_impact=financial_impact,
                    grad_cam_heatmap=rgb_result.get("grad_cam_heatmap"),
                    enable_grad_cam=enable_grad_cam,
                )
                annotated_b64 = encode_frame_base64(annotated_frame)
        except Exception as e:
            logger.warning("Visualization failed: %s", e)
            annotated_b64 = None

        # ==== ISO-9001 Summary ====
        severity = self.cost_eng.determine_severity(boxes) if boxes else SeverityLevel.LOW
        iso_summary = summary_generator.generate(
            anomaly_score=rgb_result.get("anomaly_score", 0.0),
            predicted_class=rgb_result.get("predicted_class") or "unknown",
            bounding_boxes=boxes,
            severity=severity,
            xray_triggered=decision.xray_triggered,
            xray_defects_found=xray_result.get("internal_defects_found", 0) if xray_result else 0,
            defect_area_mm2=financial_impact.defect_area_mm2,
            economic_recommendation=financial_impact.economic_recommendation,
        )

        total_ms = (time.perf_counter() - start_time) * 1000

        return {
            "validated": validated,
            "stored_image_path": stored_path,
            "rgb_result": rgb_result,
            "xray_result": xray_result,
            "fusion_result": fusion_result,
            "decision": decision,
            "product_status": product_status,
            "financial_impact": financial_impact,
            "annotated_image_base64": annotated_b64,
            "iso_summary": iso_summary,
            "processing_duration_ms": total_ms,
        }

    def run(
        self,
        db,
        image_bytes: bytes,
        filename: Optional[str],
        content_type: Optional[str],
        xray_image_bytes: Optional[bytes] = None,
        xray_filename: Optional[str] = None,
        xray_content_type: Optional[str] = None,
        enable_grad_cam: bool = False,
        object_category: Optional[str] = None,
    ) -> InspectionResponse:
        """Run full pipeline and persist to database."""
        analyzed = self.analyze(
            image_bytes, filename, content_type,
            xray_image_bytes, xray_filename, xray_content_type,
            enable_grad_cam, object_category,
        )

        validated = analyzed["validated"]
        rgb_result = analyzed["rgb_result"]
        decision = analyzed["decision"]
        financial_impact = analyzed["financial_impact"]

        repo = InspectionRepository(db)
        inspection = repo.create(
            inspection_type="adaptive",
            original_filename=validated.original_filename,
            stored_image_path=analyzed["stored_image_path"],
            image_sha256=validated.sha256,
            status=rgb_result.get("status", "completed"),
            processing_duration_ms=analyzed["processing_duration_ms"],
            predicted_class=rgb_result.get("predicted_class"),
            score=rgb_result.get("anomaly_score"),
            confidence=rgb_result.get("confidence"),
            model_id=rgb_result.get("model_id"),
            model_version=rgb_result.get("model_version"),
        )

        # Persist granular results
        repo.add_result(inspection.id, "rgb", {
            k: v for k, v in rgb_result.items()
            if k not in {"features", "feature_map", "grad_cam_heatmap", "feature_map_shape"}
        })
        if analyzed["xray_result"]:
            repo.add_result(inspection.id, "xray", {
                k: v for k, v in analyzed["xray_result"].items()
                if k not in {"features", "feature_map"}
            })
        repo.add_result(inspection.id, "adaptive", decision.model_dump())
        repo.add_result(inspection.id, "financial", financial_impact.model_dump())

        # Build response schemas
        boxes = _dict_boxes_to_schema(rgb_result.get("bounding_boxes", []))
        rgb_schema = RGBInspectionResult(
            predicted_class=rgb_result.get("predicted_class"),
            anomaly_score=rgb_result.get("anomaly_score"),
            confidence=rgb_result.get("confidence"),
            entropy=rgb_result.get("entropy"),
            bounding_boxes=boxes,
            class_scores=rgb_result.get("class_scores"),
            status=rgb_result.get("status", "completed"),
            model_id=rgb_result.get("model_id"),
            model_version=rgb_result.get("model_version"),
            processing_duration_ms=analyzed["processing_duration_ms"],
            message=rgb_result.get("message"),
            eligibility=rgb_result.get("eligibility"),
            decision_outcome=rgb_result.get("decision_outcome"),
            decision_reason=rgb_result.get("decision_reason"),
            detected_object=rgb_result.get("detected_object"),
            condition_label=rgb_result.get("condition_label"),
            is_defective=rgb_result.get("is_defective"),
        )

        xray_schema = None
        if analyzed["xray_result"]:
            xr = analyzed["xray_result"]
            xray_boxes = _dict_boxes_to_schema(xr.get("bounding_boxes", []))
            xray_schema = XRayInspectionResult(
                predicted_class=xr.get("predicted_class"),
                score=xr.get("anomaly_score", xr.get("score")),
                confidence=xr.get("confidence"),
                bounding_boxes=xray_boxes,
                internal_defects_found=xr.get("internal_defects_found", 0),
                status=xr.get("status", "completed"),
                model_id=xr.get("model_id"),
            )

        from datetime import datetime, timezone
        qc_report = QCReport(
            inspection_id=inspection.id,
            timestamp=datetime.now(timezone.utc),
            product_status=analyzed["product_status"],
            rgb_metrics=rgb_schema,
            xray_triggered=decision.xray_triggered,
            xray_metrics=xray_schema,
            adaptive_decision=decision,
            financial_impact=financial_impact,
            annotated_image_base64=analyzed.get("annotated_image_base64"),
            iso_9001_summary=analyzed.get("iso_summary"),
        )
        repo.add_result(inspection.id, "qc_report", qc_report.model_dump(mode="json"))

        def _clean_payload(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not d:
                return None
            clean = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    continue
                if isinstance(v, (np.floating, float)):
                    clean[k] = float(v)
                elif isinstance(v, (np.integer, int)):
                    clean[k] = int(v)
                else:
                    clean[k] = v
            return clean

        results = [
            {"result_kind": "rgb", "payload": _clean_payload(rgb_result)},
        ]
        if analyzed["xray_result"]:
            results.append({"result_kind": "xray", "payload": _clean_payload(analyzed["xray_result"])})
        results.append({"result_kind": "adaptive", "payload": decision.model_dump()})
        results.append({"result_kind": "financial", "payload": financial_impact.model_dump()})

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
            processing_duration_ms=analyzed["processing_duration_ms"],
            created_at=inspection.created_at,
            results=results,
            rgb=rgb_schema,
            xray=xray_schema,
            adaptive=decision,
            financial_impact=financial_impact,
            qc_report=qc_report,
            annotated_image_base64=analyzed.get("annotated_image_base64"),
            detected_object=rgb_result.get("detected_object"),
            condition_label=rgb_result.get("condition_label"),
            is_defective=rgb_result.get("is_defective"),
        )

    def inspect_stream_frame(
        self,
        frame_bytes: bytes,
        frame_id: int = 0,
        enable_grad_cam: bool = False,
    ) -> Dict[str, Any]:
        """Process a single webcam frame for WebSocket streaming.

        Returns annotated frame base64 + telemetry JSON.
        """
        import cv2
        start = time.perf_counter()

        # Decode frame
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Could not decode frame"}

        inspector = self._get_inspector()
        if inspector is None:
            return {"error": "Inspector not available"}

        # RGB inspection
        rgb_result = inspector.inspect_rgb(frame, enable_grad_cam=enable_grad_cam)

        # Quick routing (no X-ray on stream)
        decision = self.gating_router.evaluate(rgb_result=rgb_result)
        product_status = self._determine_product_status(decision, rgb_result)

        # Cost estimation
        boxes = _dict_boxes_to_schema(rgb_result.get("bounding_boxes", []))
        financial_impact = self.cost_eng.estimate_repair_cost(
            bounding_boxes=boxes,
            image_width=frame.shape[1],
            image_height=frame.shape[0],
        )

        processing_ms = (time.perf_counter() - start) * 1000
        fps = 1000.0 / processing_ms if processing_ms > 0 else 0.0

        # Render annotated frame
        annotated = self.visualizer.process(
            frame=frame,
            anomaly_score=rgb_result.get("anomaly_score", 0.0),
            confidence=rgb_result.get("confidence", 0.0),
            entropy=rgb_result.get("entropy", 0.0),
            predicted_class=rgb_result.get("predicted_class", "unknown"),
            bounding_boxes=boxes,
            product_status=product_status,
            route=decision.route,
            fps=fps,
            processing_ms=processing_ms,
            model_id=rgb_result.get("model_id"),
            financial_impact=financial_impact,
            grad_cam_heatmap=rgb_result.get("grad_cam_heatmap"),
            enable_grad_cam=enable_grad_cam,
            frame_id=frame_id,
        )
        annotated_b64 = encode_frame_base64(annotated)

        from datetime import datetime, timezone
        return {
            "annotated_frame_base64": annotated_b64,
            "telemetry": {
                "frame_id": frame_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fps": round(fps, 1),
                "anomaly_score": round(rgb_result.get("anomaly_score", 0.0), 4),
                "confidence": round(rgb_result.get("confidence", 0.0), 4),
                "entropy": round(rgb_result.get("entropy", 0.0), 4),
                "product_status": product_status.value,
                "route": decision.route.value,
                "bounding_boxes": [b.model_dump() for b in boxes],
                "financial_impact": financial_impact.model_dump() if financial_impact else None,
                "processing_ms": round(processing_ms, 1),
                "model_id": rgb_result.get("model_id"),
            },
        }

    def _determine_product_status(
        self, decision: AdaptiveDecision, rgb_result: Dict[str, Any]
    ) -> ProductStatus:
        """Map routing decision to product status."""
        if rgb_result.get("status") == "unsupported_object":
            return ProductStatus.REVIEW_REQUIRED
        if decision.route == RouteDecision.DISPATCH_TO_CONSUMER:
            return ProductStatus.GOOD_FOR_CONSUMER
        elif decision.route == RouteDecision.IMMEDIATE_SCRAP:
            return ProductStatus.DEFECTIVE_SCRAP
        elif decision.route == RouteDecision.TRIGGER_XRAY_INSPECTION:
            if decision.recommendation in {"defective_rework_or_scrap", "immediate_scrap"}:
                return ProductStatus.DEFECTIVE_SCRAP
            if decision.xray_triggered and not decision.requires_human_review:
                return ProductStatus.NEEDS_XRAY
            if decision.requires_human_review:
                return ProductStatus.REVIEW_REQUIRED
            return ProductStatus.NEEDS_XRAY
        return ProductStatus.REVIEW_REQUIRED
