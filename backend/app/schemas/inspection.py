"""Full Pydantic v2 models for inspection pipeline.

Covers: bounding boxes, anomaly scores, cost matrices, QC reports,
WebSocket telemetry, fusion results, and Grad-CAM metadata.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ProductStatus(str, Enum):
    GOOD_FOR_CONSUMER = "GOOD_FOR_CONSUMER"
    NEEDS_XRAY = "NEEDS_XRAY"
    DEFECTIVE_REWORK = "DEFECTIVE_REWORK"
    DEFECTIVE_SCRAP = "DEFECTIVE_SCRAP"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class SeverityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class RouteDecision(str, Enum):
    DISPATCH_TO_CONSUMER = "DISPATCH_TO_CONSUMER"
    TRIGGER_XRAY_INSPECTION = "TRIGGER_XRAY_INSPECTION"
    IMMEDIATE_SCRAP = "IMMEDIATE_SCRAP"


class EconomicRecommendation(str, Enum):
    SCRAP_AND_REPLACE = "SCRAP_AND_REPLACE"
    REWORK_REPAIR = "REWORK_REPAIR"
    NO_ACTION = "NO_ACTION"


# ---------------------------------------------------------------------------
# Bounding Box Models
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Pixel-space bounding box with metadata."""
    x1: float = Field(..., description="Left x coordinate")
    y1: float = Field(..., description="Top y coordinate")
    x2: float = Field(..., description="Right x coordinate")
    y2: float = Field(..., description="Bottom y coordinate")
    label: str = Field(default="defect", description="Defect type label")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    area_pct: float = Field(default=0.0, ge=0.0, le=100.0,
                            description="Area as percentage of total image")
    severity: Optional[SeverityLevel] = None
    defect_type: Optional[str] = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0, self.width * self.height)


class BoundingBoxXYWH(BaseModel):
    """Bounding box in XYWH format (x, y, width, height)."""
    x: float
    y: float
    w: float
    h: float
    label: str = "defect"
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Upload & Image Info
# ---------------------------------------------------------------------------

class UploadImageInfo(BaseModel):
    original_filename: Optional[str] = None
    stored_image_reference: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None


# ---------------------------------------------------------------------------
# RGB Inspection Results (Stage 1)
# ---------------------------------------------------------------------------

class GradCAMMetadata(BaseModel):
    """Grad-CAM explanation metadata."""
    enabled: bool = False
    target_layer: Optional[str] = None
    heatmap_base64: Optional[str] = None
    top_activation_regions: List[BoundingBox] = []


class RGBInspectionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    predicted_class: Optional[str] = None
    anomaly_score: Optional[float] = Field(None, ge=0.0,
                                           description="Image-level anomaly score")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    entropy: Optional[float] = Field(None, ge=0.0,
                                     description="Shannon entropy H(p) of softmax output")
    localization: Optional[Dict[str, Any]] = None
    bounding_boxes: List[BoundingBox] = []
    class_scores: Optional[Dict[str, float]] = None
    status: str = "completed"
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    processing_duration_ms: Optional[float] = None
    message: Optional[str] = None
    eligibility: Optional[str] = None
    decision_outcome: Optional[str] = None
    decision_reason: Optional[str] = None
    grad_cam: Optional[GradCAMMetadata] = None
    anomaly_heatmap_base64: Optional[str] = None
    detected_object: Optional[str] = None
    condition_label: Optional[str] = None
    is_defective: Optional[bool] = None


# ---------------------------------------------------------------------------
# X-Ray Inspection Results (Stage 2)
# ---------------------------------------------------------------------------

class XRayInspectionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    predicted_class: Optional[str] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    class_scores: Optional[Dict[str, float]] = None
    localization: Optional[Dict[str, Any]] = None
    bounding_boxes: List[BoundingBox] = []
    internal_defects_found: int = 0
    status: str = "completed"
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    processing_duration_ms: Optional[float] = None
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Cross-Modal Fusion
# ---------------------------------------------------------------------------

class FusionResult(BaseModel):
    """Late cross-modal fusion output."""
    fusion_score: float = Field(..., ge=0.0, le=1.0,
                                description="Fused anomaly score")
    alpha: float = Field(..., ge=0.0, le=1.0,
                         description="Gated attention weight for RGB features")
    rgb_feature_norm: Optional[float] = None
    xray_feature_norm: Optional[float] = None
    fused_bounding_boxes: List[BoundingBox] = []
    method: str = "gated_attention_late_fusion"


# ---------------------------------------------------------------------------
# Adaptive Gating Decision
# ---------------------------------------------------------------------------

class AdaptiveDecision(BaseModel):
    model_config = ConfigDict(extra="allow")

    route: RouteDecision = RouteDecision.DISPATCH_TO_CONSUMER
    recommendation: str = ""
    reason: str = ""
    explanations: List[str] = []
    rgb_available: bool = False
    xray_available: bool = False
    xray_triggered: bool = False
    requires_human_review: bool = False
    entropy_value: Optional[float] = None
    entropy_threshold: Optional[float] = None
    policy: Dict[str, Any] = {}
    fusion: Optional[FusionResult] = None


# ---------------------------------------------------------------------------
# Cost Estimation
# ---------------------------------------------------------------------------

class CostMatrix(BaseModel):
    """Configurable cost parameters for the cost engine."""
    base_material_cost_usd: float = Field(default=15.0, ge=0.0)
    rework_rate_per_unit_area: float = Field(default=50.0, ge=0.0,
                                             description="USD per unit area ratio")
    labor_cost_per_hour_usd: float = Field(default=35.0, ge=0.0)
    xray_scan_cost_usd: float = Field(default=8.50, ge=0.0)
    replacement_cost_usd: float = Field(default=45.0, ge=0.0)
    scrap_value_usd: float = Field(default=2.50, ge=0.0)
    severity_multipliers: Dict[str, float] = Field(default_factory=lambda: {
        "Low": 0.5,
        "Medium": 1.0,
        "High": 2.0,
        "Critical": 4.0,
    })
    px_to_mm_ratio: float = Field(default=0.1, gt=0.0,
                                  description="Pixel-to-millimeter conversion factor")


class FinancialImpact(BaseModel):
    """Financial impact assessment for a defective product."""
    estimated_repair_cost_usd: float = 0.0
    scrap_value_usd: float = 0.0
    xray_inspection_cost_usd: float = 0.0
    replacement_cost_usd: float = 0.0
    net_loss_usd: float = 0.0
    defect_area_ratio: float = 0.0
    defect_area_mm2: Optional[float] = None
    severity_multiplier: float = 1.0
    economic_recommendation: EconomicRecommendation = EconomicRecommendation.NO_ACTION
    cost_breakdown: Dict[str, float] = {}
    repair_vs_replace_ratio: float = 0.0


# ---------------------------------------------------------------------------
# QC Report
# ---------------------------------------------------------------------------

class QCReport(BaseModel):
    """Full structured QC report payload."""
    inspection_id: str
    timestamp: datetime
    product_status: ProductStatus
    rgb_metrics: Optional[RGBInspectionResult] = None
    xray_triggered: bool = False
    xray_metrics: Optional[XRayInspectionResult] = None
    adaptive_decision: Optional[AdaptiveDecision] = None
    financial_impact: Optional[FinancialImpact] = None
    annotated_image_base64: Optional[str] = None
    grad_cam_heatmap_base64: Optional[str] = None
    iso_9001_summary: Optional[str] = None
    fusion: Optional[FusionResult] = None


# ---------------------------------------------------------------------------
# WebSocket Telemetry
# ---------------------------------------------------------------------------

class StreamTelemetry(BaseModel):
    """Real-time telemetry sent over WebSocket alongside annotated frames."""
    frame_id: int = 0
    timestamp: str = ""
    fps: float = 0.0
    anomaly_score: float = 0.0
    confidence: float = 0.0
    entropy: float = 0.0
    product_status: ProductStatus = ProductStatus.REVIEW_REQUIRED
    route: RouteDecision = RouteDecision.DISPATCH_TO_CONSUMER
    bounding_boxes: List[BoundingBox] = []
    financial_impact: Optional[FinancialImpact] = None
    processing_ms: float = 0.0
    model_id: Optional[str] = None


class StreamFrame(BaseModel):
    """Full WebSocket response frame."""
    annotated_frame_base64: str = ""
    telemetry: StreamTelemetry = StreamTelemetry()


# ---------------------------------------------------------------------------
# Legacy-compatible response wrappers
# ---------------------------------------------------------------------------

class InspectionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    inspection_type: str
    original_filename: Optional[str] = None
    stored_image_reference: Optional[str] = None
    predicted_class: Optional[str] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    status: str
    error_details: Optional[str] = None
    processing_duration_ms: Optional[float] = None
    created_at: datetime
    results: List[Dict[str, Any]] = []
    rgb: Optional[RGBInspectionResult] = None
    xray: Optional[XRayInspectionResult] = None
    adaptive: Optional[AdaptiveDecision] = None
    financial_impact: Optional[FinancialImpact] = None
    qc_report: Optional[QCReport] = None
    annotated_image_base64: Optional[str] = None
    detected_object: Optional[str] = None
    condition_label: Optional[str] = None
    is_defective: Optional[bool] = None
    threshold: Optional[float] = None


class InspectionListResponse(BaseModel):
    total: int
    items: List[InspectionResponse]
