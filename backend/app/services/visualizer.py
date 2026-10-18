"""OpenCV-based industrial HUD visualizer.

Draws:
  - GREEN bounding boxes on verified normal items
    ("CONFIDENT GOOD PRODUCT (Conf: XX%)")
  - RED bounding boxes on localized defective sub-regions
    ("DEFECT: [Type] | Severity: [High/Med/Low]")
  - YELLOW/ORANGE bounding boxes for uncertain regions
  - Grad-CAM thermal heatmap overlay (toggleable)
  - Industrial Heads-Up Display (HUD) with real-time telemetry
  - Cost estimation overlay

Color scheme:
  - Green:  #00FF88 (normal)   | RGB(0, 255, 136)
  - Red:    #FF3366 (defect)   | RGB(255, 51, 102)
  - Yellow: #FFD700 (uncertain) | RGB(255, 215, 0)
  - Orange: #FF8C00 (internal)  | RGB(255, 140, 0)
  - Cyan:   #00E5FF (HUD)      | RGB(0, 229, 255)
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.app.schemas.inspection import (
    BoundingBox,
    EconomicRecommendation,
    FinancialImpact,
    ProductStatus,
    RouteDecision,
    SeverityLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color Palette (BGR for OpenCV)
# ---------------------------------------------------------------------------
COLOR_GREEN = (136, 255, 0)       # Normal/Good
COLOR_RED = (102, 51, 255)        # Defect
COLOR_YELLOW = (0, 215, 255)      # Uncertain
COLOR_ORANGE = (0, 140, 255)      # Internal defect (X-ray)
COLOR_CYAN = (255, 229, 0)        # HUD text
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK_BG = (30, 30, 30)      # HUD background
COLOR_SEVERITY_LOW = (0, 200, 0)
COLOR_SEVERITY_MED = (0, 200, 255)
COLOR_SEVERITY_HIGH = (0, 100, 255)
COLOR_SEVERITY_CRITICAL = (0, 0, 255)

SEVERITY_COLORS = {
    SeverityLevel.LOW: COLOR_SEVERITY_LOW,
    SeverityLevel.MEDIUM: COLOR_SEVERITY_MED,
    SeverityLevel.HIGH: COLOR_SEVERITY_HIGH,
    SeverityLevel.CRITICAL: COLOR_SEVERITY_CRITICAL,
    "Low": COLOR_SEVERITY_LOW,
    "Medium": COLOR_SEVERITY_MED,
    "High": COLOR_SEVERITY_HIGH,
    "Critical": COLOR_SEVERITY_CRITICAL,
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _put_text_with_bg(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = COLOR_WHITE,
    bg_color: Tuple[int, int, int] = COLOR_DARK_BG,
    thickness: int = 1,
    padding: int = 4,
) -> None:
    """Draw text with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(
        img,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color, -1,
    )
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def encode_frame_base64(frame: np.ndarray, fmt: str = ".jpg", quality: int = 85) -> str:
    """Encode an OpenCV frame to base64 string."""
    params = []
    if fmt == ".jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif fmt == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    success, buffer = cv2.imencode(fmt, frame, params)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def decode_base64_frame(data: str) -> Optional[np.ndarray]:
    """Decode a base64 string to an OpenCV frame."""
    try:
        buf = base64.b64decode(data)
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Bounding Box Drawer
# ---------------------------------------------------------------------------

class BoundingBoxDrawer:
    """Draw styled bounding boxes with labels on frames."""

    def draw_normal_box(
        self,
        frame: np.ndarray,
        confidence: float = 0.95,
        bbox: Optional[BoundingBox] = None,
    ) -> np.ndarray:
        """Draw GREEN box around entire object as CONFIDENT GOOD PRODUCT."""
        h, w = frame.shape[:2]
        margin = max(10, int(min(h, w) * 0.03))

        if bbox:
            x1, y1 = int(bbox.x1), int(bbox.y1)
            x2, y2 = int(bbox.x2), int(bbox.y2)
        else:
            x1, y1 = margin, margin
            x2, y2 = w - margin, h - margin

        # Draw box with corner accents
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)

        # Corner accents
        corner_len = max(15, int(min(x2 - x1, y2 - y1) * 0.1))
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), COLOR_GREEN, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), COLOR_GREEN, 3)

        # Label
        label = f"CONFIDENT GOOD PRODUCT (Conf: {confidence*100:.1f}%)"
        _put_text_with_bg(
            frame, label, (x1 + 5, y1 - 10),
            font_scale=0.55, color=COLOR_GREEN, bg_color=(0, 40, 0),
        )

        return frame

    def draw_defect_box(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        defect_type: str = "Surface Anomaly",
        severity: str = "Medium",
    ) -> np.ndarray:
        """Draw RED box around defective sub-region."""
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        sev_color = SEVERITY_COLORS.get(severity, COLOR_RED)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), sev_color, 2)

        # Crosshair at center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ch_size = 8
        cv2.line(frame, (cx - ch_size, cy), (cx + ch_size, cy), sev_color, 1)
        cv2.line(frame, (cx, cy - ch_size), (cx, cy + ch_size), sev_color, 1)

        # Label
        label = f"DEFECT: {defect_type} | Severity: {severity}"
        if bbox.area_pct > 0:
            label += f" | Area: {bbox.area_pct:.1f}%"

        _put_text_with_bg(
            frame, label, (x1 + 2, y1 - 8),
            font_scale=0.45, color=COLOR_WHITE, bg_color=sev_color,
        )

        # Confidence badge
        if bbox.confidence > 0:
            conf_text = f"{bbox.confidence*100:.0f}%"
            _put_text_with_bg(
                frame, conf_text, (x2 - 45, y2 + 18),
                font_scale=0.4, color=COLOR_WHITE, bg_color=sev_color,
            )

        return frame

    def draw_uncertain_box(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
    ) -> np.ndarray:
        """Draw YELLOW/dashed box for uncertain regions."""
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        # Dashed rectangle
        dash_length = 10
        for start in range(x1, x2, dash_length * 2):
            end = min(start + dash_length, x2)
            cv2.line(frame, (start, y1), (end, y1), COLOR_YELLOW, 2)
            cv2.line(frame, (start, y2), (end, y2), COLOR_YELLOW, 2)
        for start in range(y1, y2, dash_length * 2):
            end = min(start + dash_length, y2)
            cv2.line(frame, (x1, start), (x1, end), COLOR_YELLOW, 2)
            cv2.line(frame, (x2, start), (x2, end), COLOR_YELLOW, 2)

        label = "UNCERTAIN - Triggering Stage-2 X-Ray Scan..."
        _put_text_with_bg(
            frame, label, (x1 + 2, y1 - 8),
            font_scale=0.45, color=COLOR_YELLOW, bg_color=(40, 40, 0),
        )
        return frame

    def draw_xray_defect_box(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        defect_type: str = "Internal Void",
    ) -> np.ndarray:
        """Draw RED/ORANGE box for internal X-ray defects."""
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ORANGE, 2)

        label = f"X-RAY: {defect_type}"
        _put_text_with_bg(
            frame, label, (x1 + 2, y1 - 8),
            font_scale=0.45, color=COLOR_WHITE, bg_color=COLOR_ORANGE,
        )
        return frame


# ---------------------------------------------------------------------------
# Grad-CAM Overlay
# ---------------------------------------------------------------------------

class GradCAMOverlay:
    """Overlay Grad-CAM heatmap on frames."""

    def apply(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Apply Grad-CAM heatmap overlay.

        Args:
            frame: BGR image
            heatmap: (H, W) float32 in [0, 1]
            alpha: Blend factor
            colormap: OpenCV colormap
        """
        if heatmap is None or heatmap.size == 0:
            return frame

        h, w = frame.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored = cv2.applyColorMap(heatmap_uint8, colormap)

        blended = cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)

        # Add "GRAD-CAM" badge
        _put_text_with_bg(
            blended, "GRAD-CAM ACTIVE",
            (w - 180, 30),
            font_scale=0.5, color=COLOR_CYAN, bg_color=(40, 0, 0),
        )

        return blended


# ---------------------------------------------------------------------------
# Industrial HUD Overlay
# ---------------------------------------------------------------------------

class IndustrialHUD:
    """Heads-Up Display (HUD) overlay for industrial inspection frames."""

    def render(
        self,
        frame: np.ndarray,
        anomaly_score: float = 0.0,
        confidence: float = 0.0,
        entropy: float = 0.0,
        product_status: ProductStatus = ProductStatus.REVIEW_REQUIRED,
        route: RouteDecision = RouteDecision.DISPATCH_TO_CONSUMER,
        fps: float = 0.0,
        processing_ms: float = 0.0,
        model_id: Optional[str] = None,
        financial_impact: Optional[FinancialImpact] = None,
        frame_id: int = 0,
    ) -> np.ndarray:
        """Render the full HUD overlay on a frame."""
        h, w = frame.shape[:2]

        # --- Top-left: Status panel ---
        self._draw_status_panel(frame, product_status, route, 10, 10)

        # --- Top-right: Metrics panel ---
        self._draw_metrics_panel(
            frame, anomaly_score, confidence, entropy,
            fps, processing_ms, w - 280, 10,
        )

        # --- Bottom-left: Model info ---
        if model_id:
            _put_text_with_bg(
                frame, f"Model: {model_id}",
                (10, h - 15), font_scale=0.35, color=COLOR_CYAN,
            )

        # --- Bottom-right: Cost panel ---
        if financial_impact and financial_impact.estimated_repair_cost_usd > 0:
            self._draw_cost_panel(frame, financial_impact, w - 280, h - 100)

        # --- Bottom center: Frame counter ---
        _put_text_with_bg(
            frame, f"Frame #{frame_id}",
            (w // 2 - 40, h - 15), font_scale=0.35, color=COLOR_CYAN,
        )

        # --- Scanline effect ---
        self._draw_scanline(frame, frame_id)

        return frame

    def _draw_status_panel(
        self,
        frame: np.ndarray,
        status: ProductStatus,
        route: RouteDecision,
        x: int,
        y: int,
    ) -> None:
        """Draw product status indicator."""
        status_colors = {
            ProductStatus.GOOD_FOR_CONSUMER: COLOR_GREEN,
            ProductStatus.NEEDS_XRAY: COLOR_YELLOW,
            ProductStatus.DEFECTIVE_REWORK: COLOR_ORANGE,
            ProductStatus.DEFECTIVE_SCRAP: COLOR_RED,
            ProductStatus.REVIEW_REQUIRED: COLOR_CYAN,
        }
        status_labels = {
            ProductStatus.GOOD_FOR_CONSUMER: "PASS - Dispatch to Consumer",
            ProductStatus.NEEDS_XRAY: "HOLD - X-Ray Required",
            ProductStatus.DEFECTIVE_REWORK: "FAIL - Rework Required",
            ProductStatus.DEFECTIVE_SCRAP: "FAIL - Scrap Immediately",
            ProductStatus.REVIEW_REQUIRED: "PENDING - Human Review",
        }

        color = status_colors.get(status, COLOR_CYAN)
        label = status_labels.get(status, str(status.value))

        # Status indicator circle
        cv2.circle(frame, (x + 8, y + 12), 6, color, -1)
        cv2.circle(frame, (x + 8, y + 12), 8, color, 1)

        # Status text
        _put_text_with_bg(
            frame, label, (x + 22, y + 16),
            font_scale=0.55, color=color, bg_color=COLOR_DARK_BG,
        )

        # Route text
        route_label = f"Route: {route.value}"
        _put_text_with_bg(
            frame, route_label, (x + 22, y + 38),
            font_scale=0.4, color=COLOR_CYAN, bg_color=COLOR_DARK_BG,
        )

    def _draw_metrics_panel(
        self,
        frame: np.ndarray,
        anomaly_score: float,
        confidence: float,
        entropy: float,
        fps: float,
        processing_ms: float,
        x: int,
        y: int,
    ) -> None:
        """Draw real-time metrics panel."""
        # Background
        cv2.rectangle(frame, (x - 5, y - 5), (x + 270, y + 95), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (x - 5, y - 5), (x + 270, y + 95), COLOR_CYAN, 1)

        metrics = [
            ("Anomaly Score", f"{anomaly_score:.3f}", self._score_color(anomaly_score)),
            ("Confidence", f"{confidence*100:.1f}%", self._conf_color(confidence)),
            ("Entropy H(p)", f"{entropy:.3f}", self._entropy_color(entropy)),
            ("FPS", f"{fps:.1f}", COLOR_CYAN),
            ("Latency", f"{processing_ms:.0f}ms", COLOR_CYAN),
        ]

        for i, (name, value, color) in enumerate(metrics):
            _put_text_with_bg(
                frame, f"{name}: {value}",
                (x + 5, y + 15 + i * 18),
                font_scale=0.4, color=color, bg_color=COLOR_DARK_BG,
                thickness=1,
            )

    def _draw_cost_panel(
        self,
        frame: np.ndarray,
        impact: FinancialImpact,
        x: int,
        y: int,
    ) -> None:
        """Draw cost estimation panel."""
        cv2.rectangle(frame, (x - 5, y - 5), (x + 270, y + 90), COLOR_DARK_BG, -1)
        cv2.rectangle(frame, (x - 5, y - 5), (x + 270, y + 90), COLOR_ORANGE, 1)

        _put_text_with_bg(
            frame, "COST ANALYSIS",
            (x + 5, y + 12), font_scale=0.45, color=COLOR_ORANGE, bg_color=COLOR_DARK_BG,
        )
        _put_text_with_bg(
            frame, f"Repair: ${impact.estimated_repair_cost_usd:.2f}",
            (x + 5, y + 30), font_scale=0.4, color=COLOR_WHITE, bg_color=COLOR_DARK_BG,
        )
        _put_text_with_bg(
            frame, f"Replace: ${impact.replacement_cost_usd:.2f}",
            (x + 5, y + 48), font_scale=0.4, color=COLOR_WHITE, bg_color=COLOR_DARK_BG,
        )

        rec_color = COLOR_RED if impact.economic_recommendation == EconomicRecommendation.SCRAP_AND_REPLACE else COLOR_GREEN
        _put_text_with_bg(
            frame, f">> {impact.economic_recommendation.value}",
            (x + 5, y + 68), font_scale=0.4, color=rec_color, bg_color=COLOR_DARK_BG,
        )

    def _draw_scanline(self, frame: np.ndarray, frame_id: int) -> None:
        """Subtle animated scanline effect for industrial feel."""
        h, w = frame.shape[:2]
        scan_y = (frame_id * 3) % h
        overlay = frame.copy()
        cv2.line(overlay, (0, scan_y), (w, scan_y), COLOR_CYAN, 1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    @staticmethod
    def _score_color(score: float) -> Tuple[int, int, int]:
        if score < 0.30:
            return COLOR_GREEN
        elif score < 0.65:
            return COLOR_YELLOW
        return COLOR_RED

    @staticmethod
    def _conf_color(conf: float) -> Tuple[int, int, int]:
        if conf >= 0.85:
            return COLOR_GREEN
        elif conf >= 0.50:
            return COLOR_YELLOW
        return COLOR_RED

    @staticmethod
    def _entropy_color(entropy: float) -> Tuple[int, int, int]:
        if entropy < 0.30:
            return COLOR_GREEN
        elif entropy < 0.60:
            return COLOR_YELLOW
        return COLOR_RED


# ---------------------------------------------------------------------------
# Full Frame Processor (main orchestrator)
# ---------------------------------------------------------------------------

class FrameProcessor:
    """Process frames through the full visual overlay pipeline.

    Usage:
        processor = FrameProcessor()
        annotated = processor.process(frame, inspection_result)
    """

    def __init__(self):
        self.box_drawer = BoundingBoxDrawer()
        self.grad_cam_overlay = GradCAMOverlay()
        self.hud = IndustrialHUD()

    def process(
        self,
        frame: np.ndarray,
        anomaly_score: float = 0.0,
        confidence: float = 0.0,
        entropy: float = 0.0,
        predicted_class: str = "normal",
        bounding_boxes: Optional[List[BoundingBox]] = None,
        xray_boxes: Optional[List[BoundingBox]] = None,
        product_status: ProductStatus = ProductStatus.REVIEW_REQUIRED,
        route: RouteDecision = RouteDecision.DISPATCH_TO_CONSUMER,
        fps: float = 0.0,
        processing_ms: float = 0.0,
        model_id: Optional[str] = None,
        financial_impact: Optional[FinancialImpact] = None,
        grad_cam_heatmap: Optional[np.ndarray] = None,
        enable_grad_cam: bool = False,
        frame_id: int = 0,
        tau_low: float = 0.30,
        tau_high: float = 0.65,
    ) -> np.ndarray:
        """Process a frame with full visual overlay pipeline.

        Args:
            frame: BGR input frame
            anomaly_score: S_rgb ∈ [0, 1]
            confidence: Model confidence
            entropy: Shannon entropy H(p)
            predicted_class: "normal" or defect type
            bounding_boxes: Detected defect regions
            xray_boxes: Internal X-ray defect regions
            product_status: Current product status
            route: Adaptive routing decision
            fps: Current frames per second
            processing_ms: Inference latency
            model_id: Active model identifier
            financial_impact: Cost estimation results
            grad_cam_heatmap: Grad-CAM heatmap (H, W) in [0, 1]
            enable_grad_cam: Whether to overlay Grad-CAM
            frame_id: Current frame number
            tau_low: Low anomaly threshold
            tau_high: High anomaly threshold
        """
        annotated = frame.copy()
        bounding_boxes = bounding_boxes or []
        xray_boxes = xray_boxes or []

        # Grad-CAM overlay (before boxes so boxes are on top)
        if enable_grad_cam and grad_cam_heatmap is not None:
            annotated = self.grad_cam_overlay.apply(annotated, grad_cam_heatmap)

        # Draw bounding boxes based on anomaly score and confidence
        is_normal = predicted_class.lower() in {"normal", "ok", "good", "pass"}

        if is_normal and anomaly_score < tau_low and confidence >= 0.85:
            # GREEN box for confident normal
            annotated = self.box_drawer.draw_normal_box(
                annotated, confidence=confidence,
            )
        elif anomaly_score >= tau_high or not is_normal:
            # RED boxes for defects
            for box in bounding_boxes:
                severity = box.severity.value if box.severity else "Medium"
                defect_type = box.defect_type or box.label or "Surface Anomaly"
                annotated = self.box_drawer.draw_defect_box(
                    annotated, box,
                    defect_type=defect_type,
                    severity=severity,
                )
        elif tau_low <= anomaly_score < tau_high:
            # YELLOW boxes for uncertain
            for box in bounding_boxes:
                annotated = self.box_drawer.draw_uncertain_box(annotated, box)
            if not bounding_boxes:
                # Draw uncertain box around whole frame
                h, w = frame.shape[:2]
                full_box = BoundingBox(x1=10, y1=10, x2=w-10, y2=h-10)
                annotated = self.box_drawer.draw_uncertain_box(annotated, full_box)

        # X-ray internal defect boxes (orange)
        for box in xray_boxes:
            defect_type = box.defect_type or box.label or "Internal Void"
            annotated = self.box_drawer.draw_xray_defect_box(
                annotated, box, defect_type=defect_type,
            )

        # Industrial HUD overlay
        annotated = self.hud.render(
            annotated,
            anomaly_score=anomaly_score,
            confidence=confidence,
            entropy=entropy,
            product_status=product_status,
            route=route,
            fps=fps,
            processing_ms=processing_ms,
            model_id=model_id,
            financial_impact=financial_impact,
            frame_id=frame_id,
        )

        return annotated

    def process_and_encode(self, **kwargs) -> str:
        """Process frame and return as base64-encoded JPEG."""
        annotated = self.process(**kwargs)
        return encode_frame_base64(annotated)


# Module-level singleton
frame_processor = FrameProcessor()
