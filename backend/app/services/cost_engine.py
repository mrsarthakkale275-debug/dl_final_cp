"""Dynamic Cost Estimation Engine.

Implements the defect-to-cost calculator and repair-vs-scrap optimizer:

    C_total = C_base_material + (A_ratio × C_rework_rate)
              + (Severity_Multiplier × C_labor)
              + 𝟙_{xray} · C_xray_scan

Compare C_total against C_replacement.
If C_total > 0.75 × C_replacement → SCRAP_AND_REPLACE
Otherwise → REWORK_REPAIR

Also includes:
  - Bounding Box pixel-to-millimeter conversion for physical defect sizing
  - Multi-defect aggregation
  - Stage-aware cost accounting (surface vs internal defects)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from backend.app.schemas.inspection import (
    BoundingBox,
    CostMatrix,
    EconomicRecommendation,
    FinancialImpact,
    SeverityLevel,
)

logger = logging.getLogger(__name__)

# Default cost parameters
DEFAULT_COST_MATRIX = CostMatrix()


class CostEngine:
    """Mathematical defect-to-cost calculator and repair-vs-scrap optimizer."""

    def __init__(self, cost_matrix: Optional[CostMatrix] = None):
        self.costs = cost_matrix or DEFAULT_COST_MATRIX

    def compute_defect_area_ratio(
        self,
        bounding_boxes: List[BoundingBox],
        image_width: int,
        image_height: int,
    ) -> float:
        """Compute aggregate defect area ratio.

        A_ratio = Σ(Defect Box Area) / Object Area
        """
        total_image_area = image_width * image_height
        if total_image_area == 0:
            return 0.0

        total_defect_area = 0.0
        for box in bounding_boxes:
            box_width = max(0, box.x2 - box.x1)
            box_height = max(0, box.y2 - box.y1)
            total_defect_area += box_width * box_height

        return min(1.0, total_defect_area / total_image_area)

    def compute_defect_area_mm2(
        self,
        bounding_boxes: List[BoundingBox],
        px_to_mm_ratio: Optional[float] = None,
    ) -> float:
        """Estimate physical defect area in mm² using pixel-to-mm conversion.

        defect_area_mm2 = Σ(box_width_px × box_height_px) × (px_to_mm)²
        """
        ratio = px_to_mm_ratio or self.costs.px_to_mm_ratio
        total_area_px = 0.0

        for box in bounding_boxes:
            box_width = max(0, box.x2 - box.x1)
            box_height = max(0, box.y2 - box.y1)
            total_area_px += box_width * box_height

        return total_area_px * (ratio ** 2)

    def determine_severity(self, bounding_boxes: List[BoundingBox]) -> SeverityLevel:
        """Determine overall severity from all detected defects."""
        if not bounding_boxes:
            return SeverityLevel.LOW

        max_area_pct = max(box.area_pct for box in bounding_boxes)
        total_area_pct = sum(box.area_pct for box in bounding_boxes)

        if total_area_pct > 25.0 or max_area_pct > 15.0:
            return SeverityLevel.CRITICAL
        elif total_area_pct > 10.0 or max_area_pct > 8.0:
            return SeverityLevel.HIGH
        elif total_area_pct > 3.0 or max_area_pct > 3.0:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def get_severity_multiplier(self, severity: SeverityLevel) -> float:
        """Get the cost multiplier for a given severity level."""
        return self.costs.severity_multipliers.get(severity.value, 1.0)

    def estimate_repair_cost(
        self,
        bounding_boxes: List[BoundingBox],
        image_width: int,
        image_height: int,
        xray_triggered: bool = False,
        defect_type: Optional[str] = None,
    ) -> FinancialImpact:
        """Calculate the full financial impact of defective product.

        Formula:
            C_total = C_base_material
                      + (A_ratio × C_rework_rate)
                      + (Severity_Multiplier × C_labor)
                      + 𝟙_{xray} · C_xray_scan

        Decision:
            If C_total > 0.75 × C_replacement → SCRAP_AND_REPLACE
            Otherwise → REWORK_REPAIR
        """
        if not bounding_boxes:
            return FinancialImpact(
                economic_recommendation=EconomicRecommendation.NO_ACTION,
                replacement_cost_usd=self.costs.replacement_cost_usd,
                scrap_value_usd=self.costs.scrap_value_usd,
            )

        # Compute metrics
        area_ratio = self.compute_defect_area_ratio(
            bounding_boxes, image_width, image_height
        )
        area_mm2 = self.compute_defect_area_mm2(bounding_boxes)
        severity = self.determine_severity(bounding_boxes)
        severity_mult = self.get_severity_multiplier(severity)

        # Cost components
        c_base = self.costs.base_material_cost_usd
        c_rework = area_ratio * self.costs.rework_rate_per_unit_area
        c_labor = severity_mult * self.costs.labor_cost_per_hour_usd
        c_xray = self.costs.xray_scan_cost_usd if xray_triggered else 0.0

        c_total = c_base + c_rework + c_labor + c_xray
        c_replacement = self.costs.replacement_cost_usd
        c_scrap = self.costs.scrap_value_usd

        # Repair vs scrap decision
        repair_vs_replace = c_total / c_replacement if c_replacement > 0 else float('inf')

        if repair_vs_replace > 0.75:
            recommendation = EconomicRecommendation.SCRAP_AND_REPLACE
            net_loss = c_replacement - c_scrap
        else:
            recommendation = EconomicRecommendation.REWORK_REPAIR
            net_loss = c_total

        cost_breakdown = {
            "base_material": round(c_base, 2),
            "rework_cost": round(c_rework, 2),
            "labor_cost": round(c_labor, 2),
            "xray_scan": round(c_xray, 2),
            "total_repair": round(c_total, 2),
            "replacement": round(c_replacement, 2),
        }

        return FinancialImpact(
            estimated_repair_cost_usd=round(c_total, 2),
            scrap_value_usd=round(c_scrap, 2),
            xray_inspection_cost_usd=round(c_xray, 2),
            replacement_cost_usd=round(c_replacement, 2),
            net_loss_usd=round(net_loss, 2),
            defect_area_ratio=round(area_ratio, 4),
            defect_area_mm2=round(area_mm2, 2),
            severity_multiplier=severity_mult,
            economic_recommendation=recommendation,
            cost_breakdown=cost_breakdown,
            repair_vs_replace_ratio=round(repair_vs_replace, 4),
        )

    def batch_estimate(
        self,
        inspections: List[Dict[str, Any]],
    ) -> List[FinancialImpact]:
        """Estimate costs for a batch of inspections."""
        results = []
        for insp in inspections:
            boxes = [BoundingBox(**b) for b in insp.get("bounding_boxes", [])]
            impact = self.estimate_repair_cost(
                bounding_boxes=boxes,
                image_width=insp.get("image_width", 640),
                image_height=insp.get("image_height", 480),
                xray_triggered=insp.get("xray_triggered", False),
            )
            results.append(impact)
        return results


# ---------------------------------------------------------------------------
# ISO-9001 Quality Assurance Summary Generator
# ---------------------------------------------------------------------------

class QualitySummaryGenerator:
    """Generate 3-sentence ISO-9001 quality assurance summaries.

    Converts raw JSON telemetry into human-readable engineering certificates.
    """

    DEFECT_DESCRIPTIONS = {
        "surface_anomaly": "surface anomaly",
        "scratch": "linear scratch mark",
        "dent": "impact dent deformation",
        "crack": "thermal stress crack",
        "void": "internal void/porosity",
        "inclusion": "foreign material inclusion",
        "contamination": "surface contamination",
        "weld_distortion": "weld seam distortion",
    }

    SEVERITY_DESCRIPTIONS = {
        SeverityLevel.LOW: "minor",
        SeverityLevel.MEDIUM: "moderate",
        SeverityLevel.HIGH: "significant",
        SeverityLevel.CRITICAL: "critical",
    }

    def generate(
        self,
        anomaly_score: float,
        predicted_class: str,
        bounding_boxes: List[BoundingBox],
        severity: SeverityLevel,
        xray_triggered: bool = False,
        xray_defects_found: int = 0,
        defect_area_mm2: Optional[float] = None,
        economic_recommendation: Optional[EconomicRecommendation] = None,
    ) -> str:
        """Generate a 3-sentence ISO-9001 QA summary."""
        num_defects = len(bounding_boxes)
        severity_desc = self.SEVERITY_DESCRIPTIONS.get(severity, "unknown")

        if (predicted_class or "").lower() in {"normal", "ok", "good", "pass"}:
            return (
                f"Visual inspection completed with anomaly score {anomaly_score:.3f}. "
                f"No surface defects detected; product meets quality acceptance criteria. "
                f"Part is cleared for consumer dispatch per ISO 9001:2015 §8.6."
            )

        # Defect type description
        defect_types = set()
        for box in bounding_boxes:
            if box.defect_type:
                defect_types.add(self.DEFECT_DESCRIPTIONS.get(box.defect_type, box.defect_type))
            elif box.label:
                defect_types.add(self.DEFECT_DESCRIPTIONS.get(box.label, box.label))
        defect_str = ", ".join(defect_types) if defect_types else "unclassified defect"

        # Sentence 1: What was detected
        sentence1 = (
            f"{severity_desc.title()} {defect_str} detected on inspection surface "
            f"({num_defects} region{'s' if num_defects != 1 else ''}, "
            f"anomaly score: {anomaly_score:.3f})."
        )

        # Sentence 2: X-ray findings
        if xray_triggered and xray_defects_found > 0:
            area_str = f", estimated {defect_area_mm2:.1f}mm² affected area" if defect_area_mm2 else ""
            sentence2 = (
                f"Secondary X-ray radiographic scan confirms {xray_defects_found} "
                f"subsurface anomal{'ies' if xray_defects_found != 1 else 'y'}{area_str}."
            )
        elif xray_triggered:
            sentence2 = (
                "Secondary X-ray scan was performed; no additional internal defects confirmed."
            )
        else:
            sentence2 = (
                "X-ray subsurface inspection was not triggered based on adaptive routing policy."
            )

        # Sentence 3: Recommendation
        if economic_recommendation == EconomicRecommendation.SCRAP_AND_REPLACE:
            sentence3 = (
                "Economic analysis recommends scrap and replacement; "
                "repair cost exceeds 75% of replacement value per ISO 9001:2015 §8.7."
            )
        elif economic_recommendation == EconomicRecommendation.REWORK_REPAIR:
            sentence3 = (
                "Part is routed for rework/repair; estimated repair cost is within "
                "economic threshold per ISO 9001:2015 §8.7 nonconforming output procedure."
            )
        else:
            sentence3 = (
                "Part requires human quality engineer review before disposition "
                "per ISO 9001:2015 §8.7 nonconforming output controls."
            )

        return f"{sentence1} {sentence2} {sentence3}"


# Module-level singleton
cost_engine = CostEngine()
summary_generator = QualitySummaryGenerator()
