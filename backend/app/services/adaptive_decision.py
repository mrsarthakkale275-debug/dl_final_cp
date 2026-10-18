"""Adaptive Gating Router — the "Adaptive Multimodal" core.

Implements entropy-based and risk-based routing policy:
  Route A (DISPATCH_TO_CONSUMER): S_rgb < τ_low AND high confidence AND low entropy
  Route B (TRIGGER_XRAY_INSPECTION): Uncertain OR structural defect OR safety-critical
  Route C (IMMEDIATE_SCRAP): Catastrophic surface defect (area ratio > 25%)

Uses Shannon Entropy H(p) = -Σ p_i log(p_i) to determine model uncertainty.
When entropy exceeds threshold τ_entropy, the model "knows what it doesn't know"
and automatically routes the part to the X-ray scanner.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.schemas.inspection import (
    AdaptiveDecision,
    BoundingBox,
    FusionResult,
    ProductStatus,
    RouteDecision,
    SeverityLevel,
)


# Defect types that correlate with internal structural damage
STRUCTURAL_DEFECT_TYPES = {
    "crack", "thermal_stress_crack", "dent", "weld_distortion",
    "fracture", "delamination", "bulge",
}

# Object categories flagged as safety-critical
SAFETY_CRITICAL_CATEGORIES = {
    "pcb", "circuit_board", "weld", "pressure_vessel",
    "bearing", "turbine_blade", "brake_disc",
}


class AdaptiveGatingRouter:
    """Evaluate RGB/X-ray results and route products through the inspection pipeline.

    Thresholds:
      τ_low  = 0.30 → below this + high confidence = DISPATCH_TO_CONSUMER
      τ_high = 0.65 → above this = definite defect
      τ_entropy = 0.50 → normalized entropy above this = uncertain → trigger X-ray
      catastrophic_area_ratio = 0.25 → above this = IMMEDIATE_SCRAP
    """

    def __init__(
        self,
        tau_low: float = 0.30,
        tau_high: float = 0.65,
        confidence_threshold: float = 0.85,
        entropy_threshold: float = 0.50,
        catastrophic_area_ratio: float = 0.25,
    ) -> None:
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.catastrophic_area_ratio = catastrophic_area_ratio

    def evaluate(
        self,
        rgb_result: Optional[Dict[str, Any]] = None,
        xray_result: Optional[Dict[str, Any]] = None,
        fusion_result: Optional[Dict[str, Any]] = None,
        object_category: Optional[str] = None,
    ) -> AdaptiveDecision:
        """Evaluate inspection results and determine routing decision."""
        explanations: List[str] = []
        rgb_available = self._is_available(rgb_result)
        xray_available = self._is_available(xray_result)

        # --- No RGB data ---
        if not rgb_available:
            return AdaptiveDecision(
                route=RouteDecision.TRIGGER_XRAY_INSPECTION,
                recommendation="human_review_required",
                reason="RGB inspection data unavailable.",
                explanations=["No RGB result available for routing decision."],
                rgb_available=False,
                xray_available=xray_available,
                requires_human_review=True,
                policy=self._policy_dict(),
            )

        rgb = rgb_result or {}
        anomaly_score = float(rgb.get("anomaly_score", 0.0))
        confidence = float(rgb.get("confidence", 0.0))
        entropy = float(rgb.get("entropy", 0.0))
        predicted_class = (rgb.get("predicted_class") or "").lower()
        bounding_boxes = rgb.get("bounding_boxes", [])

        # Compute total defect area ratio
        total_area_pct = sum(
            b.get("area_pct", 0.0) if isinstance(b, dict) else b.area_pct
            for b in bounding_boxes
        )
        total_area_ratio = total_area_pct / 100.0

        is_normal = predicted_class in {"normal", "ok", "good", "pass"}
        is_defective = predicted_class in {"defective", "defect", "anomaly", "abnormal"}
        is_structural = any(
            (b.get("defect_type") if isinstance(b, dict) else b.defect_type or "").lower()
            in STRUCTURAL_DEFECT_TYPES
            for b in bounding_boxes
        ) if bounding_boxes else False
        is_safety_critical = (object_category or "").lower() in SAFETY_CRITICAL_CATEGORIES

        # --- Route C: IMMEDIATE_SCRAP (catastrophic surface defect) ---
        if total_area_ratio > self.catastrophic_area_ratio:
            explanations.append(
                f"Defect area ratio {total_area_ratio:.2%} exceeds catastrophic "
                f"threshold {self.catastrophic_area_ratio:.0%}."
            )
            explanations.append("X-ray scan skipped to save cost; product is unrepairable.")
            return AdaptiveDecision(
                route=RouteDecision.IMMEDIATE_SCRAP,
                recommendation="immediate_scrap",
                reason="Catastrophic surface defect detected. Product is irreparable.",
                explanations=explanations,
                rgb_available=True,
                xray_available=xray_available,
                xray_triggered=False,
                requires_human_review=False,
                entropy_value=entropy,
                entropy_threshold=self.entropy_threshold,
                policy=self._policy_dict(),
            )

        # --- Route A: DISPATCH_TO_CONSUMER ---
        if (anomaly_score < self.tau_low
                and confidence >= self.confidence_threshold
                and entropy < self.entropy_threshold
                and is_normal
                and not is_safety_critical):
            explanations.append(
                f"Anomaly score {anomaly_score:.3f} < τ_low={self.tau_low}."
            )
            explanations.append(
                f"Model confidence {confidence:.3f} ≥ threshold={self.confidence_threshold}."
            )
            explanations.append(
                f"Shannon entropy H(p)={entropy:.3f} < τ_entropy={self.entropy_threshold}, "
                "indicating high model certainty."
            )
            explanations.append("Product passes RGB-only inspection without X-ray.")

            product_status = ProductStatus.GOOD_FOR_CONSUMER
            return AdaptiveDecision(
                route=RouteDecision.DISPATCH_TO_CONSUMER,
                recommendation="dispatch_to_consumer",
                reason="Product passes all RGB quality thresholds with high confidence.",
                explanations=explanations,
                rgb_available=True,
                xray_available=xray_available,
                xray_triggered=False,
                requires_human_review=False,
                entropy_value=entropy,
                entropy_threshold=self.entropy_threshold,
                policy=self._policy_dict(),
            )

        # --- Route B: TRIGGER_XRAY_INSPECTION ---
        trigger_reasons = []

        # Condition 1: RGB model is uncertain
        if self.tau_low <= anomaly_score < self.tau_high:
            trigger_reasons.append(
                f"Anomaly score {anomaly_score:.3f} is in uncertain range "
                f"[{self.tau_low}, {self.tau_high})."
            )

        # Entropy-based trigger
        if entropy >= self.entropy_threshold:
            trigger_reasons.append(
                f"Shannon entropy H(p)={entropy:.3f} ≥ τ_entropy={self.entropy_threshold}. "
                "Model is uncertain, triggering X-ray for confirmation."
            )

        # Condition 2: Structural surface defect
        if is_structural:
            trigger_reasons.append(
                "Surface defect type correlates with potential internal structural damage."
            )

        # Condition 3: Safety-critical category
        if is_safety_critical:
            trigger_reasons.append(
                f"Object category '{object_category}' is flagged as safety-critical. "
                "Mandatory X-ray inspection required."
            )

        # Low confidence
        if confidence < self.confidence_threshold and confidence > 0:
            trigger_reasons.append(
                f"Model confidence {confidence:.3f} below threshold {self.confidence_threshold}."
            )

        # Defective prediction with score >= tau_high
        if is_defective and anomaly_score >= self.tau_high:
            trigger_reasons.append(
                f"Defect detected with anomaly score {anomaly_score:.3f} ≥ τ_high={self.tau_high}."
            )

        if trigger_reasons:
            explanations.extend(trigger_reasons)

            # Check if X-ray results are already available
            if xray_available:
                return self._evaluate_with_xray(
                    rgb, xray_result or {}, fusion_result,
                    anomaly_score, entropy, explanations,
                )

            return AdaptiveDecision(
                route=RouteDecision.TRIGGER_XRAY_INSPECTION,
                recommendation="trigger_xray_inspection",
                reason="RGB inspection triggered X-ray follow-up.",
                explanations=explanations,
                rgb_available=True,
                xray_available=False,
                xray_triggered=True,
                requires_human_review=False,
                entropy_value=entropy,
                entropy_threshold=self.entropy_threshold,
                policy=self._policy_dict(),
            )

        # Fallback: if no clear route, require review
        explanations.append("Result could not be confidently classified into any route.")
        return AdaptiveDecision(
            route=RouteDecision.TRIGGER_XRAY_INSPECTION,
            recommendation="human_review_required",
            reason="RGB result is ambiguous; additional inspection recommended.",
            explanations=explanations,
            rgb_available=True,
            xray_available=xray_available,
            xray_triggered=True,
            requires_human_review=True,
            entropy_value=entropy,
            entropy_threshold=self.entropy_threshold,
            policy=self._policy_dict(),
        )

    def _evaluate_with_xray(
        self,
        rgb: Dict[str, Any],
        xray: Dict[str, Any],
        fusion_result: Optional[Dict[str, Any]],
        anomaly_score: float,
        entropy: float,
        explanations: List[str],
    ) -> AdaptiveDecision:
        """Evaluate combined RGB + X-ray results with optional fusion."""
        xray_class = (xray.get("predicted_class") or "").lower()
        xray_score = float(xray.get("anomaly_score", xray.get("score", 0.0)))
        xray_defects = int(xray.get("internal_defects_found", 0))

        xray_defective = xray_class in {"defective", "defect", "void", "crack", "inclusion"}
        xray_normal = xray_class in {"normal", "ok", "good"}

        fusion = None
        if fusion_result:
            fusion = FusionResult(
                fusion_score=fusion_result.get("fusion_score", 0.0),
                alpha=fusion_result.get("alpha", 0.5),
                rgb_feature_norm=fusion_result.get("rgb_feature_norm"),
                xray_feature_norm=fusion_result.get("xray_feature_norm"),
            )

        if xray_defective or xray_defects > 0:
            explanations.append(
                f"X-ray confirmed internal defect(s): {xray_class} "
                f"(score: {xray_score:.3f}, {xray_defects} regions found)."
            )
            # Check severity
            xray_boxes = xray.get("bounding_boxes", [])
            xray_area_pct = sum(
                b.get("area_pct", 0.0) if isinstance(b, dict) else b.area_pct
                for b in xray_boxes
            )

            if xray_area_pct > 25:
                return AdaptiveDecision(
                    route=RouteDecision.IMMEDIATE_SCRAP,
                    recommendation="immediate_scrap",
                    reason="X-ray confirmed catastrophic internal defect.",
                    explanations=explanations,
                    rgb_available=True,
                    xray_available=True,
                    xray_triggered=True,
                    requires_human_review=False,
                    entropy_value=entropy,
                    entropy_threshold=self.entropy_threshold,
                    policy=self._policy_dict(),
                    fusion=fusion,
                )

            return AdaptiveDecision(
                route=RouteDecision.TRIGGER_XRAY_INSPECTION,
                recommendation="defective_rework_or_scrap",
                reason="X-ray confirmed internal defect requiring assessment.",
                explanations=explanations,
                rgb_available=True,
                xray_available=True,
                xray_triggered=True,
                requires_human_review=True,
                entropy_value=entropy,
                entropy_threshold=self.entropy_threshold,
                policy=self._policy_dict(),
                fusion=fusion,
            )

        if xray_normal:
            explanations.append(
                "X-ray scan found no internal defects. RGB surface concern may be cosmetic."
            )
            return AdaptiveDecision(
                route=RouteDecision.DISPATCH_TO_CONSUMER,
                recommendation="dispatch_after_xray_clear",
                reason="X-ray confirmed no internal defects.",
                explanations=explanations,
                rgb_available=True,
                xray_available=True,
                xray_triggered=True,
                requires_human_review=False,
                entropy_value=entropy,
                entropy_threshold=self.entropy_threshold,
                policy=self._policy_dict(),
                fusion=fusion,
            )

        explanations.append("X-ray result was inconclusive; human review required.")
        return AdaptiveDecision(
            route=RouteDecision.TRIGGER_XRAY_INSPECTION,
            recommendation="human_review_required",
            reason="Combined RGB + X-ray analysis is inconclusive.",
            explanations=explanations,
            rgb_available=True,
            xray_available=True,
            xray_triggered=True,
            requires_human_review=True,
            entropy_value=entropy,
            entropy_threshold=self.entropy_threshold,
            policy=self._policy_dict(),
            fusion=fusion,
        )

    def _is_available(self, result: Optional[Dict[str, Any]]) -> bool:
        if not result:
            return False
        if result.get("status") in {"model_unavailable", "error"}:
            return False
        return (
            result.get("predicted_class") is not None
            or result.get("anomaly_score") is not None
        )

    def _policy_dict(self) -> Dict[str, Any]:
        return {
            "tau_low": self.tau_low,
            "tau_high": self.tau_high,
            "confidence_threshold": self.confidence_threshold,
            "entropy_threshold": self.entropy_threshold,
            "catastrophic_area_ratio": self.catastrophic_area_ratio,
        }


class AdaptiveDecisionService:
    def __init__(
        self,
        rgb_confidence_threshold: float = 0.8,
        rgb_uncertainty_threshold: float = 0.5,
        tau_low: float = 0.30,
        tau_high: float = 0.65,
        **kwargs,
    ) -> None:
        self.rgb_confidence_threshold = rgb_confidence_threshold
        self.rgb_uncertainty_threshold = rgb_uncertainty_threshold
        self.router = AdaptiveGatingRouter(
            tau_low=tau_low,
            tau_high=tau_high,
            confidence_threshold=rgb_confidence_threshold,
            entropy_threshold=rgb_uncertainty_threshold,
        )

    def evaluate(
        self,
        rgb_result: Optional[Dict[str, Any]] = None,
        xray_result: Optional[Dict[str, Any]] = None,
        fusion_result: Optional[Dict[str, Any]] = None,
        object_category: Optional[str] = None,
    ) -> AdaptiveDecision:
        if not rgb_result or rgb_result.get("status") in {"model_unavailable", "error"}:
            return AdaptiveDecision(
                route=RouteDecision.TRIGGER_XRAY_INSPECTION,
                recommendation="human_review_required",
                reason="RGB inspection data unavailable.",
                explanations=["No RGB result available for routing decision."],
                rgb_available=False,
                xray_available=bool(xray_result),
                requires_human_review=True,
                policy=self.router._policy_dict(),
            )

        predicted_class = (rgb_result.get("predicted_class") or "").lower()
        confidence = float(rgb_result.get("confidence") or 0.0)

        if predicted_class in {"normal", "ok", "pass"}:
            if confidence >= self.rgb_confidence_threshold or "localization" in rgb_result:
                return AdaptiveDecision(
                    route=RouteDecision.DISPATCH_TO_CONSUMER,
                    recommendation="rgb_only_result",
                    reason="RGB inspection completed with acceptable normal prediction.",
                    explanations=["Product passes RGB inspection without X-ray."],
                    rgb_available=True,
                    xray_available=bool(xray_result),
                    xray_triggered=False,
                    requires_human_review=False,
                    policy=self.router._policy_dict(),
                )

        if predicted_class in {"defective", "defect", "anomaly", "abnormal"}:
            return AdaptiveDecision(
                route=RouteDecision.TRIGGER_XRAY_INSPECTION,
                recommendation="request_xray_inspection",
                reason="RGB inspection indicates defect; X-ray inspection requested.",
                explanations=["Defect detected in RGB inspection."],
                rgb_available=True,
                xray_available=bool(xray_result),
                xray_triggered=True,
                requires_human_review=True,
                policy=self.router._policy_dict(),
            )

        return self.router.evaluate(rgb_result, xray_result, fusion_result, object_category)
