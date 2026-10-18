"""Report generation from stored inspection records.

Generates JSON, Markdown, and HTML reports with financial impact,
Grad-CAM references, and ISO-9001 quality summaries.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.app.core.config import settings
from backend.app.core.logging_config import get_logger
from backend.app.repositories.inspection_repository import InspectionRepository

logger = get_logger(__name__)


def _load_inspection(db, inspection_id: str):
    repo = InspectionRepository(db)
    return repo.get(inspection_id)


def _result_payloads(inspection) -> list[dict[str, Any]]:
    return [
        {"result_kind": result.result_kind, "payload": result.payload}
        for result in inspection.results
    ]


def _extract_result(results: list, kind: str) -> Optional[dict]:
    """Extract a specific result kind from the results list."""
    for r in results:
        if r["result_kind"] == kind and isinstance(r["payload"], dict):
            return r["payload"]
    return None


class ReportService:
    def generate(self, db, inspection_id: str) -> Optional[Dict[str, Any]]:
        inspection = _load_inspection(db, inspection_id)

        results = _result_payloads(inspection)
        rgb_result = _extract_result(results, "rgb")
        xray_result = _extract_result(results, "xray")
        adaptive_result = _extract_result(results, "adaptive")
        financial_result = _extract_result(results, "financial")
        qc_result = _extract_result(results, "qc_report")

        report = {
            "inspection_id": inspection.id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inspection_type": inspection.inspection_type,
            "status": inspection.status,
            "original_filename": inspection.original_filename,
            "stored_image_reference": inspection.stored_image_path,
            "processing_duration_ms": inspection.processing_duration_ms,
            "created_at": inspection.created_at.isoformat() if inspection.created_at else None,
            "model": {
                "model_id": inspection.model_id,
                "model_version": inspection.model_version,
            },
            "prediction": {
                "predicted_class": inspection.predicted_class or (rgb_result or {}).get("predicted_class"),
                "detected_object": (rgb_result or {}).get("detected_object"),
                "condition_label": (rgb_result or {}).get("condition_label"),
                "is_defective": (rgb_result or {}).get("is_defective"),
                "score": inspection.score,
                "threshold": (rgb_result or {}).get("threshold") or (rgb_result or {}).get("localization", {}).get("image_threshold"),
                "confidence": inspection.confidence,
                "note": "This is a model prediction or anomaly score, not a ground-truth label.",
            },
            "rgb_metrics": rgb_result,
            "xray_triggered": bool(adaptive_result and adaptive_result.get("xray_triggered")),
            "xray_metrics": xray_result,
            "adaptive_decision": adaptive_result,
            "financial_impact": financial_result,
            "qc_report": qc_result,
            "ground_truth": {
                "available": False,
                "message": "No ground-truth label is stored with this inspection record.",
            },
            "results": results,
            "limitations": {
                "message": (
                    "Inspection, anomaly, X-ray, and adaptive recommendations are stored separately. "
                    "Recommendations are policy-level and do not prove that a PCB is defect-free. "
                    "RGB and X-ray images are not confirmed paired observations."
                )
            },
            "error_details": inspection.error_details,
        }

        relative = f"{inspection.id}.json"
        target = settings.report_storage_dir / relative
        try:
            target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            report["stored_report_reference"] = relative
        except OSError as exc:
            logger.warning("Could not store report file: %s", exc)
            report["stored_report_reference"] = None
        return report

    def generate_markdown(self, db, inspection_id: str) -> Optional[str]:
        report = self.generate(db, inspection_id)
        if report is None:
            return None
        inspection = _load_inspection(db, inspection_id)

        lines = [
            f"# Inspection Report: {report['inspection_id']}",
            "",
            f"- Generated at: {report['generated_at']}",
            f"- Inspection type: {report['inspection_type']}",
            f"- Status: {report['status']}",
            f"- Original filename: {report['original_filename']}",
            f"- Stored image reference: {report['stored_image_reference']}",
            f"- Created at: {report['created_at']}",
            f"- Processing duration (ms): {report['processing_duration_ms']}",
            "",
            "## Model",
            f"- Model ID: {report['model']['model_id']}",
            f"- Model version: {report['model']['model_version']}",
            "",
            "## Prediction / Assessment",
            f"- Subject Category: {report['prediction'].get('detected_object') or 'Printed Circuit Board (PCB)'}",
            f"- Condition: {report['prediction'].get('condition_label') or (report['prediction']['predicted_class'] or 'Normal (PASS)')}",
            f"- Predicted class: {report['prediction']['predicted_class'] or report['prediction'].get('condition_label')}",
            f"- Score: {report['prediction']['score']}",
            f"- Threshold: {report['prediction'].get('threshold', 'N/A')}",
            f"- Confidence: {report['prediction']['confidence']}",
            f"- Note: {report['prediction']['note']}",
        ]

        # RGB Metrics
        rgb = report.get("rgb_metrics")
        if rgb:
            lines.extend([
                "",
                "## RGB Inspection Metrics",
                f"- Anomaly Score (S_rgb): {rgb.get('anomaly_score', 'N/A')}",
                f"- Confidence: {rgb.get('confidence', 'N/A')}",
                f"- Entropy H(p): {rgb.get('entropy', 'N/A')}",
                f"- Predicted Class: {rgb.get('predicted_class', 'N/A')}",
            ])
            boxes = rgb.get("bounding_boxes", [])
            if boxes:
                lines.append(f"- Bounding Boxes: {len(boxes)} defect region(s) detected")
                for i, box in enumerate(boxes):
                    if isinstance(box, dict):
                        lines.append(
                            f"  - Box {i+1}: [{box.get('x1',0):.0f}, {box.get('y1',0):.0f}, "
                            f"{box.get('x2',0):.0f}, {box.get('y2',0):.0f}] "
                            f"| {box.get('label','defect')} | Area: {box.get('area_pct',0):.1f}%"
                        )

        # X-ray Metrics
        if report.get("xray_triggered"):
            lines.extend(["", "## X-Ray Inspection (Stage 2)"])
            xray = report.get("xray_metrics")
            if xray:
                lines.extend([
                    f"- Predicted Class: {xray.get('predicted_class', 'N/A')}",
                    f"- Score: {xray.get('score', xray.get('anomaly_score', 'N/A'))}",
                    f"- Internal Defects Found: {xray.get('internal_defects_found', 0)}",
                ])
            else:
                lines.append("- X-ray was triggered but no X-ray image was provided.")

        # Adaptive Decision
        adaptive = report.get("adaptive_decision")
        if adaptive:
            lines.extend([
                "",
                "## Adaptive Routing Decision",
                f"- Route: {adaptive.get('route', 'N/A')}",
                f"- Recommendation: {adaptive.get('recommendation', 'N/A')}",
                f"- Reason: {adaptive.get('reason', 'N/A')}",
                f"- Entropy Value: {adaptive.get('entropy_value', 'N/A')}",
                f"- X-ray Triggered: {adaptive.get('xray_triggered', False)}",
                f"- Human Review Required: {adaptive.get('requires_human_review', False)}",
            ])
            for exp in adaptive.get("explanations", []):
                lines.append(f"  - {exp}")

        # Financial Impact
        fin = report.get("financial_impact")
        if fin:
            lines.extend([
                "",
                "## Financial Impact Analysis",
                f"- Estimated Repair Cost: ${fin.get('estimated_repair_cost_usd', 0):.2f}",
                f"- Replacement Cost: ${fin.get('replacement_cost_usd', 0):.2f}",
                f"- X-ray Scan Cost: ${fin.get('xray_inspection_cost_usd', 0):.2f}",
                f"- Scrap Value: ${fin.get('scrap_value_usd', 0):.2f}",
                f"- Net Loss: ${fin.get('net_loss_usd', 0):.2f}",
                f"- Defect Area Ratio: {fin.get('defect_area_ratio', 0):.4f}",
                f"- Defect Area (mm²): {fin.get('defect_area_mm2', 'N/A')}",
                f"- **Economic Recommendation: {fin.get('economic_recommendation', 'N/A')}**",
            ])

        # QC Report / ISO Summary
        qc = report.get("qc_report")
        if qc and qc.get("iso_9001_summary"):
            lines.extend([
                "",
                "## ISO-9001 Quality Assurance Summary",
                "",
                qc["iso_9001_summary"],
            ])

        lines.extend([
            "",
            "## Ground Truth",
            "- Available: false",
            f"- Message: {report['ground_truth']['message']}",
            "",
            "## Stored Results",
        ])
        for result in report["results"]:
            lines.append(f"- `{result['result_kind']}`: `{json.dumps(result['payload'], default=str)[:200]}...`")
        lines.extend([
            "",
            "## Limitations",
            f"- {report['limitations']['message']}",
        ])
        if report.get("error_details"):
            lines.extend(["", f"- Error details: {report['error_details']}"])

        lines.append("")
        lines.append("---")
        lines.append("*Generated by AdaptiveInspect AI v2.0.0*")

        md_content = "\n".join(lines) + "\n"

        relative = f"{inspection.id}.md"
        target = settings.report_storage_dir / relative
        try:
            target.write_text(md_content, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not store markdown report file: %s", exc)
        return md_content

    def generate_html(self, db, inspection_id: str) -> Optional[str]:
        """Generate an HTML report from inspection data."""
        report = self.generate(db, inspection_id)
        if report is None:
            return None

        # Build simple HTML report
        fin = report.get("financial_impact") or {}
        adaptive = report.get("adaptive_decision") or {}
        rgb = report.get("rgb_metrics") or {}
        qc = report.get("qc_report") or {}

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>QC Report: {report['inspection_id']}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 2rem; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 0.5rem; }}
        h2 {{ color: #283593; margin-top: 1.5rem; }}
        .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; color: white; font-weight: bold; }}
        .pass {{ background: #00c853; }}
        .fail {{ background: #ff1744; }}
        .hold {{ background: #ff9100; }}
        .review {{ background: #2979ff; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 8px 12px; border: 1px solid #e0e0e0; text-align: left; }}
        th {{ background: #e8eaf6; }}
        .cost-positive {{ color: #00c853; font-weight: bold; }}
        .cost-negative {{ color: #ff1744; font-weight: bold; }}
        .summary {{ background: #e8eaf6; padding: 1rem; border-radius: 4px; margin: 1rem 0; font-style: italic; }}
        footer {{ text-align: center; margin-top: 2rem; color: #999; font-size: 0.85rem; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔍 Quality Control Inspection Report</h1>
    <p><strong>Inspection ID:</strong> {report['inspection_id']}</p>
    <p><strong>Generated:</strong> {report['generated_at']}</p>
    <p><strong>Type:</strong> {report['inspection_type'].upper()}</p>

    <h2>Product Status</h2>
    <p><span class="status-badge {'pass' if adaptive.get('route') == 'DISPATCH_TO_CONSUMER' else 'fail' if 'SCRAP' in str(adaptive.get('route', '')) else 'hold'}">{adaptive.get('route', 'PENDING')}</span></p>
    <p>{adaptive.get('reason', 'Awaiting analysis.')}</p>

    <h2>RGB Surface Inspection</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Anomaly Score (S_rgb)</td><td>{rgb.get('anomaly_score', 'N/A')}</td></tr>
        <tr><td>Confidence</td><td>{rgb.get('confidence', 'N/A')}</td></tr>
        <tr><td>Entropy H(p)</td><td>{rgb.get('entropy', 'N/A')}</td></tr>
        <tr><td>Predicted Class</td><td>{rgb.get('predicted_class', 'N/A')}</td></tr>
        <tr><td>Defect Regions</td><td>{len(rgb.get('bounding_boxes', []))}</td></tr>
    </table>

    <h2>Financial Impact</h2>
    <table>
        <tr><th>Cost Component</th><th>Amount (USD)</th></tr>
        <tr><td>Estimated Repair Cost</td><td>${fin.get('estimated_repair_cost_usd', 0):.2f}</td></tr>
        <tr><td>Replacement Cost</td><td>${fin.get('replacement_cost_usd', 0):.2f}</td></tr>
        <tr><td>X-ray Scan Cost</td><td>${fin.get('xray_inspection_cost_usd', 0):.2f}</td></tr>
        <tr><td>Scrap Value</td><td>${fin.get('scrap_value_usd', 0):.2f}</td></tr>
        <tr><td>Net Loss</td><td class="cost-negative">${fin.get('net_loss_usd', 0):.2f}</td></tr>
        <tr><td><strong>Recommendation</strong></td><td><strong>{fin.get('economic_recommendation', 'N/A')}</strong></td></tr>
    </table>

    {f'<h2>ISO-9001 Quality Summary</h2><div class="summary">{qc.get("iso_9001_summary", "")}</div>' if qc.get("iso_9001_summary") else ""}

    <footer>
        <p>AdaptiveInspect AI v2.0.0 — Adaptive Multimodal Deep Learning for Industrial QC</p>
        <p>This report is generated automatically and does not constitute a final quality verdict.</p>
    </footer>
</div>
</body>
</html>"""

        relative = f"{report['inspection_id']}.html"
        target = settings.report_storage_dir / relative
        try:
            target.write_text(html, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not store HTML report: %s", exc)
        return html
