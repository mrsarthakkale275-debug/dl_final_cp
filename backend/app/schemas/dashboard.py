"""Dashboard schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel


class DashboardSummary(BaseModel):
    total_inspections: int
    rgb_inspections: int
    xray_inspections: int
    adaptive_inspections: int
    normal_count: int
    defective_count: int
    unknown_count: int
    model_availability: dict[str, Any]
    recent_inspections: List[dict[str, Any]]


class ReportResponse(BaseModel):
    inspection_id: str
    generated_at: datetime
    report: dict[str, Any]
    stored_report_reference: Optional[str] = None
