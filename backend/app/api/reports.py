"""Report endpoint with JSON, Markdown, and HTML export."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.services.report_service import ReportService

router = APIRouter(tags=["reports"])
report_service = ReportService()


@router.get("/inspections/{inspection_id}/report", summary="Generate inspection report")
def get_report(
    inspection_id: str,
    format: Literal["json", "markdown", "html"] = Query("json", alias="format"),
    db: Session = Depends(get_db),
    response: Response = None,
):
    """Generate a QC report in JSON, Markdown, or HTML format.

    The report includes RGB metrics, X-ray results, adaptive routing,
    financial impact analysis, and ISO-9001 quality summary.
    """
    if format == "html":
        html = report_service.generate_html(db, inspection_id)
        if html is None:
            raise HTTPException(status_code=404, detail="Inspection not found.")
        return HTMLResponse(
            content=html,
            headers={"Content-Disposition": f'attachment; filename="{inspection_id}.html"'},
        )

    if format == "markdown":
        markdown = report_service.generate_markdown(db, inspection_id)
        if markdown is None:
            raise HTTPException(status_code=404, detail="Inspection not found.")
        return PlainTextResponse(
            content=markdown,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{inspection_id}.md"'},
        )

    report = report_service.generate(db, inspection_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Inspection not found.")
    if response is not None:
        response.headers["Content-Disposition"] = f'attachment; filename="{inspection_id}.json"'
    from backend.app.schemas.dashboard import ReportResponse
    from datetime import datetime

    stored_ref = report.pop("stored_report_reference", None)
    return ReportResponse(
        inspection_id=inspection_id,
        generated_at=datetime.fromisoformat(report["generated_at"]),
        report=report,
        stored_report_reference=stored_ref,
    )
