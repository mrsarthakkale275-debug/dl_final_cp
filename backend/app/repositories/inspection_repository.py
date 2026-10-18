"""Repository layer for inspection persistence."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.app.models.inspection import Inspection, InspectionResult


class InspectionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        *,
        inspection_type: str,
        original_filename: Optional[str] = None,
        stored_image_path: Optional[str] = None,
        image_sha256: Optional[str] = None,
        status: str = "completed",
        error_details: Optional[str] = None,
        processing_duration_ms: Optional[float] = None,
        predicted_class: Optional[str] = None,
        score: Optional[float] = None,
        confidence: Optional[float] = None,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Inspection:
        inspection = Inspection(
            inspection_type=inspection_type,
            original_filename=original_filename,
            stored_image_path=stored_image_path,
            image_sha256=image_sha256,
            status=status,
            error_details=error_details,
            processing_duration_ms=processing_duration_ms,
            predicted_class=predicted_class,
            score=score,
            confidence=confidence,
            model_id=model_id,
            model_version=model_version,
        )
        self.db.add(inspection)
        self.db.commit()
        self.db.refresh(inspection)
        return inspection

    def add_result(self, inspection_id: str, result_kind: str, payload: dict) -> InspectionResult:
        result = InspectionResult(inspection_id=inspection_id, result_kind=result_kind, payload=payload)
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        return result

    def get(self, inspection_id: str) -> Optional[Inspection]:
        return self.db.query(Inspection).filter(Inspection.id == inspection_id).first()

    def list(self, skip: int = 0, limit: int = 50, inspection_type: Optional[str] = None) -> List[Inspection]:
        from sqlalchemy.orm import joinedload
        query = self.db.query(Inspection).options(joinedload(Inspection.results))
        if inspection_type:
            query = query.filter(Inspection.inspection_type == inspection_type)
        return query.order_by(Inspection.created_at.desc()).offset(skip).limit(limit).all()

    def count(self, inspection_type: Optional[str] = None) -> int:
        query = self.db.query(Inspection)
        if inspection_type:
            query = query.filter(Inspection.inspection_type == inspection_type)
        return query.count()

    def delete(self, inspection_id: str) -> bool:
        inspection = self.get(inspection_id)
        if not inspection:
            return False
        self.db.delete(inspection)
        self.db.commit()
        return True

    def find_by_image_hash(self, image_sha256: str) -> Optional[Inspection]:
        return self.db.query(Inspection).filter(Inspection.image_sha256 == image_sha256).first()
