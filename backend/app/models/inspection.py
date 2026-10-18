"""SQLAlchemy models for inspections and result details."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.app.core.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(String, primary_key=True, default=_uuid)
    inspection_type = Column(String, nullable=False, index=True)  # rgb, xray, adaptive
    original_filename = Column(String, nullable=True)
    stored_image_path = Column(String, nullable=True)  # controlled relative path
    image_sha256 = Column(String, nullable=True, index=True)
    predicted_class = Column(String, nullable=True)
    score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    model_id = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    status = Column(String, nullable=False, default="completed")
    error_details = Column(Text, nullable=True)
    processing_duration_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.timezone.utc))

    results = relationship(
        "InspectionResult",
        back_populates="inspection",
        cascade="all, delete-orphan",
        order_by="InspectionResult.created_at",
    )


class InspectionResult(Base):
    __tablename__ = "inspection_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    inspection_id = Column(String, ForeignKey("inspections.id", ondelete="CASCADE"), nullable=False, index=True)
    result_kind = Column(String, nullable=False)  # anomaly, xray, localization, adaptive, metadata
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: dt.datetime.now(dt.timezone.utc))

    inspection = relationship("Inspection", back_populates="results")
