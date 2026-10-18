"""Model and dataset schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ModelStatusDetail(BaseModel):
    name: str
    kind: str
    available: bool
    ready: bool
    status: str
    model_id: Optional[str] = None
    version: Optional[str] = None
    weights: List[str] = []
    message: str


class ModelStatusResponse(BaseModel):
    rgb: ModelStatusDetail
    xray: ModelStatusDetail
    overall_ready: bool


class VisaDatasetStatus(BaseModel):
    name: str
    parquet_files: List[str]
    train_records: int
    test_records: int
    normal_train_records: int
    total_records: int


class XRayDatasetStatus(BaseModel):
    name: str
    source_parquet_files: int
    xray_records: int
    prepared_images: int
    labels: Dict[str, int]
    rights_status: Dict[str, int]
    permission_cleared_records: int
    excluded_records: int


class DatasetStatusResponse(BaseModel):
    rgb: VisaDatasetStatus
    xray: XRayDatasetStatus
    model_trained_from_rgb: bool
    model_trained_from_xray: bool
    notes: List[str]
