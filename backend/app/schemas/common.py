"""Common response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class ErrorResponse(BaseModel):
    error: dict[str, Any]


class HealthResponse(BaseModel):
    api: str
    status: str
    readiness: str
    database: str
    models: dict[str, str]
    timestamp: datetime


class MessageResponse(BaseModel):
    message: str
