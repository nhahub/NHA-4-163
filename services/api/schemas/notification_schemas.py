"""Request/response schemas for the notification API (Tier 4)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class NotificationResponse(BaseModel):
    """A notification returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    organization_id: uuid.UUID | None = None
    notification_type: str
    severity: str
    status: str
    title: str
    message: str
    risk_score: float | None = None
    risk_tier: str | None = None
    threshold: float | None = None
    acknowledged: bool = False
    read_at: datetime | None = None
    created_at: datetime | None = None


class NotificationCreate(BaseModel):
    """Manually create a notification for a patient."""

    model_config = ConfigDict(str_strip_whitespace=True)

    notification_type: str = Field(
        default="screening_reminder",
        pattern=r"^(risk_threshold_crossed|risk_increased|family_update|screening_reminder)$",
    )
    severity: str = Field(default="info", pattern=r"^(info|warning|critical)$")
    title: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1, max_length=1000)


class NotificationSummary(BaseModel):
    """Aggregate notification counts."""

    total: int
    unread: int
    by_severity: dict[str, int]
    by_type: dict[str, int]


class EvaluateResponse(BaseModel):
    """Result of evaluating a patient's latest risk for notifications."""

    patient_id: uuid.UUID
    created: list[NotificationResponse] = []
    message: str
