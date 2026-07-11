"""Request/response schemas for the cascade screening workflow (Tier 5)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

_MODE_PATTERN = (
    r"^(autosomal_dominant|autosomal_recessive|x_linked_recessive|"
    r"x_linked_dominant|mitochondrial)$"
)


class CascadeScreenRequest(BaseModel):
    """Request body for POST /patients/{id}/cascade-screen."""

    model_config = ConfigDict(str_strip_whitespace=True)

    inheritance_mode: str = Field(..., pattern=_MODE_PATTERN)
    condition_code: str | None = Field(default=None, max_length=50)
    condition_display: str | None = Field(default=None, max_length=255)
    penetrance: float | None = Field(default=None, ge=0.0, le=1.0)
    carrier_frequency: float | None = Field(default=None, ge=0.0, le=1.0)
    notify: bool = Field(
        default=True, description="Emit a notification for high-priority relatives."
    )


class CascadeTaskResponse(BaseModel):
    """One at-risk relative's screening task."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    screening_id: uuid.UUID
    family_member_id: uuid.UUID | None = None
    related_patient_id: uuid.UUID | None = None
    relationship_code: str
    relationship_display: str | None = None
    degree_of_relatedness: float | None = None
    priority: str
    priority_score: float
    carrier_probability: float | None = None
    affected_probability: float | None = None
    status: str
    recommended_action: str | None = None
    notes: str | None = None
    created_at: datetime | None = None


class CascadeScreeningResponse(BaseModel):
    """A cascade-screening run with its ranked tasks."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    proband_patient_id: uuid.UUID
    organization_id: uuid.UUID | None = None
    condition_code: str | None = None
    condition_display: str | None = None
    inheritance_mode: str
    penetrance: float | None = None
    task_count: int
    created_at: datetime | None = None
    tasks: list[CascadeTaskResponse] = []


class CascadeTaskUpdate(BaseModel):
    """Request body for PUT /cascade-tasks/{id}."""

    model_config = ConfigDict(str_strip_whitespace=True)

    status: str | None = Field(
        default=None,
        pattern=r"^(pending|contacted|scheduled|screened|declined|completed)$",
    )
    notes: str | None = Field(default=None, max_length=1000)
