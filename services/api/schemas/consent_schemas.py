"""Request/response schemas for patient consent management (Tier 7)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

_SCOPE_PATTERN = r"^(research|data_sharing|treatment|family_contact|genetic_testing|marketing)$"
_STATUS_PATTERN = r"^(granted|denied|withdrawn)$"
_METHOD_PATTERN = r"^(written|verbal|electronic|portal)$"


class ConsentRequest(BaseModel):
    """Request body for POST /patients/{id}/consent."""

    model_config = ConfigDict(str_strip_whitespace=True)

    scope: str = Field(..., pattern=_SCOPE_PATTERN)
    status: str = Field(..., pattern=_STATUS_PATTERN)
    method: str | None = Field(default=None, pattern=_METHOD_PATTERN)
    expires_at: datetime | None = Field(
        default=None, description="Optional expiry for a granted consent."
    )
    policy_version: str | None = Field(default=None, max_length=50)
    notes: str | None = Field(default=None, max_length=1000)


class ConsentRecordResponse(BaseModel):
    """One consent decision in a patient's history."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    organization_id: uuid.UUID | None = None
    scope: str
    status: str
    method: str | None = None
    granted_at: datetime | None = None
    expires_at: datetime | None = None
    withdrawn_at: datetime | None = None
    policy_version: str | None = None
    notes: str | None = None
    created_at: datetime | None = None


class ConsentScopeState(BaseModel):
    """Effective (current) consent for a single scope."""

    scope: str
    active: bool
    status: str | None = None
    expires_at: datetime | None = None
    decided_at: datetime | None = None


class ConsentStateResponse(BaseModel):
    """Current consent state for a patient across all scopes."""

    patient_id: uuid.UUID
    scopes: list[ConsentScopeState]
    history: list[ConsentRecordResponse] = []
