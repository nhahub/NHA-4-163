"""Request/response schemas for organization (tenant) management (Tier 4)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class OrganizationCreate(BaseModel):
    """Request body for POST /organizations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=255)
    slug: str | None = Field(
        default=None,
        max_length=100,
        pattern=r"^[a-z0-9-]+$",
        description="URL-safe identifier; derived from name if omitted.",
    )


class OrganizationResponse(BaseModel):
    """Organization record returned by the API (never includes key material)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str
    active: bool = True
    created_at: datetime | None = None


class OrganizationCreatedResponse(OrganizationResponse):
    """Returned on create/rotate — includes the one-time plaintext API key."""

    api_key: str = Field(..., description="Plaintext API key — shown only once.")
