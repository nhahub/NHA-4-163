"""Schemas for the SMART on FHIR patient portal (Tier 7)."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class SmartConfiguration(BaseModel):
    """SMART on FHIR ``.well-known/smart-configuration`` discovery document.

    A pragmatic subset advertising the endpoints and capabilities the portal
    supports.  See http://hl7.org/fhir/smart-app-launch/conformance.html.
    """

    authorization_endpoint: str
    token_endpoint: str
    grant_types_supported: list[str] = Field(default_factory=lambda: ["authorization_code"])
    scopes_supported: list[str] = Field(
        default_factory=lambda: [
            "openid",
            "fhirUser",
            "launch/patient",
            "patient/*.read",
        ]
    )
    response_types_supported: list[str] = Field(default_factory=lambda: ["code"])
    capabilities: list[str] = Field(
        default_factory=lambda: [
            "launch-standalone",
            "client-public",
            "context-standalone-patient",
            "permission-patient",
        ]
    )
    code_challenge_methods_supported: list[str] = Field(default_factory=lambda: ["S256"])


class PortalTokenRequest(BaseModel):
    """Request body for POST /portal/token (standalone patient launch).

    In a full SMART deployment the patient id would be resolved from the
    identity provider after the authorization-code exchange; here it is
    supplied directly by the trusted launch context.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    patient_id: uuid.UUID


class PortalTokenResponse(BaseModel):
    """Patient-scoped access token issued to the portal app."""

    access_token: str
    token_type: str = "bearer"  # noqa: S105 — OAuth2 token type, not a secret
    expires_in: int
    scope: str
    patient: uuid.UUID


class PortalRiskProfile(BaseModel):
    """Read-only, patient-friendly risk summary."""

    patient_id: uuid.UUID
    risk_score: float | None = None
    risk_tier: str
    band: str
    guidance: str
    model_version: str | None = None
    predicted_at: str | None = None


class PortalFamilyMember(BaseModel):
    """A de-identified relative in the patient's own pedigree view."""

    relationship_code: str
    relationship_display: str | None = None
    sex: str | None = None
    degree_of_relatedness: float | None = None
    affected: bool
    conditions: list[str] = []


class PortalFamilyResponse(BaseModel):
    """The patient's family tree, de-identified for self-service viewing."""

    patient_id: uuid.UUID
    members: list[PortalFamilyMember] = []
