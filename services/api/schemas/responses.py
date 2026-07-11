"""FastAPI response schemas (Pydantic v2).

All responses include a ``request_id`` (server-generated UUID) and a
``cached`` flag so clients can distinguish live vs. Redis-cached results.
PHI fields are never included in API responses — callers receive only
identifiers, risk scores, and aggregate statistics.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskTier = Literal["low", "moderate", "high", "very_high"]


def _risk_tier(score: float) -> RiskTier:
    """Map a calibrated probability to a four-level risk tier.

    Args:
        score: Calibrated probability in [0, 1].

    Returns:
        One of ``low``, ``moderate``, ``high``, ``very_high``.
    """
    if score < 0.25:
        return "low"
    if score < 0.50:
        return "moderate"
    if score < 0.75:
        return "high"
    return "very_high"


# ── Hereditary risk ───────────────────────────────────────────────────────────


class SHAPContribution(BaseModel):
    """Single feature's contribution to the model prediction."""

    model_config = ConfigDict(frozen=True)

    feature: str
    raw_value: float | None = None
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class HeredityRiskResponse(BaseModel):
    """Response for POST /predict/hereditary-risk."""

    model_config = ConfigDict(frozen=True)

    request_id: uuid.UUID
    patient_id: uuid.UUID
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_tier: RiskTier
    top_risk_factors: list[SHAPContribution] | None = None
    feature_date: str
    model_name: str
    model_version: str
    cached: bool


# ── Differential diagnosis ────────────────────────────────────────────────────


class DiseasePrediction(BaseModel):
    """Single disease entry in a differential diagnosis list."""

    model_config = ConfigDict(frozen=True)

    disease_code: str
    disease_name: str | None = None
    probability: float = Field(..., ge=0.0, le=1.0)


class DifferentialDiagnosisResponse(BaseModel):
    """Response for POST /predict/disease-from-symptoms and disease-from-prescription."""

    model_config = ConfigDict(frozen=True)

    request_id: uuid.UUID
    patient_id: uuid.UUID
    input_codes: list[str]
    input_type: Literal["symptoms", "prescriptions"]
    predictions: list[DiseasePrediction]
    model_version: str
    cached: bool


# ── Family risk profile ────────────────────────────────────────────────────────


class RelativeRecord(BaseModel):
    """One relative in the patient's family graph."""

    model_config = ConfigDict(frozen=True)

    relative_id: str
    relationship_code: str
    degree_of_relatedness: float
    diagnosed_icd10_codes: list[str]


class ChapterBurden(BaseModel):
    """Disease burden for one ICD-10 chapter within the family."""

    model_config = ConfigDict(frozen=True)

    affected_relative_count: int
    weighted_prevalence: float


class FamilyRiskProfileResponse(BaseModel):
    """Response for GET /patient/{patient_id}/family-risk-profile."""

    model_config = ConfigDict(frozen=True)

    request_id: uuid.UUID
    patient_id: uuid.UUID
    family_risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_tier: RiskTier
    family_size: int
    affected_relatives_count: int
    first_degree_relatives: list[RelativeRecord]
    disease_burden_by_chapter: dict[str, ChapterBurden]
    cached: bool


# ── Health ────────────────────────────────────────────────────────────────────


class ComponentStatus(BaseModel):
    status: Literal["ok", "degraded", "down"]
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    components: dict[str, ComponentStatus]
