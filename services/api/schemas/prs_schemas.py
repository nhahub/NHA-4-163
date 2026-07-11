"""Request/response schemas for polygenic risk score integration (Tier 5)."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict


class PRSPanelInfo(BaseModel):
    """Metadata describing one supported PRS panel."""

    key: str
    display: str
    condition_code: str
    baseline_prevalence: float
    snp_count: int


class PolygenicRiskResponse(BaseModel):
    """Blended polygenic + ML risk for a patient and disease."""

    model_config = ConfigDict(from_attributes=True)

    patient_id: uuid.UUID
    disease: str
    display: str
    raw_score: float
    z_score: float
    percentile: float
    odds_ratio: float
    prs_absolute_risk: float
    ml_risk: float | None = None
    blended_risk: float
    snps_used: int
    snps_available: int
    interpretation: str
