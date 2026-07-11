"""Response schemas for guideline-based screening recommendations (Tier 6)."""

from __future__ import annotations

import uuid

from pydantic import BaseModel


class GuidelineRecommendationSchema(BaseModel):
    """One actionable screening/next-step recommendation."""

    guideline_id: str
    source: str
    title: str
    recommendation: str
    urgency: str
    rationale: str


class ScreeningRecommendationsResponse(BaseModel):
    """Guideline-based recommendations for a patient."""

    patient_id: uuid.UUID
    age: int | None = None
    sex: str | None = None
    risk_score: float
    risk_tier: str | None = None
    affected_first_degree_relatives: int
    recommendations: list[GuidelineRecommendationSchema]
