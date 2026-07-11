"""Response schemas for model monitoring & fairness (Tier 6)."""

from __future__ import annotations

from pydantic import BaseModel


class DriftBinSchema(BaseModel):
    """One PSI bucket over the score range."""

    lower: float
    upper: float
    reference_pct: float
    current_pct: float
    contribution: float


class DriftResponse(BaseModel):
    """Population-stability-index drift between reference and current scores."""

    psi: float
    verdict: str
    reference_count: int
    current_count: int
    reference_window_days: int
    current_window_days: int
    bins: list[DriftBinSchema]


class GroupStatSchema(BaseModel):
    """Risk summary for one demographic subgroup."""

    group: str
    count: int
    mean_score: float
    high_risk_rate: float


class FairnessGroupResult(BaseModel):
    """Parity summary across the subgroups of one demographic attribute."""

    attribute: str
    groups: list[GroupStatSchema]
    disparate_impact_ratio: float
    statistical_parity_difference: float
    passes_four_fifths: bool
    interpretation: str


class FairnessResponse(BaseModel):
    """Fairness assessment across all requested demographic attributes."""

    predictions_evaluated: int
    high_risk_threshold: float
    attributes: list[FairnessGroupResult]
