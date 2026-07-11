"""Request/response schemas for the What-If risk simulator (Tier 6)."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class RiskFactorInfo(BaseModel):
    """Metadata describing one modifiable risk factor."""

    key: str
    display: str
    coefficient: float
    default: float
    unit: str
    kind: str


class WhatIfRequest(BaseModel):
    """Request body for a counterfactual risk simulation.

    ``baseline`` is the patient's current factor values (missing factors use
    their defaults); when omitted, the router seeds it from stored records.
    ``modifications`` are the counterfactual overrides applied on top.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    baseline: dict[str, float] | None = Field(
        default=None, description="Current factor values; unknown keys are ignored."
    )
    modifications: dict[str, float] = Field(
        default_factory=dict,
        description="Counterfactual factor overrides; unknown keys are ignored.",
    )


class FactorContributionSchema(BaseModel):
    """A single factor's log-odds contribution (SHAP-comparable)."""

    key: str
    display: str
    value: float
    log_odds_contribution: float
    delta_from_baseline: float


class WhatIfResponse(BaseModel):
    """Baseline vs simulated risk with a per-factor breakdown."""

    patient_id: uuid.UUID | None = None
    baseline_risk: float
    simulated_risk: float
    risk_delta: float
    baseline_factors: dict[str, float]
    simulated_factors: dict[str, float]
    contributions: list[FactorContributionSchema]
    interpretation: str
