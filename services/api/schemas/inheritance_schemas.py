"""Request/response schemas for the Mendelian inheritance calculator (Tier 5)."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

_MODE_PATTERN = (
    r"^(autosomal_dominant|autosomal_recessive|x_linked_recessive|"
    r"x_linked_dominant|mitochondrial)$"
)


class InheritanceRiskRequest(BaseModel):
    """Request body for POST /patients/{id}/inheritance-risk."""

    model_config = ConfigDict(str_strip_whitespace=True)

    inheritance_mode: str = Field(
        ...,
        pattern=_MODE_PATTERN,
        description="Mode of inheritance for the condition of interest.",
    )
    condition_code: str | None = Field(
        default=None, max_length=50, description="ICD-10/OMIM code of the condition."
    )
    condition_display: str | None = Field(default=None, max_length=255)
    penetrance: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Override the model's default penetrance."
    )
    carrier_frequency: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the population carrier frequency for the other allele.",
    )


class RelativeRiskResult(BaseModel):
    """Computed carrier/affected probability for a single relative."""

    family_member_id: uuid.UUID | None = None
    related_patient_id: uuid.UUID | None = None
    relationship_code: str
    relationship_display: str | None = None
    relationship_category: str
    degree_of_relatedness: float | None = None
    carrier_probability: float
    affected_probability: float
    basis: str


class InheritanceRiskResponse(BaseModel):
    """Response for the pedigree-wide Mendelian risk computation."""

    patient_id: uuid.UUID
    inheritance_mode: str
    inheritance_display: str
    penetrance: float
    carrier_frequency: float
    condition_code: str | None = None
    condition_display: str | None = None
    relatives_evaluated: int
    results: list[RelativeRiskResult] = []


class InheritanceModelInfo(BaseModel):
    """Metadata describing one supported inheritance model."""

    key: str
    display: str
    default_penetrance: float
    default_carrier_frequency: float
    sex_linked: bool
    description: str
