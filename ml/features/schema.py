"""Pydantic v2 schema for the per-patient feature vector.

This schema is the contract between the feature engineering pipeline
(Phase 4) and the ML training / serving layers (Phases 5–6).  Any change
to feature columns must be reflected here and in the feature registry.

Design notes
------------
- All optional features default to 0 / 0.0 rather than None, except
  ``age_years`` (genuinely unknown when DOB is missing) and
  ``adherence_proxy`` (undefined when no completed/stopped meds exist).
- ``shortest_path_to_affected = -1`` encodes "no affected relative found
  within 4 hops" as a sentinel distinct from depth-0 (self).
- ``feature_date`` is stored as an ISO-8601 string so it survives Delta
  table round-trips without timezone conversion issues.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatureVector(BaseModel):
    """Complete feature vector for hereditary disease risk prediction."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    # ── Identity ──────────────────────────────────────────────────────────────
    patient_id: uuid.UUID
    feature_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")

    # ── Demographics ──────────────────────────────────────────────────────────
    age_years: int | None = Field(default=None, ge=0, le=150)
    age_group: str = "unknown"
    gender_male: int = Field(default=0, ge=0, le=1)
    gender_female: int = Field(default=0, ge=0, le=1)
    gender_other_unknown: int = Field(default=0, ge=0, le=1)

    # ── Comorbidities ─────────────────────────────────────────────────────────
    comorbidity_count: int = Field(default=0, ge=0)
    hereditary_condition_count: int = Field(default=0, ge=0)
    has_cardiovascular: int = Field(default=0, ge=0, le=1)
    has_metabolic: int = Field(default=0, ge=0, le=1)
    has_neurological: int = Field(default=0, ge=0, le=1)
    has_oncological: int = Field(default=0, ge=0, le=1)
    has_haematological: int = Field(default=0, ge=0, le=1)
    has_musculoskeletal: int = Field(default=0, ge=0, le=1)
    has_respiratory: int = Field(default=0, ge=0, le=1)
    has_digestive: int = Field(default=0, ge=0, le=1)
    has_mental_health: int = Field(default=0, ge=0, le=1)
    has_genitourinary: int = Field(default=0, ge=0, le=1)
    has_infectious: int = Field(default=0, ge=0, le=1)

    # ── Medications ───────────────────────────────────────────────────────────
    active_medication_count: int = Field(default=0, ge=0)
    completed_medication_count: int = Field(default=0, ge=0)
    stopped_medication_count: int = Field(default=0, ge=0)
    distinct_medication_count: int = Field(default=0, ge=0)
    adherence_proxy: float | None = Field(default=None, ge=0.0, le=1.0)

    # ── Graph features ────────────────────────────────────────────────────────
    affected_relatives_count: int = Field(default=0, ge=0)
    weighted_family_prevalence: float = Field(default=0.0, ge=0.0)
    first_degree_affected_count: int = Field(default=0, ge=0)
    second_degree_affected_count: int = Field(default=0, ge=0)
    shortest_path_to_affected: int = Field(default=-1, ge=-1)
    family_size: int = Field(default=0, ge=0)
    family_clustering_coefficient: float = Field(default=0.0, ge=0.0, le=1.0)
