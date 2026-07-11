"""Observation ORM model — FHIR R4 Observation resource.

Observations cover lab results, vital signs, and clinical assessments.
LOINC codes are the preferred coding system for observations.

FHIR reference: https://hl7.org/fhir/R4/observation.html
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from libs.common.models.base import ActorMixin, Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from libs.common.models.encounter import Encounter
    from libs.common.models.patient import Patient


class ObservationStatus(enum.StrEnum):
    """FHIR observation-status value set."""

    REGISTERED = "registered"
    PRELIMINARY = "preliminary"
    FINAL = "final"
    AMENDED = "amended"
    CORRECTED = "corrected"
    CANCELLED = "cancelled"
    ENTERED_IN_ERROR = "entered-in-error"
    UNKNOWN = "unknown"


class Observation(UUIDPrimaryKeyMixin, TimestampMixin, ActorMixin, Base):
    """A measurement or assertion about a patient.

    The value columns are polymorphic (quantity, string, boolean, or coded)
    to mirror FHIR's ``value[x]`` pattern without requiring a separate table
    per type.  Exactly one value column should be non-null per row.
    """

    __tablename__ = "observation"

    # ── Foreign keys ──────────────────────────────────────────────────────────
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    encounter_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("encounter.id", ondelete="SET NULL"),
        index=True,
    )

    # ── FHIR Observation.status ───────────────────────────────────────────────
    status: Mapped[ObservationStatus] = mapped_column(
        Enum(ObservationStatus, name="observation_status"),
        nullable=False,
        index=True,
    )

    # ── FHIR Observation.category ─────────────────────────────────────────────
    # vital-signs, laboratory, imaging, exam, survey, social-history, activity
    category: Mapped[str | None] = mapped_column(String(100), index=True)

    # ── FHIR Observation.code (LOINC preferred) ───────────────────────────────
    code_system: Mapped[str] = mapped_column(String(255), nullable=False)
    code: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    code_display: Mapped[str | None] = mapped_column(String(500))

    # ── FHIR Observation.effective ────────────────────────────────────────────
    effective_datetime: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # ── FHIR Observation.value[x] — polymorphic ───────────────────────────────
    value_quantity: Mapped[float | None] = mapped_column(Numeric(precision=18, scale=6))
    value_unit: Mapped[str | None] = mapped_column(String(50))
    value_unit_system: Mapped[str | None] = mapped_column(String(255))
    value_string: Mapped[str | None] = mapped_column(String(500))
    value_boolean: Mapped[bool | None] = mapped_column(Boolean)
    value_codeable_code: Mapped[str | None] = mapped_column(String(50))
    value_codeable_display: Mapped[str | None] = mapped_column(String(500))

    # ── FHIR Observation.referenceRange ──────────────────────────────────────
    ref_range_low: Mapped[float | None] = mapped_column(Numeric(precision=18, scale=6))
    ref_range_high: Mapped[float | None] = mapped_column(Numeric(precision=18, scale=6))
    ref_range_text: Mapped[str | None] = mapped_column(String(255))

    # ── FHIR Observation.interpretation ──────────────────────────────────────
    # HL7 v3 ObservationInterpretation: H, L, N, A, AA, HH, LL, etc.
    interpretation: Mapped[str | None] = mapped_column(String(10))

    # ── Full FHIR resource ────────────────────────────────────────────────────
    resource_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # ── Relationships ─────────────────────────────────────────────────────────
    patient: Mapped[Patient] = relationship(back_populates="observations")
    encounter: Mapped[Encounter | None] = relationship(back_populates="observations")
