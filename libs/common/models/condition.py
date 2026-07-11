"""Condition ORM model — FHIR R4 Condition resource.

Conditions represent diagnoses.  They are the primary input to the hereditary
disease prediction model:  the ML pipeline aggregates conditions across the
family graph to compute per-disease risk scores.

FHIR reference: https://hl7.org/fhir/R4/condition.html
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from libs.common.models.base import ActorMixin, Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from libs.common.models.encounter import Encounter
    from libs.common.models.patient import Patient
    from libs.common.models.physician import Physician


class ClinicalStatus(enum.StrEnum):
    """FHIR condition-clinical value set."""

    ACTIVE = "active"
    CONFIRMED = "confirmed"
    RECURRENCE = "recurrence"
    RELAPSE = "relapse"
    INACTIVE = "inactive"
    REMISSION = "remission"
    RESOLVED = "resolved"


class VerificationStatus(enum.StrEnum):
    """FHIR condition-ver-status value set."""

    UNCONFIRMED = "unconfirmed"
    PROVISIONAL = "provisional"
    DIFFERENTIAL = "differential"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    ENTERED_IN_ERROR = "entered-in-error"


class ConditionSeverity(enum.StrEnum):
    """FHIR condition-severity value set (SNOMED CT subset)."""

    SEVERE = "severe"
    MODERATE = "moderate"
    MILD = "mild"


class Condition(UUIDPrimaryKeyMixin, TimestampMixin, ActorMixin, Base):
    """A clinical condition / diagnosis for a patient.

    Maps to FHIR R4 ``Condition`` resource.  The ``is_hereditary`` and
    ``family_history_flag`` columns are non-FHIR extensions used by the
    ML feature pipeline to select relevant training examples.
    """

    __tablename__ = "condition"

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
    recorder_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("physician.id", ondelete="SET NULL"),
        index=True,
    )

    # ── FHIR Condition.clinicalStatus ────────────────────────────────────────
    clinical_status: Mapped[ClinicalStatus] = mapped_column(
        Enum(ClinicalStatus, name="clinical_status"),
        nullable=False,
        index=True,
    )

    # ── FHIR Condition.verificationStatus ────────────────────────────────────
    verification_status: Mapped[VerificationStatus | None] = mapped_column(
        Enum(VerificationStatus, name="verification_status"), index=True
    )

    # ── FHIR Condition.severity ───────────────────────────────────────────────
    severity: Mapped[ConditionSeverity | None] = mapped_column(
        Enum(ConditionSeverity, name="condition_severity")
    )

    # ── FHIR Condition.code ───────────────────────────────────────────────────
    # Supports ICD-10, SNOMED CT.  The code_system URI disambiguates.
    # e.g., code_system="http://hl7.org/fhir/sid/icd-10", code="I10"
    code_system: Mapped[str] = mapped_column(String(255), nullable=False)
    code: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    code_display: Mapped[str | None] = mapped_column(String(500))
    code_text: Mapped[str | None] = mapped_column(String(500))

    # ── FHIR Condition.onset ──────────────────────────────────────────────────
    onset_datetime: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    # Used when exact onset date is unknown (e.g., family history self-report).
    onset_age_years: Mapped[int | None] = mapped_column(Integer)

    # ── FHIR Condition.abatement ──────────────────────────────────────────────
    abatement_datetime: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # ── Hereditary prediction extensions ─────────────────────────────────────
    # True if this condition is classified as hereditary (OMIM/ClinVar lookup).
    is_hereditary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    # True if this condition was recorded based on family history, not direct diagnosis.
    family_history_flag: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )

    # ── Full FHIR resource ────────────────────────────────────────────────────
    resource_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # ── Relationships ─────────────────────────────────────────────────────────
    patient: Mapped[Patient] = relationship(back_populates="conditions")
    encounter: Mapped[Encounter | None] = relationship(back_populates="conditions")
    recorder: Mapped[Physician | None] = relationship(back_populates="conditions")
