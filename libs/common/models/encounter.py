"""Encounter ORM model — FHIR R4 Encounter resource.

An encounter represents a patient visit or interaction with the healthcare
system (ambulatory appointment, inpatient stay, emergency visit, etc.).

FHIR reference: https://hl7.org/fhir/R4/encounter.html
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, DateTime, Enum, ForeignKey, String, Table
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from libs.common.models.base import ActorMixin, Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from libs.common.models.condition import Condition
    from libs.common.models.medication_request import MedicationRequest
    from libs.common.models.observation import Observation
    from libs.common.models.patient import Patient
    from libs.common.models.physician import Physician


class EncounterStatus(enum.StrEnum):
    """FHIR Encounter status value set."""

    PLANNED = "planned"
    ARRIVED = "arrived"
    TRIAGED = "triaged"
    IN_PROGRESS = "in-progress"
    ON_LEAVE = "onleave"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    ENTERED_IN_ERROR = "entered-in-error"
    UNKNOWN = "unknown"


# Association table for encounter ↔ physician (FHIR Encounter.participant).
encounter_participant = Table(
    "encounter_participant",
    Base.metadata,
    Column(
        "encounter_id",
        UUID(as_uuid=True),
        ForeignKey("encounter.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "physician_id",
        UUID(as_uuid=True),
        ForeignKey("physician.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class Encounter(UUIDPrimaryKeyMixin, TimestampMixin, ActorMixin, Base):
    """A patient interaction with the healthcare system.

    Maps to FHIR R4 ``Encounter`` resource.  The ``resource_json`` column
    stores the full FHIR resource for downstream FHIR API compatibility
    without requiring ORM round-trips for every field.
    """

    __tablename__ = "encounter"

    # ── FHIR Encounter.subject ────────────────────────────────────────────────
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── FHIR Encounter.status ─────────────────────────────────────────────────
    status: Mapped[EncounterStatus] = mapped_column(
        Enum(EncounterStatus, name="encounter_status"),
        nullable=False,
        index=True,
    )

    # ── FHIR Encounter.class ──────────────────────────────────────────────────
    # HL7 v3 ActCode: AMB, IMP, EMER, HH, etc.
    encounter_class: Mapped[str | None] = mapped_column(String(20), index=True)

    # ── FHIR Encounter.type ───────────────────────────────────────────────────
    type_code: Mapped[str | None] = mapped_column(String(100))
    type_display: Mapped[str | None] = mapped_column(String(255))

    # ── FHIR Encounter.serviceType ────────────────────────────────────────────
    service_type: Mapped[str | None] = mapped_column(String(255))

    # ── FHIR Encounter.period ─────────────────────────────────────────────────
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # ── FHIR Encounter.location ───────────────────────────────────────────────
    facility_name: Mapped[str | None] = mapped_column(String(255))
    facility_id: Mapped[str | None] = mapped_column(String(255))

    # ── Full FHIR resource (for FHIR API pass-through) ────────────────────────
    resource_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # ── Relationships ─────────────────────────────────────────────────────────
    patient: Mapped[Patient] = relationship(back_populates="encounters")
    participants: Mapped[list[Physician]] = relationship(
        secondary=encounter_participant, back_populates="encounters"
    )
    conditions: Mapped[list[Condition]] = relationship(back_populates="encounter")
    observations: Mapped[list[Observation]] = relationship(back_populates="encounter")
    medication_requests: Mapped[list[MedicationRequest]] = relationship(back_populates="encounter")
