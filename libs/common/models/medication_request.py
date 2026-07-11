"""MedicationRequest ORM model — FHIR R4 MedicationRequest resource.

MedicationRequests are prescriptions or medication orders.  They link a
patient to a medication (identified by RxNorm RXCUI) and are a feature input
for the prescription-to-disease prediction model (Phase 5).

FHIR reference: https://hl7.org/fhir/R4/medicationrequest.html
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from libs.common.models.base import ActorMixin, Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from libs.common.models.encounter import Encounter
    from libs.common.models.patient import Patient
    from libs.common.models.physician import Physician


class MedicationRequestStatus(enum.StrEnum):
    """FHIR medicationrequest-status value set."""

    ACTIVE = "active"
    ON_HOLD = "on-hold"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ENTERED_IN_ERROR = "entered-in-error"
    STOPPED = "stopped"
    DRAFT = "draft"
    UNKNOWN = "unknown"


class MedicationRequestIntent(enum.StrEnum):
    """FHIR medicationrequest-intent value set."""

    PROPOSAL = "proposal"
    PLAN = "plan"
    ORDER = "order"
    ORIGINAL_ORDER = "original-order"
    REFLEX_ORDER = "reflex-order"
    FILLER_ORDER = "filler-order"
    INSTANCE_ORDER = "instance-order"
    OPTION = "option"


class MedicationRequest(UUIDPrimaryKeyMixin, TimestampMixin, ActorMixin, Base):
    """An order for a medication for a patient.

    Maps to FHIR R4 ``MedicationRequest``.  The ``medication_code`` column
    stores the RxNorm RXCUI.  The code system URI in ``medication_code_system``
    allows future extension to other coding systems (NDC, ATC).
    """

    __tablename__ = "medication_request"

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
    requester_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("physician.id", ondelete="SET NULL"),
        index=True,
    )

    # ── FHIR MedicationRequest.status ─────────────────────────────────────────
    status: Mapped[MedicationRequestStatus] = mapped_column(
        Enum(MedicationRequestStatus, name="medication_request_status"),
        nullable=False,
        index=True,
    )

    # ── FHIR MedicationRequest.intent ─────────────────────────────────────────
    intent: Mapped[MedicationRequestIntent] = mapped_column(
        Enum(MedicationRequestIntent, name="medication_request_intent"),
        nullable=False,
    )

    # ── FHIR MedicationRequest.medication (RxNorm preferred) ──────────────────
    medication_code_system: Mapped[str | None] = mapped_column(String(255))
    medication_code: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    medication_display: Mapped[str | None] = mapped_column(String(500))

    # ── FHIR MedicationRequest.dosageInstruction ──────────────────────────────
    dosage_text: Mapped[str | None] = mapped_column(String(500))
    dosage_timing: Mapped[str | None] = mapped_column(String(255))
    dosage_route: Mapped[str | None] = mapped_column(String(100))
    dose_quantity: Mapped[float | None] = mapped_column(Numeric(precision=10, scale=3))
    dose_unit: Mapped[str | None] = mapped_column(String(50))

    # ── FHIR MedicationRequest.dispenseRequest ────────────────────────────────
    dispense_quantity: Mapped[float | None] = mapped_column(Numeric(precision=10, scale=3))
    dispense_unit: Mapped[str | None] = mapped_column(String(50))
    number_of_repeats: Mapped[int | None] = mapped_column(Integer)

    # ── Temporal ──────────────────────────────────────────────────────────────
    authored_on: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # ── Full FHIR resource ────────────────────────────────────────────────────
    resource_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # ── Relationships ─────────────────────────────────────────────────────────
    patient: Mapped[Patient] = relationship(back_populates="medication_requests")
    encounter: Mapped[Encounter | None] = relationship(back_populates="medication_requests")
    requester: Mapped[Physician | None] = relationship(back_populates="medication_requests")
