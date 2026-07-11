"""Patient ORM model — FHIR R4 Patient resource, PHI-aware.

COMPLIANCE NOTE: Fields marked [PHI] must be encrypted at rest using
envelope encryption before this table reaches a non-local environment.
Field-level encryption is implemented in Phase 7.  Do not deploy to staging
or production until Phase 7 encryption is active.

FHIR reference: https://hl7.org/fhir/R4/patient.html
"""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Date, DateTime, Enum, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from libs.common.models.base import (
    ActorMixin,
    Base,
    SoftDeleteMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from libs.common.models.condition import Condition
    from libs.common.models.encounter import Encounter
    from libs.common.models.family_member_history import FamilyMemberHistory
    from libs.common.models.medication_request import MedicationRequest
    from libs.common.models.observation import Observation


class AdministrativeGender(enum.StrEnum):
    """FHIR AdministrativeGender value set."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class Patient(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, ActorMixin, Base):
    """Represents a person receiving healthcare services.

    Maps to FHIR R4 ``Patient`` resource.  All PHI columns are annotated;
    Phase 7 will wrap them in transparent column-level encryption via
    ``pgcrypto`` or application-level envelope encryption.
    """

    __tablename__ = "patient"

    # ── Multi-tenant scope ────────────────────────────────────────────────────
    # Owning organization (tenant).  Nullable so single-tenant deployments and
    # pre-existing records are unaffected; org-scoped endpoints stamp/filter it.
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organization.id", ondelete="SET NULL"),
        index=True,
    )

    # ── External identifiers ──────────────────────────────────────────────────
    # [PHI] Medical Record Number or source-system identifier.
    external_id: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    # Coding system URI for external_id (e.g., "http://hospital.org/mrn").
    identifier_system: Mapped[str | None] = mapped_column(String(255))

    # ── Name [PHI] ────────────────────────────────────────────────────────────
    family_name: Mapped[str | None] = mapped_column(String(255))
    given_name: Mapped[str | None] = mapped_column(String(255))
    middle_name: Mapped[str | None] = mapped_column(String(255))

    # ── Demographics ──────────────────────────────────────────────────────────
    # [PHI] — date of birth is quasi-identifier; needed for age-based ML features.
    date_of_birth: Mapped[date | None] = mapped_column(Date, index=True)

    gender: Mapped[AdministrativeGender | None] = mapped_column(
        Enum(AdministrativeGender, name="administrative_gender"), index=True
    )

    # Ethnicity / race — used for subgroup fairness analysis (Phase 5).
    # Values follow OMB categories; stored as free text to avoid over-constraining.
    ethnicity: Mapped[str | None] = mapped_column(String(100))
    race: Mapped[str | None] = mapped_column(String(100))

    # ── Deceased ──────────────────────────────────────────────────────────────
    deceased: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false"), nullable=False
    )
    deceased_date: Mapped[date | None] = mapped_column(Date)

    # ── Contact [PHI] ─────────────────────────────────────────────────────────
    phone: Mapped[str | None] = mapped_column(String(50))
    email: Mapped[str | None] = mapped_column(String(255))

    # ── Address [PHI] ─────────────────────────────────────────────────────────
    address_line: Mapped[str | None] = mapped_column(String(500))
    city: Mapped[str | None] = mapped_column(String(255))
    state: Mapped[str | None] = mapped_column(String(100))
    postal_code: Mapped[str | None] = mapped_column(String(20))
    country: Mapped[str | None] = mapped_column(String(100), default="US")

    # ── Communication ─────────────────────────────────────────────────────────
    language: Mapped[str | None] = mapped_column(String(10), default="en")

    # ── Consent ───────────────────────────────────────────────────────────────
    # Research consent must be recorded before including this patient in any
    # de-identified analytics export.
    research_consent: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false"), nullable=False
    )
    research_consent_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # ── Cross-system references ────────────────────────────────────────────────
    # Neo4j node ID — populated by the ingestion pipeline (Phase 3).
    neo4j_node_id: Mapped[str | None] = mapped_column(String(255), index=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    conditions: Mapped[list[Condition]] = relationship(back_populates="patient")
    encounters: Mapped[list[Encounter]] = relationship(back_populates="patient")
    observations: Mapped[list[Observation]] = relationship(back_populates="patient")
    medication_requests: Mapped[list[MedicationRequest]] = relationship(back_populates="patient")
    family_member_histories: Mapped[list[FamilyMemberHistory]] = relationship(
        back_populates="patient", foreign_keys="FamilyMemberHistory.patient_id"
    )
