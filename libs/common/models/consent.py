"""Consent ORM model — granular patient consent records (Tier 7).

Patients may grant, deny, or withdraw consent at the granularity of a *scope*
(research use, cross-organisation data sharing, family contact for cascade
outreach, etc.).  Consent is **append-only**: each decision is a new row, and
the effective state for a scope is the most recent row for that scope.  This
preserves a full, auditable consent history (required for HIPAA/GDPR) rather
than mutating a single boolean on the patient.

Enforcement points already in place that read this table:
  * De-identified research export (``routers/export.py``) — a patient with a
    withdrawn/denied ``research`` consent is excluded even if the legacy
    ``Patient.research_consent`` flag is still set.

PHI note: a consent row references a ``patient_id`` and stores only the scope,
decision, method, and free-text notes.  Notes must never embed direct
identifiers.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from libs.common.models.base import ActorMixin, Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from libs.common.models.patient import Patient


class ConsentScope(enum.StrEnum):
    """A purpose for which patient data may be used.

    Kept deliberately coarse — one scope per distinct downstream use so that a
    withdrawal is unambiguous about what it stops.
    """

    RESEARCH = "research"  # inclusion in de-identified research export
    DATA_SHARING = "data_sharing"  # cross-organisation data sharing
    TREATMENT = "treatment"  # use for direct clinical care
    FAMILY_CONTACT = "family_contact"  # contact relatives for cascade screening
    GENETIC_TESTING = "genetic_testing"  # order/store genetic test results
    MARKETING = "marketing"  # outreach unrelated to care


class ConsentStatus(enum.StrEnum):
    """The patient's decision for a scope."""

    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"


class ConsentMethod(enum.StrEnum):
    """How the consent decision was captured."""

    WRITTEN = "written"
    VERBAL = "verbal"
    ELECTRONIC = "electronic"
    PORTAL = "portal"


class ConsentRecord(UUIDPrimaryKeyMixin, TimestampMixin, ActorMixin, Base):
    """One consent decision for a single patient + scope.

    Append-only: the effective consent for a scope is the row with the latest
    ``created_at``.  ``expires_at`` optionally bounds a ``granted`` decision;
    an expired grant is treated as inactive by the service layer.
    """

    __tablename__ = "consent_record"

    # ── Scope ─────────────────────────────────────────────────────────────────
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organization.id", ondelete="CASCADE"),
        index=True,
    )

    # ── Decision ──────────────────────────────────────────────────────────────
    scope: Mapped[ConsentScope] = mapped_column(
        Enum(ConsentScope, name="consent_scope"), nullable=False, index=True
    )
    status: Mapped[ConsentStatus] = mapped_column(
        Enum(ConsentStatus, name="consent_status"), nullable=False, index=True
    )
    method: Mapped[ConsentMethod | None] = mapped_column(Enum(ConsentMethod, name="consent_method"))

    # ── Validity window ───────────────────────────────────────────────────────
    granted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # ── Provenance ────────────────────────────────────────────────────────────
    # Free-form policy/version reference (e.g. "research-consent-v2") + notes.
    policy_version: Mapped[str | None] = mapped_column(String(50))
    notes: Mapped[str | None] = mapped_column(String(1000))

    # ── Relationships ─────────────────────────────────────────────────────────
    patient: Mapped[Patient] = relationship()
