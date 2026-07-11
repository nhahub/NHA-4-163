"""Cascade screening ORM models (Tier 5 — Genetics & Genomics).

Cascade screening is *the* core clinical-genetics workflow: when a patient (the
"proband") is diagnosed with a hereditary condition, at-risk blood relatives are
systematically identified, ranked, and offered testing/screening.

Two tables model this:

* :class:`CascadeScreening` — one screening run for a proband + condition. Holds
  the inheritance context (mode, penetrance) used to rank relatives.
* :class:`CascadeTask` — one outreach/screening task per at-risk relative, with a
  lifecycle status (pending → contacted → scheduled → screened / declined).

PHI note: relatives are PHI.  These rows store only relationship metadata,
probabilities, and a status — never relative names or contact details in the
clear.  Every read/write is captured by :class:`~libs.common.models.audit_log`
via the audit middleware.
"""

from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from libs.common.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from libs.common.models.patient import Patient


class CascadeTaskStatus(enum.StrEnum):
    """Lifecycle of a single cascade-screening outreach task."""

    PENDING = "pending"
    CONTACTED = "contacted"
    SCHEDULED = "scheduled"
    SCREENED = "screened"
    DECLINED = "declined"
    COMPLETED = "completed"


class CascadePriority(enum.StrEnum):
    """Outreach priority derived from relatedness × penetrance."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CascadeScreening(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One cascade-screening run for a proband's hereditary condition."""

    __tablename__ = "cascade_screening"

    # ── Scope ─────────────────────────────────────────────────────────────────
    proband_patient_id: Mapped[uuid.UUID] = mapped_column(
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

    # ── Trigger condition ─────────────────────────────────────────────────────
    condition_code: Mapped[str | None] = mapped_column(String(50), index=True)
    condition_display: Mapped[str | None] = mapped_column(String(255))
    inheritance_mode: Mapped[str] = mapped_column(String(40), nullable=False)
    penetrance: Mapped[float | None] = mapped_column(Numeric(precision=4, scale=3))

    # ── Roll-up ───────────────────────────────────────────────────────────────
    task_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ── Relationships ─────────────────────────────────────────────────────────
    tasks: Mapped[list[CascadeTask]] = relationship(
        back_populates="screening",
        cascade="all, delete-orphan",
        order_by="CascadeTask.priority_score.desc()",
    )
    proband: Mapped[Patient] = relationship()


class CascadeTask(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One at-risk relative's screening/outreach task within a cascade run."""

    __tablename__ = "cascade_task"

    screening_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cascade_screening.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Source relative (FamilyMemberHistory row) and, if they are also a patient,
    # the linked patient id.
    family_member_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("family_member_history.id", ondelete="SET NULL"),
        index=True,
    )
    related_patient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="SET NULL"),
        index=True,
    )

    # ── Relationship context ──────────────────────────────────────────────────
    relationship_code: Mapped[str] = mapped_column(String(50), nullable=False)
    relationship_display: Mapped[str | None] = mapped_column(String(100))
    degree_of_relatedness: Mapped[float | None] = mapped_column(Numeric(precision=5, scale=4))

    # ── Ranking ───────────────────────────────────────────────────────────────
    priority: Mapped[CascadePriority] = mapped_column(
        Enum(CascadePriority, name="cascade_priority"), nullable=False, index=True
    )
    priority_score: Mapped[float] = mapped_column(Numeric(precision=6, scale=4), nullable=False)
    carrier_probability: Mapped[float | None] = mapped_column(Numeric(precision=6, scale=5))
    affected_probability: Mapped[float | None] = mapped_column(Numeric(precision=6, scale=5))

    # ── Workflow ──────────────────────────────────────────────────────────────
    status: Mapped[CascadeTaskStatus] = mapped_column(
        Enum(CascadeTaskStatus, name="cascade_task_status"),
        nullable=False,
        default=CascadeTaskStatus.PENDING,
        index=True,
    )
    recommended_action: Mapped[str | None] = mapped_column(String(500))
    notes: Mapped[str | None] = mapped_column(String(1000))

    # ── Relationships ─────────────────────────────────────────────────────────
    screening: Mapped[CascadeScreening] = relationship(back_populates="tasks")
