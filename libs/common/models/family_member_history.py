"""FamilyMemberHistory ORM model — FHIR R4 FamilyMemberHistory resource.

This is the relational complement to the Neo4j family graph.  Neo4j is the
source of truth for traversal and ML feature extraction; this table provides
FHIR-compatible read access and acts as the ingestion staging area before
the graph sync pipeline writes edges to Neo4j.

``degree_of_relatedness`` is the Wright coefficient of relationship:
  - 1st degree (parent, child, sibling): 0.5
  - 2nd degree (grandparent, half-sibling, aunt/uncle): 0.25
  - 3rd degree (great-grandparent, first cousin): 0.125
  Spouse: 0.0 (not genetically related, but epidemiologically relevant).

FHIR reference: https://hl7.org/fhir/R4/familymemberhistory.html
"""

from __future__ import annotations

import enum
import uuid
from datetime import date
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from libs.common.models.base import ActorMixin, Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from libs.common.models.patient import Patient


class FamilyMemberHistoryStatus(enum.StrEnum):
    """FHIR history-status value set."""

    PARTIAL = "partial"
    COMPLETED = "completed"
    ENTERED_IN_ERROR = "entered-in-error"
    HEALTH_UNKNOWN = "health-unknown"


class FamilyMemberHistory(UUIDPrimaryKeyMixin, TimestampMixin, ActorMixin, Base):
    """One family member's medical history as reported for a patient.

    ``conditions`` stores a JSON array of FHIR-shaped condition objects
    (each with ``code``, ``outcome``, ``onset``).  This avoids a deeply
    nested FK structure for family history conditions while keeping the data
    FHIR-serialisable.

    When ``related_patient_id`` is set, this relative is an active patient in
    the system; the graph pipeline will create bidirectional edges in Neo4j.
    """

    __tablename__ = "family_member_history"
    __table_args__ = (
        CheckConstraint(
            "degree_of_relatedness >= 0 AND degree_of_relatedness <= 1",
            name="ck_fmh_degree_range",
        ),
    )

    # ── Foreign keys ──────────────────────────────────────────────────────────
    # The patient this history belongs to.
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # If this family member is also a patient in our system.
    related_patient_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="SET NULL"),
        index=True,
    )

    # ── FHIR FamilyMemberHistory.status ──────────────────────────────────────
    status: Mapped[FamilyMemberHistoryStatus] = mapped_column(
        Enum(FamilyMemberHistoryStatus, name="family_member_history_status"),
        nullable=False,
    )

    # ── FHIR FamilyMemberHistory.relationship ────────────────────────────────
    # HL7 v3 FamilyMember code: MTH (mother), FTH (father), SIB (sibling),
    # GRPRN (grandparent), CHLDINLAW (child in-law), etc.
    relationship_code: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    relationship_display: Mapped[str | None] = mapped_column(String(100))

    # ── Genetic relatedness ───────────────────────────────────────────────────
    degree_of_relatedness: Mapped[float | None] = mapped_column(Numeric(precision=5, scale=4))

    # ── FHIR FamilyMemberHistory.sex ─────────────────────────────────────────
    sex: Mapped[str | None] = mapped_column(String(20))

    # ── FHIR FamilyMemberHistory.born ────────────────────────────────────────
    born_date: Mapped[date | None] = mapped_column(Date)

    # ── FHIR FamilyMemberHistory.deceased ────────────────────────────────────
    deceased: Mapped[bool | None] = mapped_column(Boolean)
    deceased_age_years: Mapped[int | None] = mapped_column(Integer)
    deceased_date: Mapped[date | None] = mapped_column(Date)

    # ── FHIR FamilyMemberHistory.condition[] ─────────────────────────────────
    # JSON array: [{"code": {"system": ..., "code": ..., "display": ...},
    #               "outcome": ..., "onset": {...}}]
    conditions: Mapped[list[Any] | None] = mapped_column(JSONB, default=list)

    # ── Neo4j sync status ─────────────────────────────────────────────────────
    # Set to True once the graph pipeline has created the corresponding edges.
    neo4j_synced: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false"), nullable=False
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    patient: Mapped[Patient] = relationship(
        back_populates="family_member_histories",
        foreign_keys=[patient_id],
    )
    related_patient: Mapped[Patient | None] = relationship(
        foreign_keys=[related_patient_id],
    )
