"""Physician ORM model.

Physicians are referenced by Condition, Encounter, and MedicationRequest.
Created before those tables in migrations to satisfy FK ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from libs.common.models.base import ActorMixin, Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from libs.common.models.condition import Condition
    from libs.common.models.encounter import Encounter
    from libs.common.models.medication_request import MedicationRequest


class Physician(UUIDPrimaryKeyMixin, TimestampMixin, ActorMixin, Base):
    """A licensed healthcare provider.

    NPI (National Provider Identifier) is the canonical US identifier and is
    stored as a unique 10-digit string.  NUCC taxonomy codes are used for
    specialty classification (https://nucc.org).
    """

    __tablename__ = "physician"

    # US National Provider Identifier — 10 digits, unique.
    npi: Mapped[str] = mapped_column(String(10), unique=True, nullable=False, index=True)

    family_name: Mapped[str | None] = mapped_column(String(255))
    given_name: Mapped[str | None] = mapped_column(String(255))

    # Human-readable specialty label.
    specialty: Mapped[str | None] = mapped_column(String(255))
    # NUCC Health Care Provider Taxonomy code.
    specialty_code: Mapped[str | None] = mapped_column(String(20), index=True)

    # Neo4j cross-reference — populated during graph sync (Phase 3).
    neo4j_node_id: Mapped[str | None] = mapped_column(String(255))

    # ── Relationships ─────────────────────────────────────────────────────────
    conditions: Mapped[list[Condition]] = relationship(back_populates="recorder")
    encounters: Mapped[list[Encounter]] = relationship(
        secondary="encounter_participant", back_populates="participants"
    )
    medication_requests: Mapped[list[MedicationRequest]] = relationship(back_populates="requester")
