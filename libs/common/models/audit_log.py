"""AuditLog ORM model — immutable PHI access and mutation audit trail.

COMPLIANCE: This table must be append-only in production.
  - PostgreSQL RLS policy blocks UPDATE and DELETE for non-admin roles.
  - A row-level trigger enforces immutability (Phase 7, migration 0009).
  - The S3 bucket storing Parquet exports of this table uses Object Lock.

Never store PHI values in this table.  Log resource IDs and action types
only.  The ``metadata`` JSONB column may contain non-PHI context fields
(risk_score, model_version, etc.) — always run through redact_dict() before
writing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, String, func
from sqlalchemy.dialects.postgresql import INET, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from libs.common.models.base import Base


class AuditLog(Base):
    """One audit event — never updated, never deleted.

    Uses ``BigInteger`` serial PK (not UUID) so that temporal ordering is
    implicit in the PK, enabling efficient range scans for compliance exports.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # ── Identity ──────────────────────────────────────────────────────────────
    # OAuth2 subject claim of the actor (physician ID, service account, etc.).
    actor_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    # physician | researcher | admin | patient | service
    actor_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # ── Action ────────────────────────────────────────────────────────────────
    # Verb: CREATE, READ, UPDATE, DELETE, LOGIN, LOGOUT, EXPORT, PREDICT, etc.
    action: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    # FHIR resource type or internal resource name (e.g., "Patient", "Condition").
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    # UUID or other stable identifier of the affected resource — NOT PHI content.
    resource_id: Mapped[str | None] = mapped_column(String(255), index=True)

    # ── Request context ───────────────────────────────────────────────────────
    service_name: Mapped[str | None] = mapped_column(String(100), index=True)
    # Never log user-agent strings that could contain PHI.
    user_agent: Mapped[str | None] = mapped_column(String(500))
    # Redacted to /24 subnet in non-admin views to reduce re-identification risk.
    ip_address: Mapped[str | None] = mapped_column(INET)

    # ── Outcome ───────────────────────────────────────────────────────────────
    # success | failure | error
    outcome: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    outcome_detail: Mapped[str | None] = mapped_column(String(1000))

    # ── Temporal ──────────────────────────────────────────────────────────────
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # ── PHI-free metadata ─────────────────────────────────────────────────────
    # Allowed fields: model_version, prediction_score, data_source, etc.
    # Run through libs.common.phi.redact_dict() before inserting.
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB, default=dict)
