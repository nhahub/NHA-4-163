"""Notification ORM model — clinician alerts for risk and workflow events (Tier 4).

Notifications are generated when a patient's hereditary-risk score crosses a
configured threshold, when new family-history data materially changes a risk
profile, or as periodic screening reminders for high-risk patients.

PHI note: a notification stores a ``patient_id`` reference and non-PHI context
(risk score, tier, threshold).  The ``title``/``message`` text must never embed
patient names, dates of birth, or other direct identifiers.

FHIR analogue: loosely maps to the ``Communication`` / ``Flag`` resources, but
this is an internal operational construct, not a FHIR resource.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from libs.common.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from libs.common.models.patient import Patient


class NotificationType(enum.StrEnum):
    """The event that triggered a notification."""

    RISK_THRESHOLD_CROSSED = "risk_threshold_crossed"
    RISK_INCREASED = "risk_increased"
    FAMILY_UPDATE = "family_update"
    SCREENING_REMINDER = "screening_reminder"


class NotificationSeverity(enum.StrEnum):
    """Relative urgency for UI sorting and filtering."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class NotificationStatus(enum.StrEnum):
    """Lifecycle state of a notification."""

    UNREAD = "unread"
    READ = "read"
    DISMISSED = "dismissed"


class Notification(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One alert about a patient directed at clinical staff."""

    __tablename__ = "notification"

    # ── Scope ─────────────────────────────────────────────────────────────────
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Multi-tenant scope (nullable — single-tenant deployments leave it NULL).
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organization.id", ondelete="CASCADE"),
        index=True,
    )

    # ── Classification ────────────────────────────────────────────────────────
    notification_type: Mapped[NotificationType] = mapped_column(
        Enum(NotificationType, name="notification_type"), nullable=False, index=True
    )
    severity: Mapped[NotificationSeverity] = mapped_column(
        Enum(NotificationSeverity, name="notification_severity"),
        nullable=False,
        index=True,
    )
    status: Mapped[NotificationStatus] = mapped_column(
        Enum(NotificationStatus, name="notification_status"),
        nullable=False,
        default=NotificationStatus.UNREAD,
        index=True,
    )

    # ── Content (PHI-free) ────────────────────────────────────────────────────
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(String(1000), nullable=False)

    # ── Risk context ──────────────────────────────────────────────────────────
    risk_score: Mapped[float | None] = mapped_column(Numeric(precision=6, scale=5))
    risk_tier: Mapped[str | None] = mapped_column(String(20))
    threshold: Mapped[float | None] = mapped_column(Numeric(precision=6, scale=5))

    # ── Delivery ──────────────────────────────────────────────────────────────
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # ── Extra non-PHI metadata ────────────────────────────────────────────────
    context: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=dict)

    # ── Relationships ─────────────────────────────────────────────────────────
    patient: Mapped[Patient] = relationship()
