"""Notification generation and evaluation logic (Tier 4).

Encapsulates the rules that turn risk predictions and workflow events into
clinician notifications, plus the pure threshold logic so it can be unit-tested
without a database.

Trigger rules
-------------
1. ``risk_threshold_crossed`` (critical) — the latest risk score is at or above
   the alerting threshold and the previous score was below it (a fresh
   crossing), or there is no prior score and the latest is already above.
2. ``risk_increased`` (warning) — the latest score rose by at least
   ``MIN_INCREASE`` versus the previous score, without necessarily crossing the
   threshold.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.models.notification import (
    Notification,
    NotificationSeverity,
    NotificationType,
)
from libs.common.models.prediction_log import PredictionLog

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = float(os.environ.get("NOTIFY_RISK_THRESHOLD", "0.75"))
MIN_INCREASE = float(os.environ.get("NOTIFY_MIN_INCREASE", "0.15"))


@dataclass(frozen=True)
class RiskEvent:
    """A decision about whether/what notification a risk change warrants."""

    notification_type: NotificationType
    severity: NotificationSeverity
    title: str
    message: str


def evaluate_risk_change(
    current_score: float | None,
    previous_score: float | None,
    risk_tier: str | None,
    threshold: float = DEFAULT_THRESHOLD,
    min_increase: float = MIN_INCREASE,
) -> RiskEvent | None:
    """Decide which notification (if any) a risk change warrants — pure logic.

    Args:
        current_score: Latest calibrated risk probability, or ``None``.
        previous_score: Prior risk probability, or ``None`` if this is the first.
        risk_tier: Latest tier label (for the message).
        threshold: Alerting threshold in [0, 1].
        min_increase: Minimum absolute rise to warrant a ``risk_increased`` alert.

    Returns:
        A :class:`RiskEvent`, or ``None`` when no notification is warranted.
    """
    if current_score is None:
        return None

    crossed = current_score >= threshold and (previous_score is None or previous_score < threshold)
    if crossed:
        return RiskEvent(
            notification_type=NotificationType.RISK_THRESHOLD_CROSSED,
            severity=NotificationSeverity.CRITICAL,
            title="High hereditary risk detected",
            message=(
                f"Patient risk score {current_score:.0%} "
                f"({(risk_tier or 'high').replace('_', ' ')}) has crossed the "
                f"alerting threshold of {threshold:.0%}. Consider genetic "
                f"counselling referral and enhanced screening."
            ),
        )

    if previous_score is not None and (current_score - previous_score) >= min_increase:
        return RiskEvent(
            notification_type=NotificationType.RISK_INCREASED,
            severity=NotificationSeverity.WARNING,
            title="Rising hereditary risk",
            message=(
                f"Patient risk score rose from {previous_score:.0%} to "
                f"{current_score:.0%}. Review recent clinical or family-history "
                f"changes."
            ),
        )

    return None


async def _latest_two_scores(
    db: AsyncSession, patient_id: uuid.UUID
) -> tuple[PredictionLog | None, float | None]:
    """Return the latest PredictionLog and the previous score for a patient."""
    result = await db.execute(
        select(PredictionLog)
        .where(PredictionLog.patient_id == patient_id)
        .order_by(PredictionLog.predicted_at.desc())
        .limit(2)
    )
    rows = result.scalars().all()
    latest = rows[0] if rows else None
    previous = float(rows[1].risk_score) if len(rows) > 1 else None
    return latest, previous


async def evaluate_patient_notifications(
    db: AsyncSession,
    patient_id: uuid.UUID,
    organization_id: uuid.UUID | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Notification]:
    """Evaluate a patient's latest prediction and persist any warranted alerts.

    Deduplicates: if an unread notification of the same type already exists for
    the patient, no new one is created.

    Args:
        db: Async database session.
        patient_id: Patient to evaluate.
        organization_id: Owning tenant, stamped onto created notifications.
        threshold: Alerting threshold in [0, 1].

    Returns:
        The list of newly created :class:`Notification` rows (possibly empty).
    """
    latest, previous = await _latest_two_scores(db, patient_id)
    if latest is None:
        return []

    event = evaluate_risk_change(
        current_score=float(latest.risk_score),
        previous_score=previous,
        risk_tier=latest.risk_tier,
        threshold=threshold,
    )
    if event is None:
        return []

    # Dedup against an existing unread notification of the same type.
    existing = await db.execute(
        select(func.count())
        .select_from(Notification)
        .where(
            Notification.patient_id == patient_id,
            Notification.notification_type == event.notification_type,
            Notification.status == "unread",
        )
    )
    if (existing.scalar() or 0) > 0:
        return []

    notification = Notification(
        patient_id=patient_id,
        organization_id=organization_id,
        notification_type=event.notification_type,
        severity=event.severity,
        title=event.title,
        message=event.message,
        risk_score=float(latest.risk_score),
        risk_tier=latest.risk_tier,
        threshold=threshold,
    )
    db.add(notification)
    await db.flush()
    await db.refresh(notification)
    log.info(
        "Notification created: patient=%s type=%s severity=%s",
        patient_id,
        event.notification_type.value,
        event.severity.value,
    )
    return [notification]
