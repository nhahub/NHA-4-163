"""Notification endpoints (Tier 4).

GET    /notifications                        — list (filter by status/severity/patient)
GET    /notifications/summary                — aggregate counts
POST   /patients/{id}/notifications/evaluate — evaluate latest risk, create alerts
POST   /patients/{id}/notifications          — manually create a notification
POST   /notifications/{id}/read              — mark read
POST   /notifications/{id}/dismiss           — dismiss
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from libs.common.models.notification import (
    Notification,
    NotificationSeverity,
    NotificationStatus,
    NotificationType,
)
from libs.common.models.patient import Patient
from services.api.db import DbSession
from services.api.schemas.crud_schemas import PaginatedResponse
from services.api.schemas.notification_schemas import (
    EvaluateResponse,
    NotificationCreate,
    NotificationResponse,
    NotificationSummary,
)
from services.api.services.notification_service import evaluate_patient_notifications

log = logging.getLogger(__name__)

router = APIRouter(tags=["notifications"])


@router.get(
    "/notifications",
    response_model=PaginatedResponse[NotificationResponse],
    summary="List notifications",
)
async def list_notifications(
    db: DbSession,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status_filter: str | None = Query(
        default=None, alias="status", pattern=r"^(unread|read|dismissed)$"
    ),
    severity: str | None = Query(default=None, pattern=r"^(info|warning|critical)$"),
    patient_id: uuid.UUID | None = Query(default=None),
) -> PaginatedResponse[NotificationResponse]:
    """List notifications with optional filters, newest first.

    Args:
        db: Async database session.
        page: 1-indexed page number.
        page_size: Items per page.
        status_filter: Optional lifecycle status filter (``status`` query param).
        severity: Optional severity filter.
        patient_id: Optional patient filter.

    Returns:
        Paginated notifications.
    """
    query = select(Notification)
    if status_filter:
        query = query.where(Notification.status == NotificationStatus(status_filter))
    if severity:
        query = query.where(Notification.severity == NotificationSeverity(severity))
    if patient_id:
        query = query.where(Notification.patient_id == patient_id)

    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar() or 0

    offset = (page - 1) * page_size
    query = query.order_by(Notification.created_at.desc()).offset(offset).limit(page_size)
    rows = (await db.execute(query)).scalars().all()

    return PaginatedResponse(
        items=[NotificationResponse.model_validate(n) for n in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


@router.get(
    "/notifications/summary",
    response_model=NotificationSummary,
    summary="Notification counts",
)
async def notification_summary(db: DbSession) -> NotificationSummary:
    """Return aggregate notification counts by status, severity, and type.

    Args:
        db: Async database session.

    Returns:
        A :class:`NotificationSummary`.
    """
    total = (await db.execute(select(func.count()).select_from(Notification))).scalar() or 0
    unread = (
        await db.execute(
            select(func.count())
            .select_from(Notification)
            .where(Notification.status == NotificationStatus.UNREAD)
        )
    ).scalar() or 0

    sev_rows = await db.execute(
        select(Notification.severity, func.count()).group_by(Notification.severity)
    )
    by_severity = {sev.value: count for sev, count in sev_rows.all()}

    type_rows = await db.execute(
        select(Notification.notification_type, func.count()).group_by(
            Notification.notification_type
        )
    )
    by_type = {t.value: count for t, count in type_rows.all()}

    return NotificationSummary(total=total, unread=unread, by_severity=by_severity, by_type=by_type)


@router.post(
    "/patients/{patient_id}/notifications/evaluate",
    response_model=EvaluateResponse,
    summary="Evaluate latest risk and create alerts",
)
async def evaluate_notifications(patient_id: uuid.UUID, db: DbSession) -> EvaluateResponse:
    """Evaluate the patient's latest prediction and create any warranted alerts.

    Args:
        patient_id: Patient UUID.
        db: Async database session.

    Returns:
        The created notifications (possibly empty) and a summary message.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    created = await evaluate_patient_notifications(
        db, patient_id, organization_id=patient.organization_id
    )
    return EvaluateResponse(
        patient_id=patient_id,
        created=[NotificationResponse.model_validate(n) for n in created],
        message=(
            f"Created {len(created)} notification(s)."
            if created
            else "No notification warranted (no prediction, or risk unchanged/below threshold)."
        ),
    )


@router.post(
    "/patients/{patient_id}/notifications",
    response_model=NotificationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a notification manually",
)
async def create_notification(
    patient_id: uuid.UUID, body: NotificationCreate, db: DbSession
) -> NotificationResponse:
    """Manually create a notification for a patient (e.g., screening reminder).

    Args:
        patient_id: Patient UUID.
        body: Notification content.
        db: Async database session.

    Returns:
        The created notification.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    notification = Notification(
        patient_id=patient_id,
        organization_id=patient.organization_id,
        notification_type=NotificationType(body.notification_type),
        severity=NotificationSeverity(body.severity),
        title=body.title,
        message=body.message,
    )
    db.add(notification)
    await db.flush()
    await db.refresh(notification)
    return NotificationResponse.model_validate(notification)


async def _set_status(
    db: DbSession, notification_id: uuid.UUID, new_status: NotificationStatus
) -> Notification:
    """Load a notification and transition it to ``new_status``."""
    notification = await db.get(Notification, notification_id)
    if notification is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    notification.status = new_status
    if new_status == NotificationStatus.READ:
        notification.read_at = datetime.utcnow()
        notification.acknowledged = True
    await db.flush()
    await db.refresh(notification)
    return notification


@router.post(
    "/notifications/{notification_id}/read",
    response_model=NotificationResponse,
    summary="Mark a notification read",
)
async def mark_read(notification_id: uuid.UUID, db: DbSession) -> NotificationResponse:
    """Mark a notification as read.

    Args:
        notification_id: Notification UUID.
        db: Async database session.

    Returns:
        The updated notification.
    """
    n = await _set_status(db, notification_id, NotificationStatus.READ)
    return NotificationResponse.model_validate(n)


@router.post(
    "/notifications/{notification_id}/dismiss",
    response_model=NotificationResponse,
    summary="Dismiss a notification",
)
async def dismiss(notification_id: uuid.UUID, db: DbSession) -> NotificationResponse:
    """Dismiss a notification.

    Args:
        notification_id: Notification UUID.
        db: Async database session.

    Returns:
        The updated notification.
    """
    n = await _set_status(db, notification_id, NotificationStatus.DISMISSED)
    return NotificationResponse.model_validate(n)
