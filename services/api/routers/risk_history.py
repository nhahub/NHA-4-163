"""Risk history and trend tracking endpoints.

GET /patients/{id}/risk-history        — Risk score time series
GET /patients/{id}/risk-history/latest — Most recent prediction
GET /patients/{id}/risk-history/trend  — Trend analysis
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from libs.common.models.prediction_log import PredictionLog
from services.api.db import DbSession
from services.api.schemas.crud_schemas import (
    RiskHistoryEntry,
    RiskTrendResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["risk-history"])


@router.get(
    "/patients/{patient_id}/risk-history",
    response_model=list[RiskHistoryEntry],
    summary="Risk score time series",
)
async def get_risk_history(
    patient_id: uuid.UUID,
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=500),
    source: str | None = Query(default=None, pattern=r"^(api|batch|scheduled)$"),
) -> list[RiskHistoryEntry]:
    """Get the prediction history for a patient.

    Args:
        patient_id: Patient UUID.
        db: Async database session.
        limit: Maximum number of entries to return.
        source: Optional source filter (api/batch/scheduled).

    Returns:
        List of prediction log entries, newest first.
    """
    query = select(PredictionLog).where(PredictionLog.patient_id == patient_id)

    if source:
        query = query.where(PredictionLog.source == source)

    query = query.order_by(PredictionLog.predicted_at.desc()).limit(limit)
    result = await db.execute(query)
    entries = result.scalars().all()

    return [RiskHistoryEntry.model_validate(e) for e in entries]


@router.get(
    "/patients/{patient_id}/risk-history/latest",
    response_model=RiskHistoryEntry,
    summary="Most recent prediction",
)
async def get_latest_risk(patient_id: uuid.UUID, db: DbSession) -> RiskHistoryEntry:
    """Get the most recent prediction for a patient.

    Args:
        patient_id: Patient UUID.
        db: Async database session.

    Returns:
        Most recent prediction log entry.

    Raises:
        HTTPException 404: No predictions found.
    """
    query = (
        select(PredictionLog)
        .where(PredictionLog.patient_id == patient_id)
        .order_by(PredictionLog.predicted_at.desc())
        .limit(1)
    )
    result = await db.execute(query)
    entry = result.scalars().first()

    if entry is None:
        raise HTTPException(status_code=404, detail="No predictions found for this patient")

    return RiskHistoryEntry.model_validate(entry)


@router.get(
    "/patients/{patient_id}/risk-history/trend",
    response_model=RiskTrendResponse,
    summary="Risk trend analysis",
)
async def get_risk_trend(
    patient_id: uuid.UUID,
    db: DbSession,
    window: int = Query(
        default=10, ge=2, le=100, description="Number of recent predictions to analyze"
    ),
) -> RiskTrendResponse:
    """Analyze the risk trend for a patient (improving/worsening/stable).

    Compares the most recent prediction against the previous one and
    calculates the percentage change.

    Args:
        patient_id: Patient UUID.
        db: Async database session.
        window: Number of recent predictions to include in the analysis.

    Returns:
        RiskTrendResponse with trend direction, change %, and history.
    """
    query = (
        select(PredictionLog)
        .where(PredictionLog.patient_id == patient_id)
        .order_by(PredictionLog.predicted_at.desc())
        .limit(window)
    )
    result = await db.execute(query)
    entries = list(result.scalars().all())

    total = len(entries)

    if total == 0:
        return RiskTrendResponse(
            patient_id=patient_id,
            trend="insufficient_data",
            total_predictions=0,
        )

    if total == 1:
        entry = entries[0]
        return RiskTrendResponse(
            patient_id=patient_id,
            current_score=float(entry.risk_score),
            trend="insufficient_data",
            total_predictions=1,
            history=[RiskHistoryEntry.model_validate(entry)],
        )

    current = entries[0]
    previous = entries[1]
    current_score = float(current.risk_score)
    previous_score = float(previous.risk_score)

    if previous_score == 0:
        change_pct = 0.0
    else:
        change_pct = round(((current_score - previous_score) / previous_score) * 100, 2)

    # Determine trend direction
    threshold = 5.0  # 5% change threshold
    if abs(change_pct) < threshold:
        trend = "stable"
    elif change_pct > 0:
        trend = "worsening"
    else:
        trend = "improving"

    return RiskTrendResponse(
        patient_id=patient_id,
        current_score=current_score,
        previous_score=previous_score,
        trend=trend,
        change_pct=change_pct,
        total_predictions=total,
        history=[RiskHistoryEntry.model_validate(e) for e in entries],
    )
