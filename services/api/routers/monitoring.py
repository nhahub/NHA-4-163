"""Model monitoring & fairness endpoints (Tier 6 — ML Trust & Decision Support).

GET /monitoring/drift     — PSI drift between a reference and a current window
GET /monitoring/fairness  — risk-score parity across demographic subgroups

Both read from the append-only ``prediction_log`` table; the statistics are
computed by the pure, dependency-free :mod:`services.api.services.monitoring_service`.
Important for a regulated PHI model: it complements the MLflow reliability/Brier
gate required before release.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import select

from libs.common.models.patient import Patient
from libs.common.models.prediction_log import PredictionLog
from services.api.db import DbSession
from services.api.schemas.monitoring_schemas import (
    DriftBinSchema,
    DriftResponse,
    FairnessGroupResult,
    FairnessResponse,
    GroupStatSchema,
)
from services.api.services.monitoring_service import (
    fairness_report,
    population_stability_index,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["decision-support"])


@router.get(
    "/drift",
    response_model=DriftResponse,
    summary="Risk-score drift (Population Stability Index)",
)
async def get_drift(
    db: DbSession,
    reference_window_days: int = Query(default=90, ge=1, le=730),
    current_window_days: int = Query(default=30, ge=1, le=365),
    bins: int = Query(default=10, ge=2, le=50),
) -> DriftResponse:
    """Compute PSI drift between a reference and a recent score window.

    The reference window is the ``reference_window_days`` period ending where
    the current window begins; the current window is the most recent
    ``current_window_days``.

    Args:
        db: Async database session.
        reference_window_days: Length of the baseline window.
        current_window_days: Length of the recent window.
        bins: Number of score buckets.

    Returns:
        A :class:`DriftResponse`.
    """
    now = datetime.now(UTC)
    current_start = now - timedelta(days=current_window_days)
    reference_start = current_start - timedelta(days=reference_window_days)

    rows = (
        await db.execute(
            select(PredictionLog.risk_score, PredictionLog.predicted_at).where(
                PredictionLog.predicted_at >= reference_start
            )
        )
    ).all()

    reference = [float(s) for s, ts in rows if ts is not None and ts < current_start]
    current = [float(s) for s, ts in rows if ts is not None and ts >= current_start]

    result = population_stability_index(reference, current, bins=bins)
    log.info(
        "Drift computed: psi=%.3f verdict=%s ref=%d cur=%d",
        result.psi,
        result.verdict,
        result.reference_count,
        result.current_count,
    )
    return DriftResponse(
        psi=result.psi,
        verdict=result.verdict,
        reference_count=result.reference_count,
        current_count=result.current_count,
        reference_window_days=reference_window_days,
        current_window_days=current_window_days,
        bins=[DriftBinSchema(**vars(b)) for b in result.bins],
    )


def _age_band(dob: date | None) -> str:
    """Bucket a date of birth into a coarse age band (PHI-safe grouping)."""
    if dob is None:
        return "unknown"
    today = datetime.now(UTC).date()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    if age < 18:
        return "0-17"
    if age < 40:
        return "18-39"
    if age < 65:
        return "40-64"
    return "65+"


_ATTRIBUTES = ("sex", "age_band", "ethnicity", "race")


@router.get(
    "/fairness",
    response_model=FairnessResponse,
    summary="Risk-score parity across demographic subgroups",
)
async def get_fairness(
    db: DbSession,
    lookback_days: int = Query(default=180, ge=1, le=1095),
    high_risk_threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    min_group_size: int = Query(default=5, ge=1, le=1000),
) -> FairnessResponse:
    """Assess risk-score parity across sex, age band, ethnicity and race.

    Uses the most recent prediction per patient within the lookback window so
    that frequently-scored patients do not dominate the statistics.

    Args:
        db: Async database session.
        lookback_days: How far back to consider predictions.
        high_risk_threshold: Score at/above which a patient counts as high risk.
        min_group_size: Minimum subgroup size to include (avoids tiny cohorts).

    Returns:
        A :class:`FairnessResponse` with one result per demographic attribute.
    """
    since = datetime.now(UTC) - timedelta(days=lookback_days)
    rows = (
        await db.execute(
            select(
                PredictionLog.patient_id,
                PredictionLog.risk_score,
                PredictionLog.predicted_at,
                Patient.gender,
                Patient.date_of_birth,
                Patient.ethnicity,
                Patient.race,
            )
            .join(Patient, Patient.id == PredictionLog.patient_id)
            .where(PredictionLog.predicted_at >= since)
            .order_by(PredictionLog.predicted_at.desc())
        )
    ).all()

    # Keep the latest prediction per patient (rows are newest-first).
    latest: dict[str, Any] = {}
    for pid, score, _ts, gender, dob, ethnicity, race in rows:
        if pid in latest:
            continue
        latest[pid] = (float(score), gender, dob, ethnicity, race)

    grouped: dict[str, dict[str, list[float]]] = {a: {} for a in _ATTRIBUTES}
    for score, gender, dob, ethnicity, race in latest.values():
        values = {
            "sex": (gender.value if hasattr(gender, "value") else gender) or "unknown",
            "age_band": _age_band(dob),
            "ethnicity": ethnicity or "unknown",
            "race": race or "unknown",
        }
        for attr, label in values.items():
            grouped[attr].setdefault(str(label), []).append(score)

    attributes: list[FairnessGroupResult] = []
    for attr in _ATTRIBUTES:
        report = fairness_report(
            attr,
            grouped[attr],
            high_risk_threshold=high_risk_threshold,
            min_group_size=min_group_size,
        )
        attributes.append(
            FairnessGroupResult(
                attribute=report.attribute,
                groups=[GroupStatSchema(**vars(g)) for g in report.groups],
                disparate_impact_ratio=report.disparate_impact_ratio,
                statistical_parity_difference=report.statistical_parity_difference,
                passes_four_fifths=report.passes_four_fifths,
                interpretation=report.interpretation,
            )
        )

    log.info("Fairness computed over %d patients", len(latest))
    return FairnessResponse(
        predictions_evaluated=len(latest),
        high_risk_threshold=high_risk_threshold,
        attributes=attributes,
    )
