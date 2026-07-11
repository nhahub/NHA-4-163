"""Guideline-based screening recommendation endpoints (Tier 6).

GET /guidelines                                  — the recommendation catalogue
GET /patients/{id}/screening-recommendations     — actionable next steps

Maps a patient's risk, age, sex and recorded conditions onto established
screening guidelines (NCCN, USPSTF, ACC/AHA) so a clinician sees *what to do*
rather than a bare score. See :mod:`services.api.services.guideline_service`.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from libs.common.models.condition import Condition
from libs.common.models.family_member_history import FamilyMemberHistory
from libs.common.models.patient import Patient
from libs.common.models.prediction_log import PredictionLog
from services.api.db import DbSession
from services.api.schemas.guideline_schemas import (
    GuidelineRecommendationSchema,
    ScreeningRecommendationsResponse,
)
from services.api.services.guideline_service import (
    PatientContext,
    catalogue,
    recommend,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["decision-support"])


@router.get(
    "/guidelines",
    response_model=list[GuidelineRecommendationSchema],
    summary="List the guideline recommendation catalogue",
)
async def list_guidelines() -> list[GuidelineRecommendationSchema]:
    """Return every recommendation the rule base can emit.

    Returns:
        A list of :class:`GuidelineRecommendationSchema`.
    """
    return [GuidelineRecommendationSchema(**vars(r)) for r in catalogue()]


def _age_from_dob(dob: date | None) -> int | None:
    """Integer age in years from a date of birth."""
    if dob is None:
        return None
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


@router.get(
    "/patients/{patient_id}/screening-recommendations",
    response_model=ScreeningRecommendationsResponse,
    summary="Guideline-based screening recommendations for a patient",
)
async def get_screening_recommendations(
    patient_id: uuid.UUID, db: DbSession
) -> ScreeningRecommendationsResponse:
    """Return actionable, guideline-based next steps for a patient.

    The patient context is assembled from stored demographics, conditions,
    family history and the most recent risk prediction.

    Args:
        patient_id: Patient UUID.
        db: Async database session.

    Returns:
        The matching recommendations, most urgent first.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    conditions = (
        (await db.execute(select(Condition).where(Condition.patient_id == patient_id)))
        .scalars()
        .all()
    )
    codes = frozenset((c.code or "").upper() for c in conditions if c.code)
    has_hereditary = any(c.is_hereditary for c in conditions)

    members = (
        await db.execute(
            select(FamilyMemberHistory.degree_of_relatedness, FamilyMemberHistory.conditions).where(
                FamilyMemberHistory.patient_id == patient_id
            )
        )
    ).all()
    first_degree_affected = sum(
        1 for degree, conds in members if conds and degree is not None and float(degree) >= 0.5
    )

    latest = (
        await db.execute(
            select(PredictionLog.risk_score, PredictionLog.risk_tier)
            .where(PredictionLog.patient_id == patient_id)
            .order_by(PredictionLog.predicted_at.desc())
            .limit(1)
        )
    ).first()
    risk_score = float(latest[0]) if latest else 0.0
    risk_tier = latest[1] if latest else None

    gender = patient.gender.value if patient.gender is not None else None
    ctx = PatientContext(
        age=_age_from_dob(patient.date_of_birth),
        sex=gender,
        risk_score=risk_score,
        condition_codes=codes,
        has_hereditary_condition=has_hereditary,
        affected_first_degree_relatives=first_degree_affected,
    )
    recs = recommend(ctx)
    log.info(
        "Screening recommendations: patient=%s risk=%.3f matched=%d",
        patient_id,
        risk_score,
        len(recs),
    )
    return ScreeningRecommendationsResponse(
        patient_id=patient_id,
        age=ctx.age,
        sex=ctx.sex,
        risk_score=round(risk_score, 4),
        risk_tier=risk_tier,
        affected_first_degree_relatives=first_degree_affected,
        recommendations=[GuidelineRecommendationSchema(**vars(r)) for r in recs],
    )
