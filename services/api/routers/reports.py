"""Clinical PDF report endpoints.

GET /patients/{id}/report/pdf
    Generates a one-page PDF clinical summary containing patient demographics,
    the most recent hereditary-risk score/tier, active conditions, and the top
    SHAP risk factors from that prediction.

The report is assembled from persisted records (the patient row, its
conditions, and the latest ``PredictionLog`` entry) rather than triggering a
live model inference, so it is deterministic and does not depend on Neo4j/MLflow
availability at request time.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from sqlalchemy import select

from libs.common.models.condition import Condition
from libs.common.models.patient import Patient
from libs.common.models.prediction_log import PredictionLog
from services.api.db import DbSession
from services.api.services.pdf_service import ReportData, generate_patient_report

log = logging.getLogger(__name__)

router = APIRouter(prefix="/patients", tags=["reports"])

_ACTIVE_STATUSES = {"active", "recurrence", "relapse", "confirmed"}


@router.get(
    "/{patient_id}/report/pdf",
    summary="Generate a clinical risk report (PDF)",
    responses={200: {"content": {"application/pdf": {}}}},
)
async def get_patient_report_pdf(patient_id: uuid.UUID, db: DbSession) -> Response:
    """Generate and return a one-page clinical risk report as a PDF.

    Args:
        patient_id: Patient UUID.
        db: Async database session.

    Returns:
        A ``Response`` with ``application/pdf`` content.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    # ── Active conditions ─────────────────────────────────────────────────────
    cond_result = await db.execute(select(Condition).where(Condition.patient_id == patient_id))
    conditions = [
        (c.code, c.code_display or c.code_text or "", str(c.clinical_status.value))
        for c in cond_result.scalars().all()
        if c.clinical_status.value in _ACTIVE_STATUSES
    ]

    # ── Most recent prediction ────────────────────────────────────────────────
    pred_result = await db.execute(
        select(PredictionLog)
        .where(PredictionLog.patient_id == patient_id)
        .order_by(PredictionLog.predicted_at.desc())
        .limit(1)
    )
    latest = pred_result.scalars().first()

    risk_score = float(latest.risk_score) if latest is not None else None
    risk_tier = latest.risk_tier if latest is not None else None
    shap_factors = _normalise_shap(latest.shap_top_factors) if latest is not None else []

    full_name = " ".join(part for part in (patient.given_name, patient.family_name) if part)
    gender = patient.gender.value if patient.gender is not None else None

    data = ReportData(
        patient_id=str(patient.id),
        full_name=full_name,
        date_of_birth=patient.date_of_birth,
        gender=gender,
        risk_score=risk_score,
        risk_tier=risk_tier,
        conditions=conditions,
        shap_factors=shap_factors,
    )

    pdf_bytes = generate_patient_report(data)
    log.info("Generated PDF report for patient %s (%d bytes)", patient_id, len(pdf_bytes))

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": (f'attachment; filename="risk_report_{patient_id}.pdf"')},
    )


def _normalise_shap(raw: object) -> list[dict[str, Any]]:
    """Coerce a stored ``shap_top_factors`` value into a list of dicts.

    The column may hold either a JSON list of contribution dicts or a
    ``{"factors": [...]}`` wrapper depending on the writer.

    Args:
        raw: The raw JSONB value from ``PredictionLog.shap_top_factors``.

    Returns:
        A list of SHAP contribution dicts (possibly empty).
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        factors = raw.get("factors")
        return factors if isinstance(factors, list) else []
    if isinstance(raw, list):
        return raw
    return []
