"""Polygenic Risk Score endpoints (Tier 5 — Genetics & Genomics).

GET /prs/panels                     — list supported PRS panels
GET /patients/{id}/polygenic-risk   — blended PRS + ML risk for a disease

The PRS is computed from the patient's stored (annotated) variants and blended
with their latest persisted ML prediction; see
:mod:`services.api.services.prs_service`.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from libs.common.models.genetic_test import Variant, Zygosity
from libs.common.models.patient import Patient
from libs.common.models.prediction_log import PredictionLog
from services.api.db import DbSession
from services.api.schemas.prs_schemas import PolygenicRiskResponse, PRSPanelInfo
from services.api.services.prs_service import PRS_PANELS, compute_prs

log = logging.getLogger(__name__)

router = APIRouter(tags=["genetics"])

_DISEASE_PATTERN = r"^(coronary_artery_disease|type_2_diabetes|breast_cancer|alzheimer_disease)$"

_DOSAGE = {
    Zygosity.HETEROZYGOUS: 1,
    Zygosity.HEMIZYGOUS: 1,
    Zygosity.HOMOZYGOUS: 2,
    Zygosity.UNKNOWN: 1,  # a reported alt allele implies ≥1 copy
}


@router.get(
    "/prs/panels",
    response_model=list[PRSPanelInfo],
    summary="List supported PRS panels",
)
async def list_prs_panels() -> list[PRSPanelInfo]:
    """Return the supported polygenic risk panels.

    Returns:
        A list of :class:`PRSPanelInfo`.
    """
    return [
        PRSPanelInfo(
            key=p.key,
            display=p.display,
            condition_code=p.condition_code,
            baseline_prevalence=p.baseline_prevalence,
            snp_count=len(p.weights),
        )
        for p in PRS_PANELS.values()
    ]


@router.get(
    "/patients/{patient_id}/polygenic-risk",
    response_model=PolygenicRiskResponse,
    summary="Blended polygenic + ML risk for a disease",
)
async def get_polygenic_risk(
    patient_id: uuid.UUID,
    db: DbSession,
    disease: str = Query(..., pattern=_DISEASE_PATTERN),
    prs_weight: float = Query(default=0.4, ge=0.0, le=1.0),
) -> PolygenicRiskResponse:
    """Compute the patient's PRS for a disease and blend it with the ML risk.

    Risk-allele dosages are derived from the patient's stored variants (by
    rsID); the ML component is the most recent persisted prediction.

    Args:
        patient_id: Patient UUID.
        db: Async database session.
        disease: PRS panel key.
        prs_weight: Weight of the PRS signal in the log-odds blend.

    Returns:
        The blended polygenic + ML risk.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    panel = PRS_PANELS[disease]

    # Build risk-allele dosages from stored variants that appear in the panel.
    rows = (
        await db.execute(
            select(Variant.rs_id, Variant.zygosity).where(
                Variant.patient_id == patient_id,
                Variant.rs_id.in_(list(panel.weights.keys())),
            )
        )
    ).all()
    dosages: dict[str, int] = {}
    for rs_id, zygosity in rows:
        if rs_id:
            dosages[rs_id] = max(dosages.get(rs_id, 0), _DOSAGE.get(zygosity, 1))

    # Latest ML risk, if any.
    latest = (
        await db.execute(
            select(PredictionLog.risk_score)
            .where(PredictionLog.patient_id == patient_id)
            .order_by(PredictionLog.predicted_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    ml_risk = float(latest) if latest is not None else None

    result = compute_prs(disease, dosages, ml_risk=ml_risk, prs_weight=prs_weight)
    log.info(
        "PRS computed: patient=%s disease=%s percentile=%.0f blended=%.3f",
        patient_id,
        disease,
        result.percentile,
        result.blended_risk,
    )
    return PolygenicRiskResponse(
        patient_id=patient_id,
        disease=result.disease,
        display=result.display,
        raw_score=result.raw_score,
        z_score=result.z_score,
        percentile=result.percentile,
        odds_ratio=result.odds_ratio,
        prs_absolute_risk=result.prs_absolute_risk,
        ml_risk=result.ml_risk,
        blended_risk=result.blended_risk,
        snps_used=result.snps_used,
        snps_available=result.snps_available,
        interpretation=result.interpretation,
    )
