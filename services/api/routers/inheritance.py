"""Mendelian inheritance calculator endpoints (Tier 5 — Genetics & Genomics).

POST /patients/{id}/inheritance-risk — carrier/affected probabilities across pedigree
GET  /inheritance/models             — supported inheritance modes + defaults

The computation is deterministic and explainable (see
:mod:`services.api.services.inheritance_service`); it complements the ML risk
model rather than replacing it.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from libs.common.models.family_member_history import FamilyMemberHistory
from libs.common.models.patient import Patient
from services.api.db import DbSession
from services.api.schemas.inheritance_schemas import (
    InheritanceModelInfo,
    InheritanceRiskRequest,
    InheritanceRiskResponse,
    RelativeRiskResult,
)
from services.api.services.inheritance_service import (
    INHERITANCE_MODELS,
    categorise_relationship,
    compute_relative_risk,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["genetics"])


@router.get(
    "/inheritance/models",
    response_model=list[InheritanceModelInfo],
    summary="List supported inheritance models",
)
async def list_inheritance_models() -> list[InheritanceModelInfo]:
    """Return the supported inheritance modes with their default parameters.

    Returns:
        A list of :class:`InheritanceModelInfo`.
    """
    return [
        InheritanceModelInfo(
            key=m.key,
            display=m.display,
            default_penetrance=m.default_penetrance,
            default_carrier_frequency=m.default_carrier_frequency,
            sex_linked=m.sex_linked,
            description=m.description,
        )
        for m in INHERITANCE_MODELS.values()
    ]


@router.post(
    "/patients/{patient_id}/inheritance-risk",
    response_model=InheritanceRiskResponse,
    summary="Compute Mendelian carrier/affected probabilities across the pedigree",
)
async def compute_inheritance_risk(
    patient_id: uuid.UUID, body: InheritanceRiskRequest, db: DbSession
) -> InheritanceRiskResponse:
    """Compute Mendelian risk for every recorded relative of the proband.

    Args:
        patient_id: Proband (affected patient) UUID.
        body: Inheritance mode plus optional penetrance/carrier-frequency overrides.
        db: Async database session.

    Returns:
        Per-relative carrier and affected probabilities with an explanation.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    model = INHERITANCE_MODELS[body.inheritance_mode]
    penetrance = model.default_penetrance if body.penetrance is None else body.penetrance
    carrier_freq = (
        model.default_carrier_frequency
        if body.carrier_frequency is None
        else body.carrier_frequency
    )

    members = (
        (
            await db.execute(
                select(FamilyMemberHistory)
                .where(FamilyMemberHistory.patient_id == patient_id)
                .order_by(FamilyMemberHistory.degree_of_relatedness.desc())
            )
        )
        .scalars()
        .all()
    )

    results: list[RelativeRiskResult] = []
    for m in members:
        degree = float(m.degree_of_relatedness) if m.degree_of_relatedness is not None else None
        risk = compute_relative_risk(
            mode=body.inheritance_mode,
            relationship_code=m.relationship_code,
            degree_of_relatedness=degree,
            relative_sex=m.sex,
            penetrance=penetrance,
            carrier_frequency=carrier_freq,
        )
        results.append(
            RelativeRiskResult(
                family_member_id=m.id,
                related_patient_id=m.related_patient_id,
                relationship_code=m.relationship_code,
                relationship_display=m.relationship_display,
                relationship_category=categorise_relationship(m.relationship_code),
                degree_of_relatedness=degree,
                carrier_probability=risk.carrier_probability,
                affected_probability=risk.affected_probability,
                basis=risk.basis,
            )
        )

    # Rank most at-risk relatives first (by affected, then carrier probability).
    results.sort(key=lambda r: (r.affected_probability, r.carrier_probability), reverse=True)

    log.info(
        "Inheritance risk computed: patient=%s mode=%s relatives=%d",
        patient_id,
        body.inheritance_mode,
        len(results),
    )
    return InheritanceRiskResponse(
        patient_id=patient_id,
        inheritance_mode=model.key,
        inheritance_display=model.display,
        penetrance=penetrance,
        carrier_frequency=carrier_freq,
        condition_code=body.condition_code,
        condition_display=body.condition_display,
        relatives_evaluated=len(results),
        results=results,
    )
