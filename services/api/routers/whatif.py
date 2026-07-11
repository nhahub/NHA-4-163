"""What-If risk simulator endpoints (Tier 6 — ML Trust & Decision Support).

GET  /whatif/factors                 — list the modifiable risk factors
POST /patients/{id}/whatif           — recompute risk under counterfactual changes

The simulation is deterministic and explainable and performs **no writes** to
the database: it seeds a baseline factor vector from the patient's stored
records (unless one is supplied), applies the counterfactual modifications
in-memory, and reports the risk delta with a per-factor breakdown that mirrors
the production model's SHAP explanations. See
:mod:`services.api.services.whatif_service`.
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
from services.api.db import DbSession
from services.api.schemas.whatif_schemas import (
    FactorContributionSchema,
    RiskFactorInfo,
    WhatIfRequest,
    WhatIfResponse,
)
from services.api.services.whatif_service import RISK_FACTORS, simulate

log = logging.getLogger(__name__)

router = APIRouter(tags=["decision-support"])


@router.get(
    "/whatif/factors",
    response_model=list[RiskFactorInfo],
    summary="List modifiable what-if risk factors",
)
async def list_whatif_factors() -> list[RiskFactorInfo]:
    """Return the interpretable factors the simulator accepts.

    Returns:
        A list of :class:`RiskFactorInfo`.
    """
    return [
        RiskFactorInfo(
            key=f.key,
            display=f.display,
            coefficient=f.coefficient,
            default=f.default,
            unit=f.unit,
            kind=f.kind,
        )
        for f in RISK_FACTORS.values()
    ]


def _age_from_dob(dob: date | None) -> float | None:
    """Compute integer age in years from a date of birth."""
    if dob is None:
        return None
    today = date.today()
    return float(today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day)))


async def _derive_baseline(db: DbSession, patient: Patient) -> dict[str, float]:
    """Seed baseline factors from a patient's stored clinical & family records."""
    baseline: dict[str, float] = {}

    age = _age_from_dob(patient.date_of_birth)
    if age is not None:
        baseline["age_years"] = age

    conditions = (
        (await db.execute(select(Condition).where(Condition.patient_id == patient.id)))
        .scalars()
        .all()
    )
    baseline["comorbidity_count"] = float(len(conditions))
    baseline["hereditary_condition_count"] = float(sum(1 for c in conditions if c.is_hereditary))

    members = (
        (
            await db.execute(
                select(FamilyMemberHistory).where(FamilyMemberHistory.patient_id == patient.id)
            )
        )
        .scalars()
        .all()
    )
    first_deg = second_deg = affected = 0
    for m in members:
        degree = float(m.degree_of_relatedness) if m.degree_of_relatedness is not None else 0.0
        is_affected = bool(m.conditions)
        if degree >= 0.5:
            first_deg += is_affected
        elif degree >= 0.25:
            second_deg += is_affected
        affected += is_affected
    baseline["affected_first_degree_relatives"] = float(first_deg)
    baseline["affected_second_degree_relatives"] = float(second_deg)
    if members:
        baseline["family_risk_prevalence"] = round(affected / len(members), 4)

    return baseline


@router.post(
    "/patients/{patient_id}/whatif",
    response_model=WhatIfResponse,
    summary="Recompute risk under counterfactual factor changes",
)
async def simulate_what_if(
    patient_id: uuid.UUID, body: WhatIfRequest, db: DbSession
) -> WhatIfResponse:
    """Run a counterfactual ("what-if") hereditary-risk simulation.

    The baseline is taken from ``body.baseline`` when provided, otherwise seeded
    from the patient's stored records. No data is written back.

    Args:
        patient_id: Patient UUID.
        body: Optional baseline plus the counterfactual modifications.
        db: Async database session.

    Returns:
        Baseline vs simulated risk with a per-factor breakdown.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    baseline = body.baseline
    if baseline is None:
        baseline = await _derive_baseline(db, patient)

    result = simulate(baseline, body.modifications)
    log.info(
        "What-if simulated: patient=%s baseline=%.3f simulated=%.3f delta=%.3f",
        patient_id,
        result.baseline_risk,
        result.simulated_risk,
        result.risk_delta,
    )
    return WhatIfResponse(
        patient_id=patient_id,
        baseline_risk=result.baseline_risk,
        simulated_risk=result.simulated_risk,
        risk_delta=result.risk_delta,
        baseline_factors=result.baseline_factors,
        simulated_factors=result.simulated_factors,
        contributions=[FactorContributionSchema(**vars(c)) for c in result.contributions],
        interpretation=result.interpretation,
    )
