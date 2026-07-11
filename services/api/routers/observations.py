"""Clinical observation and vitals endpoints.

POST   /patients/{id}/observations — Record a single observation
POST   /patients/{id}/vitals       — Quick-entry for common vitals
GET    /patients/{id}/observations — List observations
GET    /observations/{id}          — Get single observation
PUT    /observations/{id}          — Update observation
DELETE /observations/{id}          — Delete observation
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from libs.common.models.observation import Observation, ObservationStatus
from libs.common.models.patient import Patient
from services.api.db import DbSession
from services.api.schemas.crud_schemas import (
    ObservationCreate,
    ObservationResponse,
    ObservationUpdate,
    VitalsCreate,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["observations"])

# ── LOINC codes for common vitals ─────────────────────────────────────────────
VITALS_MAP: dict[str, tuple[str, str, str]] = {
    # field_name: (LOINC code, display name, unit)
    "systolic_bp": ("8480-6", "Systolic blood pressure", "mmHg"),
    "diastolic_bp": ("8462-4", "Diastolic blood pressure", "mmHg"),
    "heart_rate": ("8867-4", "Heart rate", "/min"),
    "temperature": ("8310-5", "Body temperature", "Cel"),
    "spo2": ("2708-6", "Oxygen saturation", "%"),
    "weight": ("29463-7", "Body weight", "kg"),
    "height": ("8302-2", "Body height", "cm"),
}


@router.post(
    "/patients/{patient_id}/observations",
    response_model=ObservationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a single observation",
)
async def create_observation(
    patient_id: uuid.UUID, body: ObservationCreate, db: DbSession
) -> ObservationResponse:
    """Record a lab result, vital sign, or clinical assessment.

    Args:
        patient_id: Patient UUID.
        body: Observation data with LOINC code and value.
        db: Async database session.

    Returns:
        Created observation record.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    observation = Observation(
        patient_id=patient_id,
        encounter_id=body.encounter_id,
        status=ObservationStatus(body.status),
        category=body.category,
        code_system=body.code_system,
        code=body.code,
        code_display=body.code_display,
        effective_datetime=body.effective_datetime,
        value_quantity=body.value_quantity,
        value_unit=body.value_unit,
        value_string=body.value_string,
        value_boolean=body.value_boolean,
        interpretation=body.interpretation,
        ref_range_low=body.ref_range_low,
        ref_range_high=body.ref_range_high,
    )
    db.add(observation)
    await db.flush()
    await db.refresh(observation)
    log.info("Observation created: %s for patient %s", observation.id, patient_id)
    return ObservationResponse.model_validate(observation)


@router.post(
    "/patients/{patient_id}/vitals",
    response_model=list[ObservationResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Quick-entry for common vitals",
)
async def record_vitals(
    patient_id: uuid.UUID, body: VitalsCreate, db: DbSession
) -> list[ObservationResponse]:
    """Record common vital signs in a single call.

    Creates one Observation row per non-null vital with pre-configured
    LOINC codes.

    Args:
        patient_id: Patient UUID.
        body: Vitals data (all fields optional, only non-null values recorded).
        db: Async database session.

    Returns:
        List of created observation records.

    Raises:
        HTTPException 404: Patient not found.
        HTTPException 422: No vitals provided.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    created: list[Observation] = []
    vital_data = body.model_dump(exclude={"encounter_id", "effective_datetime"}, exclude_none=True)

    if not vital_data:
        raise HTTPException(status_code=422, detail="At least one vital sign must be provided")

    for field_name, value in vital_data.items():
        if field_name not in VITALS_MAP:
            continue
        loinc_code, display, unit = VITALS_MAP[field_name]
        obs = Observation(
            patient_id=patient_id,
            encounter_id=body.encounter_id,
            status=ObservationStatus.FINAL,
            category="vital-signs",
            code_system="http://loinc.org",
            code=loinc_code,
            code_display=display,
            effective_datetime=body.effective_datetime,
            value_quantity=float(value),
            value_unit=unit,
        )
        db.add(obs)
        created.append(obs)

    await db.flush()
    for obs in created:
        await db.refresh(obs)

    log.info("Vitals recorded: %d observations for patient %s", len(created), patient_id)
    return [ObservationResponse.model_validate(o) for o in created]


@router.get(
    "/patients/{patient_id}/observations",
    response_model=list[ObservationResponse],
    summary="List observations",
)
async def list_observations(
    patient_id: uuid.UUID,
    db: DbSession,
    category: str | None = Query(
        default=None,
        pattern=r"^(vital-signs|laboratory|imaging|exam|survey|social-history|activity)$",
    ),
    code: str | None = Query(default=None, description="Filter by LOINC code"),
) -> list[ObservationResponse]:
    """List observations for a patient, optionally filtered.

    Args:
        patient_id: Patient UUID.
        db: Async database session.
        category: Optional observation category filter.
        code: Optional LOINC code filter.

    Returns:
        List of observation records, newest first.
    """
    query = select(Observation).where(Observation.patient_id == patient_id)

    if category:
        query = query.where(Observation.category == category)
    if code:
        query = query.where(Observation.code == code)

    query = query.order_by(Observation.effective_datetime.desc())
    result = await db.execute(query)
    observations = result.scalars().all()

    return [ObservationResponse.model_validate(o) for o in observations]


@router.get(
    "/observations/{observation_id}",
    response_model=ObservationResponse,
    summary="Get single observation",
)
async def get_observation(observation_id: uuid.UUID, db: DbSession) -> ObservationResponse:
    """Get a single observation by ID.

    Args:
        observation_id: Observation UUID.
        db: Async database session.

    Returns:
        Observation record.

    Raises:
        HTTPException 404: Observation not found.
    """
    observation = await db.get(Observation, observation_id)
    if observation is None:
        raise HTTPException(status_code=404, detail="Observation not found")
    return ObservationResponse.model_validate(observation)


@router.put(
    "/observations/{observation_id}",
    response_model=ObservationResponse,
    summary="Update observation",
)
async def update_observation(
    observation_id: uuid.UUID, body: ObservationUpdate, db: DbSession
) -> ObservationResponse:
    """Update an observation's value or status.

    Args:
        observation_id: Observation UUID.
        body: Fields to update.
        db: Async database session.

    Returns:
        Updated observation record.

    Raises:
        HTTPException 404: Observation not found.
    """
    observation = await db.get(Observation, observation_id)
    if observation is None:
        raise HTTPException(status_code=404, detail="Observation not found")

    update_data = body.model_dump(exclude_none=True)
    if "status" in update_data:
        update_data["status"] = ObservationStatus(update_data["status"])

    for field, value in update_data.items():
        setattr(observation, field, value)

    await db.flush()
    await db.refresh(observation)
    log.info("Observation updated: %s", observation_id)
    return ObservationResponse.model_validate(observation)


@router.delete(
    "/observations/{observation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete observation",
)
async def delete_observation(observation_id: uuid.UUID, db: DbSession) -> None:
    """Delete an observation record.

    Args:
        observation_id: Observation UUID.
        db: Async database session.

    Raises:
        HTTPException 404: Observation not found.
    """
    observation = await db.get(Observation, observation_id)
    if observation is None:
        raise HTTPException(status_code=404, detail="Observation not found")

    await db.delete(observation)
    await db.flush()
    log.info("Observation deleted: %s", observation_id)
