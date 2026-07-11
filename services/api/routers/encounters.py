"""Encounter (visit) CRUD endpoints.

POST   /patients/{id}/encounters — Start a new encounter
GET    /patients/{id}/encounters — List patient encounters
GET    /encounters/{id}          — Get encounter with linked data
PUT    /encounters/{id}          — Update encounter
PUT    /encounters/{id}/close    — Close encounter
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from libs.common.models.condition import Condition
from libs.common.models.encounter import Encounter, EncounterStatus
from libs.common.models.medication_request import MedicationRequest
from libs.common.models.observation import Observation
from libs.common.models.patient import Patient
from services.api.db import DbSession
from services.api.schemas.crud_schemas import (
    ConditionResponse,
    EncounterCreate,
    EncounterDetailResponse,
    EncounterResponse,
    EncounterUpdate,
    MedicationResponse,
    ObservationResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["encounters"])


@router.post(
    "/patients/{patient_id}/encounters",
    response_model=EncounterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new encounter",
)
async def create_encounter(
    patient_id: uuid.UUID, body: EncounterCreate, db: DbSession
) -> EncounterResponse:
    """Start a new clinical encounter for a patient.

    Auto-sets ``period_start`` to now and ``status`` to ``in-progress``.

    Args:
        patient_id: Patient UUID.
        body: Encounter data.
        db: Async database session.

    Returns:
        Created encounter record.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    encounter = Encounter(
        patient_id=patient_id,
        status=EncounterStatus.IN_PROGRESS,
        encounter_class=body.encounter_class,
        type_code=body.type_code,
        type_display=body.type_display,
        service_type=body.service_type,
        facility_name=body.facility_name,
        facility_id=body.facility_id,
        period_start=datetime.now(UTC),
    )
    db.add(encounter)
    await db.flush()
    await db.refresh(encounter)
    log.info("Encounter started: %s for patient %s", encounter.id, patient_id)
    return EncounterResponse.model_validate(encounter)


@router.get(
    "/patients/{patient_id}/encounters",
    response_model=list[EncounterResponse],
    summary="List patient encounters",
)
async def list_encounters(
    patient_id: uuid.UUID,
    db: DbSession,
    status_filter: str | None = Query(
        default=None,
        alias="status",
        pattern=r"^(planned|arrived|triaged|in-progress|onleave|finished|cancelled)$",
    ),
) -> list[EncounterResponse]:
    """List encounters for a patient, optionally filtered by status.

    Args:
        patient_id: Patient UUID.
        db: Async database session.
        status_filter: Optional encounter status filter.

    Returns:
        List of encounter records, newest first.
    """
    query = select(Encounter).where(Encounter.patient_id == patient_id)

    if status_filter:
        query = query.where(Encounter.status == EncounterStatus(status_filter))

    query = query.order_by(Encounter.period_start.desc())
    result = await db.execute(query)
    encounters = result.scalars().all()

    return [EncounterResponse.model_validate(e) for e in encounters]


@router.get(
    "/encounters/{encounter_id}",
    response_model=EncounterDetailResponse,
    summary="Get encounter with linked clinical data",
)
async def get_encounter(encounter_id: uuid.UUID, db: DbSession) -> EncounterDetailResponse:
    """Get a single encounter with all linked conditions, observations, and medications.

    Args:
        encounter_id: Encounter UUID.
        db: Async database session.

    Returns:
        EncounterDetailResponse with linked clinical data.

    Raises:
        HTTPException 404: Encounter not found.
    """
    encounter = await db.get(Encounter, encounter_id)
    if encounter is None:
        raise HTTPException(status_code=404, detail="Encounter not found")

    # Linked conditions
    cond_result = await db.execute(select(Condition).where(Condition.encounter_id == encounter_id))
    conditions = cond_result.scalars().all()

    # Linked observations
    obs_result = await db.execute(
        select(Observation).where(Observation.encounter_id == encounter_id)
    )
    observations = obs_result.scalars().all()

    # Linked medications
    med_result = await db.execute(
        select(MedicationRequest).where(MedicationRequest.encounter_id == encounter_id)
    )
    medications = med_result.scalars().all()

    return EncounterDetailResponse(
        encounter=EncounterResponse.model_validate(encounter),
        conditions=[ConditionResponse.model_validate(c) for c in conditions],
        observations=[ObservationResponse.model_validate(o) for o in observations],
        medications=[MedicationResponse.model_validate(m) for m in medications],
    )


@router.put(
    "/encounters/{encounter_id}",
    response_model=EncounterResponse,
    summary="Update an encounter",
)
async def update_encounter(
    encounter_id: uuid.UUID, body: EncounterUpdate, db: DbSession
) -> EncounterResponse:
    """Update an encounter's status, class, or other fields.

    Args:
        encounter_id: Encounter UUID.
        body: Fields to update.
        db: Async database session.

    Returns:
        Updated encounter record.

    Raises:
        HTTPException 404: Encounter not found.
    """
    encounter = await db.get(Encounter, encounter_id)
    if encounter is None:
        raise HTTPException(status_code=404, detail="Encounter not found")

    update_data = body.model_dump(exclude_none=True)
    if "status" in update_data:
        update_data["status"] = EncounterStatus(update_data["status"])

    for field, value in update_data.items():
        setattr(encounter, field, value)

    await db.flush()
    await db.refresh(encounter)
    log.info("Encounter updated: %s", encounter_id)
    return EncounterResponse.model_validate(encounter)


@router.put(
    "/encounters/{encounter_id}/close",
    response_model=EncounterResponse,
    summary="Close an encounter",
)
async def close_encounter(encounter_id: uuid.UUID, db: DbSession) -> EncounterResponse:
    """Close an encounter by setting period_end and status to finished.

    Args:
        encounter_id: Encounter UUID.
        db: Async database session.

    Returns:
        Closed encounter record.

    Raises:
        HTTPException 404: Encounter not found.
        HTTPException 409: Encounter already closed.
    """
    encounter = await db.get(Encounter, encounter_id)
    if encounter is None:
        raise HTTPException(status_code=404, detail="Encounter not found")

    if encounter.status == EncounterStatus.FINISHED:
        raise HTTPException(status_code=409, detail="Encounter is already closed")

    encounter.status = EncounterStatus.FINISHED
    encounter.period_end = datetime.now(UTC)

    await db.flush()
    await db.refresh(encounter)
    log.info("Encounter closed: %s", encounter_id)
    return EncounterResponse.model_validate(encounter)
