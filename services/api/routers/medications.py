"""Medication CRUD endpoints.

POST   /patients/{id}/medications — Add a medication
GET    /patients/{id}/medications — List medications
PUT    /medications/{id}          — Update a medication
DELETE /medications/{id}          — Remove a medication
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from libs.common.models.medication_request import (
    MedicationRequest,
    MedicationRequestIntent,
    MedicationRequestStatus,
)
from libs.common.models.patient import Patient
from services.api.db import DbSession
from services.api.schemas.crud_schemas import (
    MedicationCreate,
    MedicationResponse,
    MedicationUpdate,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["medications"])


@router.post(
    "/patients/{patient_id}/medications",
    response_model=MedicationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a medication",
)
async def create_medication(
    patient_id: uuid.UUID, body: MedicationCreate, db: DbSession
) -> MedicationResponse:
    """Record a new medication for a patient.

    Args:
        patient_id: Patient UUID.
        body: Medication data.
        db: Async database session.

    Returns:
        Created medication record.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    medication = MedicationRequest(
        patient_id=patient_id,
        medication_code=body.medication_code,
        medication_code_system=body.medication_code_system,
        medication_display=body.medication_display,
        status=MedicationRequestStatus(body.status),
        intent=MedicationRequestIntent(body.intent),
        dosage_text=body.dosage_text,
        dosage_timing=body.dosage_timing,
        dosage_route=body.dosage_route,
        dose_quantity=body.dose_quantity,
        dose_unit=body.dose_unit,
        authored_on=body.authored_on,
    )
    db.add(medication)
    await db.flush()
    await db.refresh(medication)
    log.info("Medication created: %s for patient %s", medication.id, patient_id)
    return MedicationResponse.model_validate(medication)


@router.get(
    "/patients/{patient_id}/medications",
    response_model=list[MedicationResponse],
    summary="List medications",
)
async def list_medications(
    patient_id: uuid.UUID,
    db: DbSession,
    status_filter: str | None = Query(
        default=None,
        alias="status",
        pattern=r"^(active|on-hold|cancelled|completed|entered-in-error|stopped|draft|unknown)$",
    ),
    active_only: bool = Query(default=False),
) -> list[MedicationResponse]:
    """List all medications for a patient.

    Args:
        patient_id: Patient UUID.
        db: Async database session.
        status_filter: Optional medication status filter.
        active_only: If True, return only active medications.

    Returns:
        List of medication records.
    """
    query = select(MedicationRequest).where(MedicationRequest.patient_id == patient_id)

    if active_only:
        query = query.where(MedicationRequest.status == MedicationRequestStatus.ACTIVE)
    elif status_filter:
        query = query.where(MedicationRequest.status == MedicationRequestStatus(status_filter))

    query = query.order_by(MedicationRequest.authored_on.desc())
    result = await db.execute(query)
    meds = result.scalars().all()

    return [MedicationResponse.model_validate(m) for m in meds]


@router.put(
    "/medications/{medication_id}",
    response_model=MedicationResponse,
    summary="Update a medication",
)
async def update_medication(
    medication_id: uuid.UUID, body: MedicationUpdate, db: DbSession
) -> MedicationResponse:
    """Update a medication record (status, dosage).

    Args:
        medication_id: MedicationRequest UUID.
        body: Fields to update.
        db: Async database session.

    Returns:
        Updated medication record.

    Raises:
        HTTPException 404: Medication not found.
    """
    medication = await db.get(MedicationRequest, medication_id)
    if medication is None:
        raise HTTPException(status_code=404, detail="Medication not found")

    update_data = body.model_dump(exclude_none=True)

    if "status" in update_data:
        update_data["status"] = MedicationRequestStatus(update_data["status"])

    for field, value in update_data.items():
        setattr(medication, field, value)

    await db.flush()
    await db.refresh(medication)
    log.info("Medication updated: %s", medication_id)
    return MedicationResponse.model_validate(medication)


@router.delete(
    "/medications/{medication_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete a medication",
)
async def delete_medication(medication_id: uuid.UUID, db: DbSession) -> None:
    """Delete a medication record.

    Args:
        medication_id: MedicationRequest UUID.
        db: Async database session.

    Raises:
        HTTPException 404: Medication not found.
    """
    medication = await db.get(MedicationRequest, medication_id)
    if medication is None:
        raise HTTPException(status_code=404, detail="Medication not found")

    await db.delete(medication)
    await db.flush()
    log.info("Medication deleted: %s", medication_id)
