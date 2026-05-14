"""Condition (diagnosis) CRUD endpoints.

POST   /patients/{id}/conditions — Record a new diagnosis
GET    /patients/{id}/conditions — List patient's conditions
PUT    /conditions/{id}          — Update a condition
DELETE /conditions/{id}          — Remove a condition
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from libs.common.models.condition import (
    ClinicalStatus,
    Condition,
    ConditionSeverity,
    VerificationStatus,
)
from libs.common.models.patient import Patient
from services.api.db import DbSession
from services.api.schemas.crud_schemas import (
    ConditionCreate,
    ConditionResponse,
    ConditionUpdate,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["conditions"])


@router.post(
    "/patients/{patient_id}/conditions",
    response_model=ConditionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a new diagnosis",
)
async def create_condition(
    patient_id: uuid.UUID, body: ConditionCreate, db: DbSession
) -> ConditionResponse:
    """Add a new condition/diagnosis to a patient.

    Args:
        patient_id: Patient UUID.
        body: Condition data.
        db: Async database session.

    Returns:
        Created condition record.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    condition = Condition(
        patient_id=patient_id,
        code=body.code,
        code_system=body.code_system,
        code_display=body.code_display,
        code_text=body.code_text,
        clinical_status=ClinicalStatus(body.clinical_status),
        verification_status=(
            VerificationStatus(body.verification_status) if body.verification_status else None
        ),
        severity=ConditionSeverity(body.severity) if body.severity else None,
        is_hereditary=body.is_hereditary,
        onset_datetime=body.onset_datetime,
        onset_age_years=body.onset_age_years,
    )
    db.add(condition)
    await db.flush()
    await db.refresh(condition)
    log.info("Condition created: %s for patient %s", condition.id, patient_id)
    return ConditionResponse.model_validate(condition)


@router.get(
    "/patients/{patient_id}/conditions",
    response_model=list[ConditionResponse],
    summary="List patient conditions",
)
async def list_conditions(
    patient_id: uuid.UUID,
    db: DbSession,
    status_filter: str | None = Query(
        default=None,
        alias="status",
        pattern=r"^(active|confirmed|recurrence|relapse|inactive|remission|resolved)$",
    ),
    hereditary_only: bool = Query(default=False),
) -> list[ConditionResponse]:
    """List all conditions for a patient.

    Args:
        patient_id: Patient UUID.
        db: Async database session.
        status_filter: Optional clinical status filter.
        hereditary_only: If True, return only hereditary conditions.

    Returns:
        List of condition records.
    """
    query = select(Condition).where(Condition.patient_id == patient_id)

    if status_filter:
        query = query.where(Condition.clinical_status == ClinicalStatus(status_filter))

    if hereditary_only:
        query = query.where(Condition.is_hereditary.is_(True))

    query = query.order_by(Condition.created_at.desc())
    result = await db.execute(query)
    conditions = result.scalars().all()

    return [ConditionResponse.model_validate(c) for c in conditions]


@router.put(
    "/conditions/{condition_id}",
    response_model=ConditionResponse,
    summary="Update a condition",
)
async def update_condition(
    condition_id: uuid.UUID, body: ConditionUpdate, db: DbSession
) -> ConditionResponse:
    """Update a condition record.

    Args:
        condition_id: Condition UUID.
        body: Fields to update.
        db: Async database session.

    Returns:
        Updated condition record.

    Raises:
        HTTPException 404: Condition not found.
    """
    condition = await db.get(Condition, condition_id)
    if condition is None:
        raise HTTPException(status_code=404, detail="Condition not found")

    update_data = body.model_dump(exclude_none=True)

    if "clinical_status" in update_data:
        update_data["clinical_status"] = ClinicalStatus(update_data["clinical_status"])
    if "verification_status" in update_data:
        update_data["verification_status"] = VerificationStatus(update_data["verification_status"])
    if "severity" in update_data:
        update_data["severity"] = ConditionSeverity(update_data["severity"])

    for field, value in update_data.items():
        setattr(condition, field, value)

    await db.flush()
    await db.refresh(condition)
    log.info("Condition updated: %s", condition_id)
    return ConditionResponse.model_validate(condition)


@router.delete(
    "/conditions/{condition_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a condition",
)
async def delete_condition(condition_id: uuid.UUID, db: DbSession) -> None:
    """Delete a condition record.

    Args:
        condition_id: Condition UUID.
        db: Async database session.

    Raises:
        HTTPException 404: Condition not found.
    """
    condition = await db.get(Condition, condition_id)
    if condition is None:
        raise HTTPException(status_code=404, detail="Condition not found")

    await db.delete(condition)
    await db.flush()
    log.info("Condition deleted: %s", condition_id)
