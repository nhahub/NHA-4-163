"""Patient CRUD endpoints.

POST   /patients              — Register a new patient
GET    /patients              — List patients (paginated, searchable)
GET    /patients/{id}         — Get patient by ID
PUT    /patients/{id}         — Update patient
DELETE /patients/{id}         — Soft-delete patient
GET    /patients/{id}/summary — Full clinical summary
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from libs.common.models.condition import Condition
from libs.common.models.family_member_history import FamilyMemberHistory
from libs.common.models.medication_request import MedicationRequest
from libs.common.models.patient import AdministrativeGender, Patient
from services.api.db import DbSession
from services.api.schemas.crud_schemas import (
    ConditionResponse,
    FamilyMemberResponse,
    MedicationResponse,
    PaginatedResponse,
    PatientCreate,
    PatientResponse,
    PatientSummaryResponse,
    PatientUpdate,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/patients", tags=["patient-crud"])


@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new patient",
)
async def create_patient(body: PatientCreate, db: DbSession) -> PatientResponse:
    """Create a new patient record.

    Args:
        body: Patient data.
        db: Async database session.

    Returns:
        Created patient record with generated UUID.
    """
    patient = Patient(
        given_name=body.given_name,
        family_name=body.family_name,
        middle_name=body.middle_name,
        date_of_birth=body.date_of_birth,
        gender=AdministrativeGender(body.gender),
        ethnicity=body.ethnicity,
        race=body.race,
        phone=body.phone,
        email=body.email,
        address_line=body.address_line,
        city=body.city,
        state=body.state,
        postal_code=body.postal_code,
        country=body.country,
        language=body.language,
        external_id=body.external_id,
        identifier_system=body.identifier_system,
    )
    db.add(patient)
    await db.flush()
    await db.refresh(patient)
    log.info("Patient created: %s", patient.id)
    return PatientResponse.model_validate(patient)


@router.get(
    "",
    response_model=PaginatedResponse[PatientResponse],
    summary="List patients",
)
async def list_patients(
    db: DbSession,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search: str | None = Query(default=None, description="Search by name or external_id"),
    gender: str | None = Query(default=None, pattern=r"^(male|female|other|unknown)$"),
) -> PaginatedResponse[PatientResponse]:
    """List patients with pagination and optional search.

    Args:
        db: Async database session.
        page: Page number (1-indexed).
        page_size: Items per page.
        search: Optional search term for name/external_id.
        gender: Optional gender filter.

    Returns:
        Paginated list of patients.
    """
    query = select(Patient).where(Patient.deleted_at.is_(None))

    if search:
        like = f"%{search}%"
        query = query.where(
            (Patient.given_name.ilike(like))
            | (Patient.family_name.ilike(like))
            | (Patient.external_id.ilike(like))
        )

    if gender:
        query = query.where(Patient.gender == AdministrativeGender(gender))

    # Count
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Paginate
    offset = (page - 1) * page_size
    query = query.order_by(Patient.created_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(query)
    patients = result.scalars().all()

    return PaginatedResponse(
        items=[PatientResponse.model_validate(p) for p in patients],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


@router.get(
    "/{patient_id}",
    response_model=PatientResponse,
    summary="Get patient by ID",
)
async def get_patient(patient_id: uuid.UUID, db: DbSession) -> PatientResponse:
    """Get a single patient by UUID.

    Args:
        patient_id: Patient UUID.
        db: Async database session.

    Returns:
        Patient record.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PatientResponse.model_validate(patient)


@router.put(
    "/{patient_id}",
    response_model=PatientResponse,
    summary="Update patient",
)
async def update_patient(
    patient_id: uuid.UUID, body: PatientUpdate, db: DbSession
) -> PatientResponse:
    """Update a patient record (partial update).

    Args:
        patient_id: Patient UUID.
        body: Fields to update (only non-None fields are applied).
        db: Async database session.

    Returns:
        Updated patient record.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    update_data = body.model_dump(exclude_none=True)
    if "gender" in update_data:
        update_data["gender"] = AdministrativeGender(update_data["gender"])

    for field, value in update_data.items():
        setattr(patient, field, value)

    await db.flush()
    await db.refresh(patient)
    log.info("Patient updated: %s", patient_id)
    return PatientResponse.model_validate(patient)


@router.delete(
    "/{patient_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete patient",
)
async def delete_patient(patient_id: uuid.UUID, db: DbSession) -> None:
    """Soft-delete a patient by setting deleted_at timestamp.

    Args:
        patient_id: Patient UUID.
        db: Async database session.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient.deleted_at = datetime.utcnow()
    await db.flush()
    log.info("Patient soft-deleted: %s", patient_id)


@router.get(
    "/{patient_id}/summary",
    response_model=PatientSummaryResponse,
    summary="Full clinical summary",
)
async def get_patient_summary(patient_id: uuid.UUID, db: DbSession) -> PatientSummaryResponse:
    """Get a complete clinical summary for a patient.

    Includes conditions, medications, and family members.

    Args:
        patient_id: Patient UUID.
        db: Async database session.

    Returns:
        PatientSummaryResponse with all clinical data.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Conditions
    cond_result = await db.execute(
        select(Condition).where(Condition.patient_id == patient_id)
    )
    conditions = cond_result.scalars().all()

    # Medications
    med_result = await db.execute(
        select(MedicationRequest).where(MedicationRequest.patient_id == patient_id)
    )
    medications = med_result.scalars().all()

    # Family
    fam_result = await db.execute(
        select(FamilyMemberHistory).where(FamilyMemberHistory.patient_id == patient_id)
    )
    family = fam_result.scalars().all()

    active_meds = [m for m in medications if m.status.value == "active"]

    return PatientSummaryResponse(
        patient=PatientResponse.model_validate(patient),
        conditions=[ConditionResponse.model_validate(c) for c in conditions],
        medications=[MedicationResponse.model_validate(m) for m in medications],
        family_members=[FamilyMemberResponse.model_validate(f) for f in family],
        condition_count=len(conditions),
        active_medication_count=len(active_meds),
        family_member_count=len(family),
    )
