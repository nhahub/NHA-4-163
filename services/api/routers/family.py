"""Family relationship CRUD endpoints.

POST   /patients/{id}/family — Add a family member
GET    /patients/{id}/family — List family members
PUT    /family/{id}          — Update a family member
DELETE /family/{id}          — Remove a family link
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from libs.common.models.family_member_history import (
    FamilyMemberHistory,
    FamilyMemberHistoryStatus,
)
from libs.common.models.patient import Patient
from services.api.db import DbSession
from services.api.schemas.crud_schemas import (
    FamilyMemberCreate,
    FamilyMemberResponse,
    FamilyMemberUpdate,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["family"])


@router.post(
    "/patients/{patient_id}/family",
    response_model=FamilyMemberResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a family member",
)
async def create_family_member(
    patient_id: uuid.UUID, body: FamilyMemberCreate, db: DbSession
) -> FamilyMemberResponse:
    """Add a family member relationship for a patient.

    If ``related_patient_id`` is provided, validates that the referenced
    patient exists in the system.

    Args:
        patient_id: Patient UUID (the patient this history belongs to).
        body: Family member data.
        db: Async database session.

    Returns:
        Created family member record.

    Raises:
        HTTPException 404: Patient or related patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Validate related patient exists if provided
    if body.related_patient_id:
        related = await db.get(Patient, body.related_patient_id)
        if related is None or related.deleted_at is not None:
            raise HTTPException(status_code=404, detail="Related patient not found")

    family_member = FamilyMemberHistory(
        patient_id=patient_id,
        related_patient_id=body.related_patient_id,
        relationship_code=body.relationship_code,
        relationship_display=body.relationship_display,
        degree_of_relatedness=body.degree_of_relatedness,
        sex=body.sex,
        born_date=body.born_date,
        deceased=body.deceased,
        deceased_age_years=body.deceased_age_years,
        conditions=body.conditions,
        status=FamilyMemberHistoryStatus(body.status),
        neo4j_synced=False,
    )
    db.add(family_member)
    await db.flush()
    await db.refresh(family_member)
    log.info("Family member created: %s for patient %s", family_member.id, patient_id)
    return FamilyMemberResponse.model_validate(family_member)


@router.get(
    "/patients/{patient_id}/family",
    response_model=list[FamilyMemberResponse],
    summary="List family members",
)
async def list_family_members(patient_id: uuid.UUID, db: DbSession) -> list[FamilyMemberResponse]:
    """List all family members for a patient.

    Args:
        patient_id: Patient UUID.
        db: Async database session.

    Returns:
        List of family member records, ordered by degree of relatedness (closest first).
    """
    query = (
        select(FamilyMemberHistory)
        .where(FamilyMemberHistory.patient_id == patient_id)
        .order_by(FamilyMemberHistory.degree_of_relatedness.desc())
    )
    result = await db.execute(query)
    members = result.scalars().all()
    return [FamilyMemberResponse.model_validate(m) for m in members]


@router.put(
    "/family/{family_id}",
    response_model=FamilyMemberResponse,
    summary="Update a family member",
)
async def update_family_member(
    family_id: uuid.UUID, body: FamilyMemberUpdate, db: DbSession
) -> FamilyMemberResponse:
    """Update a family member record.

    Args:
        family_id: FamilyMemberHistory UUID.
        body: Fields to update.
        db: Async database session.

    Returns:
        Updated family member record.

    Raises:
        HTTPException 404: Family member not found.
    """
    member = await db.get(FamilyMemberHistory, family_id)
    if member is None:
        raise HTTPException(status_code=404, detail="Family member not found")

    update_data = body.model_dump(exclude_none=True)

    if "status" in update_data:
        update_data["status"] = FamilyMemberHistoryStatus(update_data["status"])

    # If linking to a different patient, validate they exist
    if "related_patient_id" in update_data and update_data["related_patient_id"]:
        related = await db.get(Patient, update_data["related_patient_id"])
        if related is None or related.deleted_at is not None:
            raise HTTPException(status_code=404, detail="Related patient not found")

    for field, value in update_data.items():
        setattr(member, field, value)

    # Mark as needing Neo4j re-sync since relationship data changed
    member.neo4j_synced = False

    await db.flush()
    await db.refresh(member)
    log.info("Family member updated: %s", family_id)
    return FamilyMemberResponse.model_validate(member)


@router.delete(
    "/family/{family_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Remove a family link",
)
async def delete_family_member(family_id: uuid.UUID, db: DbSession) -> None:
    """Delete a family member relationship.

    Args:
        family_id: FamilyMemberHistory UUID.
        db: Async database session.

    Raises:
        HTTPException 404: Family member not found.
    """
    member = await db.get(FamilyMemberHistory, family_id)
    if member is None:
        raise HTTPException(status_code=404, detail="Family member not found")

    await db.delete(member)
    await db.flush()
    log.info("Family member deleted: %s", family_id)
