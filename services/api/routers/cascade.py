"""Cascade screening workflow endpoints (Tier 5 — Genetics & Genomics).

POST /patients/{id}/cascade-screen — identify at-risk relatives, generate tasks
GET  /patients/{id}/cascade-screen — list generated screening runs + tasks
PUT  /cascade-tasks/{id}           — update outreach status (contacted/screened/...)

This is the core clinical-genetics workflow: it turns a proband's hereditary
diagnosis into a ranked, actionable outreach list.  Relatives are PHI, so every
access is captured by the audit middleware.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from libs.common.models.cascade import CascadeScreening, CascadeTask, CascadeTaskStatus
from libs.common.models.patient import Patient
from services.api.db import DbSession
from services.api.schemas.cascade_schemas import (
    CascadeScreeningResponse,
    CascadeScreenRequest,
    CascadeTaskResponse,
    CascadeTaskUpdate,
)
from services.api.services.cascade_service import generate_cascade_screening
from services.api.services.inheritance_service import INHERITANCE_MODELS

log = logging.getLogger(__name__)

router = APIRouter(tags=["genetics"])


@router.post(
    "/patients/{patient_id}/cascade-screen",
    response_model=CascadeScreeningResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Identify at-risk relatives for a proband's hereditary condition",
)
async def create_cascade_screen(
    patient_id: uuid.UUID, body: CascadeScreenRequest, db: DbSession
) -> CascadeScreeningResponse:
    """Generate a cascade-screening run for a proband.

    Args:
        patient_id: Proband (affected patient) UUID.
        body: Inheritance mode, condition, and optional overrides.
        db: Async database session.

    Returns:
        The created screening run with ranked outreach tasks.

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

    screening = await generate_cascade_screening(
        db,
        proband_patient_id=patient_id,
        inheritance_mode=body.inheritance_mode,
        penetrance=penetrance,
        carrier_frequency=carrier_freq,
        condition_code=body.condition_code,
        condition_display=body.condition_display,
        organization_id=patient.organization_id,
        notify=body.notify,
    )

    # Re-load with tasks eagerly for the response.
    loaded = (
        await db.execute(
            select(CascadeScreening)
            .where(CascadeScreening.id == screening.id)
            .options(selectinload(CascadeScreening.tasks))
        )
    ).scalar_one()
    return CascadeScreeningResponse.model_validate(loaded)


@router.get(
    "/patients/{patient_id}/cascade-screen",
    response_model=list[CascadeScreeningResponse],
    summary="List cascade-screening runs and their tasks",
)
async def list_cascade_screens(
    patient_id: uuid.UUID, db: DbSession
) -> list[CascadeScreeningResponse]:
    """List all cascade-screening runs generated for a proband, newest first.

    Args:
        patient_id: Proband UUID.
        db: Async database session.

    Returns:
        Screening runs, each with its ranked tasks.
    """
    runs = (
        (
            await db.execute(
                select(CascadeScreening)
                .where(CascadeScreening.proband_patient_id == patient_id)
                .options(selectinload(CascadeScreening.tasks))
                .order_by(CascadeScreening.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    return [CascadeScreeningResponse.model_validate(r) for r in runs]


@router.put(
    "/cascade-tasks/{task_id}",
    response_model=CascadeTaskResponse,
    summary="Update a cascade task's outreach status",
)
async def update_cascade_task(
    task_id: uuid.UUID, body: CascadeTaskUpdate, db: DbSession
) -> CascadeTaskResponse:
    """Update the status and/or notes of a cascade-screening task.

    Args:
        task_id: Cascade task UUID.
        body: Fields to update.
        db: Async database session.

    Returns:
        The updated task.

    Raises:
        HTTPException 404: Task not found.
    """
    task = await db.get(CascadeTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Cascade task not found")

    if body.status is not None:
        task.status = CascadeTaskStatus(body.status)
    if body.notes is not None:
        task.notes = body.notes

    await db.flush()
    await db.refresh(task)
    log.info("Cascade task updated: %s status=%s", task_id, task.status.value)
    return CascadeTaskResponse.model_validate(task)
