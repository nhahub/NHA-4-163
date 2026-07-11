"""Patient consent management endpoints (Tier 7 — Patient-Facing & Consent).

POST /patients/{id}/consent — record/update a consent decision for a scope
GET  /patients/{id}/consent — current consent state (per scope) + full history
GET  /consent/scopes        — list supported consent scopes

Consent recorded here is *enforced* at the de-identification/export layer
(``routers/export.py``): withdrawing ``research`` consent removes a patient from
future research exports.  Every read/write is captured by the audit middleware.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, status

from libs.common.models.consent import (
    ConsentMethod,
    ConsentScope,
    ConsentStatus,
)
from libs.common.models.patient import Patient
from services.api.db import DbSession
from services.api.schemas.consent_schemas import (
    ConsentRecordResponse,
    ConsentRequest,
    ConsentScopeState,
    ConsentStateResponse,
)
from services.api.services.consent_service import (
    get_consent_history,
    is_record_active,
    record_consent,
    resolve_effective_consent,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["consent"])


@router.get("/consent/scopes", summary="List supported consent scopes")
async def list_consent_scopes() -> dict[str, list[str]]:
    """Return the consent scopes the platform recognises.

    Returns:
        A mapping ``{"scopes": [...]}`` of scope value strings.
    """
    return {"scopes": [s.value for s in ConsentScope]}


@router.post(
    "/patients/{patient_id}/consent",
    response_model=ConsentRecordResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record or update a patient's consent for a scope",
)
async def create_consent(
    patient_id: uuid.UUID, body: ConsentRequest, db: DbSession
) -> ConsentRecordResponse:
    """Append a consent decision (grant/deny/withdraw) for a patient + scope.

    Args:
        patient_id: Patient UUID.
        body: Scope, decision, and optional method/expiry/notes.
        db: Async database session.

    Returns:
        The newly created consent record.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    record = await record_consent(
        db,
        patient_id=patient_id,
        scope=ConsentScope(body.scope),
        status=ConsentStatus(body.status),
        method=ConsentMethod(body.method) if body.method else None,
        expires_at=body.expires_at,
        policy_version=body.policy_version,
        notes=body.notes,
        organization_id=patient.organization_id,
    )

    # Keep the legacy Patient.research_consent flag in sync so existing readers
    # that still consult it agree with the granular consent record.
    if ConsentScope(body.scope) is ConsentScope.RESEARCH:
        active = is_record_active(record.status, record.expires_at)
        patient.research_consent = active
        patient.research_consent_date = record.granted_at
        await db.flush()

    return ConsentRecordResponse.model_validate(record)


@router.get(
    "/patients/{patient_id}/consent",
    response_model=ConsentStateResponse,
    summary="Get a patient's current consent state and history",
)
async def get_consent(patient_id: uuid.UUID, db: DbSession) -> ConsentStateResponse:
    """Return the effective consent per scope plus the full decision history.

    Args:
        patient_id: Patient UUID.
        db: Async database session.

    Returns:
        Current consent state for every scope with a recorded decision, and the
        newest-first history.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    history = await get_consent_history(db, patient_id)
    effective = resolve_effective_consent(history)

    scopes = [
        ConsentScopeState(
            scope=scope.value,
            active=is_record_active(record.status, record.expires_at),
            status=record.status.value,
            expires_at=record.expires_at,
            decided_at=record.created_at,
        )
        for scope, record in effective.items()
    ]
    scopes.sort(key=lambda s: s.scope)

    return ConsentStateResponse(
        patient_id=patient_id,
        scopes=scopes,
        history=[ConsentRecordResponse.model_validate(r) for r in history],
    )
