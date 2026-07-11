"""SMART on FHIR patient portal endpoints (Tier 7 — Patient-Facing).

A read-only, patient-facing surface that lets a patient view *their own* risk
profile and family tree.  It builds on the FHIR R4 API (Tier 3) and issues
patient-scoped tokens (:mod:`services.api.auth.portal_auth`) so every read is
confined to the launch patient.

    GET  /portal/.well-known/smart-configuration — SMART discovery document
    POST /portal/token                           — mint a patient-scoped token
    GET  /portal/me                              — own FHIR Patient resource
    GET  /portal/me/risk-profile                 — own risk summary (lay-friendly)
    GET  /portal/me/family                       — own pedigree (de-identified)

This is a pragmatic SMART subset: it models standalone patient launch, scoped
tokens, and read-only self-service access rather than the full OAuth2
authorization-code handshake (which needs an external IdP).
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from libs.common.models.family_member_history import FamilyMemberHistory
from libs.common.models.patient import Patient
from libs.common.models.prediction_log import PredictionLog
from services.api.auth.portal_auth import (
    CurrentPatientDep,
    create_patient_token,
)
from services.api.db import DbSession
from services.api.schemas.fhir_schemas import FHIRPatient
from services.api.schemas.portal_schemas import (
    PortalFamilyMember,
    PortalFamilyResponse,
    PortalRiskProfile,
    PortalTokenRequest,
    PortalTokenResponse,
    SmartConfiguration,
)
from services.api.services.portal_service import (
    build_risk_profile,
    deidentify_family_member,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/portal", tags=["portal"])

# Public base URL used to advertise SMART endpoints in the discovery document.
_PORTAL_BASE_URL = os.environ.get("PORTAL_BASE_URL", "").rstrip("/")


@router.get(
    "/.well-known/smart-configuration",
    response_model=SmartConfiguration,
    summary="SMART on FHIR discovery document",
)
async def smart_configuration(request: Request) -> SmartConfiguration:
    """Advertise the portal's SMART on FHIR capabilities and endpoints.

    Args:
        request: Incoming request (used to derive the base URL when
            ``PORTAL_BASE_URL`` is not configured).

    Returns:
        The SMART discovery document.
    """
    base = _PORTAL_BASE_URL or str(request.base_url).rstrip("/")
    return SmartConfiguration(
        authorization_endpoint=f"{base}/portal/authorize",
        token_endpoint=f"{base}/portal/token",
    )


@router.post(
    "/token",
    response_model=PortalTokenResponse,
    summary="Issue a patient-scoped portal access token",
)
async def issue_portal_token(body: PortalTokenRequest, db: DbSession) -> PortalTokenResponse:
    """Issue a read-only, patient-scoped SMART token for a standalone launch.

    Args:
        body: The launch patient id.
        db: Async database session.

    Returns:
        A patient-scoped bearer token bound to the launch patient.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, body.patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    token, expires_in = create_patient_token(body.patient_id)
    log.info("Portal token issued for patient=%s", body.patient_id)
    from services.api.auth.portal_auth import PATIENT_SCOPE

    return PortalTokenResponse(
        access_token=token,
        expires_in=expires_in,
        scope=PATIENT_SCOPE,
        patient=body.patient_id,
    )


async def _load_own_patient(db: DbSession, ctx: CurrentPatientDep) -> Patient:
    """Load the token's own patient row or raise 404."""
    patient = await db.get(Patient, ctx.patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient record not found")
    return patient


@router.get(
    "/me",
    response_model=FHIRPatient,
    summary="Get your own FHIR Patient resource",
)
async def get_own_patient(db: DbSession, ctx: CurrentPatientDep) -> FHIRPatient:
    """Return the authenticated patient's own record as a FHIR resource.

    Args:
        db: Async database session.
        ctx: Patient launch context from the portal token.

    Returns:
        The patient's own FHIR ``Patient`` resource.
    """
    patient = await _load_own_patient(db, ctx)
    return FHIRPatient.from_orm_patient(patient)


@router.get(
    "/me/risk-profile",
    response_model=PortalRiskProfile,
    summary="Get your own hereditary-risk summary",
)
async def get_own_risk_profile(db: DbSession, ctx: CurrentPatientDep) -> PortalRiskProfile:
    """Return a lay-friendly summary of the patient's latest risk assessment.

    Args:
        db: Async database session.
        ctx: Patient launch context from the portal token.

    Returns:
        A patient-friendly :class:`PortalRiskProfile`.
    """
    await _load_own_patient(db, ctx)  # ensure the patient still exists/active
    latest = (
        (
            await db.execute(
                select(PredictionLog)
                .where(PredictionLog.patient_id == ctx.patient_id)
                .order_by(PredictionLog.predicted_at.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )

    view = build_risk_profile(latest)
    return PortalRiskProfile(
        patient_id=ctx.patient_id,
        risk_score=view.risk_score,
        risk_tier=view.risk_tier,
        band=view.band,
        guidance=view.guidance,
        model_version=view.model_version,
        predicted_at=view.predicted_at,
    )


@router.get(
    "/me/family",
    response_model=PortalFamilyResponse,
    summary="Get your own family tree (de-identified)",
)
async def get_own_family(db: DbSession, ctx: CurrentPatientDep) -> PortalFamilyResponse:
    """Return the patient's family history, de-identified for self-viewing.

    Relatives are PHI: only relationship, sex, relatedness, and affected status
    are exposed — never a relative's name or contact details.

    Args:
        db: Async database session.
        ctx: Patient launch context from the portal token.

    Returns:
        The patient's de-identified pedigree.
    """
    await _load_own_patient(db, ctx)
    rows = (
        (
            await db.execute(
                select(FamilyMemberHistory).where(FamilyMemberHistory.patient_id == ctx.patient_id)
            )
        )
        .scalars()
        .all()
    )

    members = [PortalFamilyMember(**deidentify_family_member(r)) for r in rows]
    return PortalFamilyResponse(patient_id=ctx.patient_id, members=members)
