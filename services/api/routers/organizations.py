"""Organization (tenant) management and org-scoped access (Tier 4).

POST /organizations              — create a tenant, returns the API key ONCE
GET  /organizations              — list tenants (no key material)
POST /organizations/{id}/rotate-key — issue a new API key, invalidating the old
GET  /organizations/me           — the tenant identified by X-API-Key
GET  /organizations/me/patients  — patients scoped to the calling tenant
POST /organizations/me/patients  — create a patient owned by the calling tenant

Isolation: org-scoped endpoints stamp/filter ``Patient.organization_id`` using
the tenant resolved from the ``X-API-Key`` header.  RLS (migration 0012) is the
database-level defence-in-depth beneath this.
"""

from __future__ import annotations

import logging
import math
import re
import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from libs.common.models.organization import Organization, generate_api_key
from libs.common.models.patient import AdministrativeGender, Patient
from services.api.auth.api_key import CurrentOrgDep
from services.api.db import DbSession
from services.api.schemas.crud_schemas import (
    PaginatedResponse,
    PatientCreate,
    PatientResponse,
)
from services.api.schemas.org_schemas import (
    OrganizationCreate,
    OrganizationCreatedResponse,
    OrganizationResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/organizations", tags=["organizations"])


def _slugify(name: str) -> str:
    """Derive a URL-safe slug from an organization name."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "org"


@router.post(
    "",
    response_model=OrganizationCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an organization (tenant)",
)
async def create_organization(
    body: OrganizationCreate, db: DbSession
) -> OrganizationCreatedResponse:
    """Create a new tenant and issue its API key.

    The plaintext API key is returned exactly once in this response and is not
    recoverable afterwards — only its hash is stored.

    Args:
        body: Organization details.
        db: Async database session.

    Returns:
        The created organization plus its one-time API key.

    Raises:
        HTTPException 409: If the slug is already taken.
    """
    slug = body.slug or _slugify(body.name)
    exists = await db.execute(select(Organization).where(Organization.slug == slug))
    if exists.scalars().first() is not None:
        raise HTTPException(status_code=409, detail=f"Organization slug '{slug}' exists")

    raw_key, key_hash = generate_api_key()
    org = Organization(name=body.name, slug=slug, api_key_hash=key_hash)
    db.add(org)
    await db.flush()
    await db.refresh(org)
    log.info("Organization created: %s (%s)", org.id, slug)

    return OrganizationCreatedResponse(
        **OrganizationResponse.model_validate(org).model_dump(),
        api_key=raw_key,
    )


@router.get(
    "",
    response_model=PaginatedResponse[OrganizationResponse],
    summary="List organizations",
)
async def list_organizations(
    db: DbSession,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> PaginatedResponse[OrganizationResponse]:
    """List all tenants (never returns key material).

    Args:
        db: Async database session.
        page: 1-indexed page number.
        page_size: Items per page.

    Returns:
        Paginated organizations.
    """
    base = select(Organization).where(Organization.deleted_at.is_(None))
    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar() or 0
    rows = (
        (
            await db.execute(
                base.order_by(Organization.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        )
        .scalars()
        .all()
    )
    return PaginatedResponse(
        items=[OrganizationResponse.model_validate(o) for o in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


@router.post(
    "/{organization_id}/rotate-key",
    response_model=OrganizationCreatedResponse,
    summary="Rotate an organization's API key",
)
async def rotate_key(organization_id: uuid.UUID, db: DbSession) -> OrganizationCreatedResponse:
    """Issue a new API key for a tenant, invalidating the previous one.

    Args:
        organization_id: Organization UUID.
        db: Async database session.

    Returns:
        The organization plus its new one-time API key.

    Raises:
        HTTPException 404: Organization not found.
    """
    org = await db.get(Organization, organization_id)
    if org is None or org.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Organization not found")

    raw_key, key_hash = generate_api_key()
    org.api_key_hash = key_hash
    await db.flush()
    await db.refresh(org)
    log.info("API key rotated for organization %s", organization_id)
    return OrganizationCreatedResponse(
        **OrganizationResponse.model_validate(org).model_dump(),
        api_key=raw_key,
    )


@router.get(
    "/me",
    response_model=OrganizationResponse,
    summary="Get the calling tenant",
)
async def get_my_organization(org: CurrentOrgDep) -> OrganizationResponse:
    """Return the organization identified by the ``X-API-Key`` header.

    Args:
        org: The authenticated organization (from the API key).

    Returns:
        The tenant's own record.
    """
    return OrganizationResponse.model_validate(org)


@router.get(
    "/me/patients",
    response_model=PaginatedResponse[PatientResponse],
    summary="List the tenant's patients",
)
async def list_my_patients(
    org: CurrentOrgDep,
    db: DbSession,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> PaginatedResponse[PatientResponse]:
    """List patients owned by the calling tenant only.

    Args:
        org: The authenticated organization.
        db: Async database session.
        page: 1-indexed page number.
        page_size: Items per page.

    Returns:
        Paginated patients scoped to the tenant.
    """
    base = select(Patient).where(
        Patient.deleted_at.is_(None),
        Patient.organization_id == org.id,
    )
    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar() or 0
    rows = (
        (
            await db.execute(
                base.order_by(Patient.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        )
        .scalars()
        .all()
    )
    return PaginatedResponse(
        items=[PatientResponse.model_validate(p) for p in rows],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


@router.post(
    "/me/patients",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a patient owned by the tenant",
)
async def create_my_patient(
    body: PatientCreate, org: CurrentOrgDep, db: DbSession
) -> PatientResponse:
    """Create a patient stamped with the calling tenant's ``organization_id``.

    Args:
        body: Patient data.
        org: The authenticated organization.
        db: Async database session.

    Returns:
        The created patient.
    """
    patient = Patient(
        organization_id=org.id,
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
    log.info("Patient %s created under organization %s", patient.id, org.id)
    return PatientResponse.model_validate(patient)
