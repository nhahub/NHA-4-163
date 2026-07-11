"""FHIR R4 interoperability endpoints.

GET  /fhir/Patient/{id}          — Patient resource in FHIR JSON
GET  /fhir/Condition?patient={id} — searchset Bundle of the patient's conditions
POST /fhir/Bundle                — ingest a transaction Bundle (Patient + clinical data)

These endpoints let external EHR systems exchange data with the prediction
engine using the standard HL7 FHIR R4 wire format.  See
``services.api.schemas.fhir_schemas`` for the modelled resource subset.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from libs.common.models.condition import ClinicalStatus, Condition, VerificationStatus
from libs.common.models.patient import AdministrativeGender, Patient
from services.api.db import DbSession
from services.api.schemas.fhir_schemas import (
    FHIRBundle,
    FHIRBundleResponseEntry,
    FHIRCondition,
    FHIRPatient,
    FHIRTransactionResult,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/fhir", tags=["fhir"])


@router.get(
    "/Patient/{patient_id}",
    response_model=FHIRPatient,
    summary="Get a patient as a FHIR R4 Patient resource",
)
async def get_fhir_patient(patient_id: uuid.UUID, db: DbSession) -> FHIRPatient:
    """Return a single patient rendered as a FHIR ``Patient`` resource.

    Args:
        patient_id: Patient UUID.
        db: Async database session.

    Returns:
        FHIR ``Patient`` resource.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return FHIRPatient.from_orm_patient(patient)


@router.get(
    "/Condition",
    response_model=FHIRBundle,
    summary="Search a patient's conditions as a FHIR Bundle",
)
async def search_fhir_conditions(
    db: DbSession,
    patient: uuid.UUID = Query(..., description="Patient UUID (FHIR search param)"),
) -> FHIRBundle:
    """Return the patient's conditions as a FHIR ``searchset`` Bundle.

    Args:
        db: Async database session.
        patient: Patient UUID supplied via the ``patient`` query parameter.

    Returns:
        FHIR ``Bundle`` of ``Condition`` resources.
    """
    result = await db.execute(select(Condition).where(Condition.patient_id == patient))
    conditions = result.scalars().all()
    fhir_conditions = [FHIRCondition.from_orm_condition(c) for c in conditions]
    return FHIRBundle.searchset(fhir_conditions)


def _parse_fhir_gender(
    value: Any,
) -> AdministrativeGender:
    """Map a FHIR gender string to :class:`AdministrativeGender`."""
    try:
        return AdministrativeGender(str(value))
    except ValueError:
        return AdministrativeGender.UNKNOWN


def _extract_patient(resource: dict[str, Any]) -> Patient:
    """Build a ``Patient`` ORM row from a FHIR Patient resource dict.

    Args:
        resource: FHIR ``Patient`` resource JSON.

    Returns:
        Unpersisted :class:`Patient` instance.
    """
    names = resource.get("name") or [{}]
    name = names[0] if names else {}
    given_list = name.get("given") or []

    birth_date: date | None = None
    if resource.get("birthDate"):
        try:
            birth_date = date.fromisoformat(str(resource["birthDate"])[:10])
        except ValueError:
            birth_date = None

    identifiers = resource.get("identifier") or []
    external_id = None
    identifier_system = None
    if identifiers:
        external_id = identifiers[0].get("value")
        identifier_system = identifiers[0].get("system")

    return Patient(
        given_name=given_list[0] if given_list else None,
        middle_name=given_list[1] if len(given_list) > 1 else None,
        family_name=name.get("family"),
        gender=_parse_fhir_gender(resource.get("gender", "unknown")),
        date_of_birth=birth_date,
        deceased=bool(resource.get("deceasedBoolean", False)),
        external_id=external_id,
        identifier_system=identifier_system,
    )


def _extract_condition(resource: dict[str, Any], patient_id: uuid.UUID) -> Condition:
    """Build a ``Condition`` ORM row from a FHIR Condition resource dict.

    Args:
        resource: FHIR ``Condition`` resource JSON.
        patient_id: The patient the condition belongs to.

    Returns:
        Unpersisted :class:`Condition` instance.

    Raises:
        HTTPException 400: If the resource has no coding.
    """
    code_cc = resource.get("code") or {}
    codings = code_cc.get("coding") or []
    if not codings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Condition resource missing code.coding",
        )
    coding = codings[0]

    clinical_cc = resource.get("clinicalStatus") or {}
    clinical_codings = clinical_cc.get("coding") or [{}]
    clinical_code = clinical_codings[0].get("code", "active")
    try:
        clinical_status = ClinicalStatus(clinical_code)
    except ValueError:
        clinical_status = ClinicalStatus.ACTIVE

    onset: datetime | None = None
    if resource.get("onsetDateTime"):
        try:
            onset = datetime.fromisoformat(str(resource["onsetDateTime"]))
        except ValueError:
            onset = None

    return Condition(
        patient_id=patient_id,
        code=coding.get("code", ""),
        code_system=coding.get("system", "http://hl7.org/fhir/sid/icd-10"),
        code_display=coding.get("display"),
        code_text=code_cc.get("text"),
        clinical_status=clinical_status,
        verification_status=VerificationStatus.CONFIRMED,
        onset_datetime=onset,
    )


@router.post(
    "/Bundle",
    response_model=FHIRTransactionResult,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a FHIR transaction Bundle",
)
async def ingest_fhir_bundle(bundle: FHIRBundle, db: DbSession) -> FHIRTransactionResult:
    """Ingest a transaction Bundle containing a patient and clinical data.

    Processing order: the first ``Patient`` entry is created (or referenced),
    then every ``Condition`` entry is linked to that patient.  Conditions whose
    ``subject`` references a ``Patient`` fullUrl in the same bundle are linked
    to the newly created patient.

    Args:
        bundle: The transaction Bundle.
        db: Async database session.

    Returns:
        A ``transaction-response`` summary of created resources.

    Raises:
        HTTPException 400: If no Patient resource is present in the bundle.
    """
    created: list[FHIRBundleResponseEntry] = []

    # ── First pass: create the patient ────────────────────────────────────────
    patient_row: Patient | None = None
    patient_full_url: str | None = None
    for entry in bundle.entry:
        resource = entry.resource
        if resource.get("resourceType") == "Patient":
            patient_row = _extract_patient(resource)
            db.add(patient_row)
            await db.flush()
            await db.refresh(patient_row)
            patient_full_url = entry.fullUrl
            created.append(
                FHIRBundleResponseEntry(
                    resourceType="Patient",
                    id=str(patient_row.id),
                    status="201 Created",
                )
            )
            break

    if patient_row is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction Bundle must contain exactly one Patient resource",
        )

    # ── Second pass: create conditions linked to the patient ──────────────────
    for entry in bundle.entry:
        resource = entry.resource
        if resource.get("resourceType") != "Condition":
            continue
        condition = _extract_condition(resource, patient_row.id)
        db.add(condition)
        await db.flush()
        await db.refresh(condition)
        created.append(
            FHIRBundleResponseEntry(
                resourceType="Condition",
                id=str(condition.id),
                status="201 Created",
            )
        )

    log.info(
        "FHIR Bundle ingested: patient=%s resources=%d (bundle_patient_url=%s)",
        patient_row.id,
        len(created),
        patient_full_url,
    )
    return FHIRTransactionResult(created=created)
