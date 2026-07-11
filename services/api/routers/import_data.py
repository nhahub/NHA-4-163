"""Bulk CSV import for patients and their conditions.

POST /import/csv
    Accepts a CSV upload of patient rows (with optional condition columns),
    validates every row against the ``PatientCreate`` contract, and bulk-inserts
    the valid rows into PostgreSQL.  Invalid rows are skipped and reported back
    with their line number and error so the researcher can correct and re-upload.

Expected CSV header (superset — only ``given_name``, ``family_name``,
``date_of_birth``, ``gender`` are required):

    given_name,family_name,date_of_birth,gender,ethnicity,race,state,
    external_id,condition_code,condition_display,condition_is_hereditary
"""

from __future__ import annotations

import csv
import io
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ValidationError

from libs.common.models.condition import ClinicalStatus, Condition, VerificationStatus
from libs.common.models.patient import AdministrativeGender, Patient
from services.api.db import DbSession
from services.api.schemas.crud_schemas import PatientCreate

log = logging.getLogger(__name__)

router = APIRouter(prefix="/import", tags=["import"])

_MAX_ROWS = 10_000
_TRUE_VALUES = {"1", "true", "yes", "y", "t"}


class ImportRowError(BaseModel):
    """A single rejected row from a CSV import."""

    line: int
    error: str


class ImportResult(BaseModel):
    """Summary returned by ``POST /import/csv``."""

    total_rows: int
    patients_created: int
    conditions_created: int
    skipped: int
    errors: list[ImportRowError] = []


def _clean(row: dict[str, str]) -> dict[str, str]:
    """Strip whitespace and drop empty-string values so defaults apply."""
    return {
        k.strip(): v.strip()
        for k, v in row.items()
        if k is not None and v is not None and v.strip() != ""
    }


@router.post(
    "/csv",
    response_model=ImportResult,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk import patients (and conditions) from a CSV file",
)
async def import_csv(db: DbSession, file: UploadFile = File(...)) -> ImportResult:
    """Validate and bulk-insert patients (and optional conditions) from a CSV.

    Args:
        db: Async database session.
        file: The uploaded CSV file.

    Returns:
        An :class:`ImportResult` summarising created/skipped rows and errors.

    Raises:
        HTTPException 400: If the file is not a readable, non-empty CSV, or
            exceeds the maximum row count.
    """
    if file.content_type not in (
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
        None,
    ):
        # Be lenient — browsers report CSV inconsistently — but reject obvious non-CSV.
        if file.filename and not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Expected a .csv file")

    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"File is not valid UTF-8: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise HTTPException(status_code=400, detail="CSV has no header row")

    patients_created = 0
    conditions_created = 0
    skipped = 0
    errors: list[ImportRowError] = []
    total = 0

    for i, raw_row in enumerate(reader, start=2):  # line 1 is the header
        total += 1
        if total > _MAX_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"CSV exceeds maximum of {_MAX_ROWS} rows",
            )

        row = _clean(raw_row)
        try:
            validated = PatientCreate(
                given_name=row.get("given_name", ""),
                family_name=row.get("family_name", ""),
                # KeyError → caught below; pydantic coerces the ISO date string.
                date_of_birth=row["date_of_birth"],  # type: ignore[arg-type]
                gender=row.get("gender", "unknown"),
                ethnicity=row.get("ethnicity"),
                race=row.get("race"),
                state=row.get("state"),
                external_id=row.get("external_id"),
            )
        except (ValidationError, KeyError) as exc:
            skipped += 1
            errors.append(ImportRowError(line=i, error=_format_error(exc)))
            continue

        patient = Patient(
            given_name=validated.given_name,
            family_name=validated.family_name,
            date_of_birth=validated.date_of_birth,
            gender=AdministrativeGender(validated.gender),
            ethnicity=validated.ethnicity,
            race=validated.race,
            state=validated.state,
            external_id=validated.external_id,
        )
        db.add(patient)
        await db.flush()
        patients_created += 1

        # ── Optional condition column ─────────────────────────────────────────
        condition_code = row.get("condition_code")
        if condition_code:
            db.add(
                Condition(
                    patient_id=patient.id,
                    code=condition_code,
                    code_system=row.get("condition_code_system", "http://hl7.org/fhir/sid/icd-10"),
                    code_display=row.get("condition_display"),
                    clinical_status=ClinicalStatus.ACTIVE,
                    verification_status=VerificationStatus.CONFIRMED,
                    is_hereditary=row.get("condition_is_hereditary", "").lower() in _TRUE_VALUES,
                )
            )
            conditions_created += 1

    await db.flush()
    log.info(
        "CSV import complete: %d patients, %d conditions, %d skipped (of %d rows)",
        patients_created,
        conditions_created,
        skipped,
        total,
    )

    return ImportResult(
        total_rows=total,
        patients_created=patients_created,
        conditions_created=conditions_created,
        skipped=skipped,
        errors=errors,
    )


def _format_error(exc: Exception) -> str:
    """Render a validation/parse error into a compact human-readable message."""
    if isinstance(exc, KeyError):
        return f"missing required column: {exc.args[0]}"
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return "; ".join(parts)
    return str(exc)
