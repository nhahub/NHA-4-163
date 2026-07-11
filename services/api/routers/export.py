"""De-identified research data export.

GET /export/patients/deidentified?format=csv|json
    Streams a research-ready dataset of consented patients with all direct and
    quasi-identifiers removed or generalised per HIPAA Safe Harbor
    (``libs.common.deidentification``).

Compliance guarantees enforced here:
  - Only patients with ``research_consent = True`` are included.
  - Patient UUIDs are never emitted; a salted one-way pseudonym is used instead.
  - Exact dates of birth are reduced to an age band; names/contact are dropped.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
from collections.abc import Iterator
from datetime import date

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from libs.common.deidentification import generalise_age
from libs.common.models.condition import Condition
from libs.common.models.consent import ConsentScope
from libs.common.models.medication_request import MedicationRequest, MedicationRequestStatus
from libs.common.models.patient import Patient
from services.api.db import DbSession
from services.api.services.consent_service import patients_denying_scope

log = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["export"])

# Salt for research pseudonyms.  A stable salt yields consistent pseudonyms
# across exports (needed to join longitudinal research datasets) while the
# one-way hash prevents re-identification without the salt.
_PSEUDONYM_SALT = os.environ.get("RESEARCH_PSEUDONYM_SALT", "healthcare-research-salt")

_EXPORT_COLUMNS = [
    "research_id",
    "age_band",
    "gender",
    "ethnicity",
    "race",
    "state",
    "deceased",
    "condition_count",
    "hereditary_condition_count",
    "active_medication_count",
]


def _pseudonym(patient_id: object) -> str:
    """Return a stable, non-reversible research pseudonym for a patient ID.

    Args:
        patient_id: The internal patient UUID (any stringable form).

    Returns:
        A ``R-<16 hex>`` pseudonym derived from a salted SHA-256 hash.
    """
    digest = hashlib.sha256(f"{_PSEUDONYM_SALT}:{patient_id}".encode()).hexdigest()
    return f"R-{digest[:16]}"


def _age_band(dob: date | None) -> str | None:
    """Compute a generalised age band (decade / ``90+``) from a DOB."""
    if dob is None:
        return None
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return generalise_age(age)


async def _build_rows(db: DbSession) -> list[dict[str, object]]:
    """Build the de-identified export rows for all consented patients.

    Args:
        db: Async database session.

    Returns:
        A list of PHI-free row dicts.
    """
    # Per-patient condition counts.
    cond_counts = dict(
        row.tuple()
        for row in (
            await db.execute(
                select(Condition.patient_id, func.count()).group_by(Condition.patient_id)
            )
        ).all()
    )
    hered_counts = dict(
        row.tuple()
        for row in (
            await db.execute(
                select(Condition.patient_id, func.count())
                .where(Condition.is_hereditary.is_(True))
                .group_by(Condition.patient_id)
            )
        ).all()
    )
    med_counts = dict(
        row.tuple()
        for row in (
            await db.execute(
                select(MedicationRequest.patient_id, func.count())
                .where(MedicationRequest.status == MedicationRequestStatus.ACTIVE)
                .group_by(MedicationRequest.patient_id)
            )
        ).all()
    )

    # Tier 7 consent enforcement: exclude any patient whose *effective*
    # granular ``research`` consent has been explicitly denied or withdrawn,
    # even if the legacy ``Patient.research_consent`` flag is still set.
    denied = await patients_denying_scope(db, ConsentScope.RESEARCH)

    result = await db.execute(
        select(Patient).where(
            Patient.deleted_at.is_(None),
            Patient.research_consent.is_(True),
        )
    )
    rows: list[dict[str, object]] = []
    for p in result.scalars().all():
        if p.id in denied:
            continue
        rows.append(
            {
                "research_id": _pseudonym(p.id),
                "age_band": _age_band(p.date_of_birth),
                "gender": p.gender.value if p.gender is not None else None,
                "ethnicity": p.ethnicity,
                "race": p.race,
                "state": p.state,  # state is permitted under Safe Harbor
                "deceased": p.deceased,
                "condition_count": cond_counts.get(p.id, 0),
                "hereditary_condition_count": hered_counts.get(p.id, 0),
                "active_medication_count": med_counts.get(p.id, 0),
            }
        )
    return rows


@router.get(
    "/patients/deidentified",
    summary="Export de-identified patient dataset (CSV or JSON)",
)
async def export_deidentified_patients(
    db: DbSession,
    format: str = Query(default="csv", pattern=r"^(csv|json)$"),
) -> StreamingResponse:
    """Stream a de-identified, research-ready patient dataset.

    Args:
        db: Async database session.
        format: Output format — ``csv`` (default) or ``json``.

    Returns:
        A streaming CSV or JSON download containing no PHI.
    """
    rows = await _build_rows(db)
    log.info("De-identified export: %d consented patients as %s", len(rows), format)

    if format == "json":
        payload = json.dumps({"count": len(rows), "patients": rows}, default=str)
        return StreamingResponse(
            iter([payload]),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="patients_deidentified.json"'},
        )

    def _csv_iter() -> Iterator[str]:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=_EXPORT_COLUMNS)
        writer.writeheader()
        yield buffer.getvalue()
        for row in rows:
            buffer.seek(0)
            buffer.truncate(0)
            writer.writerow(row)
            yield buffer.getvalue()

    return StreamingResponse(
        _csv_iter(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="patients_deidentified.csv"'},
    )
