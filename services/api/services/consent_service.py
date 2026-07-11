"""Consent evaluation and recording logic (Tier 7).

Consent is append-only (see :mod:`libs.common.models.consent`): the effective
state for a scope is the most recent :class:`ConsentRecord` for that scope.
The pure helpers (:func:`is_record_active`, :func:`resolve_effective_consent`)
carry the policy and are unit-tested without a database; the ``async``
functions perform the DB reads/writes used by the router and the export layer.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.models.consent import (
    ConsentMethod,
    ConsentRecord,
    ConsentScope,
    ConsentStatus,
)

log = logging.getLogger(__name__)


def is_record_active(
    status: ConsentStatus,
    expires_at: datetime | None,
    now: datetime | None = None,
) -> bool:
    """Return whether a consent record currently grants permission — pure.

    A record is active only when its decision is ``granted`` and it has not
    passed its ``expires_at`` instant.  ``denied`` and ``withdrawn`` are never
    active.

    Args:
        status: The record's decision.
        expires_at: Optional expiry instant (timezone-aware), or ``None``.
        now: Reference instant; defaults to ``datetime.now(timezone.utc)``.

    Returns:
        ``True`` if the record grants permission at ``now``.
    """
    if status is not ConsentStatus.GRANTED:
        return False
    if expires_at is None:
        return True
    reference = now or datetime.now(UTC)
    # Guard against naive/aware comparison errors.
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    return expires_at > reference


def resolve_effective_consent(
    records: list[ConsentRecord],
    now: datetime | None = None,
) -> dict[ConsentScope, ConsentRecord]:
    """Reduce a patient's consent history to the effective record per scope.

    For each scope, the record with the latest ``created_at`` wins.  Records
    with no ``created_at`` (unpersisted) are treated as newest so freshly built
    rows resolve correctly in tests.

    Args:
        records: All consent records for a patient (any order).
        now: Unused here but accepted for symmetry with callers.

    Returns:
        Mapping of scope → most-recent :class:`ConsentRecord` for that scope.
    """
    latest: dict[ConsentScope, ConsentRecord] = {}
    for rec in records:
        current = latest.get(rec.scope)
        if current is None or _created_key(rec) >= _created_key(current):
            latest[rec.scope] = rec
    return latest


def _created_key(rec: ConsentRecord) -> datetime:
    """Sort key for consent recency; unpersisted rows sort newest."""
    return rec.created_at or datetime.max.replace(tzinfo=UTC)


async def record_consent(
    db: AsyncSession,
    patient_id: uuid.UUID,
    scope: ConsentScope,
    status: ConsentStatus,
    *,
    method: ConsentMethod | None = None,
    expires_at: datetime | None = None,
    policy_version: str | None = None,
    notes: str | None = None,
    organization_id: uuid.UUID | None = None,
    actor: str | None = None,
) -> ConsentRecord:
    """Append a new consent decision for a patient + scope.

    Sets ``granted_at`` when the decision is ``granted`` and ``withdrawn_at``
    when it is ``withdrawn`` so the timeline is self-describing.

    Args:
        db: Async database session.
        patient_id: Patient the consent belongs to.
        scope: The consent scope.
        status: The decision (granted/denied/withdrawn).
        method: How the decision was captured.
        expires_at: Optional expiry for a grant.
        policy_version: Consent policy/version reference.
        notes: PHI-free free-text notes.
        organization_id: Owning tenant, stamped onto the row.
        actor: Subject (``sub`` claim) recording the decision.

    Returns:
        The persisted :class:`ConsentRecord`.
    """
    now = datetime.now(UTC)
    record = ConsentRecord(
        patient_id=patient_id,
        organization_id=organization_id,
        scope=scope,
        status=status,
        method=method,
        granted_at=now if status is ConsentStatus.GRANTED else None,
        withdrawn_at=now if status is ConsentStatus.WITHDRAWN else None,
        expires_at=expires_at,
        policy_version=policy_version,
        notes=notes,
        created_by=actor,
        updated_by=actor,
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)
    log.info(
        "Consent recorded: patient=%s scope=%s status=%s",
        patient_id,
        scope.value,
        status.value,
    )
    return record


async def get_consent_history(db: AsyncSession, patient_id: uuid.UUID) -> list[ConsentRecord]:
    """Return all consent records for a patient, newest first.

    Args:
        db: Async database session.
        patient_id: Patient UUID.

    Returns:
        Consent records ordered by ``created_at`` descending.
    """
    result = await db.execute(
        select(ConsentRecord)
        .where(ConsentRecord.patient_id == patient_id)
        .order_by(ConsentRecord.created_at.desc())
    )
    return list(result.scalars().all())


async def has_active_consent(db: AsyncSession, patient_id: uuid.UUID, scope: ConsentScope) -> bool:
    """Check whether a patient currently grants a given consent scope.

    Args:
        db: Async database session.
        patient_id: Patient UUID.
        scope: Scope to check.

    Returns:
        ``True`` when the effective (latest) record for the scope is an active
        grant.  ``False`` when there is no record, or it is denied/withdrawn/
        expired.
    """
    history = await get_consent_history(db, patient_id)
    effective = resolve_effective_consent(history)
    record = effective.get(scope)
    if record is None:
        return False
    return is_record_active(record.status, record.expires_at)


async def patients_denying_scope(db: AsyncSession, scope: ConsentScope) -> set[uuid.UUID]:
    """Return patient IDs whose *effective* consent for a scope is not active.

    Used by enforcement points (e.g. the research export) to exclude patients
    who have explicitly denied or withdrawn a scope — even if a legacy boolean
    flag still says otherwise.  Patients with **no** record for the scope are
    *not* included here (absence of a record is not an explicit denial).

    Args:
        db: Async database session.
        scope: Scope to evaluate.

    Returns:
        Set of patient UUIDs to exclude for this scope.
    """
    result = await db.execute(
        select(ConsentRecord)
        .where(ConsentRecord.scope == scope)
        .order_by(ConsentRecord.created_at.desc())
    )
    by_patient: dict[uuid.UUID, list[ConsentRecord]] = {}
    for rec in result.scalars().all():
        by_patient.setdefault(rec.patient_id, []).append(rec)

    denied: set[uuid.UUID] = set()
    for patient_id, records in by_patient.items():
        effective = resolve_effective_consent(records).get(scope)
        if effective is not None and not is_record_active(effective.status, effective.expires_at):
            denied.add(patient_id)
    return denied
