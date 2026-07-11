"""Cascade screening logic (Tier 5 — Genetics & Genomics).

When a proband is diagnosed with a hereditary condition, this service ranks the
proband's at-risk blood relatives and materialises an outreach/screening task
per relative.  Ranking reuses the Mendelian calculator
(:mod:`services.api.services.inheritance_service`): a relative's outreach
priority is driven by their affected/carrier probability, which in turn depends
on ``degree_of_relatedness`` × condition penetrance.

The pure :func:`rank_relative` / :func:`priority_for_score` helpers carry the
policy and are unit-tested without a database; :func:`generate_cascade_screening`
performs the DB orchestration and notification side effects.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.models.cascade import (
    CascadePriority,
    CascadeScreening,
    CascadeTask,
    CascadeTaskStatus,
)
from libs.common.models.family_member_history import FamilyMemberHistory
from libs.common.models.notification import (
    Notification,
    NotificationSeverity,
    NotificationType,
)
from services.api.services.inheritance_service import (
    categorise_relationship,
    compute_relative_risk,
)

log = logging.getLogger(__name__)

# Priority thresholds on the composite priority score (affected-weighted).
HIGH_PRIORITY_THRESHOLD = 0.40
MEDIUM_PRIORITY_THRESHOLD = 0.15


@dataclass(frozen=True)
class RankedRelative:
    """A relative ranked for cascade outreach."""

    relationship_code: str
    priority: CascadePriority
    priority_score: float
    carrier_probability: float
    affected_probability: float
    recommended_action: str
    basis: str


def priority_for_score(score: float) -> CascadePriority:
    """Bucket a composite priority score into a :class:`CascadePriority`.

    Args:
        score: Composite priority score in [0, 1].

    Returns:
        The corresponding priority band.
    """
    if score >= HIGH_PRIORITY_THRESHOLD:
        return CascadePriority.HIGH
    if score >= MEDIUM_PRIORITY_THRESHOLD:
        return CascadePriority.MEDIUM
    return CascadePriority.LOW


def _recommended_action(priority: CascadePriority, condition_display: str | None) -> str:
    """Return a human-readable outreach recommendation for a priority band."""
    cond = condition_display or "the hereditary condition"
    if priority == CascadePriority.HIGH:
        return (
            f"Priority outreach: offer genetic counselling and targeted testing "
            f"for {cond} promptly."
        )
    if priority == CascadePriority.MEDIUM:
        return f"Offer genetic counselling and discuss screening for {cond}."
    return f"Inform of familial risk for {cond}; screening at routine intervals."


def rank_relative(
    relationship_code: str,
    degree_of_relatedness: float | None,
    relative_sex: str | None,
    inheritance_mode: str,
    penetrance: float,
    carrier_frequency: float | None,
    condition_display: str | None = None,
) -> RankedRelative:
    """Rank a single relative for cascade outreach — pure logic.

    The composite priority score weights affected probability most heavily
    (they may already be at risk of disease) with a smaller contribution from
    carrier probability (reproductive/at-risk-of-transmitting relevance).

    Args:
        relationship_code: HL7 v3 family-member code.
        degree_of_relatedness: Wright coefficient, or ``None``.
        relative_sex: ``male``/``female`` if known.
        inheritance_mode: Mendelian mode key.
        penetrance: Disease penetrance in [0, 1].
        carrier_frequency: Population carrier frequency override, or ``None``.
        condition_display: Human-readable condition name for the recommendation.

    Returns:
        A :class:`RankedRelative`.
    """
    risk = compute_relative_risk(
        mode=inheritance_mode,
        relationship_code=relationship_code,
        degree_of_relatedness=degree_of_relatedness,
        relative_sex=relative_sex,
        penetrance=penetrance,
        carrier_frequency=carrier_frequency,
    )
    score = round(0.7 * risk.affected_probability + 0.3 * risk.carrier_probability, 4)
    priority = priority_for_score(score)
    return RankedRelative(
        relationship_code=relationship_code,
        priority=priority,
        priority_score=score,
        carrier_probability=risk.carrier_probability,
        affected_probability=risk.affected_probability,
        recommended_action=_recommended_action(priority, condition_display),
        basis=risk.basis,
    )


async def generate_cascade_screening(
    db: AsyncSession,
    proband_patient_id: uuid.UUID,
    inheritance_mode: str,
    penetrance: float,
    carrier_frequency: float | None,
    condition_code: str | None,
    condition_display: str | None,
    organization_id: uuid.UUID | None = None,
    notify: bool = True,
) -> CascadeScreening:
    """Create a cascade-screening run with a ranked task per at-risk relative.

    Spouses/unrelated relatives (degree 0) and any relative whose composite
    score is effectively zero are excluded from outreach.

    Args:
        db: Async database session.
        proband_patient_id: The affected proband's patient id.
        inheritance_mode: Mendelian mode key.
        penetrance: Disease penetrance.
        carrier_frequency: Population carrier frequency override.
        condition_code: ICD-10/OMIM code of the trigger condition.
        condition_display: Human-readable condition name.
        organization_id: Owning tenant, stamped onto the run and notifications.
        notify: If True, emit one notification per high-priority relative.

    Returns:
        The persisted :class:`CascadeScreening` with its tasks attached.
    """
    screening = CascadeScreening(
        proband_patient_id=proband_patient_id,
        organization_id=organization_id,
        condition_code=condition_code,
        condition_display=condition_display,
        inheritance_mode=inheritance_mode,
        penetrance=penetrance,
    )
    db.add(screening)
    await db.flush()  # assign screening.id

    members = (
        (
            await db.execute(
                select(FamilyMemberHistory).where(
                    FamilyMemberHistory.patient_id == proband_patient_id
                )
            )
        )
        .scalars()
        .all()
    )

    tasks: list[CascadeTask] = []
    high_priority = 0
    for m in members:
        category = categorise_relationship(m.relationship_code)
        if category == "spouse":
            continue  # not a blood relative — no genetic cascade risk
        degree = float(m.degree_of_relatedness) if m.degree_of_relatedness is not None else None
        ranked = rank_relative(
            relationship_code=m.relationship_code,
            degree_of_relatedness=degree,
            relative_sex=m.sex,
            inheritance_mode=inheritance_mode,
            penetrance=penetrance,
            carrier_frequency=carrier_frequency,
            condition_display=condition_display,
        )
        if ranked.priority_score <= 0.0:
            continue  # no established genetic risk to this relative
        task = CascadeTask(
            screening_id=screening.id,
            family_member_id=m.id,
            related_patient_id=m.related_patient_id,
            relationship_code=m.relationship_code,
            relationship_display=m.relationship_display,
            degree_of_relatedness=degree,
            priority=ranked.priority,
            priority_score=ranked.priority_score,
            carrier_probability=ranked.carrier_probability,
            affected_probability=ranked.affected_probability,
            status=CascadeTaskStatus.PENDING,
            recommended_action=ranked.recommended_action,
        )
        db.add(task)
        tasks.append(task)
        if ranked.priority == CascadePriority.HIGH:
            high_priority += 1

    screening.task_count = len(tasks)

    if notify and high_priority > 0:
        cond = condition_display or "a hereditary condition"
        db.add(
            Notification(
                patient_id=proband_patient_id,
                organization_id=organization_id,
                notification_type=NotificationType.FAMILY_UPDATE,
                severity=NotificationSeverity.WARNING,
                title="Cascade screening: at-risk relatives identified",
                message=(
                    f"Cascade screening for {cond} identified {high_priority} "
                    f"high-priority relative(s) out of {len(tasks)} at-risk. "
                    f"Initiate genetic-counselling outreach."
                ),
            )
        )

    await db.flush()
    await db.refresh(screening)
    log.info(
        "Cascade screening generated: proband=%s mode=%s tasks=%d high=%d",
        proband_patient_id,
        inheritance_mode,
        len(tasks),
        high_priority,
    )
    return screening
