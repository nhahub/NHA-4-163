"""Real-time hereditary-risk recomputation consumer (Tier 7).

Wires the existing Kafka infrastructure to the prediction + notification
services so that a new condition, family edge, or finalised genetic test
automatically re-scores the affected patient and raises an alert when their
risk profile changes materially — rather than waiting for a clinician to
re-run a prediction by hand.

Design
------
The event *policy* (which events matter, how to read a patient id out of them)
is pure and unit-tested: :func:`parse_event` and :func:`should_recompute`.
The DB orchestration (:func:`handle_clinical_event`) delegates the actual
scoring to an injected ``recompute_fn`` so it can be exercised without a live
model, and reuses
:func:`services.api.services.notification_service.evaluate_patient_notifications`
for alerting.  The Kafka receive loop (:func:`run_consumer`) is a thin wrapper
around whichever Kafka client is installed; it is never imported at test time.

Event contract (JSON value)::

    {"event_type": "condition.created",
     "patient_id": "3f2c...uuid",
     "occurred_at": "2026-07-10T12:00:00Z",
     "payload": { ... }}
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from libs.common.models.prediction_log import PredictionLog
from services.api.services.notification_service import evaluate_patient_notifications

log = logging.getLogger(__name__)

# Kafka topic carrying clinical/family-graph change events.
DEFAULT_TOPIC = "clinical.events"
DEFAULT_GROUP_ID = "risk-recompute"

# Event types that should trigger a risk recomputation.  Anything else (e.g.
# demographic edits, read events) is ignored so we don't re-score needlessly.
RECOMPUTE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "condition.created",
        "condition.updated",
        "family.linked",
        "family.updated",
        "genetic_test.finalized",
        "observation.created",
    }
)

# A recompute function: given a session + patient id, produce (and persist) a
# fresh PredictionLog, or return None if it could not be scored.
RecomputeFn = Callable[[AsyncSession, uuid.UUID], Awaitable["PredictionLog | None"]]


@dataclass(frozen=True)
class ClinicalEvent:
    """A parsed, validated clinical-change event."""

    event_type: str
    patient_id: uuid.UUID
    occurred_at: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecomputeResult:
    """Outcome of handling a single event."""

    recomputed: bool
    patient_id: uuid.UUID | None = None
    risk_score: float | None = None
    notifications_created: int = 0
    reason: str = ""


def parse_event(
    raw: Any,
) -> ClinicalEvent | None:
    """Parse a raw Kafka value into a :class:`ClinicalEvent` — pure.

    Accepts a ``bytes``/``str`` JSON payload or an already-decoded ``dict``.
    Returns ``None`` for anything malformed (unparseable JSON, missing/invalid
    ``event_type`` or ``patient_id``) so a bad message is skipped, never fatal.

    Args:
        raw: The Kafka message value.

    Returns:
        A :class:`ClinicalEvent`, or ``None`` if the message is unusable.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(raw, dict):
        return None

    event_type = raw.get("event_type")
    if not isinstance(event_type, str) or not event_type:
        return None

    raw_patient = raw.get("patient_id")
    if raw_patient is None:
        return None
    try:
        patient_id = uuid.UUID(str(raw_patient))
    except (ValueError, TypeError):
        return None

    payload = raw.get("payload")
    return ClinicalEvent(
        event_type=event_type,
        patient_id=patient_id,
        occurred_at=raw.get("occurred_at"),
        payload=payload if isinstance(payload, dict) else {},
    )


def should_recompute(event: ClinicalEvent) -> bool:
    """Return whether an event warrants a risk recomputation — pure.

    Args:
        event: A parsed clinical event.

    Returns:
        ``True`` if the event type is in :data:`RECOMPUTE_EVENT_TYPES`.
    """
    return event.event_type in RECOMPUTE_EVENT_TYPES


async def handle_clinical_event(
    db: AsyncSession,
    event: ClinicalEvent,
    recompute_fn: RecomputeFn,
) -> RecomputeResult:
    """Recompute risk for an event's patient and raise notifications.

    Args:
        db: Async database session.
        event: The parsed clinical event.
        recompute_fn: Callable that re-scores the patient and persists a
            :class:`PredictionLog` (injected so this is testable without a
            live model).

    Returns:
        A :class:`RecomputeResult` describing what happened.
    """
    if not should_recompute(event):
        return RecomputeResult(
            recomputed=False,
            patient_id=event.patient_id,
            reason=f"ignored event type '{event.event_type}'",
        )

    prediction = await recompute_fn(db, event.patient_id)
    if prediction is None:
        return RecomputeResult(
            recomputed=False,
            patient_id=event.patient_id,
            reason="recompute produced no prediction",
        )

    notifications = await evaluate_patient_notifications(db, event.patient_id)
    log.info(
        "Recomputed risk on %s: patient=%s score=%.4f notifications=%d",
        event.event_type,
        event.patient_id,
        float(prediction.risk_score),
        len(notifications),
    )
    return RecomputeResult(
        recomputed=True,
        patient_id=event.patient_id,
        risk_score=float(prediction.risk_score),
        notifications_created=len(notifications),
        reason="ok",
    )


def run_consumer(
    recompute_fn: RecomputeFn,
    *,
    topic: str = DEFAULT_TOPIC,
    group_id: str = DEFAULT_GROUP_ID,
    bootstrap_servers: str | None = None,
) -> None:  # pragma: no cover - requires a live Kafka broker
    """Run the blocking Kafka receive loop.

    Consumes ``topic``, and for each message parses the event and, when it
    warrants it, recomputes the patient's risk in a fresh DB session.  Each
    message is committed only after successful handling so a crash re-processes
    (at-least-once) rather than dropping events.

    This function is excluded from unit tests (it needs a broker); the pure and
    DB-orchestration pieces above carry the tested logic.

    Args:
        recompute_fn: The risk-scoring callable to apply per event.
        topic: Kafka topic to consume.
        group_id: Consumer group id.
        bootstrap_servers: Override for the Kafka bootstrap servers; defaults to
            ``libs.common.config`` Kafka settings.
    """
    import asyncio

    from confluent_kafka import Consumer

    from libs.common.config import get_settings
    from libs.common.logging import configure_logging
    from services.api.db import AsyncSessionFactory

    configure_logging()
    servers = bootstrap_servers or get_settings().kafka.bootstrap_servers
    consumer = Consumer(
        {
            "bootstrap.servers": servers,
            "group.id": group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    log.info("Risk-recompute consumer subscribed to %s (group=%s)", topic, group_id)

    async def _process(raw: bytes) -> None:
        event = parse_event(raw)
        if event is None:
            log.warning("Skipping malformed clinical event")
            return
        async with AsyncSessionFactory() as session:
            try:
                await handle_clinical_event(session, event, recompute_fn)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    loop = asyncio.new_event_loop()
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log.error("Kafka error: %s", msg.error())
                continue
            try:
                loop.run_until_complete(_process(msg.value()))
                consumer.commit(msg)
            except Exception as exc:
                log.exception("Failed to handle event; will retry: %s", exc)
    finally:
        consumer.close()
        loop.close()
