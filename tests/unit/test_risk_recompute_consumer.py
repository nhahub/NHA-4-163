"""Unit tests for the real-time risk-recompute consumer logic (Tier 7)."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass

import services.consumers.risk_recompute_consumer as consumer_mod
from services.consumers.risk_recompute_consumer import (
    ClinicalEvent,
    handle_clinical_event,
    parse_event,
    should_recompute,
)

PID = uuid.uuid4()


class TestParseEvent:
    def test_parses_dict(self) -> None:
        event = parse_event({"event_type": "condition.created", "patient_id": str(PID)})
        assert event is not None
        assert event.event_type == "condition.created"
        assert event.patient_id == PID

    def test_parses_json_string(self) -> None:
        raw = json.dumps({"event_type": "family.linked", "patient_id": str(PID)})
        event = parse_event(raw)
        assert event is not None and event.patient_id == PID

    def test_parses_bytes(self) -> None:
        raw = json.dumps({"event_type": "observation.created", "patient_id": str(PID)}).encode()
        event = parse_event(raw)
        assert event is not None and event.event_type == "observation.created"

    def test_payload_preserved(self) -> None:
        event = parse_event(
            {
                "event_type": "condition.created",
                "patient_id": str(PID),
                "payload": {"code": "C50"},
            }
        )
        assert event is not None and event.payload == {"code": "C50"}

    def test_malformed_json_returns_none(self) -> None:
        assert parse_event("{not json") is None

    def test_missing_patient_returns_none(self) -> None:
        assert parse_event({"event_type": "condition.created"}) is None

    def test_bad_uuid_returns_none(self) -> None:
        assert parse_event({"event_type": "condition.created", "patient_id": "not-a-uuid"}) is None

    def test_missing_event_type_returns_none(self) -> None:
        assert parse_event({"patient_id": str(PID)}) is None

    def test_non_dict_returns_none(self) -> None:
        assert parse_event(42) is None


class TestShouldRecompute:
    def test_known_event_triggers(self) -> None:
        assert should_recompute(ClinicalEvent("condition.created", PID)) is True
        assert should_recompute(ClinicalEvent("genetic_test.finalized", PID)) is True

    def test_unknown_event_ignored(self) -> None:
        assert should_recompute(ClinicalEvent("patient.viewed", PID)) is False
        assert should_recompute(ClinicalEvent("demographics.updated", PID)) is False


@dataclass
class _FakePrediction:
    risk_score: float = 0.9


class TestHandleClinicalEvent:
    def test_ignored_event_does_not_recompute(self) -> None:
        async def _recompute(db, pid):  # pragma: no cover - must not be called
            raise AssertionError("recompute_fn should not run for ignored events")

        event = ClinicalEvent("patient.viewed", PID)
        result = asyncio.run(handle_clinical_event(None, event, _recompute))
        assert result.recomputed is False
        assert "ignored" in result.reason

    def test_recompute_returning_none_skips_notifications(self, monkeypatch) -> None:
        async def _recompute(db, pid):
            return None

        called = {"n": 0}

        async def _fake_notifications(db, pid):
            called["n"] += 1
            return []

        monkeypatch.setattr(consumer_mod, "evaluate_patient_notifications", _fake_notifications)
        event = ClinicalEvent("condition.created", PID)
        result = asyncio.run(handle_clinical_event(None, event, _recompute))
        assert result.recomputed is False
        assert called["n"] == 0

    def test_happy_path_recomputes_and_notifies(self, monkeypatch) -> None:
        async def _recompute(db, pid):
            return _FakePrediction(risk_score=0.88)

        async def _fake_notifications(db, pid):
            return ["notif1", "notif2"]

        monkeypatch.setattr(consumer_mod, "evaluate_patient_notifications", _fake_notifications)
        event = ClinicalEvent("family.linked", PID)
        result = asyncio.run(handle_clinical_event(None, event, _recompute))
        assert result.recomputed is True
        assert result.patient_id == PID
        assert result.risk_score == 0.88
        assert result.notifications_created == 2
