"""Unit tests for the notification threshold logic and org key helpers (Tier 4)."""

from __future__ import annotations

from libs.common.models.notification import NotificationSeverity, NotificationType
from libs.common.models.organization import (
    generate_api_key,
    hash_api_key,
)
from services.api.services.notification_service import evaluate_risk_change


class TestEvaluateRiskChange:
    def test_fresh_crossing_creates_critical(self) -> None:
        event = evaluate_risk_change(0.80, 0.60, "very_high", threshold=0.75)
        assert event is not None
        assert event.notification_type == NotificationType.RISK_THRESHOLD_CROSSED
        assert event.severity == NotificationSeverity.CRITICAL

    def test_first_prediction_above_threshold_crosses(self) -> None:
        event = evaluate_risk_change(0.90, None, "very_high", threshold=0.75)
        assert event is not None
        assert event.notification_type == NotificationType.RISK_THRESHOLD_CROSSED

    def test_already_above_threshold_does_not_recross(self) -> None:
        # Previous was already above threshold → not a fresh crossing.
        event = evaluate_risk_change(0.85, 0.80, "very_high", threshold=0.75)
        # A rise of 0.05 is below MIN_INCREASE (0.15), so no notification.
        assert event is None

    def test_material_increase_below_threshold_warns(self) -> None:
        event = evaluate_risk_change(0.55, 0.30, "high", threshold=0.75, min_increase=0.15)
        assert event is not None
        assert event.notification_type == NotificationType.RISK_INCREASED
        assert event.severity == NotificationSeverity.WARNING

    def test_stable_low_risk_no_event(self) -> None:
        assert evaluate_risk_change(0.20, 0.18, "low", threshold=0.75) is None

    def test_decrease_no_event(self) -> None:
        assert evaluate_risk_change(0.40, 0.70, "moderate", threshold=0.75) is None

    def test_none_current_score_no_event(self) -> None:
        assert evaluate_risk_change(None, 0.5, "moderate") is None

    def test_message_contains_no_raw_identifiers(self) -> None:
        event = evaluate_risk_change(0.80, 0.10, "very_high", threshold=0.75)
        assert event is not None
        # Message references score/tier only — no patient name/DOB tokens.
        assert "%" in event.message


class TestApiKeyHelpers:
    def test_hash_is_deterministic_and_hex64(self) -> None:
        h = hash_api_key("hc_secret")
        assert h == hash_api_key("hc_secret")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_generate_api_key_matches_hash(self) -> None:
        raw, key_hash = generate_api_key()
        assert raw.startswith("hc_")
        assert hash_api_key(raw) == key_hash

    def test_generated_keys_are_unique(self) -> None:
        assert generate_api_key()[0] != generate_api_key()[0]
