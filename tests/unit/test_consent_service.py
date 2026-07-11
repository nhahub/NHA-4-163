"""Unit tests for the consent evaluation logic (Tier 7)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from libs.common.models.consent import ConsentRecord, ConsentScope, ConsentStatus
from services.api.services.consent_service import (
    is_record_active,
    resolve_effective_consent,
)


def _rec(
    scope: ConsentScope,
    status: ConsentStatus,
    created_at: datetime,
    expires_at: datetime | None = None,
) -> ConsentRecord:
    """Build an in-memory ConsentRecord for pure-logic tests."""
    return ConsentRecord(
        scope=scope,
        status=status,
        created_at=created_at,
        expires_at=expires_at,
    )


NOW = datetime(2026, 7, 10, tzinfo=UTC)


class TestIsRecordActive:
    def test_granted_without_expiry_is_active(self) -> None:
        assert is_record_active(ConsentStatus.GRANTED, None, NOW) is True

    def test_granted_with_future_expiry_is_active(self) -> None:
        assert is_record_active(ConsentStatus.GRANTED, NOW + timedelta(days=1), NOW) is True

    def test_granted_but_expired_is_inactive(self) -> None:
        assert is_record_active(ConsentStatus.GRANTED, NOW - timedelta(days=1), NOW) is False

    def test_denied_is_inactive(self) -> None:
        assert is_record_active(ConsentStatus.DENIED, None, NOW) is False

    def test_withdrawn_is_inactive(self) -> None:
        assert is_record_active(ConsentStatus.WITHDRAWN, None, NOW) is False

    def test_naive_expiry_is_treated_as_utc(self) -> None:
        naive_future = datetime(2026, 7, 11)  # no tzinfo
        assert is_record_active(ConsentStatus.GRANTED, naive_future, NOW) is True


class TestResolveEffectiveConsent:
    def test_latest_record_per_scope_wins(self) -> None:
        old_grant = _rec(ConsentScope.RESEARCH, ConsentStatus.GRANTED, NOW)
        new_withdraw = _rec(ConsentScope.RESEARCH, ConsentStatus.WITHDRAWN, NOW + timedelta(days=5))
        effective = resolve_effective_consent([old_grant, new_withdraw])
        assert effective[ConsentScope.RESEARCH].status is ConsentStatus.WITHDRAWN

    def test_order_independent(self) -> None:
        old_grant = _rec(ConsentScope.RESEARCH, ConsentStatus.GRANTED, NOW)
        new_withdraw = _rec(ConsentScope.RESEARCH, ConsentStatus.WITHDRAWN, NOW + timedelta(days=5))
        # Reversed input order must resolve to the same effective record.
        effective = resolve_effective_consent([new_withdraw, old_grant])
        assert effective[ConsentScope.RESEARCH].status is ConsentStatus.WITHDRAWN

    def test_independent_scopes_kept_separate(self) -> None:
        research = _rec(ConsentScope.RESEARCH, ConsentStatus.GRANTED, NOW)
        sharing = _rec(ConsentScope.DATA_SHARING, ConsentStatus.DENIED, NOW)
        effective = resolve_effective_consent([research, sharing])
        assert set(effective) == {ConsentScope.RESEARCH, ConsentScope.DATA_SHARING}
        assert effective[ConsentScope.DATA_SHARING].status is ConsentStatus.DENIED

    def test_withdrawal_makes_research_inactive(self) -> None:
        # Mirrors the export-layer enforcement: a withdrawn research consent
        # resolves to an inactive effective record.
        grant = _rec(ConsentScope.RESEARCH, ConsentStatus.GRANTED, NOW)
        withdraw = _rec(ConsentScope.RESEARCH, ConsentStatus.WITHDRAWN, NOW + timedelta(days=1))
        effective = resolve_effective_consent([grant, withdraw])[ConsentScope.RESEARCH]
        assert is_record_active(effective.status, effective.expires_at, NOW) is False
