"""Unit tests for the patient-portal presentation logic (Tier 7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from services.api.services.portal_service import (
    build_risk_profile,
    deidentify_family_member,
    lay_band,
)


@dataclass
class _FakePrediction:
    risk_score: float
    risk_tier: str
    model_version: str | None = "m-1.0"
    predicted_at: datetime | None = None


@dataclass
class _FakeFamilyMember:
    relationship_code: str
    relationship_display: str | None
    sex: str | None
    degree_of_relatedness: float | None
    conditions: Any = field(default_factory=list)
    # PHI fields that must NOT leak into the portal view.
    given_name: str = "SECRET"
    family_name: str = "SECRET"


class TestLayBand:
    def test_boundaries(self) -> None:
        assert lay_band(0.0) == "low"
        assert lay_band(0.32) == "low"
        assert lay_band(0.33) == "moderate"
        assert lay_band(0.66) == "moderate"
        assert lay_band(0.67) == "high"
        assert lay_band(1.0) == "high"


class TestBuildRiskProfile:
    def test_none_prediction_is_unknown(self) -> None:
        view = build_risk_profile(None)
        assert view.band == "unknown"
        assert view.risk_score is None
        assert "No hereditary risk assessment" in view.guidance

    def test_high_score_high_band_with_guidance(self) -> None:
        view = build_risk_profile(
            _FakePrediction(0.82, "very_high", predicted_at=datetime(2026, 7, 10, tzinfo=UTC))
        )
        assert view.band == "high"
        assert view.risk_score == 0.82
        assert view.risk_tier == "very_high"
        assert "genetic counselling" in view.guidance
        assert view.predicted_at == "2026-07-10T00:00:00+00:00"

    def test_moderate_score_band(self) -> None:
        view = build_risk_profile(_FakePrediction(0.5, "moderate"))
        assert view.band == "moderate"
        assert "moderate" in view.guidance


class TestDeidentifyFamilyMember:
    def test_no_names_or_phi_leak(self) -> None:
        member = _FakeFamilyMember(
            relationship_code="MTH",
            relationship_display="Mother",
            sex="female",
            degree_of_relatedness=0.5,
            conditions=[
                {"code": {"display": "Breast cancer", "code": "C50"}},
            ],
        )
        out = deidentify_family_member(member)
        assert out["affected"] is True
        assert out["conditions"] == ["Breast cancer"]
        assert out["relationship_display"] == "Mother"
        # Ensure no PHI attributes bleed through.
        assert "SECRET" not in str(out)
        assert "given_name" not in out and "family_name" not in out

    def test_no_conditions_is_unaffected(self) -> None:
        member = _FakeFamilyMember("FTH", "Father", "male", 0.5, conditions=[])
        out = deidentify_family_member(member)
        assert out["affected"] is False
        assert out["conditions"] == []

    def test_none_conditions_handled(self) -> None:
        member = _FakeFamilyMember("SIB", "Sibling", "female", 0.5, conditions=None)
        out = deidentify_family_member(member)
        assert out["affected"] is False
        assert out["degree_of_relatedness"] == 0.5
