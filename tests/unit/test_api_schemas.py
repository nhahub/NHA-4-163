"""Unit tests for the API request/response schemas (Phase 6).

Validates:
- PredictHeredityRiskRequest — field defaults, UUID coercion, date pattern
- PredictFromSymptomsRequest / PredictFromPrescriptionRequest — list bounds
- _risk_tier() bucketing boundary conditions
- HeredityRiskResponse, FamilyRiskProfileResponse — round-trip serialisation
- SHAPContribution direction constraint
- ComponentStatus / HealthResponse construction
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from services.api.schemas.requests import (
    PredictFromPrescriptionRequest,
    PredictFromSymptomsRequest,
    PredictHeredityRiskRequest,
)
from services.api.schemas.responses import (
    ChapterBurden,
    ComponentStatus,
    DifferentialDiagnosisResponse,
    FamilyRiskProfileResponse,
    HealthResponse,
    HeredityRiskResponse,
    RelativeRecord,
    SHAPContribution,
    _risk_tier,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

_PATIENT_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")
_REQUEST_UUID = uuid.UUID("aaaabbbb-aaaa-bbbb-aaaa-bbbbaaaabbbb")


# ── _risk_tier() ──────────────────────────────────────────────────────────────


class TestRiskTier:
    def test_low_boundary(self) -> None:
        assert _risk_tier(0.0) == "low"

    def test_low_just_below_moderate(self) -> None:
        assert _risk_tier(0.249) == "low"

    def test_moderate_at_boundary(self) -> None:
        assert _risk_tier(0.25) == "moderate"

    def test_moderate_interior(self) -> None:
        assert _risk_tier(0.40) == "moderate"

    def test_high_at_boundary(self) -> None:
        assert _risk_tier(0.50) == "high"

    def test_high_interior(self) -> None:
        assert _risk_tier(0.60) == "high"

    def test_very_high_at_boundary(self) -> None:
        assert _risk_tier(0.75) == "very_high"

    def test_very_high_max(self) -> None:
        assert _risk_tier(1.0) == "very_high"


# ── PredictHeredityRiskRequest ────────────────────────────────────────────────


class TestPredictHeredityRiskRequest:
    def test_defaults(self) -> None:
        req = PredictHeredityRiskRequest(patient_id=_PATIENT_UUID)
        assert req.include_shap is True
        assert req.top_n_factors == 5
        assert req.feature_date is None

    def test_uuid_from_string(self) -> None:
        req = PredictHeredityRiskRequest(patient_id="12345678-1234-5678-1234-567812345678")
        assert req.patient_id == _PATIENT_UUID

    def test_valid_feature_date(self) -> None:
        req = PredictHeredityRiskRequest(patient_id=_PATIENT_UUID, feature_date="2024-01-15")
        assert req.feature_date == "2024-01-15"

    def test_invalid_feature_date_format(self) -> None:
        with pytest.raises(ValidationError):
            PredictHeredityRiskRequest(patient_id=_PATIENT_UUID, feature_date="15-01-2024")

    def test_top_n_factors_bounds(self) -> None:
        PredictHeredityRiskRequest(patient_id=_PATIENT_UUID, top_n_factors=1)
        PredictHeredityRiskRequest(patient_id=_PATIENT_UUID, top_n_factors=20)
        with pytest.raises(ValidationError):
            PredictHeredityRiskRequest(patient_id=_PATIENT_UUID, top_n_factors=0)
        with pytest.raises(ValidationError):
            PredictHeredityRiskRequest(patient_id=_PATIENT_UUID, top_n_factors=21)

    def test_immutable(self) -> None:
        req = PredictHeredityRiskRequest(patient_id=_PATIENT_UUID)
        with pytest.raises(ValidationError):
            req.include_shap = False  # type: ignore[misc]


# ── PredictFromSymptomsRequest ────────────────────────────────────────────────


class TestPredictFromSymptomsRequest:
    def test_valid_codes(self) -> None:
        req = PredictFromSymptomsRequest(patient_id=_PATIENT_UUID, symptom_codes=["R05.9", "J06.9"])
        assert len(req.symptom_codes) == 2

    def test_empty_codes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictFromSymptomsRequest(patient_id=_PATIENT_UUID, symptom_codes=[])

    def test_too_many_codes_rejected(self) -> None:
        codes = [f"R{i:02d}.0" for i in range(21)]
        with pytest.raises(ValidationError):
            PredictFromSymptomsRequest(patient_id=_PATIENT_UUID, symptom_codes=codes)

    def test_defaults(self) -> None:
        req = PredictFromSymptomsRequest(patient_id=_PATIENT_UUID, symptom_codes=["R05.9"])
        assert req.include_differential is True
        assert req.top_n == 5


# ── PredictFromPrescriptionRequest ────────────────────────────────────────────


class TestPredictFromPrescriptionRequest:
    def test_valid(self) -> None:
        req = PredictFromPrescriptionRequest(patient_id=_PATIENT_UUID, medication_codes=["123456"])
        assert req.medication_codes == ["123456"]

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictFromPrescriptionRequest(patient_id=_PATIENT_UUID, medication_codes=[])


# ── SHAPContribution ──────────────────────────────────────────────────────────


class TestSHAPContribution:
    def test_increases_risk(self) -> None:
        s = SHAPContribution(feature="age_years", shap_value=0.12, direction="increases_risk")
        assert s.direction == "increases_risk"

    def test_decreases_risk(self) -> None:
        s = SHAPContribution(
            feature="adherence_proxy", shap_value=-0.05, direction="decreases_risk"
        )
        assert s.direction == "decreases_risk"

    def test_invalid_direction(self) -> None:
        with pytest.raises(ValidationError):
            SHAPContribution(feature="x", shap_value=0.0, direction="neutral")  # type: ignore[arg-type]

    def test_raw_value_optional(self) -> None:
        s = SHAPContribution(feature="x", shap_value=0.0, direction="increases_risk")
        assert s.raw_value is None


# ── HeredityRiskResponse ──────────────────────────────────────────────────────


class TestHeredityRiskResponse:
    def _make(self, **overrides: object) -> HeredityRiskResponse:
        defaults: dict[str, object] = {
            "request_id": _REQUEST_UUID,
            "patient_id": _PATIENT_UUID,
            "risk_score": 0.3,
            "risk_tier": "moderate",
            "feature_date": "2024-01-15",
            "model_name": "hereditary-risk-xgboost",
            "model_version": "3",
            "cached": False,
        }
        defaults.update(overrides)
        return HeredityRiskResponse(**defaults)

    def test_basic_construction(self) -> None:
        r = self._make()
        assert r.risk_tier == "moderate"
        assert r.top_risk_factors is None

    def test_risk_score_bounds(self) -> None:
        self._make(risk_score=0.0)
        self._make(risk_score=1.0)
        with pytest.raises(ValidationError):
            self._make(risk_score=-0.01)
        with pytest.raises(ValidationError):
            self._make(risk_score=1.01)

    def test_shap_included(self) -> None:
        shap = [SHAPContribution(feature="x", shap_value=0.1, direction="increases_risk")]
        r = self._make(top_risk_factors=shap)
        assert r.top_risk_factors is not None
        assert len(r.top_risk_factors) == 1

    def test_model_dump_roundtrip(self) -> None:
        r = self._make()
        dumped = r.model_dump(mode="json")
        r2 = HeredityRiskResponse(**dumped)
        assert r2.patient_id == r.patient_id
        assert r2.risk_score == r.risk_score

    def test_cached_flag(self) -> None:
        r = self._make(cached=True)
        assert r.cached is True


# ── FamilyRiskProfileResponse ─────────────────────────────────────────────────


class TestFamilyRiskProfileResponse:
    def test_construction(self) -> None:
        profile = FamilyRiskProfileResponse(
            request_id=_REQUEST_UUID,
            patient_id=_PATIENT_UUID,
            family_risk_score=0.6,
            risk_tier="high",
            family_size=5,
            affected_relatives_count=2,
            first_degree_relatives=[
                RelativeRecord(
                    relative_id="rel-001",
                    relationship_code="IS_PARENT_OF",
                    degree_of_relatedness=0.5,
                    diagnosed_icd10_codes=["C50.9"],
                )
            ],
            disease_burden_by_chapter={
                "oncological": ChapterBurden(affected_relative_count=1, weighted_prevalence=0.5)
            },
            cached=False,
        )
        assert profile.family_size == 5
        assert profile.risk_tier == "high"
        assert len(profile.first_degree_relatives) == 1

    def test_family_risk_score_bounds(self) -> None:
        base: dict[str, object] = {
            "request_id": _REQUEST_UUID,
            "patient_id": _PATIENT_UUID,
            "risk_tier": "low",
            "family_size": 0,
            "affected_relatives_count": 0,
            "first_degree_relatives": [],
            "disease_burden_by_chapter": {},
            "cached": False,
        }
        FamilyRiskProfileResponse(**{**base, "family_risk_score": 0.0})
        FamilyRiskProfileResponse(**{**base, "family_risk_score": 1.0})
        with pytest.raises(ValidationError):
            FamilyRiskProfileResponse(**{**base, "family_risk_score": -0.01})

    def test_empty_family(self) -> None:
        profile = FamilyRiskProfileResponse(
            request_id=_REQUEST_UUID,
            patient_id=_PATIENT_UUID,
            family_risk_score=0.0,
            risk_tier="low",
            family_size=0,
            affected_relatives_count=0,
            first_degree_relatives=[],
            disease_burden_by_chapter={},
            cached=False,
        )
        assert profile.family_size == 0
        assert profile.disease_burden_by_chapter == {}


# ── HealthResponse ────────────────────────────────────────────────────────────


class TestHealthResponse:
    def test_ok_status(self) -> None:
        hr = HealthResponse(
            status="ok",
            version="0.6.0",
            components={"api": ComponentStatus(status="ok")},
        )
        assert hr.status == "ok"

    def test_degraded_status(self) -> None:
        hr = HealthResponse(
            status="degraded",
            version="0.6.0",
            components={
                "api": ComponentStatus(status="ok"),
                "model": ComponentStatus(status="degraded", detail="Model not loaded"),
            },
        )
        assert hr.components["model"].detail == "Model not loaded"

    def test_component_with_latency(self) -> None:
        cs = ComponentStatus(status="ok", latency_ms=3.7)
        assert cs.latency_ms == pytest.approx(3.7)

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            HealthResponse(status="unknown", version="0.6.0", components={})  # type: ignore[arg-type]


# ── DifferentialDiagnosisResponse ────────────────────────────────────────────


class TestDifferentialDiagnosisResponse:
    def test_construction(self) -> None:
        r = DifferentialDiagnosisResponse(
            request_id=_REQUEST_UUID,
            patient_id=_PATIENT_UUID,
            input_codes=["R05.9"],
            input_type="symptoms",
            predictions=[],
            model_version="1",
            cached=False,
        )
        assert r.input_type == "symptoms"

    def test_invalid_input_type(self) -> None:
        with pytest.raises(ValidationError):
            DifferentialDiagnosisResponse(
                request_id=_REQUEST_UUID,
                patient_id=_PATIENT_UUID,
                input_codes=["R05.9"],
                input_type="labs",  # type: ignore[arg-type]
                predictions=[],
                model_version="1",
                cached=False,
            )
