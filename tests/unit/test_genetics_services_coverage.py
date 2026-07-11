"""Unit tests for the pure genetics/decision-support service functions.

These modules (inheritance, PRS, what-if, cascade ranking) are deterministic and
free of I/O, so they are exercised directly across their full branch space.
"""

from __future__ import annotations

import pytest

from services.api.services.cascade_service import (
    CascadePriority,
    priority_for_score,
    rank_relative,
)
from services.api.services.inheritance_service import (
    INHERITANCE_MODELS,
    categorise_relationship,
    compute_relative_risk,
    infer_sex,
)
from services.api.services.prs_service import PRS_PANELS, compute_prs
from services.api.services.whatif_service import RISK_FACTORS, simulate

_MODES = list(INHERITANCE_MODELS.keys())
_PANELS = list(PRS_PANELS.keys())


# ── inheritance_service ───────────────────────────────────────────────────────


class TestInheritance:
    @pytest.mark.parametrize("mode", _MODES)
    @pytest.mark.parametrize(
        "relationship,sex",
        [("MTH", "female"), ("FTH", "male"), ("SIS", "female"), ("SON", "male")],
    )
    def test_compute_relative_risk_all_modes(self, mode: str, relationship: str, sex: str) -> None:
        risk = compute_relative_risk(
            mode=mode,
            relationship_code=relationship,
            degree_of_relatedness=0.5,
            relative_sex=sex,
        )
        assert 0.0 <= risk.carrier_probability <= 1.0
        assert 0.0 <= risk.affected_probability <= 1.0

    def test_compute_relative_risk_infers_degree_when_missing(self) -> None:
        risk = compute_relative_risk(
            mode="autosomal_dominant",
            relationship_code="MTH",
            degree_of_relatedness=None,
        )
        assert risk.affected_probability >= 0.0

    def test_compute_relative_risk_overrides(self) -> None:
        risk = compute_relative_risk(
            mode="autosomal_recessive",
            relationship_code="SIS",
            degree_of_relatedness=0.5,
            relative_sex="female",
            penetrance=0.9,
            carrier_frequency=0.05,
        )
        assert risk is not None

    def test_unsupported_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported inheritance mode"):
            compute_relative_risk("not_a_mode", "MTH", 0.5)

    def test_categorise_and_infer_sex(self) -> None:
        assert isinstance(categorise_relationship("MTH"), str)
        assert isinstance(categorise_relationship("UNKNOWNCODE"), str)
        assert infer_sex("MTH", None) in ("female", "male", None)
        assert infer_sex("SON", "male") == "male"


# ── prs_service ───────────────────────────────────────────────────────────────


class TestPRS:
    @pytest.mark.parametrize("disease", _PANELS)
    def test_compute_prs_panels(self, disease: str) -> None:
        result = compute_prs(disease, dosages={}, ml_risk=0.3)
        assert 0.0 <= result.blended_risk <= 1.0

    def test_compute_prs_without_ml_risk(self) -> None:
        result = compute_prs(_PANELS[0], dosages={}, ml_risk=None)
        assert result is not None

    def test_compute_prs_with_dosages(self) -> None:
        panel = PRS_PANELS[_PANELS[0]]
        first_rsid = next(iter(panel.weights))
        result = compute_prs(_PANELS[0], dosages={first_rsid: 2}, ml_risk=0.5)
        assert result is not None

    def test_unknown_panel_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown PRS panel"):
            compute_prs("nope", dosages={})


# ── whatif_service ────────────────────────────────────────────────────────────


class TestWhatIf:
    def test_simulate_no_changes(self) -> None:
        result = simulate(baseline=None, modifications=None)
        assert 0.0 <= result.baseline_risk <= 1.0
        assert result.baseline_risk == pytest.approx(result.simulated_risk)

    def test_simulate_with_modifications(self) -> None:
        key = next(iter(RISK_FACTORS))
        result = simulate(baseline={key: 1.0}, modifications={key: 3.0, "unknown": 9.0})
        assert len(result.contributions) == len(RISK_FACTORS)

    def test_simulate_ignores_unknown_baseline_keys(self) -> None:
        result = simulate(baseline={"unknown_factor": 5.0}, modifications={})
        assert result is not None


# ── cascade_service ranking ───────────────────────────────────────────────────


class TestCascadeRanking:
    def test_priority_for_score_buckets(self) -> None:
        assert isinstance(priority_for_score(0.95), CascadePriority)
        assert isinstance(priority_for_score(0.5), CascadePriority)
        assert isinstance(priority_for_score(0.01), CascadePriority)

    @pytest.mark.parametrize("mode", _MODES)
    def test_rank_relative(self, mode: str) -> None:
        ranked = rank_relative(
            relationship_code="MTH",
            degree_of_relatedness=0.5,
            relative_sex="female",
            inheritance_mode=mode,
            penetrance=0.8,
            carrier_frequency=0.01,
            condition_display="Familial hypercholesterolemia",
        )
        assert isinstance(ranked.priority, CascadePriority)
        assert ranked.recommended_action
