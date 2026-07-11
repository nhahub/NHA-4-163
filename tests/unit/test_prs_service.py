"""Unit tests for polygenic risk score integration (Tier 5)."""

from __future__ import annotations

import pytest

from services.api.services.prs_service import PRS_PANELS, compute_prs


class TestPanels:
    def test_panels_are_well_formed(self) -> None:
        for panel in PRS_PANELS.values():
            assert 0.0 < panel.baseline_prevalence < 1.0
            assert panel.sd > 0
            assert panel.weights


class TestComputePRS:
    def test_zero_dosage_is_baseline(self) -> None:
        result = compute_prs("type_2_diabetes", {}, ml_risk=None)
        assert result.raw_score == 0.0
        assert result.percentile == pytest.approx(50.0, abs=0.5)
        assert result.odds_ratio == pytest.approx(1.0, abs=0.01)
        # With no ML risk, blended == PRS absolute risk.
        assert result.blended_risk == result.prs_absolute_risk

    def test_risk_alleles_increase_percentile_and_or(self) -> None:
        low = compute_prs("type_2_diabetes", {})
        high = compute_prs(
            "type_2_diabetes",
            {"rs7903146": 2, "rs1801282": 2, "rs5219": 2},
        )
        assert high.percentile > low.percentile
        assert high.odds_ratio > 1.0
        assert high.prs_absolute_risk > low.prs_absolute_risk

    def test_snps_used_counts_nonzero_dosages(self) -> None:
        result = compute_prs("breast_cancer", {"rs2981582": 1, "rs3803662": 0})
        assert result.snps_used == 1
        assert result.snps_available == len(PRS_PANELS["breast_cancer"].weights)

    def test_protective_allele_lowers_risk(self) -> None:
        # APOE e2 (rs7412) has a negative beta → protective.
        protective = compute_prs("alzheimer_disease", {"rs7412": 2})
        assert protective.odds_ratio < 1.0
        assert protective.percentile < 50

    def test_blend_between_ml_and_prs(self) -> None:
        # High PRS + low ML → blended sits between the two.
        result = compute_prs(
            "type_2_diabetes",
            {"rs7903146": 2},
            ml_risk=0.20,
            prs_weight=0.5,
        )
        assert result.ml_risk == 0.20
        assert 0.0 < result.blended_risk < 1.0

    def test_prs_weight_zero_returns_ml_risk(self) -> None:
        result = compute_prs(
            "coronary_artery_disease", {"rs10757278": 2}, ml_risk=0.30, prs_weight=0.0
        )
        assert result.blended_risk == pytest.approx(0.30, abs=1e-3)

    def test_prs_weight_one_returns_prs_risk(self) -> None:
        result = compute_prs(
            "coronary_artery_disease", {"rs10757278": 2}, ml_risk=0.30, prs_weight=1.0
        )
        assert result.blended_risk == pytest.approx(result.prs_absolute_risk, abs=1e-3)

    def test_unknown_panel_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_prs("not_a_disease", {})
