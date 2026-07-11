"""Unit tests for model monitoring & fairness metrics (Tier 6)."""

from __future__ import annotations

import pytest

from services.api.services.monitoring_service import (
    calibration_report,
    fairness_report,
    population_stability_index,
)


class TestDrift:
    def test_identical_distributions_have_zero_psi(self) -> None:
        sample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = population_stability_index(sample, list(sample))
        assert result.psi == pytest.approx(0.0, abs=1e-6)
        assert result.verdict == "stable"

    def test_shifted_distribution_flags_significant(self) -> None:
        reference = [0.05] * 100
        current = [0.95] * 100
        result = population_stability_index(reference, current)
        assert result.psi > 0.25
        assert result.verdict == "significant_shift"

    def test_empty_inputs_are_stable(self) -> None:
        result = population_stability_index([], [])
        assert result.psi == 0.0
        assert result.verdict == "stable"

    def test_bins_sum_to_full_range(self) -> None:
        result = population_stability_index([0.1, 0.9], [0.2, 0.8], bins=5)
        assert len(result.bins) == 5
        assert result.bins[0].lower == 0.0
        assert result.bins[-1].upper == 1.0


class TestFairness:
    def test_equal_groups_pass_four_fifths(self) -> None:
        result = fairness_report("sex", {"male": [0.4, 0.5, 0.6], "female": [0.4, 0.5, 0.6]})
        assert result.disparate_impact_ratio == pytest.approx(1.0)
        assert result.passes_four_fifths
        assert result.statistical_parity_difference == pytest.approx(0.0)

    def test_disparate_groups_fail(self) -> None:
        result = fairness_report("sex", {"male": [0.9, 0.9, 0.9], "female": [0.2, 0.2, 0.2]})
        assert result.disparate_impact_ratio < 0.80
        assert not result.passes_four_fifths
        assert "disparity" in result.interpretation.lower()

    def test_small_groups_excluded(self) -> None:
        result = fairness_report(
            "race",
            {"a": [0.5, 0.5, 0.5], "b": [0.9]},
            min_group_size=2,
        )
        groups = {g.group for g in result.groups}
        assert "b" not in groups
        assert "a" in groups

    def test_single_group_cannot_assess(self) -> None:
        result = fairness_report("sex", {"male": [0.5, 0.6]})
        assert result.disparate_impact_ratio == 1.0
        assert "Not enough" in result.interpretation

    def test_high_risk_rate_computed(self) -> None:
        result = fairness_report("sex", {"male": [0.2, 0.8]}, high_risk_threshold=0.5)
        stat = next(g for g in result.groups if g.group == "male")
        assert stat.high_risk_rate == pytest.approx(0.5)


class TestCalibration:
    def test_perfect_calibration_low_brier(self) -> None:
        predicted = [0.0, 0.0, 1.0, 1.0]
        observed = [0, 0, 1, 1]
        result = calibration_report(predicted, observed)
        assert result.brier_score == pytest.approx(0.0)
        assert result.sample_size == 4

    def test_worst_calibration_high_brier(self) -> None:
        result = calibration_report([1.0, 0.0], [0, 1])
        assert result.brier_score == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            calibration_report([0.5], [0, 1])

    def test_empty_inputs(self) -> None:
        result = calibration_report([], [])
        assert result.brier_score == 0.0
        assert result.bins == []
