"""Unit tests for the What-If risk simulator (Tier 6)."""

from __future__ import annotations

from services.api.services.whatif_service import RISK_FACTORS, simulate


class TestFactors:
    def test_factor_catalogue_is_well_formed(self) -> None:
        for f in RISK_FACTORS.values():
            assert f.kind in ("count", "rate", "binary")
            assert f.unit


class TestSimulate:
    def test_no_modifications_leaves_risk_unchanged(self) -> None:
        result = simulate({"age_years": 50}, {})
        assert result.baseline_risk == result.simulated_risk
        assert result.risk_delta == 0.0

    def test_defaults_used_when_baseline_missing(self) -> None:
        result = simulate(None, {})
        # All defaults → a stable, low-moderate baseline in (0, 1).
        assert 0.0 < result.baseline_risk < 1.0
        assert result.baseline_factors["age_years"] == RISK_FACTORS["age_years"].default

    def test_adding_hereditary_condition_increases_risk(self) -> None:
        result = simulate({}, {"hereditary_condition_count": 2})
        assert result.simulated_risk > result.baseline_risk
        assert result.risk_delta > 0

    def test_affected_relative_toggle_raises_risk(self) -> None:
        result = simulate(
            {"affected_first_degree_relatives": 0},
            {"affected_first_degree_relatives": 1},
        )
        assert result.risk_delta > 0

    def test_contributions_ranked_by_delta(self) -> None:
        result = simulate({}, {"hereditary_condition_count": 3})
        top = result.contributions[0]
        assert top.key == "hereditary_condition_count"
        assert top.delta_from_baseline > 0
        # Contributions are sorted by absolute delta, descending.
        deltas = [abs(c.delta_from_baseline) for c in result.contributions]
        assert deltas == sorted(deltas, reverse=True)

    def test_rate_factor_is_clamped(self) -> None:
        result = simulate({}, {"family_risk_prevalence": 5.0})
        assert result.simulated_factors["family_risk_prevalence"] == 1.0

    def test_binary_factor_snaps_to_zero_or_one(self) -> None:
        result = simulate({}, {"carrier_variant_present": 0.9})
        assert result.simulated_factors["carrier_variant_present"] == 1.0

    def test_unknown_keys_ignored(self) -> None:
        result = simulate({"not_a_factor": 99}, {"also_bogus": 3})
        assert "not_a_factor" not in result.baseline_factors
        assert result.risk_delta == 0.0

    def test_negative_counts_clamped_to_zero(self) -> None:
        result = simulate({}, {"comorbidity_count": -5})
        assert result.simulated_factors["comorbidity_count"] == 0.0

    def test_interpretation_mentions_direction(self) -> None:
        up = simulate({}, {"hereditary_condition_count": 2})
        assert "increases" in up.interpretation
