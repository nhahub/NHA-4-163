"""Unit tests for the cascade-screening ranking logic (Tier 5)."""

from __future__ import annotations

from libs.common.models.cascade import CascadePriority
from services.api.services.cascade_service import (
    priority_for_score,
    rank_relative,
)


class TestPriorityBanding:
    def test_high_medium_low_bands(self) -> None:
        assert priority_for_score(0.9) == CascadePriority.HIGH
        assert priority_for_score(0.25) == CascadePriority.MEDIUM
        assert priority_for_score(0.05) == CascadePriority.LOW


class TestRankRelative:
    def test_first_degree_dominant_is_high_priority(self) -> None:
        ranked = rank_relative(
            relationship_code="SON",
            degree_of_relatedness=0.5,
            relative_sex="male",
            inheritance_mode="autosomal_dominant",
            penetrance=1.0,
            carrier_frequency=0.001,
            condition_display="Lynch syndrome",
        )
        assert ranked.priority == CascadePriority.HIGH
        assert ranked.affected_probability == 0.5
        assert "Lynch syndrome" in ranked.recommended_action

    def test_distant_relative_lower_priority_than_close(self) -> None:
        close = rank_relative("SON", 0.5, "male", "autosomal_dominant", 0.8, 0.001, None)
        distant = rank_relative("COUSN", 0.125, "female", "autosomal_dominant", 0.8, 0.001, None)
        assert close.priority_score > distant.priority_score

    def test_score_weights_affected_over_carrier(self) -> None:
        # Recessive sibling: affected 0.25, carrier 0.5 → 0.7*0.25 + 0.3*0.5 = 0.325.
        ranked = rank_relative("SIB", 0.5, "female", "autosomal_recessive", 1.0, 0.02, None)
        assert ranked.priority_score == 0.325
        assert ranked.priority == CascadePriority.MEDIUM

    def test_unaffected_path_scores_zero(self) -> None:
        # Son of an affected male under X-linked recessive is not at risk.
        ranked = rank_relative("SON", 0.5, "male", "x_linked_recessive", 1.0, 0.001, None)
        assert ranked.priority_score == 0.0
        assert ranked.priority == CascadePriority.LOW
