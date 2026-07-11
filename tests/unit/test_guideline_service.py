"""Unit tests for guideline-based screening recommendations (Tier 6)."""

from __future__ import annotations

from services.api.services.guideline_service import (
    PatientContext,
    catalogue,
    recommend,
)


def _ctx(**kw) -> PatientContext:
    base = {
        "age": None,
        "sex": None,
        "risk_score": 0.0,
        "condition_codes": frozenset(),
        "has_hereditary_condition": False,
        "affected_first_degree_relatives": 0,
    }
    base.update(kw)
    return PatientContext(**base)


class TestRecommend:
    def test_breast_cancer_history_triggers_brca(self) -> None:
        recs = recommend(_ctx(sex="female", risk_score=0.6, condition_codes=frozenset({"C50"})))
        ids = {r.guideline_id for r in recs}
        assert "nccn-hboc-testing" in ids

    def test_colorectal_history_triggers_lynch(self) -> None:
        recs = recommend(
            _ctx(
                risk_score=0.55,
                condition_codes=frozenset({"C18"}),
                affected_first_degree_relatives=1,
            )
        )
        ids = {r.guideline_id for r in recs}
        assert "nccn-lynch-testing" in ids

    def test_average_risk_crc_screening_by_age(self) -> None:
        recs = recommend(_ctx(age=50, sex="male"))
        ids = {r.guideline_id for r in recs}
        assert "uspstf-crc-45" in ids

    def test_no_crc_screening_when_already_diagnosed(self) -> None:
        recs = recommend(_ctx(age=50, condition_codes=frozenset({"C18"})))
        ids = {r.guideline_id for r in recs}
        assert "uspstf-crc-45" not in ids

    def test_mammography_recommended_for_women_40_plus(self) -> None:
        recs = recommend(_ctx(age=45, sex="female"))
        ids = {r.guideline_id for r in recs}
        assert "uspstf-mammo-40" in ids

    def test_generic_referral_when_high_risk_no_match(self) -> None:
        recs = recommend(_ctx(age=25, sex="male", risk_score=0.9))
        ids = {r.guideline_id for r in recs}
        assert "genetics-referral" in ids

    def test_low_risk_young_patient_gets_nothing_urgent(self) -> None:
        recs = recommend(_ctx(age=25, sex="male", risk_score=0.1))
        assert "genetics-referral" not in {r.guideline_id for r in recs}

    def test_results_sorted_by_urgency(self) -> None:
        recs = recommend(
            _ctx(
                age=50,
                sex="female",
                risk_score=0.6,
                condition_codes=frozenset({"C50"}),
            )
        )
        rank = {"urgent": 0, "soon": 1, "routine": 2}
        urgencies = [rank[r.urgency] for r in recs]
        assert urgencies == sorted(urgencies)


class TestCatalogue:
    def test_catalogue_is_non_empty_and_unique(self) -> None:
        cat = catalogue()
        ids = [r.guideline_id for r in cat]
        assert ids
        assert len(ids) == len(set(ids))
