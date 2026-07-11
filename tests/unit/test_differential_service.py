"""Unit tests for the knowledge-based differential-diagnosis service (Tier 4)."""

from __future__ import annotations

from services.api.services.differential_service import (
    infer_from_medications,
    infer_from_symptoms,
)


class TestSymptomInference:
    def test_returns_ranked_diseases(self) -> None:
        results = infer_from_symptoms(["R05.9"])  # cough
        assert results
        codes = [c for c, _, _ in results]
        assert "J06" in codes  # URI is the top association for cough

    def test_probabilities_sum_to_one(self) -> None:
        results = infer_from_symptoms(["R05.9", "R50.9"])
        total = sum(prob for _, _, prob in results)
        assert abs(total - 1.0) < 1e-6

    def test_diabetes_symptoms_rank_e11_first(self) -> None:
        results = infer_from_symptoms(["R63.1", "R35"])  # polydipsia + polyuria
        assert results[0][0] == "E11"

    def test_unknown_codes_return_empty(self) -> None:
        assert infer_from_symptoms(["ZZZ.9"]) == []

    def test_top_n_limits_results(self) -> None:
        results = infer_from_symptoms(["R05.9", "R07.9", "R51", "R10.9"], top_n=2)
        assert len(results) <= 2

    def test_category_fallback_matches_dotless(self) -> None:
        # "R05" (no decimal) should still resolve via the category fallback.
        assert infer_from_symptoms(["R05"]) == infer_from_symptoms(["R05.9"])


class TestMedicationInference:
    def test_metformin_implies_diabetes(self) -> None:
        results = infer_from_medications(["860975"])
        assert results[0][0] == "E11"
        assert results[0][2] == 1.0

    def test_names_are_populated(self) -> None:
        results = infer_from_medications(["197361"])  # Lisinopril
        names = [name for _, name, _ in results]
        assert "Essential hypertension" in names

    def test_unknown_medication_returns_empty(self) -> None:
        assert infer_from_medications(["000000"]) == []

    def test_multiple_meds_aggregate(self) -> None:
        results = infer_from_medications(["860975", "197361"])
        codes = [c for c, _, _ in results]
        assert "E11" in codes and "I10" in codes
