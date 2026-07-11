"""Unit tests for patient identity resolution.

No Spark session required for the similarity functions — those are tested as
pure Python.  Spark integration tests live in tests/integration/.
"""

from __future__ import annotations

import pytest

from pipelines.spark.streaming.transforms.identity_resolution import (
    MATCH_THRESHOLD,
    PatientRecord,
    _build_block_key,
    _jaro,
    _jaro_winkler,
    _score_pair,
)


class TestJaro:
    def test_identical_strings(self) -> None:
        assert _jaro("smith", "smith") == 1.0

    def test_empty_strings(self) -> None:
        assert _jaro("", "") == 0.0
        assert _jaro("", "abc") == 0.0
        assert _jaro("abc", "") == 0.0

    def test_completely_different(self) -> None:
        assert _jaro("abc", "xyz") < 0.4

    def test_known_value(self) -> None:
        # MARTHA / MARHTA — classic Jaro example ≈ 0.944
        score = _jaro("MARTHA", "MARHTA")
        assert 0.93 < score < 0.96

    def test_symmetric(self) -> None:
        assert abs(_jaro("jones", "johnson") - _jaro("johnson", "jones")) < 1e-9


class TestJaroWinkler:
    def test_identical(self) -> None:
        assert _jaro_winkler("john", "john") == 1.0

    def test_prefix_boost_applied(self) -> None:
        # Jaro-Winkler >= Jaro when strings share prefix
        jaro = _jaro("JOHN", "JOHNY")
        jw = _jaro_winkler("JOHN", "JOHNY")
        assert jw >= jaro

    def test_no_prefix_no_boost(self) -> None:
        # Completely different strings — no boost
        jaro = _jaro("abc", "xyz")
        jw = _jaro_winkler("abc", "xyz")
        assert abs(jaro - jw) < 0.05

    def test_case_insensitive(self) -> None:
        upper = _jaro_winkler("SMITH", "smith")
        assert upper == pytest.approx(1.0)


class TestBlockKey:
    def test_with_dob_and_postal(self) -> None:
        rec = PatientRecord("id1", "Smith", "John", "1985-06-15", "male", "90210")
        key = _build_block_key(rec)
        assert key == "1985|902"

    def test_with_dob_no_postal(self) -> None:
        rec = PatientRecord("id1", "Smith", "John", "1985-06-15", "male", None)
        key = _build_block_key(rec)
        assert key == "1985|"

    def test_no_dob_returns_none(self) -> None:
        rec = PatientRecord("id1", "Smith", "John", None, "male", "90210")
        key = _build_block_key(rec)
        assert key is None


class TestScorePair:
    def _make(self, **kwargs: object) -> PatientRecord:
        defaults = {
            "patient_id": "pid1",
            "family_name": "Smith",
            "given_name": "John",
            "date_of_birth": "1985-06-15",
            "gender": "male",
            "postal_code": "90210",
        }
        defaults.update(kwargs)
        return PatientRecord(**defaults)  # type: ignore[arg-type]

    def test_identical_records_score_one(self) -> None:
        a = self._make(patient_id="a")
        b = self._make(patient_id="b")
        assert _score_pair(a, b) == pytest.approx(1.0, abs=0.01)

    def test_completely_different_score_low(self) -> None:
        a = self._make(
            family_name="Smith", given_name="John", date_of_birth="1985-01-01", gender="male"
        )
        b = self._make(
            patient_id="b",
            family_name="Xyz",
            given_name="Abc",
            date_of_birth="1960-12-31",
            gender="female",
        )
        assert _score_pair(a, b) < MATCH_THRESHOLD

    def test_typo_in_family_name_still_above_threshold(self) -> None:
        a = self._make(patient_id="a")
        b = self._make(patient_id="b", family_name="Smyth")  # one char off
        score = _score_pair(a, b)
        assert score >= MATCH_THRESHOLD

    def test_different_gender_reduces_score(self) -> None:
        a = self._make(patient_id="a")
        b = self._make(patient_id="b", gender="female")
        score = _score_pair(a, b)
        assert score < _score_pair(a, self._make(patient_id="b"))

    def test_none_names_handled(self) -> None:
        a = PatientRecord("a", None, None, "1985-06-15", "male", None)
        b = PatientRecord("b", None, None, "1985-06-15", "male", None)
        score = _score_pair(a, b)
        # Both have matching DOB + gender but 0 name similarity
        assert 0.0 <= score <= 1.0
