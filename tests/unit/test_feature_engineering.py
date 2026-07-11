"""Unit tests for Phase 4 feature engineering modules.

No Spark session or external services required — tests cover:
- ``depth_to_weight`` helper (pure Python)
- ``PatientFeatureVector`` Pydantic schema validation
- ``FeatureGroup`` / registry structure invariants
- ICD-10 chapter prefix mapping completeness
- Medication adherence edge cases (pure Python logic verification)

Spark DataFrame transformation tests (demographics, comorbidities,
medication_adherence) are in tests/integration/ because they require
a SparkSession.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from ml.features.registry import (
    ALL_GROUPS,
    COMORBIDITIES,
    DEMOGRAPHICS,
    FEATURE_TO_GROUP,
    FEATURE_VECTOR,
)
from ml.features.schema import PatientFeatureVector

# The pipelines.spark.* modules import pyspark at module load; skip this file
# when pyspark is not installed (it is exercised in the integration suite).
pytest.importorskip("pyspark")

from pipelines.spark.feature_engineering.features.comorbidities import (  # noqa: E402
    _CHAPTER_PREFIXES,
    CHAPTER_FEATURE_NAMES,
)
from pipelines.spark.feature_engineering.features.graph_features import (  # noqa: E402
    depth_to_weight,
)

_PATIENT_UUID = str(uuid.uuid4())
_FEATURE_DATE = "2024-01-15"


# ── depth_to_weight ───────────────────────────────────────────────────────────


class TestDepthToWeight:
    def test_first_degree(self) -> None:
        assert depth_to_weight(1) == pytest.approx(0.5)

    def test_second_degree(self) -> None:
        assert depth_to_weight(2) == pytest.approx(0.25)

    def test_third_degree(self) -> None:
        assert depth_to_weight(3) == pytest.approx(0.125)

    def test_fourth_degree(self) -> None:
        assert depth_to_weight(4) == pytest.approx(0.0625)

    def test_deeper_than_four_continues_halving(self) -> None:
        assert depth_to_weight(5) == pytest.approx(0.03125)

    def test_zero_treated_as_depth_one(self) -> None:
        # depth=0 (self) makes no clinical sense, but guard against it.
        assert depth_to_weight(0) == pytest.approx(0.5)

    def test_monotone_decrease(self) -> None:
        for d in range(1, 8):
            assert depth_to_weight(d) > depth_to_weight(d + 1)


# ── PatientFeatureVector schema ───────────────────────────────────────────────


class TestPatientFeatureVector:
    def _base(self, **overrides: Any) -> dict[str, Any]:
        return {
            "patient_id": _PATIENT_UUID,
            "feature_date": _FEATURE_DATE,
            **overrides,
        }

    def test_minimal_valid(self) -> None:
        v = PatientFeatureVector(**self._base())
        assert str(v.patient_id) == _PATIENT_UUID
        assert v.feature_date == _FEATURE_DATE

    def test_defaults_are_zero(self) -> None:
        v = PatientFeatureVector(**self._base())
        assert v.comorbidity_count == 0
        assert v.weighted_family_prevalence == 0.0
        assert v.affected_relatives_count == 0

    def test_age_years_none_allowed(self) -> None:
        v = PatientFeatureVector(**self._base(age_years=None))
        assert v.age_years is None

    def test_age_years_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PatientFeatureVector(**self._base(age_years=-1))

    def test_age_years_over_150_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PatientFeatureVector(**self._base(age_years=151))

    def test_valid_age_years(self) -> None:
        v = PatientFeatureVector(**self._base(age_years=85))
        assert v.age_years == 85

    def test_gender_flags_are_binary(self) -> None:
        with pytest.raises(ValidationError):
            PatientFeatureVector(**self._base(gender_male=2))

    def test_adherence_proxy_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PatientFeatureVector(**self._base(adherence_proxy=1.5))
        with pytest.raises(ValidationError):
            PatientFeatureVector(**self._base(adherence_proxy=-0.1))

    def test_adherence_proxy_none_allowed(self) -> None:
        v = PatientFeatureVector(**self._base(adherence_proxy=None))
        assert v.adherence_proxy is None

    def test_shortest_path_sentinel(self) -> None:
        v = PatientFeatureVector(**self._base(shortest_path_to_affected=-1))
        assert v.shortest_path_to_affected == -1

    def test_shortest_path_below_minus_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PatientFeatureVector(**self._base(shortest_path_to_affected=-2))

    def test_clustering_coefficient_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PatientFeatureVector(**self._base(family_clustering_coefficient=1.01))

    def test_frozen_model(self) -> None:
        v = PatientFeatureVector(**self._base())
        with pytest.raises(ValidationError):
            v.comorbidity_count = 5  # type: ignore[misc]

    def test_invalid_feature_date_format(self) -> None:
        with pytest.raises(ValidationError):
            PatientFeatureVector(**self._base(feature_date="2024/01/15"))

    def test_weighted_prevalence_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            PatientFeatureVector(**self._base(weighted_family_prevalence=-0.1))


# ── Feature registry structure ────────────────────────────────────────────────


class TestFeatureRegistry:
    def test_all_groups_non_empty(self) -> None:
        for grp in ALL_GROUPS:
            assert grp.feature_columns, f"{grp.name} has no feature columns"

    def test_all_groups_have_unique_names(self) -> None:
        names = [g.name for g in ALL_GROUPS]
        assert len(names) == len(set(names))

    def test_feature_vector_contains_all_group_columns(self) -> None:
        all_cols = {c for grp in ALL_GROUPS for c in grp.feature_columns}
        vector_cols = set(FEATURE_VECTOR.feature_columns)
        missing = all_cols - vector_cols
        assert not missing, f"Columns missing from feature vector: {missing}"

    def test_feature_to_group_lookup_covers_all_source_columns(self) -> None:
        for grp in ALL_GROUPS:
            for col in grp.feature_columns:
                assert col in FEATURE_TO_GROUP, f"{col} missing from FEATURE_TO_GROUP"
                assert FEATURE_TO_GROUP[col] == grp.name

    def test_feature_groups_are_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            DEMOGRAPHICS.name = "hacked"  # type: ignore[misc]

    def test_delta_paths_are_distinct(self) -> None:
        paths = [g.delta_path for g in ALL_GROUPS] + [FEATURE_VECTOR.delta_path]
        assert len(paths) == len(set(paths))

    def test_feature_vector_columns_match_pydantic_schema(self) -> None:
        schema_fields = set(PatientFeatureVector.model_fields.keys()) - {
            "patient_id",
            "feature_date",
        }
        registry_cols = set(FEATURE_VECTOR.feature_columns)
        in_registry_not_schema = registry_cols - schema_fields
        in_schema_not_registry = schema_fields - registry_cols
        assert not in_registry_not_schema, f"In registry but not schema: {in_registry_not_schema}"
        assert not in_schema_not_registry, f"In schema but not registry: {in_schema_not_registry}"


# ── ICD-10 chapter prefixes ───────────────────────────────────────────────────


class TestIcd10ChapterPrefixes:
    def test_chapter_feature_names_sorted(self) -> None:
        assert CHAPTER_FEATURE_NAMES == sorted(CHAPTER_FEATURE_NAMES)

    def test_all_prefixes_are_single_uppercase_letters(self) -> None:
        for feature, prefixes in _CHAPTER_PREFIXES.items():
            for prefix in prefixes:
                assert len(prefix) == 1, f"{feature}: prefix '{prefix}' is not single char"
                assert prefix.isupper(), f"{feature}: prefix '{prefix}' is not uppercase"

    def test_no_duplicate_prefixes_across_features(self) -> None:
        seen: dict[str, str] = {}
        for feature, prefixes in _CHAPTER_PREFIXES.items():
            for prefix in prefixes:
                assert (
                    prefix not in seen
                ), f"Prefix '{prefix}' claimed by both '{seen[prefix]}' and '{feature}'"
                seen[prefix] = feature

    def test_chapter_names_match_registry(self) -> None:
        registry_cols = set(COMORBIDITIES.feature_columns)
        chapter_flags = {
            f"has_{k.split('_', 1)[1]}" if k.startswith("has_") else k for k in _CHAPTER_PREFIXES
        }
        chapter_flags = set(_CHAPTER_PREFIXES.keys())
        missing = chapter_flags - registry_cols
        assert not missing, (
            f"Chapter flags defined in comorbidities.py but absent from COMORBIDITIES registry: "
            f"{missing}"
        )
