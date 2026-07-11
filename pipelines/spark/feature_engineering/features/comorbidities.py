"""Comorbidity feature extraction from the PostgreSQL conditions table.

Maps active ICD-10 codes to ICD-10-chapter-level binary flags, counts
total active conditions, and counts hereditary conditions separately.
Only active/relapsed/recurring conditions are counted — resolved and
inactive conditions are excluded to reflect *current* burden.
"""

from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType

# Map feature flag name → tuple of ICD-10 first-character prefixes.
# Chapter boundaries follow ICD-10 CM 2024 tabular index.
_CHAPTER_PREFIXES: dict[str, tuple[str, ...]] = {
    "has_infectious": ("A", "B"),
    "has_oncological": ("C",),
    "has_haematological": ("D",),
    "has_metabolic": ("E",),
    "has_mental_health": ("F",),
    "has_neurological": ("G",),
    "has_cardiovascular": ("I",),
    "has_respiratory": ("J",),
    "has_digestive": ("K",),
    "has_musculoskeletal": ("M",),
    "has_genitourinary": ("N",),
}

# Ordered list of chapter feature column names (stable for ML pipelines).
CHAPTER_FEATURE_NAMES: list[str] = sorted(_CHAPTER_PREFIXES.keys())

_ACTIVE_STATUSES = ("active", "recurrence", "relapse")


def build_comorbidity_features(conditions_df: DataFrame) -> DataFrame:
    """Aggregate condition records into per-patient comorbidity features.

    Args:
        conditions_df: Raw rows from the PostgreSQL ``conditions`` table —
            must contain ``patient_id`` (str), ``clinical_status`` (str),
            ``icd10_code`` (str), ``is_hereditary`` (bool/int).

    Returns:
        DataFrame keyed by ``patient_id`` with columns:
        ``comorbidity_count``, ``hereditary_condition_count``,
        and one INT flag per ICD-10 chapter (e.g. ``has_cardiovascular``).
        Patients with no active conditions receive count=0 and all flags=0.
    """
    active = conditions_df.filter(F.col("clinical_status").isin(*_ACTIVE_STATUSES))

    # Aggregate counts
    counts_df = active.groupBy("patient_id").agg(
        F.count("*").cast(IntegerType()).alias("comorbidity_count"),
        F.sum(F.col("is_hereditary").cast(IntegerType()))
        .cast(IntegerType())
        .alias("hereditary_condition_count"),
    )

    # Build chapter-level flags: 1 if the patient has ≥1 active condition
    # in that chapter, 0 otherwise.
    first_char = F.col("icd10_code").substr(1, 1)
    chapter_aggs = [
        F.max(F.when(first_char.isin(*prefixes), 1).otherwise(0))
        .cast(IntegerType())
        .alias(feature_name)
        for feature_name, prefixes in _CHAPTER_PREFIXES.items()
    ]
    chapters_df = active.groupBy("patient_id").agg(*chapter_aggs)

    return counts_df.join(chapters_df, on="patient_id", how="full").fillna(0)
