"""Medication adherence feature extraction.

Derives per-patient features from the ``medication_requests`` table:
active count, completed count, stopped count, distinct medication count,
and a crude proxy adherence ratio.

Note on adherence_proxy
-----------------------
True medication adherence (MPR / PDC) requires pharmacy dispensing records
with fill dates, which are Phase 6 scope.  The proxy here uses FHIR
MedicationRequest status transitions:

    adherence_proxy = completed / (completed + stopped)

This captures whether prescriptions were finished vs. discontinued but
understates adherence for patients who are still actively taking meds.
Treat as an ordinal indicator, not a clinical measurement.
"""

from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, IntegerType

_STOPPED_STATUSES = ("stopped", "cancelled", "entered-in-error", "on-hold")


def build_medication_features(medication_requests_df: DataFrame) -> DataFrame:
    """Compute medication-based features per patient.

    Args:
        medication_requests_df: Raw rows from the PostgreSQL
            ``medication_requests`` table — must contain ``patient_id``
            (str), ``status`` (str), ``medication_code`` (str).

    Returns:
        DataFrame keyed by ``patient_id`` with columns:
        ``active_medication_count``, ``completed_medication_count``,
        ``stopped_medication_count``, ``distinct_medication_count``,
        ``adherence_proxy`` (DOUBLE, NULL when no completed/stopped meds).
    """
    agg_df = medication_requests_df.groupBy("patient_id").agg(
        F.sum(F.when(F.col("status") == "active", 1).otherwise(0))
        .cast(IntegerType())
        .alias("active_medication_count"),
        F.sum(F.when(F.col("status") == "completed", 1).otherwise(0))
        .cast(IntegerType())
        .alias("completed_medication_count"),
        F.sum(F.when(F.col("status").isin(*_STOPPED_STATUSES), 1).otherwise(0))
        .cast(IntegerType())
        .alias("stopped_medication_count"),
        F.countDistinct("medication_code").cast(IntegerType()).alias("distinct_medication_count"),
    )

    denominator = F.col("completed_medication_count") + F.col("stopped_medication_count")

    return agg_df.withColumn(
        "adherence_proxy",
        F.when(
            denominator > 0,
            (F.col("completed_medication_count") / denominator).cast(DoubleType()),
        ).otherwise(F.lit(None).cast(DoubleType())),
    )
