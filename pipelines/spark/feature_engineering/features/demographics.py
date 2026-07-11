"""Patient demographic feature extraction.

Computes age-based and gender encoding features from the patient table.
All features are point-in-time safe via the ``as_of_date`` parameter so
training labels computed at different dates are not contaminated with
future demographic information.
"""

from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType


def build_demographics_features(patients_df: DataFrame, as_of_date: str) -> DataFrame:
    """Compute demographic features for all active (non-deleted) patients.

    Args:
        patients_df: Raw patient rows from PostgreSQL — must contain columns
            ``id``, ``date_of_birth`` (DATE or NULL), ``gender`` (str),
            ``deleted_at`` (TIMESTAMP or NULL).
        as_of_date: ISO-8601 date string (``YYYY-MM-DD``) used as the
            reference point for age computation.  Pass the pipeline run date
            to keep features point-in-time correct.

    Returns:
        DataFrame with one row per active patient and columns:
        ``patient_id``, ``age_years`` (INT, NULL when DOB unknown),
        ``age_group`` (VARCHAR), ``gender_male``, ``gender_female``,
        ``gender_other_unknown`` (all INT 0/1 flags).
    """
    ref_date = F.to_date(F.lit(as_of_date))

    age_expr = F.when(
        F.col("date_of_birth").isNotNull(),
        F.floor(F.datediff(ref_date, F.col("date_of_birth")) / 365.25).cast(IntegerType()),
    ).otherwise(F.lit(None).cast(IntegerType()))

    age_group_expr = (
        F.when(F.col("age_years").isNull(), "unknown")
        .when(F.col("age_years") < 18, "pediatric")
        .when(F.col("age_years") < 35, "young_adult")
        .when(F.col("age_years") < 50, "middle_age")
        .when(F.col("age_years") < 65, "older_adult")
        .otherwise("elderly")
    )

    return (
        patients_df.filter(F.col("deleted_at").isNull())
        .select("id", "date_of_birth", "gender")
        .withColumn("age_years", age_expr)
        .withColumn("age_group", age_group_expr)
        .withColumn("gender_male", (F.col("gender") == "male").cast(IntegerType()))
        .withColumn("gender_female", (F.col("gender") == "female").cast(IntegerType()))
        .withColumn(
            "gender_other_unknown",
            F.col("gender").isin("other", "unknown").cast(IntegerType()),
        )
        .select(
            F.col("id").alias("patient_id"),
            "age_years",
            "age_group",
            "gender_male",
            "gender_female",
            "gender_other_unknown",
        )
    )
