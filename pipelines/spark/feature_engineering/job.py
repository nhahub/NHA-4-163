"""Healthcare Feature Engineering Batch Job.

Reads clinical data from PostgreSQL and graph features from Neo4j,
computes per-patient feature vectors, and writes partitioned Delta
tables to MinIO for downstream ML training and serving.

Architecture
------------
1. **Postgres reads** — JDBC (PostgreSQL driver included in SPARK_PACKAGES).
   Each table is read in a single parallel scan; no pagination needed for
   dev-scale data.  Production would use partitionColumn for parallelism.

2. **Graph features** — extracted via bulk Cypher queries (Python neo4j
   driver on the Spark driver node, not inside executors).  Results are
   collected to the driver, then broadcast as a Spark DataFrame.  This
   avoids opening N connections from executors to Neo4j and is correct
   for graphs that fit in driver memory (< 10 M patients × 8 cols ≈ 800 MB).

3. **Delta writes** — one Delta table per feature group + one joined
   table.  Each run MERGEs on (patient_id, feature_date) so re-runs are
   idempotent.  The joined table is created last, after all groups succeed.

Run locally (from project root):
    spark-submit \\
      --master spark://localhost:7077 \\
      --packages "${SPARK_PACKAGES}" \\
      --py-files libs.zip \\
      pipelines/spark/feature_engineering/job.py \\
      --as-of-date 2024-01-01

Environment variables (from .env / Docker Compose):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_DELTA
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from typing import Any

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from libs.common.config import get_settings  # noqa: E402
from libs.common.logging import configure_logging  # noqa: E402
from ml.features.registry import (  # noqa: E402
    ALL_GROUPS,
    COMORBIDITIES,
    DEMOGRAPHICS,
    FEATURE_VECTOR,
    GRAPH,
    MEDICATIONS,
)
from pipelines.spark.feature_engineering.features.comorbidities import (  # noqa: E402
    build_comorbidity_features,
)
from pipelines.spark.feature_engineering.features.demographics import (  # noqa: E402
    build_demographics_features,
)
from pipelines.spark.feature_engineering.features.graph_features import (  # noqa: E402
    GraphFeatureRow,
    extract_all_graph_features,
    run_gds_write_projection,
)
from pipelines.spark.feature_engineering.features.medication_adherence import (  # noqa: E402
    build_medication_features,
)

configure_logging(service_name="feature-engineering")
log = logging.getLogger(__name__)

# ── Spark schema for graph feature rows (driver-collected → DataFrame) ────────
_GRAPH_SCHEMA = StructType(
    [
        StructField("patient_id", StringType(), False),
        StructField("affected_relatives_count", IntegerType(), True),
        StructField("weighted_family_prevalence", DoubleType(), True),
        StructField("first_degree_affected_count", IntegerType(), True),
        StructField("second_degree_affected_count", IntegerType(), True),
        StructField("shortest_path_to_affected", IntegerType(), True),
        StructField("family_size", IntegerType(), True),
        StructField("family_clustering_coefficient", DoubleType(), True),
    ]
)


# ── Session factory ───────────────────────────────────────────────────────────


def _build_spark_session(minio_endpoint: str, access_key: str, secret_key: str) -> SparkSession:
    """Create a Delta-enabled SparkSession configured for MinIO S3A."""
    builder = (
        SparkSession.builder.appName("healthcare-feature-engineering")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.hadoop.fs.s3a.endpoint", minio_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        # Delta Lake write performance
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
    )
    # Add the JVM jars pip-installed PySpark does not bundle:
    #   - PostgreSQL JDBC driver (source-table reads)
    #   - hadoop-aws + AWS SDK bundle (S3A writes to MinIO)
    # Resolved from Maven at session start alongside the Delta packages.
    # hadoop-aws must match PySpark's bundled Hadoop version (3.3.4 for 3.5.x).
    extra_packages = [
        "org.postgresql:postgresql:42.7.3",
        "org.apache.hadoop:hadoop-aws:3.3.4",
    ]
    return configure_spark_with_delta_pip(builder, extra_packages=extra_packages).getOrCreate()


# ── Postgres JDBC helpers ─────────────────────────────────────────────────────


def _jdbc_read(
    spark: SparkSession,
    jdbc_url: str,
    query: str,
    user: str,
    password: str,
) -> DataFrame:
    """Read a PostgreSQL query result into a Spark DataFrame via JDBC."""
    return spark.read.jdbc(
        url=jdbc_url,
        table=f"({query}) AS t",
        properties={
            "user": user,
            "password": password,
            "driver": "org.postgresql.Driver",
        },
    )


# ── Delta MERGE helper ────────────────────────────────────────────────────────


def _delta_merge(
    spark: SparkSession,
    df: DataFrame,
    table_path: str,
    merge_keys: list[str],
) -> None:
    """MERGE df into a Delta table, upserting on merge_keys.

    Creates the table on first run (append mode).  Subsequent runs
    perform MERGE so the job is idempotent when replayed.

    Args:
        spark: Active SparkSession.
        df: DataFrame to merge (must include partition column ``feature_date``).
        table_path: Full S3A path to the Delta table.
        merge_keys: Column names used for the MERGE condition.
    """
    if not DeltaTable.isDeltaTable(spark, table_path):
        (df.write.format("delta").partitionBy("feature_date").mode("overwrite").save(table_path))
        log.info("Created new Delta table at %s", table_path)
        return

    delta_tbl = DeltaTable.forPath(spark, table_path)
    merge_condition = " AND ".join(f"target.{k} = source.{k}" for k in merge_keys)
    update_set: dict[str, Any] = {c: F.col(f"source.{c}") for c in df.columns}

    (
        delta_tbl.alias("target")
        .merge(df.alias("source"), merge_condition)
        .whenMatchedUpdate(set=update_set)
        .whenNotMatchedInsertAll()
        .execute()
    )
    log.info("Merged %d rows into %s", df.count(), table_path)


# ── Graph features → Spark DataFrame ─────────────────────────────────────────


def _graph_features_df(
    spark: SparkSession,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> DataFrame:
    """Extract Neo4j graph features and return as a Spark DataFrame."""
    log.info("Running GDS projection…")
    run_gds_write_projection(neo4j_uri, neo4j_user, neo4j_password)

    log.info("Extracting bulk graph features from Neo4j…")
    rows: list[GraphFeatureRow] = extract_all_graph_features(neo4j_uri, neo4j_user, neo4j_password)
    log.info("Extracted graph features for %d patients", len(rows))
    return spark.createDataFrame(rows, schema=_GRAPH_SCHEMA)  # type: ignore[arg-type]


# ── Main ──────────────────────────────────────────────────────────────────────


def main(as_of_date: str, delta_base: str) -> None:
    """Run the feature engineering pipeline for a given reference date.

    Args:
        as_of_date: ISO-8601 date string (``YYYY-MM-DD``) — all features
            are computed as-of this date for point-in-time correctness.
        delta_base: S3A base path (e.g. ``s3a://healthcare-delta``).
    """
    settings = get_settings()
    pg = settings.postgres
    n4j = settings.neo4j
    minio = settings.minio

    spark = _build_spark_session(
        minio_endpoint=str(minio.endpoint),
        access_key=minio.access_key,
        secret_key=minio.secret_key.get_secret_value(),
    )
    spark.sparkContext.setLogLevel("WARN")

    jdbc_url = f"jdbc:postgresql://{pg.host}:{pg.port}/{pg.db}"
    jdbc_props = {"user": pg.user, "password": pg.password.get_secret_value()}

    log.info("Feature engineering run — as_of_date=%s", as_of_date)

    # ── Step 1: Read source tables ────────────────────────────────────────────
    # Table names are singular (see libs/common/models); the condition ICD-10
    # value is stored in ``code`` (with ``code_system``), aliased to the
    # ``icd10_code`` column name the feature builders expect.
    patients_df = _jdbc_read(
        spark,
        jdbc_url,
        "SELECT id, date_of_birth, gender, deleted_at FROM patient",
        **jdbc_props,
    )
    conditions_df = _jdbc_read(
        spark,
        jdbc_url,
        "SELECT patient_id, clinical_status, code AS icd10_code, is_hereditary FROM condition",
        **jdbc_props,
    )
    meds_df = _jdbc_read(
        spark,
        jdbc_url,
        "SELECT patient_id, status, medication_code FROM medication_request",
        **jdbc_props,
    )

    # ── Step 2: Build feature DataFrames ──────────────────────────────────────
    demog_df = build_demographics_features(patients_df, as_of_date).withColumn(
        "feature_date", F.lit(as_of_date)
    )
    comor_df = build_comorbidity_features(conditions_df).withColumn(
        "feature_date", F.lit(as_of_date)
    )
    meds_feat_df = build_medication_features(meds_df).withColumn("feature_date", F.lit(as_of_date))
    graph_df = _graph_features_df(
        spark,
        neo4j_uri=n4j.uri,
        neo4j_user=n4j.user,
        neo4j_password=n4j.password.get_secret_value(),
    ).withColumn("feature_date", F.lit(as_of_date))

    # ── Step 3: Write individual feature groups ───────────────────────────────
    group_dfs: dict[str, DataFrame] = {
        DEMOGRAPHICS.name: demog_df,
        COMORBIDITIES.name: comor_df,
        MEDICATIONS.name: meds_feat_df,
        GRAPH.name: graph_df,
    }
    merge_keys = ["patient_id", "feature_date"]
    for grp in ALL_GROUPS:
        path = f"{delta_base}/{grp.delta_path}"
        log.info("Writing feature group '%s' → %s", grp.name, path)
        _delta_merge(spark, group_dfs[grp.name], path, merge_keys)

    # ── Step 4: Build and write joined feature vector ─────────────────────────
    vector_df = (
        demog_df.alias("d")
        .join(comor_df.drop("feature_date").alias("c"), on="patient_id", how="left")
        .join(meds_feat_df.drop("feature_date").alias("m"), on="patient_id", how="left")
        .join(graph_df.drop("feature_date").alias("g"), on="patient_id", how="left")
        .fillna(0)
    )
    vector_path = f"{delta_base}/{FEATURE_VECTOR.delta_path}"
    log.info("Writing joined feature vector → %s", vector_path)
    _delta_merge(spark, vector_df, vector_path, merge_keys)

    log.info("Feature engineering complete for as_of_date=%s", as_of_date)
    spark.stop()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healthcare feature engineering batch job")
    parser.add_argument(
        "--as-of-date",
        default=str(date.today()),
        help="Reference date (YYYY-MM-DD).  Defaults to today.",
    )
    parser.add_argument(
        "--delta-base",
        default=os.environ.get("DELTA_BASE", "s3a://healthcare-delta"),
        help="S3A base path for Delta tables.",
    )
    args = parser.parse_args()
    main(as_of_date=args.as_of_date, delta_base=args.delta_base)
