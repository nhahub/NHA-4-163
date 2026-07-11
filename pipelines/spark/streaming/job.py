"""Healthcare Hereditary Disease Prediction — Kafka Structured Streaming Job.

Reads all five Kafka topics in a single stream, routes events by topic inside
``foreachBatch``, validates with Pydantic, then writes to three sinks:
  1. Neo4j  — graph nodes and relationships
  2. Postgres — FHIR-aligned relational records
  3. Delta Lake — raw layer (immutable) + processed layer (MERGE) + DLQ

Architecture notes
------------------
- **One stream, multiple topics** — Spark reads from all topics with a single
  ``readStream`` (``subscribe`` option). This keeps resource usage low for the
  local dev cluster and simplifies checkpoint management.
- **``foreachBatch``** — chosen over multiple ``writeStream`` outputs because
  it allows: (a) writing to non-streaming sinks (Neo4j, psycopg2 JDBC),
  (b) conditional writes (valid vs. DLQ), (c) shared batch context.
- **Confluent wire format** — Avro messages in Kafka carry a 5-byte prefix
  (1 magic byte + 4-byte big-endian schema ID). We strip it before passing
  to ``from_avro()``. Schemas are fetched from Schema Registry at job startup.
- **Idempotency** — All writes are MERGE/UPSERT keyed on the event's UUID so
  the job can be replayed from any checkpoint offset without duplicates.
- **Checkpointing** — stored in Delta Lake (S3/MinIO) for durability.

Run locally:
    spark-submit \\
      --master spark://localhost:7077 \\
      --packages org.apache.spark:spark-avro_2.12:3.5.1,\\
                 org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,\\
                 io.delta:delta-spark_2.12:3.2.0,\\
                 org.neo4j:neo4j-connector-apache-spark_2.12:5.3.1_for_spark_3,\\
                 org.postgresql:postgresql:42.7.3 \\
      --py-files libs.zip \\
      pipelines/spark/streaming/job.py
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import requests
from delta import configure_spark_with_delta_pip
from pydantic import ValidationError
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.avro.functions import from_avro

# ── Project root on path (for libs.common imports) ────────────────────────────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from libs.common.config import get_settings  # noqa: E402
from libs.common.logging import configure_logging  # noqa: E402
from pipelines.spark.streaming.validators import (  # noqa: E402
    DiagnosisAddedEvent,
    ObservationRecordedEvent,
    PatientCreatedEvent,
    PrescriptionIssuedEvent,
    RelativeLinkedEvent,
)
from pipelines.spark.streaming.writers.delta_writer import DeltaWriter  # noqa: E402
from pipelines.spark.streaming.writers.neo4j_writer import Neo4jWriter  # noqa: E402
from pipelines.spark.streaming.writers.postgres_writer import PostgresWriter  # noqa: E402

configure_logging()
log = logging.getLogger(__name__)

TOPICS = [
    "patient.created",
    "diagnosis.added",
    "prescription.issued",
    "relative.linked",
    "observation.recorded",
]


# ---------------------------------------------------------------------------
# Schema Registry helpers
# ---------------------------------------------------------------------------


def _fetch_schema(registry_url: str, topic: str) -> str:
    """Fetch the latest Avro schema string for a topic from Schema Registry.

    Args:
        registry_url: Base URL (e.g., ``http://localhost:8081``).
        topic: Kafka topic name.

    Returns:
        JSON schema string.

    Raises:
        requests.HTTPError: If schema not found (404) or registry unreachable.
    """
    url = f"{registry_url}/subjects/{topic}-value/versions/latest"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()["schema"]


def _load_all_schemas(registry_url: str) -> dict[str, str]:
    """Load all topic schemas from the Schema Registry at startup.

    Args:
        registry_url: Base Schema Registry URL.

    Returns:
        Dict mapping topic name → schema JSON string.
    """
    schemas: dict[str, str] = {}
    for topic in TOPICS:
        try:
            schemas[topic] = _fetch_schema(registry_url, topic)
            log.info("Loaded schema", extra={"topic": topic})
        except requests.HTTPError as exc:
            log.error(
                "Could not load schema — topic may not be registered yet",
                extra={"topic": topic, "error": str(exc)},
            )
            raise
    return schemas


# ---------------------------------------------------------------------------
# Event parsing + validation helpers
# ---------------------------------------------------------------------------


def _validate_rows(rows: list[dict], model: Any, topic: str) -> tuple[list[dict], list[dict]]:
    """Validate a list of row dicts against a Pydantic model.

    Args:
        rows: List of Row.asDict() outputs.
        model: Pydantic BaseModel subclass.
        topic: Topic name (for DLQ error messages).

    Returns:
        ``(valid, invalid)`` where ``invalid`` rows have an ``error_message`` key.
    """
    valid, invalid = [], []
    for row in rows:
        try:
            model(**row)
            valid.append(row)
        except ValidationError as exc:
            row["error_message"] = exc.json()
            row["source_topic"] = topic
            invalid.append(row)
    return valid, invalid


# ---------------------------------------------------------------------------
# Per-topic batch processors
# ---------------------------------------------------------------------------


def _process_patient_batch(
    df: DataFrame,
    schemas: dict[str, str],
    neo4j: Neo4jWriter,
    pg: PostgresWriter,
    delta: DeltaWriter,
    batch_id: int,
) -> None:
    """Process a micro-batch of ``patient.created`` events."""
    topic = "patient.created"
    raw_df = df.filter(F.col("topic") == topic)
    if raw_df.rdd.isEmpty():
        return

    delta.write_raw(raw_df, topic)

    parsed = raw_df.select(
        from_avro(F.expr("substring(value, 6)"), schemas[topic]).alias("d"),
        F.col("timestamp").alias("kafka_ts"),
    ).select("d.*", "kafka_ts")

    rows = parsed.collect()
    valid_rows, invalid_rows = _validate_rows(
        [r.asDict() for r in rows], PatientCreatedEvent, topic
    )

    if invalid_rows:
        dlq_df = parsed.sparkSession.createDataFrame(invalid_rows)
        delta.write_dlq(dlq_df, topic, batch_id)

    if valid_rows:
        valid_df = parsed.sparkSession.createDataFrame(valid_rows, schema=parsed.schema)
        neo4j.upsert_patient_nodes(valid_df)
        pg.upsert_patients(valid_df)
        delta.merge_processed(valid_df, "patient", "patient_id")


def _process_diagnosis_batch(
    df: DataFrame,
    schemas: dict[str, str],
    neo4j: Neo4jWriter,
    pg: PostgresWriter,
    delta: DeltaWriter,
    batch_id: int,
) -> None:
    """Process a micro-batch of ``diagnosis.added`` events."""
    topic = "diagnosis.added"
    raw_df = df.filter(F.col("topic") == topic)
    if raw_df.rdd.isEmpty():
        return

    delta.write_raw(raw_df, topic)

    parsed = raw_df.select(
        from_avro(F.expr("substring(value, 6)"), schemas[topic]).alias("d"),
    ).select("d.*")

    rows = parsed.collect()
    valid_rows, invalid_rows = _validate_rows(
        [r.asDict() for r in rows], DiagnosisAddedEvent, topic
    )

    if invalid_rows:
        delta.write_dlq(parsed.sparkSession.createDataFrame(invalid_rows), topic, batch_id)

    if valid_rows:
        valid_df = parsed.sparkSession.createDataFrame(valid_rows, schema=parsed.schema)
        neo4j.upsert_disease_nodes(valid_df)
        neo4j.upsert_diagnosed_with_relationships(valid_df)
        pg.upsert_conditions(valid_df)
        delta.merge_processed(valid_df, "condition", "condition_id")


def _process_prescription_batch(
    df: DataFrame,
    schemas: dict[str, str],
    neo4j: Neo4jWriter,
    pg: PostgresWriter,
    delta: DeltaWriter,
    batch_id: int,
) -> None:
    """Process a micro-batch of ``prescription.issued`` events."""
    topic = "prescription.issued"
    raw_df = df.filter(F.col("topic") == topic)
    if raw_df.rdd.isEmpty():
        return

    delta.write_raw(raw_df, topic)
    parsed = raw_df.select(
        from_avro(F.expr("substring(value, 6)"), schemas[topic]).alias("d"),
    ).select("d.*")

    rows = parsed.collect()
    valid_rows, invalid_rows = _validate_rows(
        [r.asDict() for r in rows], PrescriptionIssuedEvent, topic
    )

    if invalid_rows:
        delta.write_dlq(parsed.sparkSession.createDataFrame(invalid_rows), topic, batch_id)

    if valid_rows:
        valid_df = parsed.sparkSession.createDataFrame(valid_rows, schema=parsed.schema)
        pg.upsert_medication_requests(valid_df)
        delta.merge_processed(valid_df, "medication_request", "medication_request_id")


def _process_relative_batch(
    df: DataFrame,
    schemas: dict[str, str],
    neo4j: Neo4jWriter,
    pg: PostgresWriter,
    delta: DeltaWriter,
    batch_id: int,
) -> None:
    """Process a micro-batch of ``relative.linked`` events."""
    topic = "relative.linked"
    raw_df = df.filter(F.col("topic") == topic)
    if raw_df.rdd.isEmpty():
        return

    delta.write_raw(raw_df, topic)
    parsed = raw_df.select(
        from_avro(F.expr("substring(value, 6)"), schemas[topic]).alias("d"),
    ).select("d.*")

    rows = parsed.collect()
    valid_rows, invalid_rows = _validate_rows(
        [r.asDict() for r in rows], RelativeLinkedEvent, topic
    )

    if invalid_rows:
        delta.write_dlq(parsed.sparkSession.createDataFrame(invalid_rows), topic, batch_id)

    if valid_rows:
        valid_df = parsed.sparkSession.createDataFrame(valid_rows, schema=parsed.schema)
        neo4j.upsert_relative_relationships(valid_df)
        pg.upsert_family_member_history(valid_df)
        delta.merge_processed(valid_df, "family_member_history", "fmh_id")


def _process_observation_batch(
    df: DataFrame,
    schemas: dict[str, str],
    neo4j: Neo4jWriter,
    pg: PostgresWriter,
    delta: DeltaWriter,
    batch_id: int,
) -> None:
    """Process a micro-batch of ``observation.recorded`` events."""
    topic = "observation.recorded"
    raw_df = df.filter(F.col("topic") == topic)
    if raw_df.rdd.isEmpty():
        return

    delta.write_raw(raw_df, topic)
    parsed = raw_df.select(
        from_avro(F.expr("substring(value, 6)"), schemas[topic]).alias("d"),
    ).select("d.*")

    rows = parsed.collect()
    valid_rows, invalid_rows = _validate_rows(
        [r.asDict() for r in rows], ObservationRecordedEvent, topic
    )

    if invalid_rows:
        delta.write_dlq(parsed.sparkSession.createDataFrame(invalid_rows), topic, batch_id)

    if valid_rows:
        valid_df = parsed.sparkSession.createDataFrame(valid_rows, schema=parsed.schema)
        pg.upsert_observations(valid_df)
        delta.merge_processed(valid_df, "observation", "observation_id")


# ---------------------------------------------------------------------------
# SparkSession builder
# ---------------------------------------------------------------------------


def build_spark(settings: Any) -> SparkSession:
    """Build a Delta-enabled SparkSession with S3/MinIO configuration.

    Args:
        settings: Project ``Settings`` object.

    Returns:
        Configured ``SparkSession``.
    """
    minio = settings.minio
    endpoint = str(minio.endpoint)
    access_key = minio.access_key
    secret_key = minio.secret_key.get_secret_value()

    builder = (
        SparkSession.builder.appName("healthcare-streaming")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        # S3A / MinIO configuration
        .config("spark.hadoop.fs.s3a.endpoint", endpoint)
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        # Streaming micro-batch interval: 30 seconds
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
    )

    return configure_spark_with_delta_pip(builder).getOrCreate()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the Kafka → Neo4j/Postgres/Delta streaming pipeline."""
    settings = get_settings()
    spark = build_spark(settings)
    spark.sparkContext.setLogLevel("WARN")

    registry_url = str(settings.kafka.schema_registry_url).rstrip("/")
    schemas = _load_all_schemas(registry_url)

    neo4j_writer = Neo4jWriter.from_settings(settings)
    pg_writer = PostgresWriter.from_settings(settings)
    delta_writer = DeltaWriter.from_settings(spark, settings)

    checkpoint_location = f"s3a://{settings.minio.bucket_delta}/checkpoints/streaming-job"

    # ── Read all topics in one stream ─────────────────────────────────────────
    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka.bootstrap_servers)
        .option("subscribe", ",".join(TOPICS))
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .option("kafka.group.id", "healthcare-streaming-job")
        .load()
        .withColumn("event_timestamp", F.col("timestamp"))
    )

    # ── foreachBatch: route + validate + write ────────────────────────────────
    def process_batch(batch_df: DataFrame, batch_id: int) -> None:
        log.info("Processing micro-batch", extra={"batch_id": batch_id})
        kwargs = {
            "schemas": schemas,
            "neo4j": neo4j_writer,
            "pg": pg_writer,
            "delta": delta_writer,
            "batch_id": batch_id,
        }
        _process_patient_batch(batch_df, **kwargs)
        _process_diagnosis_batch(batch_df, **kwargs)
        _process_prescription_batch(batch_df, **kwargs)
        _process_relative_batch(batch_df, **kwargs)
        _process_observation_batch(batch_df, **kwargs)

    query = (
        kafka_df.writeStream.foreachBatch(process_batch)
        .option("checkpointLocation", checkpoint_location)
        .trigger(processingTime="30 seconds")
        .start()
    )

    log.info("Streaming job started — awaiting termination")
    query.awaitTermination()


if __name__ == "__main__":
    main()
