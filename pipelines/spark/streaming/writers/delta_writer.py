"""Delta Lake writer for the healthcare data lake.

Writes three partitioned Delta tables:
  - ``{bucket}/raw/{topic}/``          — immutable raw events, append-only
  - ``{bucket}/processed/{entity}/``   — validated, typed records (MERGE)
  - ``{bucket}/dlq/{topic}/``          — invalid events (dead-letter queue)

Partition strategy:
  - Raw:       partition by (year, month, day) on event_timestamp
  - Processed: partition by (year, month) on event_timestamp
  - DLQ:       partition by (year, month) on event_timestamp

Schema evolution (``mergeSchema=true``) is enabled for processed tables only.
Raw tables are append-only with a fixed schema to preserve the original record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from delta import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

log = logging.getLogger(__name__)


@dataclass
class DeltaConfig:
    """Storage configuration for Delta Lake paths."""

    base_path: str
    """S3/MinIO path prefix, e.g. ``s3a://healthcare-delta``."""


class DeltaWriter:
    """Writes raw, processed, and dead-letter data to Delta Lake."""

    def __init__(self, spark: SparkSession, config: DeltaConfig) -> None:
        self._spark = spark
        self._cfg = config

    # ── Raw layer (append-only) ───────────────────────────────────────────────

    def write_raw(self, df: DataFrame, topic: str) -> None:
        """Append raw Kafka events to the raw Delta table for a topic.

        The raw layer preserves the original event bytes for audit / replay.
        Schema changes are NOT allowed — use ``write_processed`` for evolved data.

        Args:
            df: DataFrame including kafka metadata columns (topic, partition,
                offset, timestamp) plus the raw value bytes.
            topic: Kafka topic name (used as the sub-path).
        """
        if df.rdd.isEmpty():
            return

        path = f"{self._cfg.base_path}/raw/{topic.replace('.', '_')}"
        enriched = (
            df.withColumn("_year", F.year("event_timestamp"))
            .withColumn("_month", F.month("event_timestamp"))
            .withColumn("_day", F.dayofmonth("event_timestamp"))
        )

        (
            enriched.write.format("delta")
            .mode("append")
            .partitionBy("_year", "_month", "_day")
            .option("mergeSchema", "false")
            .save(path)
        )
        log.info("Wrote raw events", extra={"topic": topic, "path": path})

    # ── Processed layer (MERGE / upsert) ─────────────────────────────────────

    def merge_processed(
        self,
        df: DataFrame,
        entity: str,
        merge_key: str,
    ) -> None:
        """MERGE (upsert) validated records into the processed Delta table.

        Creates the table on first run.  Subsequent runs use Delta's MERGE
        to update existing records and insert new ones.

        Args:
            df: Validated, typed DataFrame.
            entity: Entity name used as sub-path (e.g., ``patient``, ``condition``).
            merge_key: Column name to match on (usually the primary key UUID).
        """
        if df.rdd.isEmpty():
            return

        path = f"{self._cfg.base_path}/processed/{entity}"
        enriched = df.withColumn("_year", F.year("event_timestamp")).withColumn(
            "_month", F.month("event_timestamp")
        )

        if DeltaTable.isDeltaTable(self._spark, path):
            target = DeltaTable.forPath(self._spark, path)
            (
                target.alias("target")
                .merge(
                    enriched.alias("source"),
                    f"target.`{merge_key}` = source.`{merge_key}`",
                )
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
        else:
            (
                enriched.write.format("delta")
                .mode("overwrite")
                .partitionBy("_year", "_month")
                .option("mergeSchema", "true")
                .save(path)
            )

        log.info(
            "Merged processed records",
            extra={"entity": entity, "path": path, "merge_key": merge_key},
        )

    # ── Dead-letter queue ─────────────────────────────────────────────────────

    def write_dlq(self, df: DataFrame, topic: str, batch_id: int) -> None:
        """Append invalid / unprocessable records to the dead-letter queue table.

        DLQ records include the original raw event plus an ``error_message``
        column explaining why validation failed.  They are never deleted.

        Args:
            df: DataFrame of invalid records.  Must include ``error_message``
                and ``event_timestamp`` columns.
            topic: Originating Kafka topic.
            batch_id: Spark micro-batch ID (for traceability).
        """
        if df.rdd.isEmpty():
            return

        path = f"{self._cfg.base_path}/dlq/{topic.replace('.', '_')}"
        enriched = (
            df.withColumn("_batch_id", F.lit(batch_id))
            .withColumn("_year", F.year("event_timestamp"))
            .withColumn("_month", F.month("event_timestamp"))
        )

        (
            enriched.write.format("delta")
            .mode("append")
            .partitionBy("_year", "_month")
            .option("mergeSchema", "true")
            .save(path)
        )
        count = enriched.count()
        log.warning(
            "Wrote DLQ records",
            extra={"topic": topic, "batch_id": batch_id, "count": count},
        )

    @classmethod
    def from_settings(cls, spark: SparkSession, settings: Any) -> DeltaWriter:
        """Construct from the project Settings object.

        Args:
            spark: Active SparkSession.
            settings: ``libs.common.config.Settings`` instance.

        Returns:
            Configured ``DeltaWriter``.
        """
        bucket = settings.minio.bucket_delta
        str(settings.minio.endpoint)
        path = f"s3a://{bucket}"
        return cls(spark, DeltaConfig(base_path=path))
