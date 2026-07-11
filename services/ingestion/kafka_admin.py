"""Kafka topic and schema registry bootstrapper.

Run once before starting any producers or the Spark Streaming job:
    python services/ingestion/kafka_admin.py

Creates all required topics with appropriate partition counts and retention,
then registers each Avro schema with the Schema Registry.

Design decisions:
- 6 partitions per topic: allows up to 6 Spark streaming tasks per topic.
  Increase in production based on throughput requirements.
- Replication factor 1 for local dev; set to 3 for production.
- retention.ms = 7 days (604800000 ms) — enough for replay window.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import requests
from confluent_kafka.admin import AdminClient, NewTopic

from libs.common.config import get_settings
from libs.common.logging import configure_logging

configure_logging()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic definitions
# ---------------------------------------------------------------------------

TOPICS: list[dict[str, Any]] = [
    {
        "name": "patient.created",
        "partitions": 6,
        "replication_factor": 1,
        "config": {
            "retention.ms": "604800000",  # 7 days
            "cleanup.policy": "delete",
            "compression.type": "snappy",
        },
    },
    {
        "name": "diagnosis.added",
        "partitions": 6,
        "replication_factor": 1,
        "config": {
            "retention.ms": "604800000",
            "cleanup.policy": "delete",
            "compression.type": "snappy",
        },
    },
    {
        "name": "prescription.issued",
        "partitions": 6,
        "replication_factor": 1,
        "config": {
            "retention.ms": "604800000",
            "cleanup.policy": "delete",
            "compression.type": "snappy",
        },
    },
    {
        "name": "relative.linked",
        "partitions": 4,
        "replication_factor": 1,
        "config": {
            "retention.ms": "604800000",
            "cleanup.policy": "delete",
            "compression.type": "snappy",
        },
    },
    {
        "name": "observation.recorded",
        "partitions": 6,
        "replication_factor": 1,
        "config": {
            "retention.ms": "604800000",
            "cleanup.policy": "delete",
            "compression.type": "snappy",
        },
    },
    # Dead-letter queue — catches events that fail validation in the stream job.
    {
        "name": "healthcare.dlq",
        "partitions": 2,
        "replication_factor": 1,
        "config": {
            "retention.ms": "2592000000",  # 30 days
            "cleanup.policy": "delete",
        },
    },
]

# Maps topic name → Avro schema file path (relative to project root).
SCHEMA_MAP: dict[str, str] = {
    "patient.created": "schemas/avro/patient_created.avsc",
    "diagnosis.added": "schemas/avro/diagnosis_added.avsc",
    "prescription.issued": "schemas/avro/prescription_issued.avsc",
    "relative.linked": "schemas/avro/relative_linked.avsc",
    "observation.recorded": "schemas/avro/observation_recorded.avsc",
}


# ---------------------------------------------------------------------------
# Topic administration
# ---------------------------------------------------------------------------


def create_topics(bootstrap_servers: str) -> None:
    """Create all configured topics.  Idempotent — skips existing topics.

    Args:
        bootstrap_servers: Kafka bootstrap servers string.
    """
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    existing = set(admin.list_topics(timeout=10).topics.keys())

    new_topics = [
        NewTopic(
            t["name"],
            num_partitions=t["partitions"],
            replication_factor=t["replication_factor"],
            config=t["config"],
        )
        for t in TOPICS
        if t["name"] not in existing
    ]

    if not new_topics:
        log.info("All topics already exist — skipping creation")
        return

    futures = admin.create_topics(new_topics)
    for topic_name, future in futures.items():
        try:
            future.result()
            log.info("Created topic", extra={"topic": topic_name})
        except Exception as exc:
            log.error("Failed to create topic", extra={"topic": topic_name, "error": str(exc)})
            raise


# ---------------------------------------------------------------------------
# Schema Registry
# ---------------------------------------------------------------------------


def register_schema(registry_url: str, topic: str, schema_path: str) -> int:
    """Register (or update) an Avro value schema for a topic.

    Uses the default subject naming convention: ``{topic}-value``.

    Args:
        registry_url: Base URL of the Confluent Schema Registry.
        topic: Kafka topic name.
        schema_path: Path to the ``.avsc`` file, relative to project root.

    Returns:
        The schema ID assigned by the registry.

    Raises:
        FileNotFoundError: If the ``.avsc`` file does not exist.
        requests.HTTPError: If the registry returns a non-2xx response.
    """
    avsc_path = Path(schema_path)
    if not avsc_path.exists():
        raise FileNotFoundError(f"Avro schema not found: {avsc_path}")

    schema_str = avsc_path.read_text()
    subject = f"{topic}-value"
    url = f"{registry_url}/subjects/{subject}/versions"

    response = requests.post(
        url,
        json={"schema": schema_str, "schemaType": "AVRO"},
        headers={"Content-Type": "application/vnd.schemaregistry.v1+json"},
        timeout=10,
    )
    response.raise_for_status()
    schema_id: int = response.json()["id"]
    log.info("Registered schema", extra={"subject": subject, "schema_id": schema_id})
    return schema_id


def get_latest_schema(registry_url: str, topic: str) -> tuple[int, str]:
    """Fetch the latest schema for a topic from the Schema Registry.

    Args:
        registry_url: Base URL of the Confluent Schema Registry.
        topic: Kafka topic name.

    Returns:
        Tuple of (schema_id, schema_json_string).

    Raises:
        requests.HTTPError: If the subject does not exist or registry is unreachable.
    """
    subject = f"{topic}-value"
    url = f"{registry_url}/subjects/{subject}/versions/latest"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    body = response.json()
    return body["id"], body["schema"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Bootstrap Kafka topics and register all Avro schemas."""
    settings = get_settings()
    bootstrap_servers = settings.kafka.bootstrap_servers
    registry_url = str(settings.kafka.schema_registry_url).rstrip("/")

    log.info("Creating Kafka topics", extra={"bootstrap_servers": bootstrap_servers})
    create_topics(bootstrap_servers)

    log.info("Registering Avro schemas", extra={"registry_url": registry_url})
    for topic, schema_path in SCHEMA_MAP.items():
        try:
            register_schema(registry_url, topic, schema_path)
        except Exception as exc:
            log.error(
                "Schema registration failed",
                extra={"topic": topic, "error": str(exc)},
            )
            sys.exit(1)

    log.info("Kafka bootstrap complete")


if __name__ == "__main__":
    main()
