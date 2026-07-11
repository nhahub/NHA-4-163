"""Airflow DAG — Batch FHIR Bundle / CSV Ingestion.

Reads FHIR R4 bundles or CSV exports from MinIO (S3-compatible),
validates with Great Expectations, loads into Postgres + Neo4j,
then marks Neo4j-synced FamilyMemberHistory records via Kafka events.

Schedule: daily at 02:00 UTC (off-peak, avoids overlap with streaming job).

Task graph:
  check_source_files
    → validate_data_quality
      → load_postgres
        → sync_neo4j
          → run_identity_resolution
            → notify_complete

Idempotency: DAG is safe to re-run.  Postgres uses INSERT … ON CONFLICT.
Neo4j writes use MERGE.  MinIO reads are non-destructive (objects stay).

Variables (set in Airflow UI → Admin → Variables):
  BATCH_INGESTION_SOURCE_PREFIX  — MinIO prefix to scan (default: fhir/pending/)
  BATCH_INGESTION_MAX_FILES      — Max files per run (default: 500)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from airflow.decorators import dag, task
from airflow.models import Variable

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "email_on_failure": False,
    "email_on_retry": False,
}


@dag(
    dag_id="batch_fhir_ingestion",
    description="Daily batch ingestion of FHIR bundles from MinIO → Postgres + Neo4j",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["ingestion", "fhir", "batch", "phase3"],
)
def batch_fhir_ingestion() -> None:
    """DAG definition for batch FHIR bundle ingestion."""

    # ── Task: check_source_files ──────────────────────────────────────────────

    @task
    def check_source_files() -> list[str]:
        """List FHIR bundle files available in MinIO for ingestion.

        Returns:
            List of MinIO object keys to process in this run.

        Raises:
            AirflowSkipException: If no files are available.
        """
        import boto3
        from airflow.exceptions import AirflowSkipException

        from libs.common.config import get_settings

        settings = get_settings()
        minio = settings.minio
        prefix = Variable.get("BATCH_INGESTION_SOURCE_PREFIX", default_var="fhir/pending/")
        max_files = int(Variable.get("BATCH_INGESTION_MAX_FILES", default_var="500"))

        s3 = boto3.client(
            "s3",
            endpoint_url=str(minio.endpoint),
            aws_access_key_id=minio.access_key,
            aws_secret_access_key=minio.secret_key.get_secret_value(),
        )

        paginator = s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=minio.bucket_raw, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
                if len(keys) >= max_files:
                    break
            if len(keys) >= max_files:
                break

        if not keys:
            raise AirflowSkipException(f"No files found at prefix {prefix!r}")

        log.info("Found batch files", extra={"count": len(keys), "prefix": prefix})
        return keys

    # ── Task: validate_data_quality ───────────────────────────────────────────

    @task
    def validate_data_quality(object_keys: list[str]) -> dict[str, Any]:
        """Download a sample of files and run Great Expectations suites.

        Validates up to 100 records from the batch before full ingestion.
        Returns validation summary that downstream tasks check.

        Args:
            object_keys: List of MinIO object keys from ``check_source_files``.

        Returns:
            Dict with keys: ``success``, ``validated_count``, ``failures``.
        """

        import boto3

        from libs.common.config import get_settings
        from libs.common.quality import validate_diagnosis_records, validate_patient_records

        settings = get_settings()
        minio = settings.minio
        s3 = boto3.client(
            "s3",
            endpoint_url=str(minio.endpoint),
            aws_access_key_id=minio.access_key,
            aws_secret_access_key=minio.secret_key.get_secret_value(),
        )

        patient_records, diagnosis_records = [], []

        # Sample up to 50 files for quality check
        for key in object_keys[:50]:
            try:
                obj = s3.get_object(Bucket=minio.bucket_raw, Key=key)
                bundle = json.loads(obj["Body"].read())
            except Exception as exc:
                log.warning("Could not read file", extra={"key": key, "error": str(exc)})
                continue

            for entry in bundle.get("entry", []):
                resource = entry.get("resource", {})
                rt = resource.get("resourceType")
                if rt == "Patient":
                    patient_records.append(_fhir_patient_to_dict(resource))
                elif rt == "Condition":
                    diagnosis_records.append(_fhir_condition_to_dict(resource))

        patient_result = validate_patient_records(patient_records[:100])
        diagnosis_result = validate_diagnosis_records(diagnosis_records[:100])

        all_failures = patient_result.failures + diagnosis_result.failures
        overall_success = patient_result.success and diagnosis_result.success

        if not overall_success:
            log.error("Data quality validation failed", extra={"failures": all_failures})
        else:
            log.info("Data quality validation passed")

        return {
            "success": overall_success,
            "validated_count": len(patient_records) + len(diagnosis_records),
            "failures": all_failures,
        }

    # ── Task: load_postgres ───────────────────────────────────────────────────

    @task
    def load_postgres(object_keys: list[str], quality_result: dict[str, Any]) -> int:
        """Load all FHIR bundle records from MinIO into PostgreSQL.

        Skips loading if data quality validation failed.

        Args:
            object_keys: List of MinIO object keys.
            quality_result: Output from ``validate_data_quality``.

        Returns:
            Total number of records loaded.
        """
        if not quality_result["success"]:
            log.warning("Skipping Postgres load due to quality failures")
            return 0

        import boto3
        import psycopg2

        from libs.common.config import get_settings

        settings = get_settings()
        minio = settings.minio
        pg = settings.postgres

        s3 = boto3.client(
            "s3",
            endpoint_url=str(minio.endpoint),
            aws_access_key_id=minio.access_key,
            aws_secret_access_key=minio.secret_key.get_secret_value(),
        )
        conn = psycopg2.connect(
            host=pg.host,
            port=pg.port,
            dbname=pg.db,
            user=pg.user,
            password=pg.password.get_secret_value(),
        )

        total = 0
        try:
            with conn:
                with conn.cursor() as cur:
                    for key in object_keys:
                        try:
                            obj = s3.get_object(Bucket=minio.bucket_raw, Key=key)
                            bundle = json.loads(obj["Body"].read())
                        except Exception as exc:
                            log.warning(
                                "Skipping unreadable file", extra={"key": key, "error": str(exc)}
                            )
                            continue

                        for entry in bundle.get("entry", []):
                            resource = entry.get("resource", {})
                            rt = resource.get("resourceType")
                            if rt == "Patient":
                                _upsert_patient_pg(cur, _fhir_patient_to_dict(resource))
                                total += 1
                            elif rt == "Condition":
                                _upsert_condition_pg(cur, _fhir_condition_to_dict(resource))
                                total += 1
        finally:
            conn.close()

        log.info("Loaded records into Postgres", extra={"total": total})
        return total

    # ── Task: sync_neo4j ──────────────────────────────────────────────────────

    @task
    def sync_neo4j(records_loaded: int) -> int:
        """Push unsynced FamilyMemberHistory records into Neo4j as graph edges.

        Reads all rows where neo4j_synced = false from Postgres, creates
        the corresponding edges in Neo4j using MERGE, then marks them synced.

        Args:
            records_loaded: Count from ``load_postgres`` (for logging).

        Returns:
            Number of FamilyMemberHistory records synced.
        """
        import psycopg2
        from neo4j import GraphDatabase

        from libs.common.config import get_settings

        settings = get_settings()
        pg = settings.postgres
        neo4j_cfg = settings.neo4j

        conn = psycopg2.connect(
            host=pg.host,
            port=pg.port,
            dbname=pg.db,
            user=pg.user,
            password=pg.password.get_secret_value(),
        )
        driver = GraphDatabase.driver(
            neo4j_cfg.uri,
            auth=(neo4j_cfg.user, neo4j_cfg.password.get_secret_value()),
        )

        synced = 0
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id, patient_id, related_patient_id, relationship_code, "
                    "degree_of_relatedness FROM family_member_history "
                    "WHERE neo4j_synced = false LIMIT 1000"
                )
                rows = cur.fetchall()

                with driver.session() as session:
                    for row in rows:
                        fmh_id, patient_id, related_id, rel_code, degree = row
                        if not related_id:
                            continue
                        _merge_family_edge(session, patient_id, related_id, rel_code, degree)
                        synced += 1

                if rows:
                    ids = [str(r[0]) for r in rows if r[2]]
                    if ids:
                        cur.execute(
                            "UPDATE family_member_history SET neo4j_synced = true "
                            "WHERE id = ANY(%s::uuid[])",
                            (ids,),
                        )
        finally:
            conn.close()
            driver.close()

        log.info("Synced FamilyMemberHistory to Neo4j", extra={"count": synced})
        return synced

    # ── Task: run_identity_resolution ─────────────────────────────────────────

    @task
    def run_identity_resolution(records_synced: int) -> dict[str, Any]:
        """Run identity resolution on recently ingested patients.

        Finds probable duplicate patient records using blocking + Jaro-Winkler.
        Results are logged as warnings; no automatic merge is performed.

        Args:
            records_synced: Count from ``sync_neo4j`` (for logging).

        Returns:
            Dict with ``candidate_pairs`` count.
        """
        import psycopg2
        from pyspark.sql import SparkSession

        from libs.common.config import get_settings
        from pipelines.spark.streaming.transforms.identity_resolution import (
            MATCH_THRESHOLD,
            resolve_identities,
        )

        settings = get_settings()
        pg = settings.postgres
        conn = psycopg2.connect(
            host=pg.host,
            port=pg.port,
            dbname=pg.db,
            user=pg.user,
            password=pg.password.get_secret_value(),
        )

        # Pull patients modified in the last 7 days
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id AS patient_id, family_name, given_name, "
                "date_of_birth::text, gender, postal_code "
                "FROM patient WHERE updated_at > NOW() - INTERVAL '7 days' "
                "AND deleted_at IS NULL LIMIT 50000"
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
        conn.close()

        if len(rows) < 2:
            return {"candidate_pairs": 0}

        # Minimal Spark session for identity resolution
        spark = (
            SparkSession.builder.appName("identity-resolution-airflow")
            .master("local[2]")
            .getOrCreate()
        )
        df = spark.createDataFrame(rows)
        candidates = resolve_identities(df, spark)
        count = candidates.count()
        spark.stop()

        if count > 0:
            log.warning(
                "Identity resolution flagged probable duplicates",
                extra={"candidate_pairs": count, "threshold": MATCH_THRESHOLD},
            )
        return {"candidate_pairs": count}

    # ── Wire tasks ────────────────────────────────────────────────────────────

    files = check_source_files()
    quality = validate_data_quality(files)
    loaded = load_postgres(files, quality)
    synced = sync_neo4j(loaded)
    run_identity_resolution(synced)


# ---------------------------------------------------------------------------
# FHIR → dict helpers (minimal, not a full FHIR parser)
# ---------------------------------------------------------------------------


def _fhir_patient_to_dict(resource: dict) -> dict:
    """Extract flat fields from a FHIR Patient resource."""
    name = (resource.get("name") or [{}])[0]
    dob = resource.get("birthDate")
    return {
        "patient_id": resource.get("id", ""),
        "family_name": name.get("family"),
        "given_name": (name.get("given") or [""])[0],
        "date_of_birth": dob,
        "gender": resource.get("gender"),
        "research_consent": False,
        "deceased": resource.get("deceasedBoolean", False),
        "event_id": resource.get("id", ""),
        "event_timestamp": datetime.now(UTC),
        "source_system": "fhir-batch",
        "event_version": "1.0",
    }


def _fhir_condition_to_dict(resource: dict) -> dict:
    """Extract flat fields from a FHIR Condition resource."""
    coding = (resource.get("code", {}).get("coding") or [{}])[0]
    subject_ref = resource.get("subject", {}).get("reference", "/")
    patient_id = subject_ref.split("/")[-1]
    return {
        "condition_id": resource.get("id", ""),
        "patient_id": patient_id,
        "clinical_status": resource.get("clinicalStatus", {})
        .get("coding", [{}])[0]
        .get("code", "active"),
        "verification_status": None,
        "code_system": coding.get("system", ""),
        "code": coding.get("code", ""),
        "code_display": coding.get("display"),
        "is_hereditary": False,
        "family_history_flag": False,
        "event_id": resource.get("id", ""),
        "event_timestamp": datetime.now(UTC),
        "source_system": "fhir-batch",
        "event_version": "1.0",
    }


def _upsert_patient_pg(cur: Any, data: dict) -> None:
    cur.execute(
        """
        INSERT INTO patient (id, family_name, given_name, date_of_birth, gender,
                             deceased, research_consent, created_at, updated_at)
        VALUES (%(patient_id)s, %(family_name)s, %(given_name)s, %(date_of_birth)s,
                %(gender)s, %(deceased)s, %(research_consent)s, NOW(), NOW())
        ON CONFLICT (id) DO UPDATE SET
            family_name  = EXCLUDED.family_name,
            given_name   = EXCLUDED.given_name,
            updated_at   = NOW()
        """,
        data,
    )


def _upsert_condition_pg(cur: Any, data: dict) -> None:
    cur.execute(
        """
        INSERT INTO condition (id, patient_id, clinical_status, code_system, code,
                               code_display, is_hereditary, family_history_flag,
                               created_at, updated_at)
        VALUES (%(condition_id)s, %(patient_id)s, %(clinical_status)s, %(code_system)s,
                %(code)s, %(code_display)s, %(is_hereditary)s, %(family_history_flag)s,
                NOW(), NOW())
        ON CONFLICT (id) DO UPDATE SET
            clinical_status = EXCLUDED.clinical_status,
            updated_at      = NOW()
        """,
        data,
    )


def _merge_family_edge(
    session: Any,
    patient_id: str,
    related_id: str,
    rel_code: str,
    degree: float | None,
) -> None:
    """MERGE a family relationship edge in Neo4j."""
    _HL7_MAP = {
        "MTH": "PARENT_OF",
        "FTH": "PARENT_OF",
        "CHILD": "CHILD_OF",
        "SON": "CHILD_OF",
        "DAU": "CHILD_OF",
        "SIB": "SIBLING_OF",
        "BRO": "SIBLING_OF",
        "SIS": "SIBLING_OF",
        "HUSB": "SPOUSE_OF",
        "WIFE": "SPOUSE_OF",
        "SPS": "SPOUSE_OF",
    }
    rel_type = _HL7_MAP.get(rel_code, "RELATED_TO")
    cypher = (
        f"MATCH (a:Patient {{id: $pid}}) "
        f"MATCH (b:Patient {{id: $rid}}) "
        f"MERGE (a)-[r:{rel_type}]->(b) "
        f"SET r.degree_of_relatedness = $degree"
    )
    session.run(cypher, pid=patient_id, rid=related_id, degree=degree)


# Instantiate the DAG
batch_fhir_ingestion()
