"""PostgreSQL Spark writer — upserts FHIR records via JDBC.

Uses ``INSERT … ON CONFLICT DO UPDATE`` (PostgreSQL UPSERT) executed via JDBC
``foreachPartition`` rather than Spark's built-in JDBC writer so we can:
  1. Use server-side UPSERT semantics (idempotent).
  2. Avoid loading the full DataFrame into the driver for row-by-row inserts.
  3. Control transaction boundaries per partition.

The ``psycopg2`` driver is used directly inside ``foreachPartition`` because
the Spark JDBC writer only supports INSERT/OVERWRITE, not ON CONFLICT.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import psycopg2
from pyspark.sql import DataFrame, Row

log = logging.getLogger(__name__)


@dataclass
class PostgresConfig:
    """JDBC-compatible Postgres connection parameters."""

    host: str
    port: int
    database: str
    user: str
    password: str

    def dsn(self) -> str:
        """Return a psycopg2 DSN dict-style connection string."""
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password}"
        )


class PostgresWriter:
    """Writes Spark DataFrames to PostgreSQL using UPSERT semantics."""

    def __init__(self, config: PostgresConfig) -> None:
        self._cfg = config

    # ── Patient ───────────────────────────────────────────────────────────────

    def upsert_patients(self, df: DataFrame) -> None:
        """INSERT … ON CONFLICT UPDATE patient rows.

        Uses patient_id (UUID) as the conflict key.  PHI fields are passed
        through without encryption here — Phase 7 wraps the column in
        pgcrypto before the same UPSERT fires.

        Args:
            df: DataFrame with PatientCreated event fields.
        """
        if df.rdd.isEmpty():
            return

        sql = """
            INSERT INTO patient (
                id, external_id, identifier_system,
                family_name, given_name, middle_name,
                date_of_birth, gender, deceased,
                research_consent, created_at, updated_at
            ) VALUES (
                %(patient_id)s, %(external_id)s, %(identifier_system)s,
                %(family_name)s, %(given_name)s, %(middle_name)s,
                %(date_of_birth)s, %(gender)s, %(deceased)s,
                %(research_consent)s, %(event_timestamp)s, %(event_timestamp)s
            )
            ON CONFLICT (id) DO UPDATE SET
                external_id        = EXCLUDED.external_id,
                family_name        = EXCLUDED.family_name,
                given_name         = EXCLUDED.given_name,
                date_of_birth      = EXCLUDED.date_of_birth,
                gender             = EXCLUDED.gender,
                deceased           = EXCLUDED.deceased,
                research_consent   = EXCLUDED.research_consent,
                updated_at         = EXCLUDED.updated_at
        """
        self._upsert_partition(df, sql)
        log.info("Upserted patients", extra={"count": df.count()})

    # ── Condition ─────────────────────────────────────────────────────────────

    def upsert_conditions(self, df: DataFrame) -> None:
        """INSERT … ON CONFLICT UPDATE condition rows.

        Args:
            df: DataFrame with DiagnosisAdded event fields.
        """
        if df.rdd.isEmpty():
            return

        sql = """
            INSERT INTO condition (
                id, patient_id, encounter_id, recorder_id,
                clinical_status, verification_status, severity,
                code_system, code, code_display,
                onset_datetime, onset_age_years,
                is_hereditary, family_history_flag,
                created_at, updated_at
            ) VALUES (
                %(condition_id)s, %(patient_id)s, %(encounter_id)s, %(recorder_id)s,
                %(clinical_status)s, %(verification_status)s, %(severity)s,
                %(code_system)s, %(code)s, %(code_display)s,
                %(onset_datetime)s, %(onset_age_years)s,
                %(is_hereditary)s, %(family_history_flag)s,
                %(event_timestamp)s, %(event_timestamp)s
            )
            ON CONFLICT (id) DO UPDATE SET
                clinical_status     = EXCLUDED.clinical_status,
                verification_status = EXCLUDED.verification_status,
                severity            = EXCLUDED.severity,
                onset_datetime      = EXCLUDED.onset_datetime,
                is_hereditary       = EXCLUDED.is_hereditary,
                updated_at          = EXCLUDED.updated_at
        """
        self._upsert_partition(df, sql)

    # ── MedicationRequest ─────────────────────────────────────────────────────

    def upsert_medication_requests(self, df: DataFrame) -> None:
        """INSERT … ON CONFLICT UPDATE medication_request rows.

        Args:
            df: DataFrame with PrescriptionIssued event fields.
        """
        if df.rdd.isEmpty():
            return

        sql = """
            INSERT INTO medication_request (
                id, patient_id, encounter_id, requester_id,
                status, intent,
                medication_code_system, medication_code, medication_display,
                dosage_text, dosage_route, dose_quantity, dose_unit,
                authored_on, created_at, updated_at
            ) VALUES (
                %(medication_request_id)s, %(patient_id)s, %(encounter_id)s, %(requester_id)s,
                %(status)s, %(intent)s,
                %(medication_code_system)s, %(medication_code)s, %(medication_display)s,
                %(dosage_text)s, %(dosage_route)s, %(dose_quantity)s, %(dose_unit)s,
                %(authored_on)s, %(event_timestamp)s, %(event_timestamp)s
            )
            ON CONFLICT (id) DO UPDATE SET
                status              = EXCLUDED.status,
                dosage_text         = EXCLUDED.dosage_text,
                updated_at          = EXCLUDED.updated_at
        """
        self._upsert_partition(df, sql)

    # ── FamilyMemberHistory ───────────────────────────────────────────────────

    def upsert_family_member_history(self, df: DataFrame) -> None:
        """INSERT … ON CONFLICT UPDATE family_member_history rows.

        Args:
            df: DataFrame with RelativeLinked event fields.
        """
        if df.rdd.isEmpty():
            return

        sql = """
            INSERT INTO family_member_history (
                id, patient_id, related_patient_id,
                status, relationship_code, relationship_display,
                degree_of_relatedness, sex,
                born_date, deceased, deceased_age_years,
                conditions, neo4j_synced,
                created_at, updated_at
            ) VALUES (
                %(fmh_id)s, %(patient_id)s, %(related_patient_id)s,
                'partial', %(relationship_code)s, %(relationship_display)s,
                %(degree_of_relatedness)s, %(sex)s,
                %(born_date)s, %(deceased)s, %(deceased_age_years)s,
                %(conditions_json)s::jsonb, false,
                %(event_timestamp)s, %(event_timestamp)s
            )
            ON CONFLICT (id) DO UPDATE SET
                related_patient_id    = EXCLUDED.related_patient_id,
                degree_of_relatedness = EXCLUDED.degree_of_relatedness,
                conditions            = EXCLUDED.conditions,
                updated_at            = EXCLUDED.updated_at
        """
        self._upsert_partition(df, sql)

    # ── Observation ───────────────────────────────────────────────────────────

    def upsert_observations(self, df: DataFrame) -> None:
        """INSERT … ON CONFLICT UPDATE observation rows.

        Args:
            df: DataFrame with ObservationRecorded event fields.
        """
        if df.rdd.isEmpty():
            return

        sql = """
            INSERT INTO observation (
                id, patient_id, encounter_id,
                status, category,
                code_system, code, code_display,
                effective_datetime,
                value_quantity, value_unit, value_string,
                value_boolean, value_codeable_code, value_codeable_display,
                ref_range_low, ref_range_high, interpretation,
                created_at, updated_at
            ) VALUES (
                %(observation_id)s, %(patient_id)s, %(encounter_id)s,
                %(status)s, %(category)s,
                %(code_system)s, %(code)s, %(code_display)s,
                %(effective_datetime)s,
                %(value_quantity)s, %(value_unit)s, %(value_string)s,
                %(value_boolean)s, %(value_codeable_code)s, %(value_codeable_display)s,
                %(ref_range_low)s, %(ref_range_high)s, %(interpretation)s,
                %(event_timestamp)s, %(event_timestamp)s
            )
            ON CONFLICT (id) DO UPDATE SET
                status           = EXCLUDED.status,
                value_quantity   = EXCLUDED.value_quantity,
                interpretation   = EXCLUDED.interpretation,
                updated_at       = EXCLUDED.updated_at
        """
        self._upsert_partition(df, sql)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _upsert_partition(self, df: DataFrame, sql: str) -> None:
        """Execute ``sql`` for every row in each partition of ``df``.

        Each partition opens one connection and runs all rows in a single
        transaction.  On error the partition is rolled back and the exception
        propagates to the Spark task, triggering a retry.

        Args:
            df: Source DataFrame.
            sql: Parameterised psycopg2 SQL string.
        """
        dsn = self._cfg.dsn()

        def write_partition(rows: Iterator[Row]) -> None:
            conn = psycopg2.connect(dsn)
            try:
                with conn:
                    with conn.cursor() as cur:
                        for row in rows:
                            cur.execute(sql, row.asDict())
            finally:
                conn.close()

        df.foreachPartition(write_partition)

    @classmethod
    def from_settings(cls, settings: Any) -> PostgresWriter:
        """Construct from the project Settings object.

        Args:
            settings: ``libs.common.config.Settings`` instance.

        Returns:
            Configured ``PostgresWriter``.
        """
        pg = settings.postgres
        cfg = PostgresConfig(
            host=pg.host,
            port=pg.port,
            database=pg.db,
            user=pg.user,
            password=pg.password.get_secret_value(),
        )
        return cls(cfg)
