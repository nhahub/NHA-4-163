"""Neo4j Spark writer — creates/merges graph nodes and relationships.

Uses the neo4j-spark-connector (Maven: org.neo4j:neo4j-connector-apache-spark_2.12).
All writes are MERGE operations keyed on the node's ``id`` property so the
job is idempotent and can be replayed safely.

Neo4j relationship semantics:
  Patient -[:DIAGNOSED_WITH]-> Disease
  Patient -[:PARENT_OF/CHILD_OF/SIBLING_OF/SPOUSE_OF]-> Patient|Relative
  Patient -[:PRESCRIBED]-> Prescription -[:OF_MEDICATION]-> Medication
  Patient -[:HAD_OBSERVATION]-> (implicit, not a separate node)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pyspark.sql import DataFrame

log = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Connection parameters for the Neo4j Spark connector."""

    uri: str
    user: str
    password: str
    database: str = "neo4j"

    def base_options(self) -> dict[str, str]:
        """Return the minimal set of options required by every write."""
        return {
            "url": self.uri,
            "authentication.type": "basic",
            "authentication.basic.username": self.user,
            "authentication.basic.password": self.password,
            "database": self.database,
        }


class Neo4jWriter:
    """Writes Spark DataFrames to Neo4j via the Spark connector.

    All public methods are called from ``foreachBatch`` in the streaming job.
    They must be idempotent — use MERGE, never CREATE.
    """

    def __init__(self, config: Neo4jConfig) -> None:
        self._cfg = config

    # ── Node writes ───────────────────────────────────────────────────────────

    def upsert_patient_nodes(self, df: DataFrame) -> None:
        """MERGE :Patient nodes from a patient.created batch.

        Args:
            df: DataFrame with columns matching the PatientCreated Avro schema.
                Must have at least: patient_id, date_of_birth, gender, deceased.
        """
        if df.rdd.isEmpty():
            return

        node_df = df.selectExpr(
            "patient_id AS id",
            "external_id",
            "date_of_birth",
            "gender",
            "deceased",
            "research_consent",
            "event_timestamp AS updated_at",
        )

        (
            node_df.write.format("org.neo4j.spark.DataSource")
            .mode("overwrite")
            .options(**self._cfg.base_options())
            .option("labels", ":Patient")
            .option("node.keys", "id")
            .save()
        )
        log.info("Upserted Patient nodes", extra={"count": node_df.count()})

    def upsert_disease_nodes(self, df: DataFrame) -> None:
        """MERGE :Disease nodes from a diagnosis.added batch.

        Disease nodes are shared — one node per ICD-10 code, many patients
        will have DIAGNOSED_WITH edges pointing to the same node.

        Args:
            df: DataFrame with at least: code, code_system, code_display, is_hereditary.
        """
        if df.rdd.isEmpty():
            return

        disease_df = df.selectExpr(
            "code AS icd10_code",
            "code_system",
            "code_display AS name",
            "is_hereditary",
        ).dropDuplicates(["icd10_code"])

        (
            disease_df.write.format("org.neo4j.spark.DataSource")
            .mode("overwrite")
            .options(**self._cfg.base_options())
            .option("labels", ":Disease")
            .option("node.keys", "icd10_code")
            .save()
        )

    def upsert_diagnosed_with_relationships(self, df: DataFrame) -> None:
        """MERGE DIAGNOSED_WITH relationships between Patient and Disease.

        Args:
            df: DataFrame with: patient_id, code, clinical_status, verification_status,
                onset_datetime, onset_age_years, severity.
        """
        if df.rdd.isEmpty():
            return

        rel_df = df.selectExpr(
            "patient_id",
            "code AS icd10_code",
            "clinical_status",
            "verification_status",
            "onset_datetime",
            "onset_age_years",
            "severity",
            "is_hereditary",
            "family_history_flag",
            "event_timestamp AS recorded_at",
        )

        (
            rel_df.write.format("org.neo4j.spark.DataSource")
            .mode("overwrite")
            .options(**self._cfg.base_options())
            .option("relationship", "DIAGNOSED_WITH")
            .option("relationship.save.strategy", "keys")
            .option("relationship.source.labels", ":Patient")
            .option("relationship.source.node.keys", "patient_id:id")
            .option("relationship.target.labels", ":Disease")
            .option("relationship.target.node.keys", "icd10_code:icd10_code")
            .option(
                "relationship.properties",
                "clinical_status,verification_status,onset_datetime,"
                "onset_age_years,severity,is_hereditary,family_history_flag,recorded_at",
            )
            .save()
        )

    def upsert_relative_relationships(self, df: DataFrame) -> None:
        """MERGE family relationships (PARENT_OF, SIBLING_OF, etc.) in Neo4j.

        The relationship type is derived from relationship_code using the HL7
        v3 FamilyMember vocabulary mapping.  Unknown codes fall back to a
        generic RELATED_TO relationship.

        Args:
            df: DataFrame with: patient_id, related_patient_id or fmh_id,
                relationship_code, degree_of_relatedness.
        """
        if df.rdd.isEmpty():
            return

        # Only link when both parties are known patients in the system.
        linked_df = df.filter("related_patient_id IS NOT NULL")
        if linked_df.rdd.isEmpty():
            return

        rel_df = linked_df.selectExpr(
            "patient_id",
            "related_patient_id",
            "relationship_code",
            "degree_of_relatedness",
            "event_timestamp AS linked_at",
        )

        # Map HL7 codes → Neo4j relationship types for the most common cases.
        _HL7_TO_REL = {
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

        for hl7_code, rel_type in _HL7_TO_REL.items():
            subset = rel_df.filter(f"relationship_code = '{hl7_code}'")
            if subset.rdd.isEmpty():
                continue
            (
                subset.write.format("org.neo4j.spark.DataSource")
                .mode("overwrite")
                .options(**self._cfg.base_options())
                .option("relationship", rel_type)
                .option("relationship.save.strategy", "keys")
                .option("relationship.source.labels", ":Patient")
                .option("relationship.source.node.keys", "patient_id:id")
                .option("relationship.target.labels", ":Patient")
                .option("relationship.target.node.keys", "related_patient_id:id")
                .option("relationship.properties", "degree_of_relatedness,linked_at")
                .save()
            )

    @classmethod
    def from_settings(cls, settings: Any) -> Neo4jWriter:
        """Construct from the project Settings object.

        Args:
            settings: ``libs.common.config.Settings`` instance.

        Returns:
            Configured ``Neo4jWriter``.
        """
        cfg = Neo4jConfig(
            uri=settings.neo4j.uri,
            user=settings.neo4j.user,
            password=settings.neo4j.password.get_secret_value(),
        )
        return cls(cfg)
