"""Live feature computation for the prediction API.

At inference time, features are computed on-the-fly from the operational
stores (Postgres + Neo4j), then cached in Redis for 24 hours.  This
avoids replicating the Spark feature engineering pipeline into the serving
layer while keeping per-request latency low (~20–40 ms uncached, < 2 ms
cached).

Feature computation mirrors Phase 4 logic but uses direct SQL/Cypher
rather than Spark DataFrames.

PHI note: patient demographics are used for feature computation (age from
DOB, gender encoding) but the raw values are never logged or stored in the
response; only the encoded numeric features are returned.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import date
from typing import Any

log = logging.getLogger(__name__)

_ICD10_CHAPTERS: dict[str, str] = {
    "A": "has_infectious",
    "B": "has_infectious",
    "C": "has_oncological",
    "D": "has_haematological",
    "E": "has_metabolic",
    "F": "has_mental_health",
    "G": "has_neurological",
    "I": "has_cardiovascular",
    "J": "has_respiratory",
    "K": "has_digestive",
    "M": "has_musculoskeletal",
    "N": "has_genitourinary",
}

_AGE_GROUP_ORDER = ["unknown", "pediatric", "young_adult", "middle_age", "older_adult", "elderly"]
_AGE_GROUP_ORDINAL = {g: i for i, g in enumerate(_AGE_GROUP_ORDER)}

_ACTIVE_STATUSES = ("active", "recurrence", "relapse")
_STOPPED_STATUSES = ("stopped", "cancelled", "entered-in-error", "on-hold")

_FAMILY_RELS = "HAS_RELATIVE|IS_PARENT_OF|HAS_CHILD|IS_SIBLING_OF"

# ── Postgres queries ──────────────────────────────────────────────────────────

# Table/column names follow the canonical ORM schema (singular tables).
# ``condition.code`` holds the diagnosis code (ICD-10 or SNOMED); the ICD-10
# chapter flag is derived from its leading letter — non-ICD codes yield no flag.
_DEMOGRAPHICS_SQL = """
SELECT date_of_birth, gender
FROM patient
WHERE id = %(patient_id)s AND deleted_at IS NULL
"""

_CONDITIONS_SQL = """
SELECT clinical_status, code AS icd10_code, is_hereditary
FROM condition
WHERE patient_id = %(patient_id)s
"""

_MEDICATIONS_SQL = """
SELECT status, medication_code
FROM medication_request
WHERE patient_id = %(patient_id)s
"""

# ── Neo4j queries ─────────────────────────────────────────────────────────────

_GRAPH_PREVALENCE_CYPHER = f"""
MATCH (p:Patient {{id: $pid}})
OPTIONAL MATCH (p)-[rels:{_FAMILY_RELS}*1..4]-(rel)
WHERE (rel:Patient OR rel:Relative)
  AND (rel)-[:DIAGNOSED_WITH]->(:Disease)
WITH rel, MIN(size(rels)) AS min_depth
RETURN
  count(DISTINCT rel)                                AS affected_relatives_count,
  sum(CASE min_depth
        WHEN 1 THEN 0.5 WHEN 2 THEN 0.25 WHEN 3 THEN 0.125 ELSE 0.0625
      END)                                           AS weighted_family_prevalence,
  sum(CASE WHEN min_depth = 1 THEN 1 ELSE 0 END)   AS first_degree_affected_count,
  sum(CASE WHEN min_depth = 2 THEN 1 ELSE 0 END)   AS second_degree_affected_count
"""

_GRAPH_PATH_CYPHER = f"""
MATCH (p:Patient {{id: $pid}})
CALL {{
  WITH p
  OPTIONAL MATCH path = shortestPath(
    (p)-[:{_FAMILY_RELS}*1..4]-(affected)
  )
  WHERE (affected:Patient OR affected:Relative)
    AND (affected)-[:DIAGNOSED_WITH]->(:Disease)
    AND affected <> p
  RETURN path LIMIT 1
}}
RETURN CASE WHEN path IS NULL THEN -1 ELSE length(path) END AS shortest_path_to_affected
"""

_GRAPH_SIZE_CYPHER = f"""
MATCH (p:Patient {{id: $pid}})
OPTIONAL MATCH (p)-[:{_FAMILY_RELS}*1..4]-(rel)
WHERE rel:Patient OR rel:Relative
RETURN count(DISTINCT rel) AS family_size
"""

_GRAPH_CLUSTER_CYPHER = """
MATCH (p:Patient {id: $pid})
RETURN COALESCE(p.gds_clustering_coefficient, 0.0) AS family_clustering_coefficient
"""

_FAMILY_PROFILE_CYPHER = f"""
MATCH (p:Patient {{id: $pid}})
OPTIONAL MATCH (p)-[r:{_FAMILY_RELS}]-(rel)
WHERE rel:Patient OR rel:Relative
OPTIONAL MATCH (rel)-[:DIAGNOSED_WITH]->(d:Disease)
WITH rel,
     type(r)                              AS rel_type,
     COALESCE(r.degree_of_relatedness, 0.5) AS degree,
     collect(DISTINCT d.icd10_code)       AS diseases
RETURN rel.id             AS relative_id,
       rel_type            AS relationship_code,
       degree              AS degree_of_relatedness,
       diseases            AS diagnosed_icd10_codes
ORDER BY degree DESC
LIMIT 50
"""


# ── Pure helpers ──────────────────────────────────────────────────────────────


def _compute_age_features(dob: date | None, as_of: date) -> tuple[int | None, str, int, int, int]:
    """Return (age_years, age_group_ordinal, gender_male, gender_female, gender_other)."""
    if dob is None:
        return None, "unknown", 0, 0, 0
    age = math.floor((as_of - dob).days / 365.25)
    if age < 18:
        grp = "pediatric"
    elif age < 35:
        grp = "young_adult"
    elif age < 50:
        grp = "middle_age"
    elif age < 65:
        grp = "older_adult"
    else:
        grp = "elderly"
    return age, grp, 0, 0, 0  # gender set by caller


def _chapter_flag(icd10_code: str) -> str | None:
    if not icd10_code:
        return None
    return _ICD10_CHAPTERS.get(icd10_code[0].upper())


# ── Main feature computation ──────────────────────────────────────────────────


def compute_features_sync(
    patient_id: str,
    postgres_dsn: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """Compute the full feature vector for a single patient synchronously.

    Runs three Postgres queries (demographics, conditions, medications)
    and four Neo4j Cypher queries to produce the same feature set as the
    Phase 4 Spark batch pipeline.

    Args:
        patient_id: Patient UUID string.
        postgres_dsn: PostgreSQL DSN (``postgresql://user:pw@host/db``).
        neo4j_uri: Bolt URI for Neo4j.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        as_of_date: Reference date for age calculation (ISO-8601).
            Defaults to today.

    Returns:
        Dict of feature names to values (matches Phase 4 / PatientFeatureVector fields).

    Raises:
        LookupError: If the patient_id is not found in Postgres.
    """
    import psycopg2
    import psycopg2.extras
    from neo4j import GraphDatabase

    as_of = date.fromisoformat(as_of_date) if as_of_date else date.today()
    pid = {"patient_id": patient_id}

    # ── Postgres ──────────────────────────────────────────────────────────────
    conn = psycopg2.connect(postgres_dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_DEMOGRAPHICS_SQL, pid)
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise LookupError(f"Patient not found: {patient_id}")

    dob = row["date_of_birth"]
    gender = (row["gender"] or "unknown").lower()
    age_years, age_grp, _, _, _ = _compute_age_features(dob, as_of)

    conn = psycopg2.connect(postgres_dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_CONDITIONS_SQL, pid)
            cond_rows = cur.fetchall()
            cur.execute(_MEDICATIONS_SQL, pid)
            med_rows = cur.fetchall()
    finally:
        conn.close()

    # ── Demographics features ─────────────────────────────────────────────────
    feats: dict[str, Any] = {
        "age_years": age_years,
        "age_group": _AGE_GROUP_ORDINAL.get(age_grp, 0),
        "gender_male": 1 if gender == "male" else 0,
        "gender_female": 1 if gender == "female" else 0,
        "gender_other_unknown": 1 if gender in ("other", "unknown") else 0,
    }

    # ── Comorbidity features ──────────────────────────────────────────────────
    active = [r for r in cond_rows if r["clinical_status"] in _ACTIVE_STATUSES]
    chapter_flags: dict[str, int] = dict.fromkeys(set(_ICD10_CHAPTERS.values()), 0)
    for r in active:
        flag = _chapter_flag(r["icd10_code"] or "")
        if flag:
            chapter_flags[flag] = 1

    feats.update(
        {
            "comorbidity_count": len(active),
            "hereditary_condition_count": sum(1 for r in active if r["is_hereditary"]),
            **chapter_flags,
        }
    )

    # ── Medication features ───────────────────────────────────────────────────
    n_active = sum(1 for r in med_rows if r["status"] == "active")
    n_completed = sum(1 for r in med_rows if r["status"] == "completed")
    n_stopped = sum(1 for r in med_rows if r["status"] in _STOPPED_STATUSES)
    n_distinct = len({r["medication_code"] for r in med_rows if r["medication_code"]})
    denom = n_completed + n_stopped
    feats.update(
        {
            "active_medication_count": n_active,
            "completed_medication_count": n_completed,
            "stopped_medication_count": n_stopped,
            "distinct_medication_count": n_distinct,
            "adherence_proxy": (n_completed / denom) if denom > 0 else None,
        }
    )

    # ── Graph features ────────────────────────────────────────────────────────
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session() as session:
            prev = dict(session.run(_GRAPH_PREVALENCE_CYPHER, pid=patient_id).single() or {})
            path_rec = session.run(_GRAPH_PATH_CYPHER, pid=patient_id).single()
            size_rec = session.run(_GRAPH_SIZE_CYPHER, pid=patient_id).single()
            clus_rec = session.run(_GRAPH_CLUSTER_CYPHER, pid=patient_id).single()
    finally:
        driver.close()

    feats.update(
        {
            "affected_relatives_count": int(prev.get("affected_relatives_count") or 0),
            "weighted_family_prevalence": float(prev.get("weighted_family_prevalence") or 0.0),
            "first_degree_affected_count": int(prev.get("first_degree_affected_count") or 0),
            "second_degree_affected_count": int(prev.get("second_degree_affected_count") or 0),
            "shortest_path_to_affected": int(
                path_rec["shortest_path_to_affected"] if path_rec else -1
            ),
            "family_size": int(size_rec["family_size"] if size_rec else 0),
            "family_clustering_coefficient": float(
                clus_rec["family_clustering_coefficient"] if clus_rec else 0.0
            ),
        }
    )

    return feats


async def compute_features(
    patient_id: str,
    postgres_dsn: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """Async wrapper — runs ``compute_features_sync`` in a thread pool.

    Args:
        patient_id: Patient UUID string.
        postgres_dsn: PostgreSQL DSN.
        neo4j_uri: Neo4j Bolt URI.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        as_of_date: Reference date (ISO-8601). Defaults to today.

    Returns:
        Feature dict.
    """
    return await asyncio.to_thread(
        compute_features_sync,
        patient_id,
        postgres_dsn,
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        as_of_date,
    )


async def get_family_profile_sync(
    patient_id: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> list[dict[str, Any]]:
    """Query Neo4j for first-degree relatives and their diagnoses.

    Args:
        patient_id: Patient UUID string.
        neo4j_uri: Bolt URI for Neo4j.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.

    Returns:
        List of relative record dicts.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session() as session:
            records = list(session.run(_FAMILY_PROFILE_CYPHER, pid=patient_id))
    finally:
        driver.close()

    return [
        {
            "relative_id": str(r["relative_id"] or ""),
            "relationship_code": str(r["relationship_code"] or ""),
            "degree_of_relatedness": float(r["degree_of_relatedness"] or 0.5),
            "diagnosed_icd10_codes": [c for c in (r["diagnosed_icd10_codes"] or []) if c],
        }
        for r in records
        if r["relative_id"]
    ]
