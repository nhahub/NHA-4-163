"""Graph feature extraction using Neo4j Cypher and optional GDS algorithms.

Computes per-patient features from the patient family-relationship graph:

* **Weighted family disease prevalence** — sum of Wright coefficients for all
  relatives (up to 4 hops) who have any hereditary disease diagnosis.
  Weights: 1st-degree = 0.5, 2nd-degree = 0.25, 3rd-degree = 0.125,
  4th-degree = 0.0625.  Each relative is counted once at their closest
  degree to avoid double-counting via multiple paths.

* **Affected relative counts** — total, 1st-degree, 2nd-degree.

* **Shortest path depth** — minimum hop count to the nearest affected
  relative (-1 when none exists within 4 hops).

* **Family network size** — distinct relatives reachable within 4 hops.

* **Family clustering coefficient** — via ``gds.localClusteringCoefficient``
  written to the ``gds_clustering_coefficient`` node property before this
  function is called.  Defaults to 0.0 when GDS is unavailable.

All features are computed with **bulk Cypher queries** (one query covers all
patients) to avoid the N+1 round-trip overhead of per-patient queries.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from neo4j import Driver, GraphDatabase

log = logging.getLogger(__name__)

# ── Cypher constants ──────────────────────────────────────────────────────────

_FAMILY_RELS = "HAS_RELATIVE|IS_PARENT_OF|HAS_CHILD|IS_SIBLING_OF"

# Returns one row per patient with prevalence, count, and degree breakdowns.
# MIN(size(rels)) gives the *closest* degree when multiple paths exist.
_PREVALENCE_QUERY = f"""
MATCH (p:Patient)
OPTIONAL MATCH (p)-[rels:{_FAMILY_RELS}*1..4]-(relative)
WHERE (relative:Patient OR relative:Relative)
  AND (relative)-[:DIAGNOSED_WITH]->(:Disease)
WITH p, relative, MIN(size(rels)) AS min_depth
WITH p,
  count(DISTINCT relative) AS affected_relatives_count,
  sum(
    CASE min_depth
      WHEN 1 THEN 0.5
      WHEN 2 THEN 0.25
      WHEN 3 THEN 0.125
      ELSE 0.0625
    END
  ) AS weighted_family_prevalence,
  sum(CASE WHEN min_depth = 1 THEN 1 ELSE 0 END) AS first_degree_affected_count,
  sum(CASE WHEN min_depth = 2 THEN 1 ELSE 0 END) AS second_degree_affected_count
RETURN p.id AS patient_id,
       COALESCE(affected_relatives_count, 0)    AS affected_relatives_count,
       COALESCE(weighted_family_prevalence, 0.0) AS weighted_family_prevalence,
       COALESCE(first_degree_affected_count, 0)  AS first_degree_affected_count,
       COALESCE(second_degree_affected_count, 0) AS second_degree_affected_count
"""

# Returns shortest path length to any affected relative (-1 = no path found).
_SHORTEST_PATH_QUERY = f"""
MATCH (p:Patient)
CALL {{
  WITH p
  OPTIONAL MATCH path = shortestPath(
    (p)-[:{_FAMILY_RELS}*1..4]-(affected)
  )
  WHERE (affected:Patient OR affected:Relative)
    AND (affected)-[:DIAGNOSED_WITH]->(:Disease)
    AND affected <> p
  RETURN path
  LIMIT 1
}}
RETURN p.id AS patient_id,
       CASE WHEN path IS NULL THEN -1 ELSE length(path) END
         AS shortest_path_to_affected
"""

# Returns family network size (distinct relatives within 4 hops).
_FAMILY_SIZE_QUERY = f"""
MATCH (p:Patient)
OPTIONAL MATCH (p)-[:{_FAMILY_RELS}*1..4]-(relative)
WHERE relative:Patient OR relative:Relative
WITH p, count(DISTINCT relative) AS family_size
RETURN p.id AS patient_id, COALESCE(family_size, 0) AS family_size
"""

# Reads the GDS-written clustering coefficient property.
_CLUSTERING_QUERY = """
MATCH (p:Patient)
RETURN p.id AS patient_id,
       COALESCE(p.gds_clustering_coefficient, 0.0) AS family_clustering_coefficient
"""

# GDS graph projection name (ephemeral — dropped after write).
_GDS_GRAPH_NAME = "patient_family_graph"

_GDS_PROJECT_QUERY = f"""
CALL gds.graph.project(
  '{_GDS_GRAPH_NAME}',
  ['Patient', 'Relative'],
  {{
    HAS_RELATIVE:  {{ orientation: 'UNDIRECTED' }},
    IS_PARENT_OF:  {{ orientation: 'UNDIRECTED' }},
    HAS_CHILD:     {{ orientation: 'UNDIRECTED' }},
    IS_SIBLING_OF: {{ orientation: 'UNDIRECTED' }}
  }}
)
YIELD graphName, nodeCount, relationshipCount
"""

_GDS_WRITE_QUERY = f"""
CALL gds.localClusteringCoefficient.write(
  '{_GDS_GRAPH_NAME}',
  {{ writeProperty: 'gds_clustering_coefficient' }}
)
YIELD nodeCount, averageClusteringCoefficient
"""

_GDS_DROP_QUERY = f"CALL gds.graph.drop('{_GDS_GRAPH_NAME}') YIELD graphName"


# ── TypedDict for type-safe return values ─────────────────────────────────────


class GraphFeatureRow(TypedDict):
    """One row in the graph features output."""

    patient_id: str
    affected_relatives_count: int
    weighted_family_prevalence: float
    first_degree_affected_count: int
    second_degree_affected_count: int
    shortest_path_to_affected: int
    family_size: int
    family_clustering_coefficient: float


# ── Public API ────────────────────────────────────────────────────────────────


def run_gds_write_projection(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> bool:
    """Project the family graph and write GDS clustering coefficients.

    Projects 'Patient' and 'Relative' nodes with all family relationship
    types (undirected), runs ``localClusteringCoefficient``, and writes
    the result to the ``gds_clustering_coefficient`` node property.

    Args:
        neo4j_uri: Bolt URI for Neo4j.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.

    Returns:
        True if GDS projection succeeded, False if GDS is unavailable.
        On False, graph feature extraction will default clustering to 0.0.
    """
    driver: Driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session() as session:
            # Drop stale projection if present from a previous failed run.
            try:
                session.run(_GDS_DROP_QUERY)
            except Exception as exc:
                # No projection to drop — safe to ignore.
                log.debug("GDS drop projection skipped: %s", exc)

            result = session.run(_GDS_PROJECT_QUERY).single()
            node_count = result["nodeCount"] if result else 0
            log.info("GDS graph projected: %d nodes", node_count)

            write_result = session.run(_GDS_WRITE_QUERY).single()
            avg_cc = write_result["averageClusteringCoefficient"] if write_result else 0.0
            log.info("GDS clustering coefficient written, average=%.4f", avg_cc)

            session.run(_GDS_DROP_QUERY)
            return True
    except Exception:
        log.warning(
            "Neo4j GDS unavailable — family_clustering_coefficient will default to 0.0; "
            "install the GDS plugin to enable this feature"
        )
        return False
    finally:
        driver.close()


def extract_all_graph_features(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> list[GraphFeatureRow]:
    """Extract graph features for all patients in bulk.

    Runs three Cypher queries that each return one row per Patient node,
    then merges the results by patient_id.  Missing patients in any query
    result default to zeros.

    Args:
        neo4j_uri: Bolt URI for Neo4j.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.

    Returns:
        List of GraphFeatureRow dicts, one per patient node in Neo4j.

    Raises:
        neo4j.exceptions.ServiceUnavailable: when Neo4j is unreachable.
    """
    driver: Driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session() as session:
            prevalence_rows: dict[str, dict[str, Any]] = {
                r["patient_id"]: dict(r) for r in session.run(_PREVALENCE_QUERY)
            }
            path_rows: dict[str, int] = {
                r["patient_id"]: int(r["shortest_path_to_affected"])
                for r in session.run(_SHORTEST_PATH_QUERY)
            }
            size_rows: dict[str, int] = {
                r["patient_id"]: int(r["family_size"]) for r in session.run(_FAMILY_SIZE_QUERY)
            }
            clustering_rows: dict[str, float] = {
                r["patient_id"]: float(r["family_clustering_coefficient"])
                for r in session.run(_CLUSTERING_QUERY)
            }

        all_ids = set(prevalence_rows) | set(path_rows) | set(size_rows) | set(clustering_rows)
        results: list[GraphFeatureRow] = []
        for pid in all_ids:
            prev = prevalence_rows.get(pid, {})
            results.append(
                GraphFeatureRow(
                    patient_id=pid,
                    affected_relatives_count=int(prev.get("affected_relatives_count", 0)),
                    weighted_family_prevalence=float(prev.get("weighted_family_prevalence", 0.0)),
                    first_degree_affected_count=int(prev.get("first_degree_affected_count", 0)),
                    second_degree_affected_count=int(prev.get("second_degree_affected_count", 0)),
                    shortest_path_to_affected=path_rows.get(pid, -1),
                    family_size=size_rows.get(pid, 0),
                    family_clustering_coefficient=clustering_rows.get(pid, 0.0),
                )
            )
        return results
    finally:
        driver.close()


def depth_to_weight(depth: int) -> float:
    """Convert pedigree hop depth to Wright relatedness coefficient.

    Args:
        depth: Number of graph hops (1 = parent/sibling/child, etc.).

    Returns:
        Genetic relatedness coefficient (0.5 → 0.0625).
    """
    return 0.5 ** max(depth, 1)
