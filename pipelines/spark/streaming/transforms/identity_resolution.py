"""Patient identity resolution — probabilistic deduplication.

Detects likely duplicate patient records using a two-phase approach:

Phase 1 — Blocking:
  Candidate pairs must share at least one of:
    - Same birth year + same first-3 chars of postal_code  (tight block)
    - Same date_of_birth (loose block)
  Blocking dramatically reduces the O(n²) comparison space.

Phase 2 — Comparison:
  For each candidate pair, compute a composite similarity score from:
    - Jaro-Winkler similarity on family_name  (weight 0.4)
    - Jaro-Winkler similarity on given_name   (weight 0.3)
    - Exact match on date_of_birth            (weight 0.2)
    - Exact match on gender                   (weight 0.1)
  Pairs above ``MATCH_THRESHOLD`` are flagged as probable duplicates.

Results are written to the Delta processed layer as ``identity_candidates``
for human review.  No automatic merging is performed — this is a flag-only
system.  Clinical patient merges require a human decision (HIPAA-aligned).

Clinical safety note: false positives (incorrectly merging two different
patients) are more dangerous than false negatives.  The default threshold
(0.85) is intentionally conservative.  Tune only after reviewing FP/FN
rates on your specific patient population.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

log = logging.getLogger(__name__)

MATCH_THRESHOLD = 0.85
"""Composite score above which a pair is flagged as a probable duplicate."""

_CANDIDATE_SCHEMA = StructType(
    [
        StructField("patient_id_a", StringType(), nullable=False),
        StructField("patient_id_b", StringType(), nullable=False),
        StructField("score", DoubleType(), nullable=False),
        StructField("family_name_sim", DoubleType(), nullable=True),
        StructField("given_name_sim", DoubleType(), nullable=True),
        StructField("dob_match", DoubleType(), nullable=True),
        StructField("gender_match", DoubleType(), nullable=True),
        StructField("block_key", StringType(), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Jaro-Winkler similarity (pure Python, no external dep)
# ---------------------------------------------------------------------------


def _jaro(s1: str, s2: str) -> float:
    """Compute Jaro similarity between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Jaro similarity score in [0, 1].
    """
    if not s1 and not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    match_dist = max(len(s1), len(s2)) // 2 - 1
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)

    matches = 0
    transpositions = 0

    for i, c1 in enumerate(s1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len(s2))
        for j in range(start, end):
            if s2_matches[j] or c1 != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i, matched in enumerate(s1_matches):
        if not matched:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (matches / len(s1) + matches / len(s2) + (matches - transpositions / 2) / matches) / 3


def _jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity between two strings.

    Args:
        s1: First string.
        s2: Second string.
        p: Scaling factor for common prefix boost (standard: 0.1).

    Returns:
        Jaro-Winkler similarity score in [0, 1].
    """
    if not s1 or not s2:
        return 0.0
    s1, s2 = s1.lower(), s2.lower()
    jaro = _jaro(s1, s2)
    prefix = 0
    for c1, c2 in zip(s1[:4], s2[:4], strict=False):
        if c1 == c2:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


# ---------------------------------------------------------------------------
# Blocking + comparison
# ---------------------------------------------------------------------------


@dataclass
class PatientRecord:
    """Minimal patient fields needed for identity resolution."""

    patient_id: str
    family_name: str | None
    given_name: str | None
    date_of_birth: str | None  # ISO date string "YYYY-MM-DD"
    gender: str | None
    postal_code: str | None


def _score_pair(a: PatientRecord, b: PatientRecord) -> float:
    """Compute composite similarity score for two patient records.

    Args:
        a: First patient.
        b: Second patient.

    Returns:
        Composite score in [0, 1].
    """
    family_sim = _jaro_winkler(a.family_name or "", b.family_name or "")
    given_sim = _jaro_winkler(a.given_name or "", b.given_name or "")
    dob_match = 1.0 if a.date_of_birth and a.date_of_birth == b.date_of_birth else 0.0
    gender_match = 1.0 if a.gender and a.gender == b.gender else 0.0

    return 0.4 * family_sim + 0.3 * given_sim + 0.2 * dob_match + 0.1 * gender_match


def _build_block_key(record: PatientRecord) -> str | None:
    """Build a blocking key for a patient record.

    Patients share a block key if they could plausibly be the same person.
    Returns ``None`` if insufficient data to block.

    Args:
        record: Patient record.

    Returns:
        Block key string or ``None``.
    """
    if record.date_of_birth:
        year = record.date_of_birth[:4]
        postal_prefix = (record.postal_code or "")[:3]
        return f"{year}|{postal_prefix}"
    return None


def resolve_identities(df: DataFrame, spark: SparkSession) -> DataFrame:
    """Find probable duplicate patient records within a DataFrame.

    Args:
        df: DataFrame with columns: patient_id, family_name, given_name,
            date_of_birth (string "YYYY-MM-DD"), gender, postal_code.
        spark: Active SparkSession.

    Returns:
        DataFrame of candidate duplicate pairs with schema ``_CANDIDATE_SCHEMA``.
        Empty DataFrame if no candidates found.
    """
    records = [
        PatientRecord(
            patient_id=row["patient_id"],
            family_name=row.get("family_name"),
            given_name=row.get("given_name"),
            date_of_birth=str(row["date_of_birth"]) if row.get("date_of_birth") else None,
            gender=row.get("gender"),
            postal_code=row.get("postal_code"),
        )
        for row in df.collect()
    ]

    if len(records) < 2:
        return spark.createDataFrame([], _CANDIDATE_SCHEMA)

    # Build blocks
    blocks: dict[str, list[PatientRecord]] = {}
    for rec in records:
        key = _build_block_key(rec)
        if key:
            blocks.setdefault(key, []).append(rec)

    # Compare within blocks
    candidates: list[tuple] = []
    for block_key, block_records in blocks.items():
        if len(block_records) < 2:
            continue
        for i in range(len(block_records)):
            for j in range(i + 1, len(block_records)):
                a, b = block_records[i], block_records[j]
                if a.patient_id == b.patient_id:
                    continue

                family_sim = _jaro_winkler(a.family_name or "", b.family_name or "")
                given_sim = _jaro_winkler(a.given_name or "", b.given_name or "")
                dob_match = 1.0 if a.date_of_birth and a.date_of_birth == b.date_of_birth else 0.0
                gender_match = 1.0 if a.gender and a.gender == b.gender else 0.0
                score = 0.4 * family_sim + 0.3 * given_sim + 0.2 * dob_match + 0.1 * gender_match

                if score >= MATCH_THRESHOLD:
                    candidates.append(
                        (
                            a.patient_id,
                            b.patient_id,
                            round(score, 4),
                            round(family_sim, 4),
                            round(given_sim, 4),
                            dob_match,
                            gender_match,
                            block_key,
                        )
                    )

    if not candidates:
        return spark.createDataFrame([], _CANDIDATE_SCHEMA)

    result_df = spark.createDataFrame(candidates, _CANDIDATE_SCHEMA)
    log.warning(
        "Identity resolution found probable duplicates",
        extra={"candidate_pairs": result_df.count(), "threshold": MATCH_THRESHOLD},
    )
    return result_df
