"""Patient-centric endpoints.

GET /patient/{patient_id}/family-risk-profile
    Returns the patient's family graph summary: relatives, disease burden
    by ICD-10 chapter, weighted prevalence, and an aggregate family risk
    score derived from the graph features.  Cached in Redis for 24 hours.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from libs.common.config import get_settings
from services.api.deps import CacheDep
from services.api.schemas.responses import (
    ChapterBurden,
    FamilyRiskProfileResponse,
    RelativeRecord,
    _risk_tier,
)
from services.api.services.feature_service import (
    _ICD10_CHAPTERS,
)

router = APIRouter(prefix="/patient", tags=["patients"])

# Map feature-flag names back to chapter label strings
_CHAPTER_LABEL: dict[str, str] = {v: v.removeprefix("has_") for v in set(_ICD10_CHAPTERS.values())}


def _build_profile_response(
    request_id: str,
    patient_id: str,
    relatives: list[dict[str, Any]],
    *,
    cached: bool,
) -> FamilyRiskProfileResponse:
    """Assemble a FamilyRiskProfileResponse from raw relative records.

    Computes:
    - ``family_risk_score`` as a weighted average of degree-of-relatedness
      among affected relatives (capped at 1.0).
    - ``disease_burden_by_chapter`` counts and weighted prevalences per
      ICD-10 chapter across all relatives.

    Args:
        request_id: Server-generated request UUID string.
        patient_id: Patient UUID string.
        relatives: List of relative record dicts from Neo4j.
        cached: Whether the response came from cache.

    Returns:
        Assembled FamilyRiskProfileResponse.
    """
    affected = [r for r in relatives if r["diagnosed_icd10_codes"]]
    affected_count = len(affected)

    # Weighted family risk score: sum of relatedness weights for affected relatives
    raw_score = sum(r["degree_of_relatedness"] for r in affected)
    family_risk_score = min(raw_score, 1.0)

    # Per-chapter burden
    chapter_burden: dict[str, dict[str, Any]] = {}
    for rel in affected:
        weight = rel["degree_of_relatedness"]
        seen_chapters: set[str] = set()
        for code in rel["diagnosed_icd10_codes"]:
            chapter_flag = _ICD10_CHAPTERS.get((code or "")[:1].upper())
            if chapter_flag and chapter_flag not in seen_chapters:
                seen_chapters.add(chapter_flag)
                entry = chapter_burden.setdefault(chapter_flag, {"count": 0, "weighted": 0.0})
                entry["count"] += 1
                entry["weighted"] += weight

    disease_burden_by_chapter = {
        _CHAPTER_LABEL.get(flag, flag): ChapterBurden(
            affected_relative_count=v["count"],
            weighted_prevalence=round(v["weighted"], 4),
        )
        for flag, v in chapter_burden.items()
    }

    first_degree = [
        RelativeRecord(
            relative_id=r["relative_id"],
            relationship_code=r["relationship_code"],
            degree_of_relatedness=r["degree_of_relatedness"],
            diagnosed_icd10_codes=r["diagnosed_icd10_codes"],
        )
        for r in relatives
        if r["degree_of_relatedness"] >= 0.5
    ]

    return FamilyRiskProfileResponse(
        request_id=uuid.UUID(request_id),
        patient_id=uuid.UUID(patient_id),
        family_risk_score=family_risk_score,
        risk_tier=_risk_tier(family_risk_score),
        family_size=len(relatives),
        affected_relatives_count=affected_count,
        first_degree_relatives=first_degree,
        disease_burden_by_chapter=disease_burden_by_chapter,
        cached=cached,
    )


@router.get(
    "/{patient_id}/family-risk-profile",
    response_model=FamilyRiskProfileResponse,
    summary="Family risk profile",
    description=(
        "Returns the patient's family disease burden: relatives, weighted "
        "hereditary prevalence, and an aggregate family risk score. "
        "Cached 24 hours per patient."
    ),
)
async def get_family_risk_profile(
    patient_id: uuid.UUID,
    cache: CacheDep,
    request: Request,
) -> FamilyRiskProfileResponse:
    """Return the family risk profile for a patient.

    **Cache behaviour**: results are cached in Redis for 24 hours keyed by
    ``patient_id``.  Graph changes (new diagnoses, new relatives) only
    propagate after the TTL expires.

    Args:
        patient_id: Patient UUID (path parameter).
        cache: Injected CacheService.
        request: FastAPI request (used for request_id).

    Returns:
        FamilyRiskProfileResponse.

    Raises:
        HTTPException 503: If Neo4j or Postgres are unreachable.
    """
    import asyncio
    import uuid as _uuid

    pid_str = str(patient_id)
    request_id = str(getattr(request.state, "request_id", _uuid.uuid4()))
    cache_key = cache.family_profile_key(pid_str)

    cached_data = await cache.get_json(cache_key)
    if cached_data:
        cached_data["cached"] = True
        cached_data["request_id"] = request_id
        return FamilyRiskProfileResponse(**cached_data)

    settings = get_settings()
    n4j = settings.neo4j

    try:
        relatives = await asyncio.to_thread(
            _get_family_profile_sync_wrapper,
            pid_str,
            n4j.uri,
            n4j.user,
            n4j.password.get_secret_value(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Graph query failed: {exc}",
        ) from exc

    response = _build_profile_response(request_id, pid_str, relatives, cached=False)

    payload = response.model_dump(mode="json")
    payload.pop("request_id", None)
    await cache.set_json(cache_key, payload, cache.TTL_FEATURES)

    return response


def _get_family_profile_sync_wrapper(
    patient_id: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> list[dict[str, Any]]:
    """Synchronous wrapper so asyncio.to_thread can call the Neo4j query.

    Args:
        patient_id: Patient UUID string.
        neo4j_uri: Bolt URI for Neo4j.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.

    Returns:
        List of relative record dicts.
    """
    from neo4j import GraphDatabase

    from services.api.services.feature_service import _FAMILY_PROFILE_CYPHER

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
