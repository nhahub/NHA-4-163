"""Health and readiness endpoints.

GET /health  — liveness probe: returns 200 if the process is alive.
GET /ready   — readiness probe: checks all downstream dependencies.

Kubernetes convention: liveness failure → restart container;
readiness failure → remove from load balancer without restarting.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from services.api.schemas.responses import ComponentStatus, HealthResponse

router = APIRouter(tags=["health"])

_VERSION = "0.6.0"  # Updated each phase


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 if the API process is running.",
)
async def health(request: Request) -> HealthResponse:
    """Simple liveness check — always returns ok if the process is up.

    Returns:
        HealthResponse with status 'ok'.
    """
    model_ok = request.app.state.model_service.is_loaded
    return HealthResponse(
        status="ok" if model_ok else "degraded",
        version=_VERSION,
        components={
            "api": ComponentStatus(status="ok"),
            "model": ComponentStatus(
                status="ok" if model_ok else "degraded",
                detail=None if model_ok else "Model not loaded",
            ),
        },
    )


@router.get(
    "/ready",
    summary="Readiness probe",
    description="Checks Postgres, Neo4j, Redis, and the ML model.",
)
async def ready(request: Request) -> JSONResponse:
    """Deep readiness check — verifies all dependencies are reachable.

    Returns:
        200 with full component status when all critical dependencies
        are healthy; 503 if any critical component is down.
    """
    from libs.common.config import get_settings

    settings = get_settings()
    components: dict[str, ComponentStatus] = {}
    # Report API itself as a component so readiness probes include it
    components["api"] = ComponentStatus(status="ok")
    overall_ok = True

    # ── Redis ─────────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    redis_ok = await request.app.state.cache_service.ping()
    components["redis"] = ComponentStatus(
        status="ok" if redis_ok else "down",
        latency_ms=round((time.monotonic() - t0) * 1000, 1),
    )
    if not redis_ok:
        overall_ok = False

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        import asyncio

        await asyncio.to_thread(_ping_postgres, settings.postgres.sync_dsn)
        components["postgres"] = ComponentStatus(
            status="ok", latency_ms=round((time.monotonic() - t0) * 1000, 1)
        )
    except Exception as exc:
        components["postgres"] = ComponentStatus(status="down", detail=str(exc)[:80])
        overall_ok = False

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        import asyncio

        await asyncio.to_thread(
            _ping_neo4j,
            settings.neo4j.uri,
            settings.neo4j.user,
            settings.neo4j.password.get_secret_value(),
        )
        components["neo4j"] = ComponentStatus(
            status="ok", latency_ms=round((time.monotonic() - t0) * 1000, 1)
        )
    except Exception as exc:
        components["neo4j"] = ComponentStatus(status="down", detail=str(exc)[:80])
        overall_ok = False

    # ── Model ─────────────────────────────────────────────────────────────────
    model_ok = request.app.state.model_service.is_loaded
    components["model"] = ComponentStatus(
        status="ok" if model_ok else "degraded",
        detail=None if model_ok else "No model loaded",
    )
    if not model_ok:
        overall_ok = False

    result = HealthResponse(
        status="ok" if overall_ok else "degraded",
        version=_VERSION,
        components=components,
    )
    code = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(content=result.model_dump(), status_code=code)


# ── Sync helpers (run in thread pool) ────────────────────────────────────────


def _ping_postgres(dsn: str) -> None:
    import psycopg2

    conn = psycopg2.connect(dsn, connect_timeout=3)
    conn.close()


def _ping_neo4j(uri: str, user: str, password: str) -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            session.run("RETURN 1").single()
    finally:
        driver.close()
