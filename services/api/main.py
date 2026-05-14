"""FastAPI application factory for the Healthcare Prediction API.

Exposes:
  - POST /auth/token                             — JWT authentication
  - POST /predict/hereditary-risk               — hereditary risk prediction
  - POST /predict/disease-from-symptoms         — symptom differential
  - POST /predict/disease-from-prescription     — prescription differential
  - GET  /patient/{patient_id}/family-risk-profile
  - GET  /health
  - GET  /ready
  - GET  /metrics                               — Prometheus scrape endpoint

The lifespan context manager initialises application-scoped singletons
(Redis cache client, MLflow model) before the first request arrives and
tears them down cleanly on shutdown.

Middleware stack (outermost → innermost):
  CORSMiddleware → AuditLogMiddleware → RateLimitMiddleware
  → PrometheusMetricsMiddleware → routes

Environment variables (required unless noted):
  MLFLOW_TRACKING_URI          MLflow tracking server URI
  MODEL_NAME                   Registry model name (default: hereditary-risk-xgboost)
  MODEL_STAGE                  Registry stage (default: Staging)
  REDIS_HOST / REDIS_PORT /
  REDIS_PASSWORD               Redis connection (see libs.common.config)
  POSTGRES_* / NEO4J_*         Database connections
  APP_ENV                      development | staging | production
  JWT_SECRET_KEY               JWT signing secret (min 32 chars)
  ENCRYPTION_KEY               Fernet key for PHI field encryption
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from libs.common.config import get_settings
from libs.common.logging import configure_logging
from services.api.middleware.audit import AuditLogMiddleware
from services.api.middleware.metrics import MODEL_LOADED, PrometheusMetricsMiddleware
from services.api.middleware.rate_limit import RateLimitMiddleware
from services.api.routers import health, patients, predictions
from services.api.routers.auth import router as auth_router
from services.api.routers.conditions import router as conditions_router
from services.api.routers.family import router as family_router
from services.api.routers.medications import router as medications_router
from services.api.routers.metrics_router import router as metrics_router
from services.api.routers.patient_crud import router as patient_crud_router
from services.api.services.cache_service import CacheService
from services.api.services.model_service import ModelService

log = logging.getLogger(__name__)

_MODEL_NAME = os.environ.get("MODEL_NAME", "hereditary-risk-xgboost")
_MODEL_STAGE = os.environ.get("MODEL_STAGE", "Staging")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise and tear down application-scoped singletons.

    Runs at startup:
    1. Configure structured JSON logging.
    2. Connect to Redis and store CacheService on app.state.
    3. Load ML model from MLflow Model Registry into ModelService.
    4. Set model_loaded Prometheus gauge.

    Runs at shutdown:
    5. Clear model_loaded gauge.
    6. Close the Redis connection pool.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the application (request handling phase).
    """
    configure_logging()

    settings = get_settings()

    # ── Redis ─────────────────────────────────────────────────────────────────
    import redis.asyncio as aioredis

    log.info("Connecting to Redis at %s:%d", settings.redis.host, settings.redis.port)
    redis_client: aioredis.Redis = aioredis.from_url(  # type: ignore[type-arg]
        settings.redis.url,
        encoding="utf-8",
        decode_responses=True,
    )
    cache = CacheService(redis_client)
    app.state.cache_service = cache
    log.info("Redis connected")

    # ── ML model ──────────────────────────────────────────────────────────────
    model = ModelService()
    app.state.model_service = model
    MODEL_LOADED.set(0)
    try:
        model.load(
            tracking_uri=str(settings.mlflow.tracking_uri),
            model_name=_MODEL_NAME,
            stage=_MODEL_STAGE,
        )
        MODEL_LOADED.set(1)
        log.info("Model loaded: %s/%s", _MODEL_NAME, _MODEL_STAGE)
    except Exception as exc:
        log.warning("Model failed to load at startup: %s", exc)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    MODEL_LOADED.set(0)
    await redis_client.aclose()
    log.info("Redis connection closed")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application.

    All middleware and routers are registered here.  This function is
    importable by tests (which can override ``app.state`` directly) and
    by the Docker ASGI entry-point.

    Returns:
        Configured FastAPI instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Healthcare Hereditary Disease Prediction API",
        version="1.0.0",
        description=(
            "Predicts hereditary disease risk from patient demographics, "
            "comorbidities, medications, and family graph structure. "
            "HIPAA-compliant. JWT authentication required."
        ),
        lifespan=_lifespan,
        docs_url="/docs" if settings.app.env != "production" else None,
        redoc_url="/redoc" if settings.app.env != "production" else None,
        openapi_url="/openapi.json" if settings.app.env != "production" else None,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    origins = (
        ["*"]
        if settings.app.env == "development"
        else ["https://healthcare-internal.example.com"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # ── Audit logging ─────────────────────────────────────────────────────────
    app.add_middleware(AuditLogMiddleware, dsn=settings.postgres.sync_dsn)

    # ── Rate limiting ─────────────────────────────────────────────────────────
    app.add_middleware(RateLimitMiddleware)

    # ── Prometheus metrics (innermost — measures true handler latency) ─────────
    app.add_middleware(PrometheusMetricsMiddleware)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(metrics_router)   # /metrics — no auth, excluded from audit
    app.include_router(auth_router)
    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(patients.router)
    app.include_router(patient_crud_router)   # Patient CRUD
    app.include_router(conditions_router)     # Condition CRUD
    app.include_router(family_router)         # Family relationship CRUD
    app.include_router(medications_router)    # Medication CRUD

    return app


# ASGI entry-point — used by uvicorn and BentoML
app = create_app()
