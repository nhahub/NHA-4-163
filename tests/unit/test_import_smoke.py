"""Import-smoke tests.

Importing every application module executes its module-level code — schema field
definitions, router registration, constants and class bodies — which both guards
against import-time regressions and gives the pure-declarative modules (Pydantic
schemas especially) their baseline coverage.
"""

from __future__ import annotations

import importlib

import pytest

# Modules that import cleanly without a running service or GPU/Spark runtime.
_APP_MODULES = [
    # FastAPI app (pulls in every router + middleware on import)
    "services.api.main",
    "services.api.db",
    "services.api.deps",
    # Schemas (pure Pydantic — import covers the field definitions)
    "services.api.schemas.crud_schemas",
    "services.api.schemas.requests",
    "services.api.schemas.responses",
    "services.api.schemas.fhir_schemas",
    "services.api.schemas.genetics_schemas",
    "services.api.schemas.cascade_schemas",
    "services.api.schemas.consent_schemas",
    "services.api.schemas.portal_schemas",
    "services.api.schemas.inheritance_schemas",
    "services.api.schemas.guideline_schemas",
    "services.api.schemas.monitoring_schemas",
    "services.api.schemas.notification_schemas",
    "services.api.schemas.org_schemas",
    "services.api.schemas.pedigree_schemas",
    "services.api.schemas.prs_schemas",
    "services.api.schemas.whatif_schemas",
    # Middleware
    "services.api.middleware.rate_limit",
    "services.api.middleware.audit",
    "services.api.middleware.metrics",
    # Auth
    "services.api.auth.jwt",
    "services.api.auth.rbac",
    "services.api.auth.api_key",
    "services.api.auth.portal_auth",
    "services.api.auth.models",
    # Services
    "services.api.services.model_service",
    "services.api.services.cache_service",
    "services.api.services.feature_service",
    "services.api.services.monitoring_service",
    "services.api.services.notification_service",
    "services.api.services.consent_service",
    "services.ingestion.kafka_admin",
    # Shared libs
    "libs.common.quality",
    "libs.common.config",
]


@pytest.mark.parametrize("module_name", _APP_MODULES)
def test_module_imports(module_name: str) -> None:
    """Every listed application module imports without error."""
    module = importlib.import_module(module_name)
    assert module is not None


def test_fastapi_app_builds() -> None:
    """The FastAPI app object is constructed with its full router set."""
    from services.api.main import app

    assert app.title
    # A representative subset of routes should be registered.
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    assert any("/health" in p for p in paths)
