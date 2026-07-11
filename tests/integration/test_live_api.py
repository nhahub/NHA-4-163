"""Smoke tests for the live API container."""

from __future__ import annotations

import pytest

from tests.integration.conftest import LiveServiceEndpoints, _http_get

pytestmark = pytest.mark.integration


def test_api_health_endpoint(live_endpoints: LiveServiceEndpoints) -> None:
    """The live API should answer /health with an OK-style payload."""
    response = _http_get(f"{live_endpoints.api_base_url}/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded", "healthy"}
    assert "components" in payload


def test_api_ready_endpoint(live_endpoints: LiveServiceEndpoints) -> None:
    """The live API should answer /ready and report component status."""
    response = _http_get(f"{live_endpoints.api_base_url}/ready")
    assert response.status_code in {200, 503}
    payload = response.json()
    assert payload["status"] in {"ok", "degraded", "ready"}
    assert {"api", "redis", "postgres", "neo4j", "model"}.issubset(payload["components"].keys())


def test_api_metrics_endpoint_exposes_prometheus_text(live_endpoints: LiveServiceEndpoints) -> None:
    """The live API should expose Prometheus metrics text."""
    response = _http_get(f"{live_endpoints.api_base_url}/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")
    assert "# HELP" in response.text or "# TYPE" in response.text
