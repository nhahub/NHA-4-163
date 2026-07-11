"""Smoke tests for backing services used by the API."""

from __future__ import annotations

import os

import pytest

from tests.integration.conftest import LiveServiceEndpoints, _http_get

pytestmark = pytest.mark.integration


def test_mlflow_health(live_endpoints: LiveServiceEndpoints) -> None:
    """MLflow should be reachable at its root endpoint."""
    response = _http_get(live_endpoints.mlflow_base_url)
    assert response.status_code == 200
    assert "mlflow" in response.text.lower()


def test_neo4j_browser_responds(live_endpoints: LiveServiceEndpoints) -> None:
    """Neo4j Browser should respond on the local browser port."""
    response = _http_get(live_endpoints.neo4j_browser_url)
    assert response.status_code in {200, 302, 303}


def test_redis_port_is_open(live_endpoints: LiveServiceEndpoints) -> None:
    """Redis should accept TCP connections when Compose is healthy."""
    import redis

    client = redis.Redis(
        host=live_endpoints.redis_host,
        port=live_endpoints.redis_port,
        password=os.environ.get("REDIS_PASSWORD", "change_me_redis"),
        decode_responses=True,
    )
    try:
        assert client.ping() is True
    finally:
        client.close()
