"""Shared pytest configuration.

Auto-applies the ``unit`` / ``integration`` markers based on a test's location
so the suites can be selected with ``-m unit`` / ``-m integration`` without every
module having to declare ``pytestmark`` by hand.

Also seeds dummy secrets into the environment (before any application module is
imported) so that ``libs.common.config`` settings — which require secrets from
the environment — can be constructed under test without a real ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Test environment secrets (must be set before app modules import config) ────
_ENCRYPTION_KEY = (
    "AfwbrqjgC2GZhjmpDwEk-C2pek3soYubE1PblA55wUs="  # gitleaks:allow test-only dummy Fernet key
)
os.environ.setdefault("POSTGRES_PASSWORD", "test_postgres_pw")
os.environ.setdefault("NEO4J_PASSWORD", "test_neo4j_pw")
os.environ.setdefault("MINIO_SECRET_KEY", "test_minio_secret")
os.environ.setdefault("REDIS_PASSWORD", "test_redis_pw")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_at_least_32_chars_long")
os.environ.setdefault("APP_SECRET_KEY", "test_app_secret_key_at_least_32_chars_long")
os.environ.setdefault("ENCRYPTION_KEY", _ENCRYPTION_KEY)

import pytest  # noqa: E402 — imported after env setup on purpose

_TESTS_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tag collected tests with a marker derived from their sub-directory."""
    for item in items:
        try:
            rel = Path(str(item.fspath)).relative_to(_TESTS_ROOT)
        except ValueError:
            continue
        top = rel.parts[0] if rel.parts else ""
        if top == "unit":
            item.add_marker(pytest.mark.unit)
        elif top == "integration":
            item.add_marker(pytest.mark.integration)
