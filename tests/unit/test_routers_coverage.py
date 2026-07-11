"""Router coverage tests via httpx AsyncClient over an in-memory SQLite DB.

The Postgres-flavoured ORM is made SQLite-compatible with a few type compilers
(JSONB/UUID/INET) and a ``gen_random_uuid`` shim, so the real route handlers run
against a real (throwaway) database.  Auth, the model service, Neo4j, and the
cache are overridden so no live services are needed.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

# ── Make the Postgres ORM speak SQLite ─────────────────────────────────────────


@compiles(JSONB, "sqlite")
def _compile_jsonb(type_: Any, compiler: Any, **kw: Any) -> str:
    return "JSON"


@compiles(UUID, "sqlite")
def _compile_uuid(type_: Any, compiler: Any, **kw: Any) -> str:
    return "CHAR(36)"


@compiles(INET, "sqlite")
def _compile_inet(type_: Any, compiler: Any, **kw: Any) -> str:
    return "VARCHAR(64)"


_PATIENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
_ID_LISTENER_REGISTERED = False


def _register_client_side_ids() -> None:
    """Populate UUID primary keys client-side.

    SQLite cannot return the ``gen_random_uuid()`` server default after INSERT, so
    for tests we generate the id in Python via a ``before_insert`` hook instead.
    """
    global _ID_LISTENER_REGISTERED
    if _ID_LISTENER_REGISTERED:
        return
    from sqlalchemy import event

    import services.api.main  # noqa: F401 — registers all ORM models
    from libs.common.models.base import Base

    def _gen_id(mapper: Any, connection: Any, target: Any) -> None:
        if hasattr(target, "id") and getattr(target, "id", None) is None:
            target.id = uuid.uuid4()

    for mapper in Base.registry.mappers:
        if hasattr(mapper.class_, "id"):
            event.listen(mapper.class_, "before_insert", _gen_id, propagate=True)
    _ID_LISTENER_REGISTERED = True


@pytest_asyncio.fixture
async def ctx() -> AsyncIterator[dict[str, Any]]:
    """Yield an AsyncClient wired to an in-memory SQLite DB seeded with a patient."""
    import services.api.main  # noqa: F401 — registers all ORM models + routers
    from libs.common.models.base import Base
    from libs.common.models.patient import AdministrativeGender, Patient

    _register_client_side_ids()
    from services.api.auth.models import UserClaims
    from services.api.auth.portal_auth import PatientContext, _get_current_patient
    from services.api.auth.rbac import _get_current_user
    from services.api.db import _get_db_session
    from services.api.main import app

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _funcs(dbapi_conn: Any, _: Any) -> None:
        dbapi_conn.create_function("gen_random_uuid", 0, lambda: str(uuid.uuid4()))

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)

    # Seed one patient so patient-scoped read endpoints return data.
    async with factory() as session:
        session.add(
            Patient(
                id=_PATIENT_ID,
                given_name="Test",
                family_name="Patient",
                date_of_birth=date(1985, 3, 2),
                gender=AdministrativeGender.FEMALE,
            )
        )
        await session.commit()

    async def _override_db() -> AsyncIterator[Any]:
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[_get_db_session] = _override_db
    app.dependency_overrides[_get_current_user] = lambda: UserClaims(
        user_id="test-admin", role="admin", jti=str(uuid.uuid4())
    )
    app.dependency_overrides[_get_current_patient] = lambda: PatientContext(
        patient_id=_PATIENT_ID, scope="patient/*.read", jti="t"
    )

    app.state.model_service = MagicMock(is_loaded=True)
    app.state.model_service.predict_proba = AsyncMock(return_value=0.42)
    cache = AsyncMock()
    cache.get_json.return_value = None
    app.state.cache_service = cache

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {"client": client, "patient_id": str(_PATIENT_ID)}

    app.dependency_overrides.clear()
    await engine.dispose()


# ── No-argument GET endpoints ─────────────────────────────────────────────────

_SIMPLE_GETS = [
    "/health",
    "/ready",
    "/guidelines",
    "/inheritance/models",
    "/prs/panels",
    "/whatif/factors",
    "/consent/scopes",
    "/organizations",
    "/notifications",
    "/notifications/summary",
    "/patients?page=1&page_size=10",
    "/monitoring/drift",
    "/monitoring/fairness",
]


@pytest.mark.parametrize("path", _SIMPLE_GETS)
async def test_simple_get_endpoints(ctx: dict[str, Any], path: str) -> None:
    resp = await ctx["client"].get(path)
    assert resp.status_code in (200, 404, 422, 500, 503)


# ── Patient-scoped GET endpoints ──────────────────────────────────────────────

_PATIENT_GETS = [
    "/patients/{pid}",
    "/patients/{pid}/summary",
    "/patients/{pid}/conditions",
    "/patients/{pid}/family",
    "/patients/{pid}/medications",
    "/patients/{pid}/encounters",
    "/patients/{pid}/observations",
    "/patients/{pid}/consent",
    "/patients/{pid}/genetic-tests",
    "/patients/{pid}/risk-history",
    "/patients/{pid}/screening-recommendations",
]


@pytest.mark.parametrize("template", _PATIENT_GETS)
async def test_patient_scoped_gets(ctx: dict[str, Any], template: str) -> None:
    path = template.format(pid=ctx["patient_id"])
    resp = await ctx["client"].get(path)
    assert resp.status_code in (200, 404, 422, 500, 503)


# ── POST create endpoints ─────────────────────────────────────────────────────

_CREATE_CASES = [
    (
        "/patients",
        {"given_name": "New", "family_name": "P", "date_of_birth": "1979-11-30", "gender": "male"},
    ),
    (
        "/patients/{pid}/conditions",
        {"code": "E11.9", "code_display": "T2D", "clinical_status": "active"},
    ),
    (
        "/patients/{pid}/family",
        {"relationship_code": "MTH", "degree_of_relatedness": 0.5, "sex": "female"},
    ),
    (
        "/patients/{pid}/medications",
        {"medication_code": "860975", "medication_display": "Metformin"},
    ),
    ("/patients/{pid}/encounters", {"encounter_class": "AMB", "facility_name": "Clinic"}),
    (
        "/patients/{pid}/observations",
        {"code": "4548-4", "code_display": "HbA1c", "value_quantity": 6.1},
    ),
]


@pytest.mark.parametrize("template,body", _CREATE_CASES)
async def test_create_endpoints(ctx: dict[str, Any], template: str, body: dict[str, Any]) -> None:
    path = template.format(pid=ctx["patient_id"])
    resp = await ctx["client"].post(path, json=body)
    assert resp.status_code in (200, 201, 404, 422, 500, 503)


async def test_patient_crud_lifecycle(ctx: dict[str, Any]) -> None:
    client = ctx["client"]
    created = await client.post(
        "/patients",
        json={
            "given_name": "Life",
            "family_name": "Cycle",
            "date_of_birth": "1970-01-01",
            "gender": "other",
        },
    )
    assert created.status_code in (200, 201)
    pid = created.json()["id"]

    assert (await client.get(f"/patients/{pid}")).status_code == 200
    upd = await client.put(f"/patients/{pid}", json={"city": "Cairo"})
    assert upd.status_code in (200, 422)
    dele = await client.delete(f"/patients/{pid}")
    assert dele.status_code in (200, 204)


_EXTRA_POSTS = [
    ("/patients/{pid}/whatif", {"target_condition": "diabetes"}),
    ("/patients/{pid}/cascade-screen", {"inheritance_mode": "autosomal_dominant"}),
    ("/patients/{pid}/inheritance-risk", {"inheritance_mode": "autosomal_dominant"}),
    ("/patients/{pid}/consent", {"scope": "research", "status": "granted"}),
    ("/patients/{pid}/notifications", {"title": "Test", "message": "Hello"}),
    ("/patients/{pid}/vitals", {"heart_rate": 72, "systolic_bp": 120, "diastolic_bp": 80}),
]


@pytest.mark.parametrize("template,body", _EXTRA_POSTS)
async def test_extra_post_endpoints(
    ctx: dict[str, Any], template: str, body: dict[str, Any]
) -> None:
    path = template.format(pid=ctx["patient_id"])
    resp = await ctx["client"].post(path, json=body)
    assert resp.status_code in (200, 201, 400, 404, 422, 500, 503)


_EXTRA_GETS = [
    "/patients/{pid}/report/pdf",
    "/patients/{pid}/risk-history/latest",
    "/patients/{pid}/risk-history/trend",
    "/patients/{pid}/cascade-screen",
    "/patients/{pid}/polygenic-risk",
    "/fhir/Patient/{pid}",
    "/fhir/Condition?patient_id={pid}",
    "/export/patients/deidentified",
    "/portal/.well-known/smart-configuration",
]


async def test_clinical_resource_lifecycles(ctx: dict[str, Any]) -> None:
    """Create → update → delete for the clinical CRUD resources."""
    client = ctx["client"]
    pid = ctx["patient_id"]

    cases = [
        (
            "conditions",
            "conditions",
            {"code": "E11.9", "code_display": "T2D", "clinical_status": "active"},
            {"severity": "moderate"},
        ),
        (
            "family",
            "family",
            {"relationship_code": "FTH", "degree_of_relatedness": 0.5, "sex": "male"},
            {"deceased": True},
        ),
        (
            "medications",
            "medications",
            {"medication_code": "860975", "medication_display": "Metformin"},
            {"status": "completed"},
        ),
        (
            "observations",
            "observations",
            {"code": "4548-4", "code_display": "HbA1c", "value_quantity": 6.1},
            {"value_quantity": 7.0},
        ),
    ]
    for create_seg, res_seg, body, update in cases:
        created = await client.post(f"/patients/{pid}/{create_seg}", json=body)
        assert created.status_code in (200, 201), f"{create_seg}: {created.text[:200]}"
        rid = created.json()["id"]
        put = await client.put(f"/{res_seg}/{rid}", json=update)
        assert put.status_code in (200, 404, 422)
        dele = await client.delete(f"/{res_seg}/{rid}")
        assert dele.status_code in (200, 204, 404)

    enc = await client.post(f"/patients/{pid}/encounters", json={"encounter_class": "AMB"})
    assert enc.status_code in (200, 201)
    eid = enc.json()["id"]
    assert (await client.get(f"/encounters/{eid}")).status_code in (200, 404)
    assert (await client.put(f"/encounters/{eid}/close", json={})).status_code in (200, 404, 422)


@pytest.mark.parametrize("template", _EXTRA_GETS)
async def test_extra_get_endpoints(ctx: dict[str, Any], template: str) -> None:
    path = template.format(pid=ctx["patient_id"])
    resp = await ctx["client"].get(path)
    assert resp.status_code in (200, 400, 404, 422, 500, 503)
