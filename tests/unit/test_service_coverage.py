"""Unit tests for service-layer, middleware, and data-quality helpers.

Covers the pure/mockable logic that does not require a live database or broker:
Great Expectations suites, the model-service value coercion, the request-scoped
middleware helpers, and the Kafka admin helpers (with the client mocked).
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
from starlette.requests import Request

from libs.common.quality import (
    validate_diagnosis_records,
    validate_observation_records,
    validate_patient_records,
)


def _make_request(path: str, *, method: str = "GET", client: str | None = "203.0.113.7") -> Request:
    """Build a minimal Starlette Request for middleware-helper tests."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"user-agent", b"pytest")],
        "client": (client, 12345) if client else None,
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


# ── libs/common/quality.py ────────────────────────────────────────────────────


class TestQualitySuites:
    def test_validate_patient_records_success(self) -> None:
        records = [
            {
                "patient_id": str(uuid.uuid4()),
                "event_id": str(uuid.uuid4()),
                "gender": "female",
                "date_of_birth": "1990-05-01",
            }
        ]
        result = validate_patient_records(records)
        assert result.success is True
        assert result.statistics["evaluated"] > 0

    def test_validate_patient_records_bad_uuid_fails_critical(self) -> None:
        records = [
            {
                "patient_id": "not-a-uuid",
                "event_id": str(uuid.uuid4()),
                "gender": "female",
                "date_of_birth": "1990-05-01",
            }
        ]
        result = validate_patient_records(records)
        assert result.success is False
        assert any(f["severity"] == "critical" for f in result.failures)

    def test_empty_records_short_circuit(self) -> None:
        assert validate_patient_records([]).success is True

    def test_validate_diagnosis_and_observation_run(self) -> None:
        diag = [
            {
                "patient_id": str(uuid.uuid4()),
                "event_id": str(uuid.uuid4()),
                "icd10_code": "E11.9",
                "clinical_status": "active",
            }
        ]
        obs = [
            {
                "patient_id": str(uuid.uuid4()),
                "event_id": str(uuid.uuid4()),
                "loinc_code": "4548-4",
                "value": 5.6,
            }
        ]
        assert validate_diagnosis_records(diag) is not None
        assert validate_observation_records(obs) is not None


# ── services/api/services/model_service.py ────────────────────────────────────


class TestModelService:
    def test_to_float_coerces(self) -> None:
        from services.api.services.model_service import _to_float

        assert _to_float(3) == 3.0
        assert _to_float(2.5) == 2.5
        assert _to_float(None) == 0.0
        assert _to_float("nan") == 0.0

    def test_fresh_service_not_loaded(self) -> None:
        from services.api.services.model_service import ModelService

        svc = ModelService()
        assert svc.is_loaded is False

    def test_predict_proba_sync_with_mocked_model(self) -> None:
        from services.api.services.model_service import ModelInfo, ModelService

        svc = ModelService()
        svc._xgb_model = MagicMock()
        svc._xgb_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        svc.info = ModelInfo(
            model_name="m",
            version="1",
            run_id="r",
            feature_names=["age_years", "gender_male"],
        )
        proba = svc.predict_proba_sync({"age_years": 50, "gender_male": 1})
        assert proba == 0.8

    def test_predict_proba_sync_unloaded_raises(self) -> None:
        import pytest

        from services.api.services.model_service import ModelService

        with pytest.raises(RuntimeError):
            ModelService().predict_proba_sync({"age_years": 1})


# ── middleware helpers ────────────────────────────────────────────────────────


class TestMiddlewareHelpers:
    def test_rate_limit_match_rule(self) -> None:
        from services.api.middleware.rate_limit import _match_rule

        assert _match_rule("/auth/token") is not None
        assert _match_rule("/predict/hereditary-risk") is not None
        assert _match_rule("/totally/unmatched/path") is None

    def test_rate_limit_identifier_ip_fallback(self) -> None:
        from services.api.middleware.rate_limit import _identifier

        ident = _identifier(_make_request("/predict/x"), use_ip=True)
        assert "203.0.113.7" in ident

    def test_metrics_normalise_path(self) -> None:
        from services.api.main import app
        from services.api.middleware.metrics import _normalise_path

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/health",
            "raw_path": b"/health",
            "query_string": b"",
            "headers": [],
            "client": ("203.0.113.7", 1),
            "app": app,
        }
        normalised = _normalise_path(Request(scope))
        assert isinstance(normalised, str)

    def test_audit_pure_helpers(self) -> None:
        from services.api.middleware.audit import (
            _get_client_ip,
            _outcome,
            _resource_type,
        )

        assert _outcome(200) == "success"
        assert _outcome(500) != "success"
        assert isinstance(_resource_type("/patient/abc/family-risk-profile"), str)
        assert _get_client_ip(_make_request("/x")) == "203.0.113.7"

    def test_audit_client_ip_forwarded_header(self) -> None:
        from services.api.middleware.audit import _get_client_ip

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/x",
            "headers": [(b"x-forwarded-for", b"198.51.100.9, 10.0.0.1")],
            "client": ("10.0.0.1", 5),
            "query_string": b"",
        }
        assert _get_client_ip(Request(scope)) == "198.51.100.9"


# ── services/ingestion/kafka_admin.py ─────────────────────────────────────────


class TestKafkaAdmin:
    def test_create_topics_invokes_admin_client(self) -> None:
        from services.ingestion import kafka_admin

        fake_future = MagicMock()
        fake_future.result.return_value = None
        fake_admin = MagicMock()
        fake_admin.create_topics.return_value = {t["name"]: fake_future for t in kafka_admin.TOPICS}

        with patch.object(kafka_admin, "AdminClient", return_value=fake_admin):
            kafka_admin.create_topics("localhost:9092")

        fake_admin.create_topics.assert_called_once()

    def test_register_schema_posts_to_registry(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from services.ingestion import kafka_admin

        avsc = tmp_path / "patients.avsc"
        avsc.write_text('{"type": "record", "name": "P", "fields": []}')

        fake_resp = MagicMock()
        fake_resp.json.return_value = {"id": 42}
        fake_resp.raise_for_status.return_value = None
        with patch.object(kafka_admin.requests, "post", return_value=fake_resp) as post:
            schema_id = kafka_admin.register_schema("http://localhost:8081", "patients", str(avsc))
        assert schema_id == 42
        post.assert_called_once()
