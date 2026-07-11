"""Additional unit tests for service helpers exercised with mocks.

Targets the model-service MLflow load path, the Kafka admin helpers, and the
async cache service — all driven with fakes so no live MLflow/Kafka/Redis is
needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── model_service.load ────────────────────────────────────────────────────────


class TestModelServiceLoad:
    def _mlflow_stack(self, feature_names: list[str] | None) -> MagicMock:
        model = MagicMock()
        if feature_names is not None:
            model.feature_names_in_ = feature_names
        else:
            del model.feature_names_in_
        mv = SimpleNamespace(version="3", run_id="run-xyz")
        client = MagicMock()
        client.get_latest_versions.return_value = [mv]
        client.get_run.return_value = SimpleNamespace(
            data=SimpleNamespace(tags={"feature_columns": "age_years,gender_male"})
        )
        return model, client

    def test_load_success(self) -> None:
        import mlflow

        from services.api.services.model_service import ModelService

        model, client = self._mlflow_stack(["age_years", "gender_male"])
        with (
            patch.object(mlflow, "set_tracking_uri"),
            patch("mlflow.xgboost.load_model", return_value=model),
            patch("mlflow.tracking.MlflowClient", return_value=client),
        ):
            svc = ModelService()
            svc.load("http://mlflow:5000", "hereditary-risk-xgboost", "Staging")

        assert svc.is_loaded
        assert svc.info is not None
        assert svc.info.version == "3"
        assert svc.info.feature_names == ["age_years", "gender_male"]

    def test_load_falls_back_to_tag_features(self) -> None:
        import mlflow

        from services.api.services.model_service import ModelService

        model, client = self._mlflow_stack(None)
        with (
            patch.object(mlflow, "set_tracking_uri"),
            patch("mlflow.xgboost.load_model", return_value=model),
            patch("mlflow.tracking.MlflowClient", return_value=client),
        ):
            svc = ModelService()
            svc.load("http://mlflow:5000")
        assert svc.info is not None
        assert "age_years" in svc.info.feature_names

    def test_load_model_error_raises_runtime(self) -> None:
        import mlflow

        from services.api.services.model_service import ModelService

        with (
            patch.object(mlflow, "set_tracking_uri"),
            patch("mlflow.xgboost.load_model", side_effect=ValueError("boom")),
        ):
            with pytest.raises(RuntimeError, match="Failed to load model"):
                ModelService().load("http://mlflow:5000")

    def test_load_no_versions_raises(self) -> None:
        import mlflow

        from services.api.services.model_service import ModelService

        model = MagicMock()
        model.feature_names_in_ = ["age_years"]
        client = MagicMock()
        client.get_latest_versions.return_value = []
        with (
            patch.object(mlflow, "set_tracking_uri"),
            patch("mlflow.xgboost.load_model", return_value=model),
            patch("mlflow.tracking.MlflowClient", return_value=client),
        ):
            with pytest.raises(RuntimeError, match="No model version"):
                ModelService().load("http://mlflow:5000")


# ── kafka_admin ───────────────────────────────────────────────────────────────


class TestKafkaAdmin:
    def test_get_latest_schema(self) -> None:
        from services.ingestion import kafka_admin

        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"id": 7, "schema": '{"type":"record"}'}
        with patch.object(kafka_admin.requests, "get", return_value=resp) as get:
            schema_id, schema = kafka_admin.get_latest_schema("http://reg:8081", "patients")
        assert schema_id == 7
        assert "record" in schema
        get.assert_called_once()

    def test_register_schema_missing_file_raises(self) -> None:
        from services.ingestion import kafka_admin

        with pytest.raises(FileNotFoundError):
            kafka_admin.register_schema("http://reg:8081", "patients", "does_not_exist.avsc")

    def test_register_schema_http_error_propagates(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from services.ingestion import kafka_admin

        avsc = tmp_path / "p.avsc"
        avsc.write_text('{"type":"record","name":"P","fields":[]}')
        resp = MagicMock()
        resp.raise_for_status.side_effect = RuntimeError("409 conflict")
        with (
            patch.object(kafka_admin.requests, "post", return_value=resp),
            pytest.raises(RuntimeError),
        ):
            kafka_admin.register_schema("http://reg:8081", "patients", str(avsc))

    def test_main_bootstraps_topics_and_schemas(self) -> None:
        from services.ingestion import kafka_admin

        fake_settings = SimpleNamespace(
            kafka=SimpleNamespace(
                bootstrap_servers="localhost:9092",
                schema_registry_url="http://reg:8081/",
            )
        )
        with (
            patch.object(kafka_admin, "get_settings", return_value=fake_settings),
            patch.object(kafka_admin, "create_topics") as create,
            patch.object(kafka_admin, "register_schema", return_value=1) as reg,
        ):
            kafka_admin.main()
        create.assert_called_once_with("localhost:9092")
        assert reg.call_count == len(kafka_admin.SCHEMA_MAP)

    def test_main_exits_on_schema_failure(self) -> None:
        from services.ingestion import kafka_admin

        fake_settings = SimpleNamespace(
            kafka=SimpleNamespace(
                bootstrap_servers="localhost:9092",
                schema_registry_url="http://reg:8081",
            )
        )
        with (
            patch.object(kafka_admin, "get_settings", return_value=fake_settings),
            patch.object(kafka_admin, "create_topics"),
            patch.object(kafka_admin, "register_schema", side_effect=RuntimeError("x")),
            pytest.raises(SystemExit),
        ):
            kafka_admin.main()


# ── cache_service (async) ─────────────────────────────────────────────────────


class TestCacheService:
    def _svc(self, redis: Any) -> Any:  # type: ignore[valid-type]
        from services.api.services.cache_service import CacheService

        return CacheService(redis)

    async def test_get_json_hit_and_miss(self) -> None:
        redis = AsyncMock()
        redis.get.return_value = None
        svc = self._svc(redis)
        assert await svc.get_json("k") is None

        redis.get.return_value = json.dumps({"a": 1})
        assert await svc.get_json("k") == {"a": 1}

    async def test_get_json_invalid_evicts(self) -> None:
        redis = AsyncMock()
        redis.get.return_value = "{not json"
        svc = self._svc(redis)
        assert await svc.get_json("k") is None
        redis.delete.assert_awaited_once()

    async def test_set_delete_ping(self) -> None:
        redis = AsyncMock()
        svc = self._svc(redis)
        await svc.set_json("k", {"a": 1}, ttl=60)
        redis.setex.assert_awaited_once()
        await svc.delete("k")
        redis.delete.assert_awaited()
        assert await svc.ping() is True

    async def test_ping_failure_returns_false(self) -> None:
        redis = AsyncMock()
        redis.ping.side_effect = ConnectionError("down")
        assert await self._svc(redis).ping() is False

    def test_key_builders_are_deterministic(self) -> None:
        from services.api.services.cache_service import CacheService

        assert CacheService.hereditary_key("p", "2024-01-01").startswith("predict:hereditary:")
        assert CacheService.features_key("p", "2024-01-01").startswith("features:")
        assert CacheService.family_profile_key("p").startswith("profile:family:")
        # Order-independent hashing for code lists.
        assert CacheService.symptom_key("p", ["b", "a"]) == CacheService.symptom_key(
            "p", ["a", "b"]
        )
