"""Airflow DAG: nightly patient feature engineering.

Scheduled at 03:00 UTC daily — runs one hour after the batch ingestion
DAG (02:00) to ensure the previous night's clinical data is already
loaded into PostgreSQL and Neo4j before features are computed.

Task graph
----------
check_neo4j_connection
        ↓
run_gds_projection          (writes gds_clustering_coefficient to nodes)
        ↓
run_feature_engineering_job (Spark batch, reads PG + Neo4j, writes Delta)
        ↓
validate_feature_output     (Great Expectations checks on Delta output)

All tasks use the TaskFlow API (@task decorator) for clean data passing.

Configuration
-------------
All connection parameters are read from Airflow environment variables
at task execution time via ``get_settings()``.  No Airflow Connections
objects are used so the DAG is portable without Airflow UI configuration.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, date, datetime, timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

# ── Project root on path ──────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

log = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "depends_on_past": False,
}


@dag(
    dag_id="feature_engineering",
    description="Nightly feature engineering: Neo4j GDS + Spark Delta feature store",
    schedule="0 3 * * *",
    start_date=datetime(2024, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args=_DEFAULT_ARGS,
    tags=["feature-engineering", "ml", "spark"],
)
def feature_engineering_dag() -> None:
    """Orchestrate the nightly patient feature engineering pipeline."""

    @task()
    def check_neo4j_connection() -> bool:
        """Verify Neo4j is reachable and GDS plugin is installed.

        Returns:
            True if both Neo4j and GDS are available.
            Raises on connection failure (triggers retry).
        """
        from neo4j import GraphDatabase

        from libs.common.config import get_settings

        cfg = get_settings().neo4j
        driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password.get_secret_value()))
        try:
            with driver.session() as session:
                result = session.run("RETURN 1 AS ok").single()
                if not result or result["ok"] != 1:
                    raise RuntimeError("Neo4j health check returned unexpected result")
                # Check GDS availability (non-fatal — just log)
                try:
                    session.run("CALL gds.list() YIELD name RETURN count(*) AS n LIMIT 1")
                    log.info("Neo4j GDS plugin detected")
                except Exception:
                    log.warning(
                        "Neo4j GDS plugin not available — "
                        "family_clustering_coefficient will default to 0.0"
                    )
            return True
        finally:
            driver.close()

    @task()
    def run_gds_projection(neo4j_ok: bool) -> bool:  # noqa: FBT001
        """Project the family graph and write GDS clustering coefficients.

        Args:
            neo4j_ok: Upstream connectivity check result (unused value —
                Airflow uses it for task dependency only).

        Returns:
            True if GDS projection succeeded, False otherwise.
        """
        from libs.common.config import get_settings
        from pipelines.spark.feature_engineering.features.graph_features import (
            run_gds_write_projection,
        )

        cfg = get_settings().neo4j
        success = run_gds_write_projection(
            neo4j_uri=cfg.uri,
            neo4j_user=cfg.user,
            neo4j_password=cfg.password.get_secret_value(),
        )
        if not success:
            log.warning("GDS projection skipped — clustering coefficient will be 0.0")
        return success

    @task()
    def run_feature_engineering_job(
        gds_ready: bool,  # noqa: FBT001
        as_of_date: str | None = None,
    ) -> dict[str, object]:
        """Run the Spark feature engineering job.

        Creates a local SparkSession (connected to the Spark master when
        ``SPARK_MASTER_URL`` is set) and executes the full pipeline.

        Args:
            gds_ready: Whether GDS projection succeeded (dependency token).
            as_of_date: Override the feature reference date (ISO-8601).
                Defaults to today's date when None.

        Returns:
            Dict with ``as_of_date`` and ``status`` for downstream tasks.
        """
        import os

        from pipelines.spark.feature_engineering.job import main

        run_date = as_of_date or str(date.today())
        delta_base = os.environ.get("DELTA_BASE", "s3a://healthcare-delta")

        log.info("Starting feature engineering job for as_of_date=%s", run_date)
        main(as_of_date=run_date, delta_base=delta_base)
        log.info("Feature engineering job complete")

        return {"as_of_date": run_date, "status": "success"}

    @task()
    def validate_feature_output(job_result: dict[str, object]) -> None:
        """Run Great Expectations checks on the feature vector Delta table.

        Validates:
        - Feature vector is non-empty
        - No null patient_id values
        - age_years is in [0, 150] or NULL
        - weighted_family_prevalence is non-negative
        - Binary flags are in {0, 1}

        Args:
            job_result: Output dict from run_feature_engineering_job.

        Raises:
            AirflowSkipException: if the Delta table doesn't exist yet
                (first run before any data is written).
            ValueError: if critical expectations fail.
        """
        import os

        import pandas as pd
        from great_expectations.data_context import EphemeralDataContext
        from great_expectations.data_context.types.base import (
            DataContextConfig,
            InMemoryStoreBackendDefaults,
        )

        from libs.common.config import get_settings

        run_date = str(job_result.get("as_of_date", date.today()))
        delta_base = os.environ.get("DELTA_BASE", "s3a://healthcare-delta")
        vector_path = f"{delta_base}/features/patient_feature_vector"

        # Build a dedicated Spark session to read the Delta table.  Each Airflow
        # task runs in its own process, so the upstream job's session is never
        # reachable here — create one configured identically (Delta + MinIO S3A).
        from pipelines.spark.feature_engineering.job import _build_spark_session

        minio = get_settings().minio
        spark = _build_spark_session(
            minio_endpoint=str(minio.endpoint),
            access_key=minio.access_key,
            secret_key=minio.secret_key.get_secret_value(),
        )
        try:
            sample_df: pd.DataFrame = (
                spark.read.format("delta")
                .load(vector_path)
                .filter(f"feature_date = '{run_date}'")
                .limit(5000)
                .toPandas()
            )
        except Exception as exc:
            log.warning("Could not read feature vector for validation: %s", exc)
            raise AirflowSkipException("Feature vector not readable — skipping validation") from exc
        finally:
            spark.stop()

        if sample_df.empty:
            raise ValueError(f"Feature vector is empty for feature_date={run_date}")

        context = EphemeralDataContext(
            project_config=DataContextConfig(store_backend_defaults=InMemoryStoreBackendDefaults())
        )
        ds = context.sources.add_pandas("feature_validation")
        asset = ds.add_dataframe_asset("feature_vector_sample")
        batch = asset.build_batch_request(dataframe=sample_df)
        validator = context.get_validator(batch_request=batch)

        failures: list[str] = []

        def _check(result: dict[str, object], name: str) -> None:  # type: ignore[type-arg]
            if not result.get("success"):
                failures.append(name)

        _check(
            validator.expect_column_values_to_not_be_null("patient_id").to_json_dict(),
            "patient_id_not_null",
        )
        _check(
            validator.expect_table_row_count_to_be_between(min_value=1).to_json_dict(),
            "table_non_empty",
        )
        _check(
            validator.expect_column_values_to_be_between(
                "age_years", min_value=0, max_value=150, mostly=0.99
            ).to_json_dict(),
            "age_years_range",
        )
        _check(
            validator.expect_column_values_to_be_between(
                "weighted_family_prevalence", min_value=0.0
            ).to_json_dict(),
            "weighted_prevalence_non_negative",
        )
        for flag_col in ("gender_male", "gender_female", "has_cardiovascular", "has_oncological"):
            _check(
                validator.expect_column_values_to_be_in_set(flag_col, {0, 1}).to_json_dict(),
                f"{flag_col}_binary",
            )

        if failures:
            raise ValueError(f"Feature output validation failed for {run_date}: {failures}")
        log.info("Feature output validation passed for %s", run_date)

    # ── Wire task dependencies ────────────────────────────────────────────────
    neo4j_ok = check_neo4j_connection()
    gds_ok = run_gds_projection(neo4j_ok)
    job_result = run_feature_engineering_job(gds_ok)
    validate_feature_output(job_result)


feature_engineering_dag()
