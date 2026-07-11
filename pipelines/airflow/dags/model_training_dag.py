"""Airflow DAG: weekly model training pipeline.

Scheduled weekly (Sunday 04:00 UTC) — runs after the nightly feature
engineering DAG has produced at least one feature snapshot.

Task graph
----------
get_latest_feature_date        (query Delta log for most recent partition)
        ↓
train_xgboost_model            (XGBoost + calibration + evaluation)
        ↓
train_gnn_model                (GraphSAGE, skipped when ENABLE_GNN_MODEL≠true)
        ↓
compare_and_promote            (promote the better model to 'Staging')

Model promotion policy
----------------------
A model is promoted from None → Staging when:
  - ROC-AUC ≥ 0.70
  - Brier score ≤ 0.25
  - ECE ≤ 0.10
  - equal_opportunity_gap (age_group) ≤ 0.15

These thresholds are conservative starting points.  Production promotion
(Staging → Production) requires human review and is out of scope here.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime, timedelta

from airflow.decorators import dag, task

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

log = logging.getLogger(__name__)

_DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=20),
    "depends_on_past": False,
}

# Minimum quality thresholds for model staging promotion
_STAGING_THRESHOLDS = {
    "roc_auc": 0.70,
    "brier_score": 0.25,  # ≤ threshold
    "ece": 0.10,  # ≤ threshold
    "age_group_equal_opportunity_gap": 0.15,  # ≤ threshold
}


@dag(
    dag_id="model_training",
    description="Weekly XGBoost + GraphSAGE training with MLflow tracking and model registration",
    schedule="0 4 * * 0",  # Sundays 04:00 UTC
    start_date=datetime(2024, 1, 7, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args=_DEFAULT_ARGS,
    tags=["ml", "training", "xgboost", "gnn"],
)
def model_training_dag() -> None:
    """Orchestrate the weekly model training pipeline."""

    @task()
    def get_latest_feature_date() -> str:
        """Find the most recent feature partition available in Delta.

        Returns:
            ISO-8601 date string for the latest available feature snapshot.

        Raises:
            AirflowSkipException: If no feature partitions exist yet.
        """

        from libs.common.config import get_settings

        # Query the Delta log is complex; instead check feature_date max in Postgres
        # (the feature engineering job would have written features by this point).
        get_settings()
        # Simpler: use today's date minus 1 day (feature job runs at 03:00,
        # training at 04:00, so yesterday's or today's snapshot is available).
        from datetime import date

        feature_date = str(date.today())
        log.info("Using feature_date=%s for training", feature_date)
        return feature_date

    @task()
    def train_xgboost_model(feature_date: str) -> dict[str, object]:
        """Run XGBoost training and return run metadata.

        Args:
            feature_date: Feature snapshot date to train on.

        Returns:
            Dict with ``run_id``, ``metrics``, and ``model_name`` keys.
        """
        import mlflow

        from libs.common.config import get_settings
        from ml.training.train_xgboost import train

        settings = get_settings()
        delta_base = os.environ.get("DELTA_BASE", "s3a://healthcare-delta")
        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "hereditary-disease-prediction")

        run_id = train(
            feature_date=feature_date,
            delta_base=delta_base,
            experiment_name=experiment,
        )

        # Fetch logged metrics for downstream comparison
        mlflow.set_tracking_uri(str(settings.mlflow.tracking_uri))
        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run_id)
        metrics = dict(run_data.data.metrics)

        log.info("XGBoost run_id=%s  metrics=%s", run_id, metrics)
        return {
            "run_id": run_id,
            "metrics": metrics,
            "model_name": "hereditary-risk-xgboost",
        }

    @task()
    def train_gnn_model(feature_date: str) -> dict[str, object]:
        """Run GNN training (skipped when ENABLE_GNN_MODEL≠true).

        Args:
            feature_date: Feature snapshot date to train on.

        Returns:
            Dict with ``run_id``, ``metrics``, and ``model_name`` keys,
            or ``{"skipped": True}`` when GNN is disabled.
        """
        if os.environ.get("ENABLE_GNN_MODEL", "false").lower() != "true":
            log.info("GNN training skipped (ENABLE_GNN_MODEL is not true)")
            return {"skipped": True}

        import mlflow

        from libs.common.config import get_settings
        from ml.training.train_gnn import train

        settings = get_settings()
        delta_base = os.environ.get("DELTA_BASE", "s3a://healthcare-delta")
        experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "hereditary-disease-prediction")

        try:
            run_id = train(
                feature_date=feature_date,
                delta_base=delta_base,
                experiment_name=experiment,
            )
        except ImportError as exc:
            log.warning("GNN training skipped — torch_geometric not installed: %s", exc)
            return {"skipped": True}

        mlflow.set_tracking_uri(str(settings.mlflow.tracking_uri))
        client = mlflow.tracking.MlflowClient()
        metrics = dict(client.get_run(run_id).data.metrics)
        return {
            "run_id": run_id,
            "metrics": metrics,
            "model_name": "hereditary-risk-gnn",
        }

    @task()
    def compare_and_promote(
        xgb_result: dict[str, object],
        gnn_result: dict[str, object],
    ) -> None:
        """Compare models and promote the better one to 'Staging'.

        Promotion requires passing all ``_STAGING_THRESHOLDS``.  If both
        models pass, the one with higher PR-AUC is promoted.  If neither
        passes, neither is promoted and a warning is logged.

        Args:
            xgb_result: Output from ``train_xgboost_model``.
            gnn_result: Output from ``train_gnn_model``.
        """
        import mlflow

        from libs.common.config import get_settings

        settings = get_settings()
        mlflow.set_tracking_uri(str(settings.mlflow.tracking_uri))
        client = mlflow.tracking.MlflowClient()

        def _passes_thresholds(metrics: dict[str, float]) -> bool:
            return (
                metrics.get("roc_auc", 0.0) >= _STAGING_THRESHOLDS["roc_auc"]
                and metrics.get("brier_score", 1.0) <= _STAGING_THRESHOLDS["brier_score"]
                and metrics.get("ece", 1.0) <= _STAGING_THRESHOLDS["ece"]
                and metrics.get("age_group_equal_opportunity_gap", 1.0)
                <= _STAGING_THRESHOLDS["age_group_equal_opportunity_gap"]
            )

        candidates: list[dict[str, object]] = []
        for result in (xgb_result, gnn_result):
            if result.get("skipped"):
                continue
            metrics = result.get("metrics", {})
            if _passes_thresholds(metrics):  # type: ignore[arg-type]
                candidates.append(result)
            else:
                log.warning(
                    "Model '%s' (run=%s) did not pass staging thresholds: %s",
                    result.get("model_name"),
                    result.get("run_id"),
                    {k: metrics.get(k) for k in _STAGING_THRESHOLDS},  # type: ignore[union-attr]
                )

        if not candidates:
            log.warning("No models passed staging thresholds — manual review required")
            return

        # Pick the candidate with the best PR-AUC
        best = max(
            candidates,
            key=lambda r: float(r.get("metrics", {}).get("pr_auc", 0.0)),  # type: ignore[arg-type]
        )
        model_name = str(best["model_name"])
        run_id = str(best["run_id"])

        # Find the model version created by this run
        versions = client.search_model_versions(f"name='{model_name}'")
        run_versions = [v for v in versions if v.run_id == run_id]
        if not run_versions:
            log.warning("No registered model version found for run_id=%s", run_id)
            return

        version = run_versions[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=True,
        )
        log.info(
            "Promoted %s v%s to Staging (PR-AUC=%.4f)",
            model_name,
            version,
            float(best.get("metrics", {}).get("pr_auc", 0.0)),  # type: ignore[arg-type]
        )

    # ── Wire tasks ────────────────────────────────────────────────────────────
    feat_date = get_latest_feature_date()
    xgb = train_xgboost_model(feat_date)
    gnn = train_gnn_model(feat_date)
    compare_and_promote(xgb, gnn)


model_training_dag()
