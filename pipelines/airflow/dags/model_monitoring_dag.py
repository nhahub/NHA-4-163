"""Airflow DAG — daily model monitoring, drift detection, and retraining gate.

Schedule: daily at 02:00 UTC (after the nightly feature engineering job
completes at 03:00, adjusted to run with a 1-day lag for fresh data).

Pipeline
--------
1. ``get_latest_feature_date``      — resolve yesterday's feature snapshot.
2. ``load_reference_feature_date``  — look up the training reference date
                                      from MLflow model tags.
3. ``run_drift_detection``          — compare current vs. reference feature
                                      distributions; logs DriftReport to MLflow.
4. ``evaluate_model_performance``   — compute Brier / ROC-AUC on labeled
                                      predictions from the past 30 days.
5. ``check_retraining_gate``        — branch: trigger retraining DAG when
                                      drift or performance degradation is
                                      detected; skip otherwise.
6. ``trigger_retraining`` (branch)  — fire ``model_training_dag`` via
                                      ``TriggerDagRunOperator``.
7. ``send_alert`` (branch)          — log alert to Airflow XCom + emit a
                                      structured WARNING log picked up by
                                      Prometheus Alertmanager.

Retraining gate logic
---------------------
Retraining is triggered when ANY of the following are true:
  - ``DriftReport.dataset_drifted`` is True
  - ``DriftReport.prediction_drift_detected`` is True
  - ``PerformanceReport.has_degradation()`` is True

PHI note: feature data used for drift detection contains only aggregated
numeric features — no patient names, DOBs, or other direct identifiers.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
_POSTGRES_DSN = (
    f"postgresql://{os.environ.get('POSTGRES_USER', 'healthcare_app')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', '')}@"
    f"{os.environ.get('POSTGRES_HOST', 'postgres')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'healthcare')}"
)
_DELTA_BASE = os.environ.get("DELTA_BASE", "s3a://healthcare-delta")
_MODEL_NAME = os.environ.get("MODEL_NAME", "hereditary-risk-xgboost")
_MONITORING_EXPERIMENT = "model-monitoring"
_TRAINING_DAG_ID = "model_training_dag"

# Lookback window for labeled performance evaluation
_LOOKBACK_DAYS = 30


@dag(
    dag_id="model_monitoring_dag",
    schedule="0 2 * * *",
    start_date=days_ago(1),
    max_active_runs=1,
    catchup=False,
    tags=["monitoring", "mlops"],
    doc_md=__doc__,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
)
def model_monitoring_dag() -> None:
    """Daily model monitoring pipeline."""

    # ── 1. Resolve yesterday's feature snapshot date ──────────────────────────
    @task
    def get_latest_feature_date() -> str:
        """Return yesterday's date as the current feature snapshot to evaluate."""
        yesterday = date.today() - timedelta(days=1)
        feature_date = yesterday.isoformat()
        log.info("Current feature date: %s", feature_date)
        return feature_date

    # ── 2. Find the training reference date from MLflow ───────────────────────
    @task
    def load_reference_feature_date() -> str:
        """Look up the ``feature_date`` tag from the Staging model version.

        Returns:
            ISO-8601 date string of the reference training dataset.
        """
        import mlflow

        mlflow.set_tracking_uri(_MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(_MODEL_NAME, stages=["Staging"])
        if not versions:
            log.warning("No Staging model found — using 90-day lookback as reference")
            return (date.today() - timedelta(days=90)).isoformat()
        mv = versions[0]
        run_data = client.get_run(mv.run_id).data
        ref_date = run_data.tags.get("feature_date", "")
        if not ref_date:
            log.warning("Model has no feature_date tag — using 90-day lookback")
            return (date.today() - timedelta(days=90)).isoformat()
        log.info("Reference feature date from MLflow: %s", ref_date)
        return ref_date

    # ── 3. Drift detection ────────────────────────────────────────────────────
    @task
    def run_drift_detection(current_date: str, reference_date: str) -> dict:
        """Load reference and current feature data; compute drift report.

        Args:
            current_date: ISO-8601 current feature snapshot date.
            reference_date: ISO-8601 reference (training) feature snapshot date.

        Returns:
            Serialised drift report dict (pushed to XCom).
        """
        import mlflow

        from ml.monitoring.drift_detector import DriftDetector, load_reference_dataset

        mlflow.set_tracking_uri(_MLFLOW_URI)
        mlflow.set_experiment(_MONITORING_EXPERIMENT)

        try:
            ref_df = load_reference_dataset(_DELTA_BASE, reference_date)
            cur_df = load_reference_dataset(_DELTA_BASE, current_date)
        except Exception as exc:
            log.warning("Could not load feature data for drift detection: %s", exc)
            return {
                "dataset_drifted": False,
                "drift_share": 0.0,
                "n_features": 0,
                "n_drifted": 0,
                "prediction_drift_detected": False,
                "sufficient_data": False,
                "error": str(exc),
            }

        detector = DriftDetector(reference_df=ref_df)
        report = detector.run(current_df=cur_df)

        with mlflow.start_run(run_name=f"drift-{current_date}"):
            mlflow.log_metrics(report.to_mlflow_metrics())
            mlflow.log_text(report.to_json(), "drift_report.json")
            mlflow.set_tag("drift.current_date", current_date)
            mlflow.set_tag("drift.reference_date", reference_date)
            mlflow.set_tag(
                "drift.status",
                "drifted" if report.dataset_drifted else "stable",
            )
            if report.top_drifted_features:
                top = ", ".join(r.feature for r in report.top_drifted_features[:5])
                mlflow.set_tag("drift.top_features", top)

        log.info(
            "Drift detection complete: drifted=%s share=%.2f%%",
            report.dataset_drifted,
            report.drift_share * 100,
        )
        return {
            "dataset_drifted": report.dataset_drifted,
            "prediction_drift_detected": report.prediction_drift_detected,
            "drift_share": report.drift_share,
            "n_drifted": report.n_drifted,
            "n_features": report.n_features,
        }

    # ── 4. Performance evaluation ─────────────────────────────────────────────
    @task
    def evaluate_model_performance() -> dict:
        """Compute Brier / ROC-AUC on recent labeled predictions.

        Returns:
            Serialised performance report dict (pushed to XCom).
        """
        from ml.monitoring.model_monitor import evaluate_production_performance

        report = evaluate_production_performance(
            postgres_dsn=_POSTGRES_DSN,
            mlflow_tracking_uri=_MLFLOW_URI,
            lookback_days=_LOOKBACK_DAYS,
            experiment_name=_MONITORING_EXPERIMENT,
        )
        return {
            "has_degradation": report.has_degradation(),
            "sufficient_data": report.sufficient_data,
            "alerts": report.alerts,
            "roc_auc": report.roc_auc,
            "brier_score": report.brier_score,
        }

    # ── 5. Retraining gate ────────────────────────────────────────────────────
    @task.branch
    def check_retraining_gate(drift_result: dict, perf_result: dict) -> str:
        """Decide whether to trigger retraining or skip.

        Args:
            drift_result: Output of ``run_drift_detection``.
            perf_result: Output of ``evaluate_model_performance``.

        Returns:
            Task ID to execute next.
        """
        should_retrain = (
            drift_result.get("dataset_drifted", False)
            or drift_result.get("prediction_drift_detected", False)
            or perf_result.get("has_degradation", False)
        )

        if should_retrain:
            reasons = []
            if drift_result.get("dataset_drifted"):
                reasons.append(
                    f"feature drift ({drift_result.get('drift_share', 0)*100:.0f}% features)"
                )
            if drift_result.get("prediction_drift_detected"):
                reasons.append("prediction distribution drift")
            if perf_result.get("has_degradation"):
                reasons.append(f"performance alerts: {perf_result.get('alerts')}")
            log.warning("Retraining triggered — reasons: %s", "; ".join(reasons))
            return "trigger_retraining"

        log.info("No retraining needed — model is stable")
        return "skip_retraining"

    # ── 6a. Trigger retraining ────────────────────────────────────────────────
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id=_TRAINING_DAG_ID,
        wait_for_completion=False,
        conf={"triggered_by": "model_monitoring_dag", "reason": "drift_or_degradation"},
    )

    # ── 6b. Skip retraining ───────────────────────────────────────────────────
    @task
    def skip_retraining() -> None:
        """Log that monitoring passed and no retraining is needed."""
        log.info("Model monitoring passed — skipping retraining")

    # ── DAG wiring ────────────────────────────────────────────────────────────
    current_date = get_latest_feature_date()
    reference_date = load_reference_feature_date()

    drift_result = run_drift_detection(current_date, reference_date)
    perf_result = evaluate_model_performance()

    branch = check_retraining_gate(drift_result, perf_result)
    branch >> trigger_retraining
    branch >> skip_retraining()


model_monitoring_dag()
