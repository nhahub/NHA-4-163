"""Production model performance monitoring.

Computes performance metrics on recent predictions that have received
ground-truth labels (e.g., confirmed hereditary diagnoses from the EHR).

Workflow
--------
1. Pull recent predictions from a ``predictions_log`` table (written by
   the API after each inference — see Phase 9 schema).
2. Join with confirmed labels from the ``conditions`` table (patients
   with a newly confirmed hereditary condition since the prediction date).
3. Compute Brier score, ROC-AUC, and calibration metrics.
4. Compare against staging thresholds.
5. Log results to MLflow and return a ``PerformanceReport``.

This module is called weekly by the model monitoring Airflow DAG and
can also be run ad-hoc::

    python -m ml.monitoring.model_monitor \
        --postgres-dsn postgresql://... \
        --mlflow-uri http://localhost:5000 \
        --lookback-days 30
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

log = logging.getLogger(__name__)

# Thresholds copied from model_training_dag — alert if any are violated
_ALERT_THRESHOLDS: dict[str, float] = {
    "roc_auc": 0.70,  # minimum acceptable AUC
    "brier_score": 0.25,  # maximum acceptable Brier score
    "ece": 0.10,  # maximum acceptable Expected Calibration Error
}

_MIN_SAMPLES_FOR_EVALUATION = 30  # skip evaluation if fewer labeled examples


@dataclass
class PerformanceReport:
    """Results of a production performance evaluation run.

    Attributes:
        evaluation_date: Date of the evaluation run.
        lookback_days: Number of days of predictions evaluated.
        n_predictions: Total predictions evaluated.
        n_labeled: Predictions with confirmed ground-truth labels.
        roc_auc: Area under the ROC curve (None if insufficient data).
        brier_score: Mean Brier score (None if insufficient data).
        ece: Expected Calibration Error (None if insufficient data).
        alerts: List of threshold violations found.
        sufficient_data: True when ``n_labeled >= _MIN_SAMPLES_FOR_EVALUATION``.
    """

    evaluation_date: str
    lookback_days: int
    n_predictions: int
    n_labeled: int
    roc_auc: float | None
    brier_score: float | None
    ece: float | None
    alerts: list[str]
    sufficient_data: bool

    def to_mlflow_metrics(self) -> dict[str, float]:
        """Return flat dict for ``mlflow.log_metrics``.

        Returns:
            Dict with ``monitor/`` prefix keys.
        """
        m: dict[str, float] = {
            "monitor/n_predictions": float(self.n_predictions),
            "monitor/n_labeled": float(self.n_labeled),
            "monitor/sufficient_data": float(self.sufficient_data),
        }
        if self.roc_auc is not None:
            m["monitor/roc_auc"] = self.roc_auc
        if self.brier_score is not None:
            m["monitor/brier_score"] = self.brier_score
        if self.ece is not None:
            m["monitor/ece"] = self.ece
        m["monitor/n_alerts"] = float(len(self.alerts))
        return m

    def has_degradation(self) -> bool:
        """Return True if any performance threshold is violated."""
        return len(self.alerts) > 0


def _expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        y_true: Binary ground-truth labels.
        y_proba: Predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        ECE as a float in [0, 1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        if hi == 1.0:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_proba[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)


def _load_labeled_predictions(
    postgres_dsn: str,
    lookback_days: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Query recent predictions joined with confirmed labels.

    Joins the ``predictions_log`` table (written at inference time) with
    the ``conditions`` table to get ground-truth labels for predictions
    that now have a confirmed hereditary diagnosis.

    Args:
        postgres_dsn: Sync PostgreSQL DSN.
        lookback_days: Number of days of history to evaluate.

    Returns:
        ``(y_true, y_proba)`` arrays, or ``None`` if insufficient data.
    """
    import psycopg2
    import psycopg2.extras

    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

    sql = """
        SELECT
            pl.risk_score                       AS predicted_proba,
            CASE
                WHEN c.patient_id IS NOT NULL THEN 1
                ELSE 0
            END                                 AS label
        FROM predictions_log pl
        LEFT JOIN LATERAL (
            SELECT DISTINCT patient_id
            FROM conditions
            WHERE patient_id = pl.patient_id
              AND is_hereditary = TRUE
              AND clinical_status IN ('active', 'recurrence', 'relapse')
              AND recorded_date >= pl.predicted_at::date
        ) c ON TRUE
                WHERE pl.predicted_at >= %(cutoff)s
          AND pl.prediction_type = 'hereditary_risk'
    """

    try:
        conn = psycopg2.connect(postgres_dsn)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, {"cutoff": cutoff})
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as exc:
        log.warning("Could not load labeled predictions: %s", exc)
        return None

    if not rows:
        return None

    y_true = np.array([r["label"] for r in rows], dtype=np.float32)
    y_proba = np.array([float(r["predicted_proba"]) for r in rows], dtype=np.float32)
    return y_true, y_proba


def evaluate_production_performance(
    postgres_dsn: str,
    mlflow_tracking_uri: str,
    lookback_days: int = 30,
    experiment_name: str = "model-monitoring",
) -> PerformanceReport:
    """Evaluate model performance on recent labeled production predictions.

    Args:
        postgres_dsn: Sync PostgreSQL DSN.
        mlflow_tracking_uri: MLflow tracking server URI.
        lookback_days: Days of history to evaluate.
        experiment_name: MLflow experiment name for logging results.

    Returns:
        ``PerformanceReport`` with metrics and any threshold alerts.
    """
    import mlflow
    from sklearn.metrics import roc_auc_score

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    today = date.today().isoformat()
    labeled = _load_labeled_predictions(postgres_dsn, lookback_days)

    if labeled is None:
        log.warning("No labeled predictions found for the past %d days", lookback_days)
        report = PerformanceReport(
            evaluation_date=today,
            lookback_days=lookback_days,
            n_predictions=0,
            n_labeled=0,
            roc_auc=None,
            brier_score=None,
            ece=None,
            alerts=[],
            sufficient_data=False,
        )
        with mlflow.start_run(run_name=f"monitor-{today}"):
            mlflow.log_metrics(report.to_mlflow_metrics())
            mlflow.set_tag("monitor.status", "no_data")
        return report

    y_true, y_proba = labeled
    n = len(y_true)
    n_pos = int(y_true.sum())
    sufficient = n >= _MIN_SAMPLES_FOR_EVALUATION and n_pos > 0 and n_pos < n

    roc_auc = brier = ece = None
    alerts: list[str] = []

    if sufficient:
        roc_auc = float(roc_auc_score(y_true, y_proba))
        brier = float(np.mean((y_proba - y_true) ** 2))
        ece = _expected_calibration_error(y_true, y_proba)

        if roc_auc < _ALERT_THRESHOLDS["roc_auc"]:
            alerts.append(f"ROC-AUC {roc_auc:.3f} < threshold {_ALERT_THRESHOLDS['roc_auc']}")
        if brier > _ALERT_THRESHOLDS["brier_score"]:
            alerts.append(f"Brier score {brier:.3f} > threshold {_ALERT_THRESHOLDS['brier_score']}")
        if ece > _ALERT_THRESHOLDS["ece"]:
            alerts.append(f"ECE {ece:.3f} > threshold {_ALERT_THRESHOLDS['ece']}")

    report = PerformanceReport(
        evaluation_date=today,
        lookback_days=lookback_days,
        n_predictions=n,
        n_labeled=n_pos,
        roc_auc=roc_auc,
        brier_score=brier,
        ece=ece,
        alerts=alerts,
        sufficient_data=sufficient,
    )

    with mlflow.start_run(run_name=f"monitor-{today}"):
        mlflow.log_metrics(report.to_mlflow_metrics())
        mlflow.set_tag("monitor.status", "degraded" if alerts else "ok")
        if alerts:
            mlflow.set_tag("monitor.alerts", "; ".join(alerts))

    log.info(
        "Performance report: roc_auc=%.3f brier=%.3f ece=%.3f alerts=%d",
        roc_auc or 0,
        brier or 0,
        ece or 0,
        len(alerts),
    )
    return report


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Evaluate production model performance")
    parser.add_argument("--postgres-dsn", required=True)
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--experiment", default="model-monitoring")
    args = parser.parse_args()

    report = evaluate_production_performance(
        postgres_dsn=args.postgres_dsn,
        mlflow_tracking_uri=args.mlflow_uri,
        lookback_days=args.lookback_days,
        experiment_name=args.experiment,
    )
    if report.has_degradation():
        print(f"DEGRADATION DETECTED: {report.alerts}", file=sys.stderr)  # noqa: T201 — CLI output
        sys.exit(1)
    print(f"OK — roc_auc={report.roc_auc}, brier={report.brier_score}")  # noqa: T201 — CLI output
