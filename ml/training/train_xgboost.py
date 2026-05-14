"""XGBoost training script for hereditary disease risk prediction.

Full pipeline:
    1. Load feature vector from Delta feature store
    2. Load binary labels from PostgreSQL
    3. Build dataset (encode categoricals, impute nulls)
    4. Patient-ID stratified train / calibration / val / test split
    5. Optional Optuna HPO (``--hpo`` flag)
    6. Train XGBoost with early stopping on val set
    7. Platt-scale calibration on calibration split
    8. Evaluation (ROC-AUC, PR-AUC, Brier, ECE)
    9. Fairness analysis (age_group, gender)
   10. Log all parameters, metrics, and artifacts to MLflow
   11. Register model in MLflow Model Registry

Run from project root:
    python ml/training/train_xgboost.py \\
        --feature-date 2024-01-01 \\
        --delta-base s3a://healthcare-delta \\
        [--hpo] [--hpo-trials 50] [--experiment xgb-experiment]

Environment variables (from .env):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from libs.common.config import get_settings  # noqa: E402
from libs.common.logging import configure_logging  # noqa: E402
from ml.models.calibration import CalibrationMethod, calibrate  # noqa: E402
from ml.models.xgboost_model import HeredityXGBModel, XGBConfig  # noqa: E402
from ml.training.dataset import (  # noqa: E402
    apply_split,
    build_dataset,
    load_feature_vector,
    load_labels,
    patient_id_split,
)
from ml.training.evaluate import evaluate_binary_classifier  # noqa: E402
from ml.training.fairness import compute_fairness_report  # noqa: E402

configure_logging(service_name="train-xgboost")
log = logging.getLogger(__name__)

_MODEL_NAME = "hereditary-risk-xgboost"


# ── Optional Optuna HPO ───────────────────────────────────────────────────────

def _run_hpo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    n_trials: int = 30,
) -> XGBConfig:
    """Run Optuna hyperparameter search and return the best XGBConfig.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        feature_names: Feature column names.
        n_trials: Number of Optuna trials.

    Returns:
        ``XGBConfig`` with the best hyperparameters found.

    Raises:
        ImportError: If ``optuna`` is not installed.
    """
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("Install 'optuna' for HPO: pip install optuna") from exc

    from sklearn.metrics import average_precision_score

    def objective(trial: optuna.Trial) -> float:
        cfg = XGBConfig(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 1.0),
        )
        model = HeredityXGBModel(cfg)
        model.fit(X_train, y_train, X_val, y_val, feature_names)
        proba = model.predict_proba(X_val)
        return float(average_precision_score(y_val, proba))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params

    log.info("HPO complete — best PR-AUC=%.4f  params=%s", study.best_value, best)
    return XGBConfig(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        learning_rate=best["learning_rate"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        min_child_weight=best["min_child_weight"],
        gamma=best["gamma"],
    )


# ── Training entry point ──────────────────────────────────────────────────────

def train(
    feature_date: str,
    delta_base: str,
    experiment_name: str,
    config: XGBConfig | None = None,
    run_hpo: bool = False,
    hpo_trials: int = 30,
    spark: object | None = None,
) -> str:
    """Train, calibrate, evaluate, and register the XGBoost model.

    Args:
        feature_date: ISO-8601 date of the feature snapshot to train on.
        delta_base: S3A base path for the Delta feature store.
        experiment_name: MLflow experiment name.
        config: Override ``XGBConfig``; defaults to ``XGBConfig()`` then HPO.
        run_hpo: Whether to run Optuna HPO before final training.
        hpo_trials: Number of HPO trials (used when ``run_hpo=True``).
        spark: Active ``SparkSession`` for Delta reads (optional).

    Returns:
        MLflow run ID for the completed training run.
    """
    settings = get_settings()
    pg = settings.postgres

    # ── Data loading ──────────────────────────────────────────────────────────
    feat_path = f"{delta_base}/features/patient_feature_vector"
    log.info("Loading features from %s  date=%s", feat_path, feature_date)
    features_df = load_feature_vector(feat_path, feature_date, spark=spark)
    labels_df = load_labels(pg.sync_dsn)

    X, y, feat_cols, patient_ids = build_dataset(features_df, labels_df)

    # ── Patient-ID split (train / calibration / val / test) ───────────────────
    # Calibration is a 10% hold-out from the train pool (after test/val removed).
    train_ids, val_ids, test_ids = patient_id_split(patient_ids, y, val_size=0.15, test_size=0.15)
    # Carve calibration out of train_ids
    from sklearn.model_selection import train_test_split as _tts
    train_ids_final, cal_ids = _tts(
        train_ids,
        test_size=0.12,
        stratify=y[np.isin(patient_ids, train_ids)],
        random_state=42,
    )

    X_train, y_train = apply_split(patient_ids, X, y, train_ids_final)
    X_cal, y_cal = apply_split(patient_ids, X, y, cal_ids)
    X_val, y_val = apply_split(patient_ids, X, y, val_ids)
    X_test, y_test = apply_split(patient_ids, X, y, test_ids)

    log.info(
        "Splits — train: %d  cal: %d  val: %d  test: %d",
        len(X_train), len(X_cal), len(X_val), len(X_test),
    )

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(str(settings.mlflow.tracking_uri))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"xgboost-{feature_date}") as run:
        run_id = run.info.run_id
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("feature_date", feature_date)

        # ── HPO ───────────────────────────────────────────────────────────────
        if run_hpo:
            log.info("Running HPO (%d trials)…", hpo_trials)
            config = _run_hpo(X_train, y_train, X_val, y_val, feat_cols, hpo_trials)
        cfg = config or XGBConfig()

        # ── Training ──────────────────────────────────────────────────────────
        model = HeredityXGBModel(cfg)
        model.fit(X_train, y_train, X_val, y_val, feat_cols)
        mlflow.log_params(model.params_dict())

        # ── Calibration ───────────────────────────────────────────────────────
        calibrated = calibrate(model.predict_proba, X_cal, y_cal, CalibrationMethod.SIGMOID)
        mlflow.log_metrics({
            "brier_before_calibration": calibrated.brier_before,
            "brier_after_calibration": calibrated.brier_after,
        })

        # ── Evaluation ────────────────────────────────────────────────────────
        test_proba = calibrated.predict_proba(X_test)
        eval_result = evaluate_binary_classifier(y_test, test_proba)
        mlflow.log_metrics(eval_result.to_mlflow_metrics())

        # ── Fairness ──────────────────────────────────────────────────────────
        test_df = pd.DataFrame(X_test, columns=feat_cols)
        for sensitive_col in ("age_group", "gender_male"):
            if sensitive_col in test_df.columns:
                report = compute_fairness_report(y_test, test_proba, test_df[sensitive_col])
                mlflow.log_metrics(report.to_mlflow_metrics(prefix=sensitive_col))
                report_csv = report.to_dataframe().to_csv(index=False)
                mlflow.log_text(report_csv, f"fairness_{sensitive_col}.csv")

        # ── Feature importances ───────────────────────────────────────────────
        importances = model.feature_importances
        mlflow.log_text(
            json.dumps(importances, indent=2),
            "feature_importances.json",
        )

        # ── SHAP ──────────────────────────────────────────────────────────────
        try:
            import shap
            import matplotlib.pyplot as plt

            shap_vals = model.shap_values(X_test[:500])
            fig, _ = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_vals, X_test[:500], feature_names=feat_cols, show=False)
            mlflow.log_figure(fig, "shap_summary.png")
            plt.close(fig)
        except Exception as exc:
            log.warning("SHAP plot skipped: %s", exc)

        # ── Calibration plot ──────────────────────────────────────────────────
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(
                eval_result.calibration_mean_pred,
                eval_result.calibration_fraction_pos,
                marker="o",
                label=f"XGBoost (ECE={eval_result.ece:.3f})",
            )
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title("Reliability diagram")
            ax.legend()
            mlflow.log_figure(fig, "reliability_diagram.png")
            plt.close(fig)
        except Exception as exc:
            log.warning("Calibration plot skipped: %s", exc)

        # ── Model registration ────────────────────────────────────────────────
        mlflow.xgboost.log_model(
            model._model,
            artifact_path="xgboost_model",
            registered_model_name=_MODEL_NAME,
        )
        log.info(
            "Run complete — run_id=%s  ROC-AUC=%.4f  PR-AUC=%.4f  Brier=%.4f",
            run_id, eval_result.roc_auc, eval_result.pr_auc, eval_result.brier_score,
        )

    return run_id


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost hereditary risk model")
    parser.add_argument("--feature-date", required=True, help="Feature snapshot date YYYY-MM-DD")
    parser.add_argument(
        "--delta-base",
        default=os.environ.get("DELTA_BASE", "s3a://healthcare-delta"),
    )
    parser.add_argument(
        "--experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "hereditary-disease-prediction"),
    )
    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO before training")
    parser.add_argument("--hpo-trials", type=int, default=30)
    args = parser.parse_args()

    rid = train(
        feature_date=args.feature_date,
        delta_base=args.delta_base,
        experiment_name=args.experiment,
        run_hpo=args.hpo,
        hpo_trials=args.hpo_trials,
    )
    log.info("MLflow run_id: %s", rid)
