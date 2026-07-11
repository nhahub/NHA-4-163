"""Unit tests for the ML training / model / monitoring pure logic.

Exercises the numpy/pandas dataset construction, the XGBoost wrapper on a tiny
synthetic set, and the monitoring report helpers — all without MLflow, a GPU,
or a database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.models.xgboost_model import HeredityXGBModel, XGBConfig
from ml.monitoring.model_monitor import PerformanceReport, _expected_calibration_error
from ml.training.dataset import (
    apply_split,
    build_dataset,
    create_synthetic_dataset,
    patient_id_split,
)

# ── dataset.py ────────────────────────────────────────────────────────────────


class TestDataset:
    def test_create_synthetic_dataset_shapes(self) -> None:
        X, y = create_synthetic_dataset(n_patients=120, random_state=1)
        assert len(X) == 120
        assert len(y) == 120
        assert "patient_id" in X.columns
        assert set(np.unique(y)).issubset({0, 1})

    def test_create_synthetic_dataset_is_deterministic(self) -> None:
        x1, y1 = create_synthetic_dataset(n_patients=50, random_state=7)
        x2, y2 = create_synthetic_dataset(n_patients=50, random_state=7)
        pd.testing.assert_frame_equal(x1, x2)
        assert (y1.values == y2.values).all()

    def test_build_dataset_and_split(self) -> None:
        X_df, y = create_synthetic_dataset(n_patients=200, random_state=3)
        labels = pd.DataFrame({"patient_id": X_df["patient_id"], "label": y.values})
        X, y_arr, feat_names, pids = build_dataset(X_df, labels)
        assert X.shape[0] == 200
        assert X.dtype == np.float32
        assert len(feat_names) > 0
        assert len(pids) == 200

        train_ids, val_ids, test_ids = patient_id_split(pids, y_arr, random_state=3)
        # No patient appears in more than one split.
        assert not (set(train_ids) & set(val_ids))
        assert not (set(train_ids) & set(test_ids))
        assert not (set(val_ids) & set(test_ids))

        X_train, y_train = apply_split(pids, X, y_arr, train_ids)
        assert X_train.shape[0] == len(y_train)
        assert X_train.shape[0] == len(train_ids)

    def test_build_dataset_empty_join_raises(self) -> None:
        feats = pd.DataFrame({"patient_id": [1, 2], "age_years": [30, 40]})
        labels = pd.DataFrame({"patient_id": [98, 99], "label": [0, 1]})
        with pytest.raises(ValueError, match="join is empty"):
            build_dataset(feats, labels)


# ── xgboost_model.py ──────────────────────────────────────────────────────────


class TestHeredityXGBModel:
    @staticmethod
    def _fit_tiny() -> tuple[HeredityXGBModel, np.ndarray, list[str]]:
        X_df, y = create_synthetic_dataset(n_patients=200, random_state=5)
        labels = pd.DataFrame({"patient_id": X_df["patient_id"], "label": y.values})
        X, y_arr, feat_names, pids = build_dataset(X_df, labels)
        train_ids, val_ids, _ = patient_id_split(pids, y_arr, random_state=5)
        X_tr, y_tr = apply_split(pids, X, y_arr, train_ids)
        X_val, y_val = apply_split(pids, X, y_arr, val_ids)
        model = HeredityXGBModel(XGBConfig(n_estimators=15, early_stopping_rounds=5))
        model.fit(X_tr, y_tr, X_val, y_val, feat_names)
        return model, X_val, feat_names

    def test_config_defaults(self) -> None:
        cfg = XGBConfig()
        assert cfg.n_estimators == 500
        assert cfg.eval_metric == "aucpr"

    def test_fit_predict_and_interpret(self) -> None:
        model, X_val, feat_names = self._fit_tiny()
        proba = model.predict_proba(X_val)
        assert proba.shape[0] == X_val.shape[0]
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

        importances = model.feature_importances
        assert set(importances) == set(feat_names)

        params = model.params_dict()
        assert params["n_estimators"] == "15"

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            HeredityXGBModel().predict_proba(np.zeros((2, 3), dtype=np.float32))

    def test_fit_empty_raises(self) -> None:
        empty = np.zeros((0, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            HeredityXGBModel().fit(empty, np.array([]), empty, np.array([]), ["a", "b", "c"])


# ── model_monitor.py ──────────────────────────────────────────────────────────


class TestPerformanceReport:
    def _report(self, alerts: list[str]) -> PerformanceReport:
        return PerformanceReport(
            evaluation_date="2026-01-01",
            lookback_days=30,
            n_predictions=100,
            n_labeled=80,
            roc_auc=0.82,
            brier_score=0.15,
            ece=0.05,
            alerts=alerts,
            sufficient_data=True,
        )

    def test_to_mlflow_metrics(self) -> None:
        metrics = self._report([]).to_mlflow_metrics()
        assert metrics["monitor/roc_auc"] == 0.82
        assert metrics["monitor/n_alerts"] == 0.0
        assert metrics["monitor/n_predictions"] == 100.0

    def test_has_degradation(self) -> None:
        assert self._report(["roc_auc below floor"]).has_degradation() is True
        assert self._report([]).has_degradation() is False

    def test_expected_calibration_error(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.3, 0.6, 0.4])
        ece = _expected_calibration_error(y_true, y_proba, n_bins=5)
        assert 0.0 <= ece <= 1.0

    def test_ece_perfect_calibration_is_low(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.0, 1.0, 0.0, 1.0])
        assert _expected_calibration_error(y_true, y_proba) == pytest.approx(0.0, abs=1e-9)
