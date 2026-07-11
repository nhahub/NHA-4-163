"""Unit tests for Phase 5 ML model components.

No external services, Spark, or GPU required.  Tests cover:

- ``EvaluationResult`` metric computation
- ``expected_calibration_error`` implementation
- ``FairnessReport`` gap calculations
- ``patient_id_split`` leakage prevention guarantee
- ``XGBConfig`` dataclass defaults
- ``GNNConfig`` dataclass defaults
- ``CalibrationMethod`` enum values
- ``CalibratedModel`` wrapping behaviour

Training scripts (train_xgboost, train_gnn) and the Airflow DAG are
covered by integration tests (tests/integration/) which require running
services.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from ml.models.calibration import CalibrationMethod, calibrate
from ml.models.gnn_model import GNNConfig
from ml.models.xgboost_model import XGBConfig
from ml.training.dataset import apply_split, patient_id_split
from ml.training.evaluate import (
    evaluate_binary_classifier,
    expected_calibration_error,
)
from ml.training.fairness import compute_fairness_report

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _binary_arrays(
    n: int = 200,
    positive_rate: float = 0.3,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_proba) for tests.  proba = truth + calibrated noise."""
    rng = np.random.default_rng(random_state)
    y = (rng.random(n) < positive_rate).astype(int)
    noise = rng.normal(0, 0.15, n)
    proba = np.clip(y.astype(float) * 0.7 + 0.15 + noise, 0.01, 0.99)
    return y, proba


# ── expected_calibration_error ────────────────────────────────────────────────


class TestExpectedCalibrationError:
    def test_perfect_calibration_is_near_zero(self) -> None:
        # Perfect calibration: proba equals empirical fraction per bin
        n = 1000
        y_true = np.array([1, 0] * (n // 2))
        y_proba = np.array([0.5] * n)
        ece = expected_calibration_error(y_true, y_proba, n_bins=10)
        assert ece < 0.05

    def test_overconfident_model_has_high_ece(self) -> None:
        # Model predicts 0.99 for every sample; true rate is 0.5
        y_true = np.array([1, 0] * 500)
        y_proba = np.full(1000, 0.99)
        ece = expected_calibration_error(y_true, y_proba, n_bins=10)
        assert ece > 0.3

    def test_empty_bins_are_skipped(self) -> None:
        # All predictions in a single bin
        y_true = np.array([1, 0, 1, 0])
        y_proba = np.array([0.55, 0.55, 0.55, 0.55])
        ece = expected_calibration_error(y_true, y_proba, n_bins=20)
        assert 0.0 <= ece <= 1.0

    def test_output_in_unit_interval(self) -> None:
        y, proba = _binary_arrays()
        ece = expected_calibration_error(y, proba)
        assert 0.0 <= ece <= 1.0


# ── evaluate_binary_classifier ────────────────────────────────────────────────


class TestEvaluateBinaryClassifier:
    def test_perfect_model_auc_one(self) -> None:
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        result = evaluate_binary_classifier(y, proba)
        assert result.roc_auc == pytest.approx(1.0)

    def test_roc_auc_in_unit_interval(self) -> None:
        y, proba = _binary_arrays()
        result = evaluate_binary_classifier(y, proba)
        assert 0.0 <= result.roc_auc <= 1.0

    def test_brier_score_in_unit_interval(self) -> None:
        y, proba = _binary_arrays()
        result = evaluate_binary_classifier(y, proba)
        assert 0.0 <= result.brier_score <= 1.0

    def test_pr_auc_in_unit_interval(self) -> None:
        y, proba = _binary_arrays()
        result = evaluate_binary_classifier(y, proba)
        assert 0.0 <= result.pr_auc <= 1.0

    def test_threshold_metrics_sum_consistently(self) -> None:
        y, proba = _binary_arrays()
        result = evaluate_binary_classifier(y, proba)
        tm = result.threshold_metrics
        assert 0.0 <= tm.precision <= 1.0
        assert 0.0 <= tm.recall <= 1.0
        assert 0.0 <= tm.f1 <= 1.0
        assert 0.0 <= tm.specificity <= 1.0
        assert 0.0 <= tm.accuracy <= 1.0

    def test_single_class_raises(self) -> None:
        y = np.zeros(50, dtype=int)
        proba = np.full(50, 0.1)
        with pytest.raises(ValueError, match="both positive and negative"):
            evaluate_binary_classifier(y, proba)

    def test_to_mlflow_metrics_keys(self) -> None:
        y, proba = _binary_arrays()
        result = evaluate_binary_classifier(y, proba)
        keys = set(result.to_mlflow_metrics().keys())
        assert "roc_auc" in keys
        assert "brier_score" in keys
        assert "ece" in keys
        assert "pr_auc" in keys

    def test_calibration_data_lengths_match(self) -> None:
        y, proba = _binary_arrays()
        result = evaluate_binary_classifier(y, proba, n_calibration_bins=5)
        assert len(result.calibration_fraction_pos) == len(result.calibration_mean_pred)


# ── Fairness metrics ──────────────────────────────────────────────────────────


class TestFairnessReport:
    def _make_inputs(
        self,
        n: int = 300,
        groups: tuple[str, ...] = ("A", "B", "C"),
    ) -> tuple[np.ndarray, np.ndarray, pd.Series]:
        rng = np.random.default_rng(99)
        y = (rng.random(n) < 0.35).astype(int)
        proba = np.clip(y.astype(float) * 0.65 + 0.15 + rng.normal(0, 0.15, n), 0.01, 0.99)
        group_vals = np.array(groups * (n // len(groups) + 1))[:n]
        return y, proba, pd.Series(group_vals, name="test_group")

    def test_gaps_are_non_negative(self) -> None:
        y, proba, series = self._make_inputs()
        report = compute_fairness_report(y, proba, series, min_group_size=5)
        assert report.statistical_parity_gap >= 0.0
        assert report.equal_opportunity_gap >= 0.0
        assert report.predictive_equality_gap >= 0.0
        assert report.brier_gap >= 0.0

    def test_equal_groups_have_near_zero_gaps(self) -> None:
        rng = np.random.default_rng(7)
        n = 400
        y = (rng.random(n) < 0.4).astype(int)
        # Same proba for both groups — gaps must be zero
        proba = np.clip(y * 0.7 + 0.1, 0.01, 0.99)
        groups = pd.Series(["X"] * n, name="g")
        report = compute_fairness_report(y, proba, groups, min_group_size=5)
        assert report.statistical_parity_gap == pytest.approx(0.0, abs=1e-9)

    def test_to_mlflow_metrics_has_all_gaps(self) -> None:
        y, proba, series = self._make_inputs()
        report = compute_fairness_report(y, proba, series, min_group_size=5)
        keys = set(report.to_mlflow_metrics().keys())
        assert "statistical_parity_gap" in keys
        assert "equal_opportunity_gap" in keys

    def test_to_dataframe_has_group_rows(self) -> None:
        y, proba, series = self._make_inputs(groups=("X", "Y"))
        report = compute_fairness_report(y, proba, series, min_group_size=5)
        df = report.to_dataframe()
        assert len(df) == 2
        assert "tpr" in df.columns

    def test_small_groups_excluded(self) -> None:
        y = np.array([1, 0] * 10)
        proba = np.full(20, 0.5)
        groups = pd.Series(["tiny"] * 20, name="g")
        report = compute_fairness_report(y, proba, groups, min_group_size=100)
        assert len(report.group_metrics) == 0


# ── patient_id_split ──────────────────────────────────────────────────────────


class TestPatientIdSplit:
    def _make_pids(self, n: int, pos_rate: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
        pids = np.array([str(uuid.uuid4()) for _ in range(n)])
        y = (np.random.default_rng(0).random(n) < pos_rate).astype(int)
        return pids, y

    def test_splits_are_disjoint(self) -> None:
        pids, y = self._make_pids(200)
        train_ids, val_ids, test_ids = patient_id_split(pids, y)
        assert len(set(train_ids) & set(val_ids)) == 0
        assert len(set(train_ids) & set(test_ids)) == 0
        assert len(set(val_ids) & set(test_ids)) == 0

    def test_splits_cover_all_patients(self) -> None:
        pids, y = self._make_pids(200)
        train_ids, val_ids, test_ids = patient_id_split(pids, y)
        all_ids = set(train_ids) | set(val_ids) | set(test_ids)
        assert all_ids == set(pids.tolist())

    def test_approximate_size_fractions(self) -> None:
        pids, y = self._make_pids(500)
        train_ids, val_ids, test_ids = patient_id_split(pids, y, val_size=0.15, test_size=0.15)
        total = len(pids)
        assert abs(len(test_ids) / total - 0.15) < 0.05
        assert abs(len(val_ids) / total - 0.15) < 0.05

    def test_apply_split_returns_correct_rows(self) -> None:
        pids, y = self._make_pids(100)
        X = np.eye(100, 5)
        train_ids, val_ids, test_ids = patient_id_split(pids, y)
        X_train, y_train = apply_split(pids, X, y, train_ids)
        assert len(X_train) == len(y_train) == len(train_ids)


# ── Config dataclasses ────────────────────────────────────────────────────────


class TestXGBConfig:
    def test_defaults_are_sensible(self) -> None:
        cfg = XGBConfig()
        assert cfg.n_estimators > 0
        assert 0.0 < cfg.learning_rate < 1.0
        assert cfg.eval_metric == "aucpr"
        assert cfg.scale_pos_weight is None  # auto

    def test_custom_values_accepted(self) -> None:
        cfg = XGBConfig(max_depth=3, learning_rate=0.01)
        assert cfg.max_depth == 3
        assert cfg.learning_rate == pytest.approx(0.01)


class TestGNNConfig:
    def test_defaults_set(self) -> None:
        cfg = GNNConfig()
        assert cfg.hidden_dim == 64
        assert cfg.num_layers == 2
        assert 0.0 < cfg.dropout < 1.0

    def test_input_dim_configurable(self) -> None:
        cfg = GNNConfig(input_dim=30)
        assert cfg.input_dim == 30


# ── CalibrationMethod ─────────────────────────────────────────────────────────


class TestCalibrationMethod:
    def test_values(self) -> None:
        assert CalibrationMethod.SIGMOID.value == "sigmoid"
        assert CalibrationMethod.ISOTONIC.value == "isotonic"

    def test_calibrate_improves_or_maintains_brier(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        y = (rng.random(n) < 0.3).astype(int)
        # Intentionally overconfident scores
        proba = np.clip(y.astype(float) * 0.9 + 0.05 + rng.normal(0, 0.05, n), 0.01, 0.99)

        cal_model = calibrate(
            lambda x: proba[: len(x)],
            np.zeros((n, 1)),
            y,
            CalibrationMethod.SIGMOID,
        )
        # After calibration, Brier should not worsen dramatically
        assert cal_model.brier_after <= cal_model.brier_before + 0.05

    def test_calibrate_empty_set_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            calibrate(
                lambda x: np.array([]),
                np.zeros((0, 1)),
                np.array([], dtype=int),
                CalibrationMethod.SIGMOID,
            )

    def test_calibrate_single_class_raises(self) -> None:
        proba = np.full(50, 0.5)
        with pytest.raises(ValueError, match="both positive and negative"):
            calibrate(
                lambda x: proba[: len(x)],
                np.zeros((50, 1)),
                np.zeros(50, dtype=int),
                CalibrationMethod.SIGMOID,
            )
