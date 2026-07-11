"""Unit tests for Phase 8 — Observability & MLOps monitoring.

Covers:
- ml/monitoring/drift_detector.py — DriftDetector, DriftReport, FeatureDriftResult
- ml/monitoring/model_monitor.py  — PerformanceReport, _expected_calibration_error
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

# ── fixtures ───────────────────────────────────────────────────────────────────


def _ref_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.normal(45, 12, n).clip(18, 90),
            "bmi": rng.normal(27, 5, n).clip(15, 50),
            "num_relatives": rng.integers(0, 6, n).astype(float),
            "gender": rng.choice(["M", "F"], n),
            "patient_id": [f"p{i}" for i in range(n)],
            "label": rng.integers(0, 2, n).astype(float),
        }
    )


def _same_dist_df(ref: pd.DataFrame, n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Current data drawn from the same distribution as reference — no drift."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.normal(45, 12, n).clip(18, 90),
            "bmi": rng.normal(27, 5, n).clip(15, 50),
            "num_relatives": rng.integers(0, 6, n).astype(float),
            "gender": rng.choice(["M", "F"], n),
            "patient_id": [f"q{i}" for i in range(n)],
            "label": rng.integers(0, 2, n).astype(float),
        }
    )


def _shifted_df(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Current data with heavily shifted distributions — expect drift."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.normal(70, 5, n).clip(18, 90),  # mean 70 vs 45
            "bmi": rng.normal(35, 3, n).clip(15, 50),  # mean 35 vs 27
            "num_relatives": rng.integers(3, 7, n).astype(float),
            "gender": rng.choice(["M", "F", "Other"], n),  # extra category
            "patient_id": [f"r{i}" for i in range(n)],
            "label": rng.integers(0, 2, n).astype(float),
        }
    )


# ── DriftDetector: basic behaviour ────────────────────────────────────────────


class TestDriftDetectorNoDrift:
    def test_returns_drift_report(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector, DriftReport

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_same_dist_df(ref))
        assert isinstance(report, DriftReport)

    def test_no_drift_same_distribution(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_same_dist_df(ref))
        assert report.dataset_drifted is False

    def test_feature_count_excludes_patient_id_and_label(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_same_dist_df(ref))
        # patient_id and label excluded; age, bmi, num_relatives, gender remain
        assert report.n_features == 4

    def test_drift_share_between_zero_and_one(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_same_dist_df(ref))
        assert 0.0 <= report.drift_share <= 1.0

    def test_reference_and_current_n(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df(n=200)
        cur = _same_dist_df(ref, n=150)
        det = DriftDetector(reference_df=ref)
        report = det.run(cur)
        assert report.reference_n == 200
        assert report.current_n == 150


class TestDriftDetectorWithDrift:
    def test_dataset_drifted_on_shifted_distribution(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_shifted_df())
        assert report.dataset_drifted is True

    def test_n_drifted_greater_than_half_on_shifted(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_shifted_df())
        assert report.n_drifted > report.n_features // 2

    def test_feature_results_populated(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_shifted_df())
        assert len(report.feature_results) == report.n_features

    def test_drift_share_equals_n_drifted_over_n_features(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_shifted_df())
        expected = report.n_drifted / report.n_features
        assert abs(report.drift_share - expected) < 1e-9


# ── DriftDetector: prediction drift ───────────────────────────────────────────


class TestPredictionDrift:
    def test_no_prediction_drift_when_col_absent(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_same_dist_df(ref), prediction_col="score")
        assert report.prediction_drift_detected is False
        assert report.prediction_p_value is None

    def test_no_prediction_drift_same_distribution(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        rng = np.random.default_rng(0)
        ref = _ref_df()
        ref["score"] = rng.uniform(0, 1, len(ref))
        cur = _same_dist_df(ref)
        cur["score"] = rng.uniform(0, 1, len(cur))
        det = DriftDetector(reference_df=ref)
        report = det.run(cur, prediction_col="score")
        assert report.prediction_p_value is not None
        # Same distribution should yield high p-value (no drift)
        assert report.prediction_p_value > 0.01

    def test_prediction_drift_detected_on_shift(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        rng = np.random.default_rng(1)
        n = 300
        ref = pd.DataFrame({"age": rng.normal(45, 5, n), "score": rng.beta(2, 5, n)})
        cur = pd.DataFrame({"age": rng.normal(45, 5, n), "score": rng.beta(5, 2, n)})
        det = DriftDetector(reference_df=ref)
        report = det.run(cur, prediction_col="score")
        assert report.prediction_drift_detected is True


# ── DriftDetector: categorical feature handling ────────────────────────────────


class TestCategoricalDrift:
    def test_categorical_auto_detected_for_object_dtype(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        assert "gender" in det._cat_features

    def test_categorical_not_in_numeric_features(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_same_dist_df(ref))
        gender_result = next((r for r in report.feature_results if r.feature == "gender"), None)
        assert gender_result is not None
        assert gender_result.test_name == "chi2_ks"

    def test_numeric_uses_ks_test(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_same_dist_df(ref))
        age_result = next(r for r in report.feature_results if r.feature == "age")
        assert age_result.test_name == "ks"

    def test_explicit_categorical_override(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref, categorical_features=["age", "bmi"])
        assert "age" in det._cat_features
        assert "bmi" in det._cat_features

    def test_numeric_result_has_means(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = _ref_df()
        det = DriftDetector(reference_df=ref)
        report = det.run(_same_dist_df(ref))
        age_result = next(r for r in report.feature_results if r.feature == "age")
        assert age_result.reference_mean is not None
        assert age_result.current_mean is not None
        assert abs(age_result.reference_mean - 45) < 5  # roughly 45


# ── DriftDetector: small-sample guard ─────────────────────────────────────────


class TestSmallSampleGuard:
    def test_skips_feature_with_fewer_than_5_values(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        cur = pd.DataFrame({"x": [4.0, 5.0, 6.0]})
        det = DriftDetector(reference_df=ref)
        report = det.run(cur)
        assert report.n_features == 0

    def test_analyses_feature_at_exactly_five_values(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        cur = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
        det = DriftDetector(reference_df=ref)
        report = det.run(cur)
        assert report.n_features == 1


# ── FeatureDriftResult ─────────────────────────────────────────────────────────


class TestFeatureDriftResult:
    def test_drift_detected_when_p_below_threshold(self) -> None:
        from ml.monitoring.drift_detector import FeatureDriftResult

        r = FeatureDriftResult(
            feature="age",
            drift_detected=True,
            statistic=0.9,
            p_value=0.001,
            test_name="ks",
            reference_mean=45.0,
            current_mean=70.0,
        )
        assert r.drift_detected is True

    def test_no_drift_when_p_above_threshold(self) -> None:
        from ml.monitoring.drift_detector import FeatureDriftResult

        r = FeatureDriftResult(
            feature="bmi",
            drift_detected=False,
            statistic=0.05,
            p_value=0.8,
            test_name="ks",
            reference_mean=27.0,
            current_mean=27.5,
        )
        assert r.drift_detected is False


# ── DriftReport ────────────────────────────────────────────────────────────────


class TestDriftReport:
    def _make_report(
        self,
        n_features: int = 4,
        n_drifted: int = 2,
        dataset_drifted: bool = False,
        prediction_drift: bool = False,
        pred_p: float | None = None,
    ):
        from ml.monitoring.drift_detector import DriftReport, FeatureDriftResult

        results = [
            FeatureDriftResult(
                feature=f"f{i}",
                drift_detected=(i < n_drifted),
                statistic=0.5 if i < n_drifted else 0.1,
                p_value=0.001 if i < n_drifted else 0.5,
                test_name="ks",
            )
            for i in range(n_features)
        ]
        return DriftReport(
            dataset_drifted=dataset_drifted,
            drift_share=n_drifted / n_features if n_features else 0.0,
            n_features=n_features,
            n_drifted=n_drifted,
            feature_results=results,
            prediction_drift_detected=prediction_drift,
            prediction_p_value=pred_p,
            reference_n=200,
            current_n=150,
        )

    def test_to_mlflow_metrics_contains_required_keys(self) -> None:
        report = self._make_report()
        m = report.to_mlflow_metrics()
        for key in (
            "drift/dataset_drifted",
            "drift/drift_share",
            "drift/n_features",
            "drift/n_drifted",
            "drift/prediction_drifted",
        ):
            assert key in m, f"Missing key: {key}"

    def test_to_mlflow_metrics_all_values_are_float(self) -> None:
        report = self._make_report(prediction_drift=True, pred_p=0.03)
        for k, v in report.to_mlflow_metrics().items():
            assert isinstance(v, float), f"{k} is not float"

    def test_to_mlflow_metrics_per_feature_keys(self) -> None:
        report = self._make_report(n_features=3, n_drifted=1)
        m = report.to_mlflow_metrics()
        assert "drift/feature/f0/p_value" in m
        assert "drift/feature/f1/p_value" in m

    def test_to_mlflow_metrics_prediction_p_value_included_when_set(self) -> None:
        report = self._make_report(prediction_drift=True, pred_p=0.04)
        m = report.to_mlflow_metrics()
        assert "drift/prediction_p_value" in m
        assert abs(m["drift/prediction_p_value"] - 0.04) < 1e-9

    def test_to_mlflow_metrics_prediction_p_value_absent_when_none(self) -> None:
        report = self._make_report(pred_p=None)
        assert "drift/prediction_p_value" not in report.to_mlflow_metrics()

    def test_to_json_is_valid_json(self) -> None:
        report = self._make_report()
        parsed = json.loads(report.to_json())
        assert "dataset_drifted" in parsed
        assert "feature_results" in parsed

    def test_to_json_round_trips_n_features(self) -> None:
        report = self._make_report(n_features=5, n_drifted=2)
        parsed = json.loads(report.to_json())
        assert parsed["n_features"] == 5
        assert parsed["n_drifted"] == 2

    def test_top_drifted_features_sorted_ascending_p_value(self) -> None:
        report = self._make_report(n_features=4, n_drifted=2)
        top = report.top_drifted_features
        assert all(r.drift_detected for r in top)
        p_values = [r.p_value for r in top]
        assert p_values == sorted(p_values)

    def test_top_drifted_features_excludes_no_drift(self) -> None:
        report = self._make_report(n_features=4, n_drifted=1)
        assert len(report.top_drifted_features) == 1


# ── _expected_calibration_error ───────────────────────────────────────────────


class TestExpectedCalibrationError:
    def _ece(self, y_true, y_proba, n_bins: int = 10) -> float:
        from ml.monitoring.model_monitor import _expected_calibration_error

        return _expected_calibration_error(
            np.array(y_true, dtype=float),
            np.array(y_proba, dtype=float),
            n_bins=n_bins,
        )

    def test_perfect_calibration_is_near_zero(self) -> None:
        # Perfect calibration: predicted probability == empirical frequency per bin
        rng = np.random.default_rng(0)
        y_proba = rng.uniform(0, 1, 1000)
        y_true = (rng.uniform(0, 1, 1000) < y_proba).astype(float)
        ece = self._ece(y_true, y_proba)
        assert ece < 0.05

    def test_constant_overconfident_has_high_ece(self) -> None:
        # Always predict 1.0, but only 50% are positive
        y_true = np.array([1.0, 0.0] * 50)
        y_proba = np.ones(100)
        ece = self._ece(y_true, y_proba)
        assert ece > 0.3

    def test_ece_in_zero_one_range(self) -> None:
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 2, 200).astype(float)
        y_proba = rng.uniform(0, 1, 200)
        ece = self._ece(y_true, y_proba)
        assert 0.0 <= ece <= 1.0

    def test_empty_bins_skipped_no_error(self) -> None:
        # All predictions in one bin — other bins are empty
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_proba = np.array([0.45, 0.55, 0.48, 0.52])
        ece = self._ece(y_true, y_proba, n_bins=10)
        assert math.isfinite(ece)

    def test_ece_symmetric_around_midpoint(self) -> None:
        # Predicting 0.2 for all but 80% are positive should give same ECE
        # as predicting 0.8 for all but 20% are positive
        y_true_a = np.array([1.0] * 80 + [0.0] * 20)
        y_proba_a = np.full(100, 0.2)
        y_true_b = np.array([0.0] * 80 + [1.0] * 20)
        y_proba_b = np.full(100, 0.8)
        ece_a = self._ece(y_true_a, y_proba_a)
        ece_b = self._ece(y_true_b, y_proba_b)
        assert abs(ece_a - ece_b) < 1e-9

    def test_known_value(self) -> None:
        # 100 samples, all in [0.4, 0.6) bin (single bin with n_bins=5)
        # bin confidence = 0.5, bin accuracy = 0.6 → ECE = 1.0 * |0.6 - 0.5| = 0.1
        y_true = np.array([1.0] * 60 + [0.0] * 40)
        y_proba = np.full(100, 0.5)
        ece = self._ece(y_true, y_proba, n_bins=5)
        assert abs(ece - 0.1) < 1e-9


# ── PerformanceReport ──────────────────────────────────────────────────────────


class TestPerformanceReport:
    def _make_report(
        self,
        roc_auc: float | None = 0.80,
        brier: float | None = 0.15,
        ece: float | None = 0.05,
        alerts: list[str] | None = None,
        sufficient_data: bool = True,
        n: int = 100,
        n_labeled: int = 40,
    ):
        from ml.monitoring.model_monitor import PerformanceReport

        return PerformanceReport(
            evaluation_date="2026-04-23",
            lookback_days=30,
            n_predictions=n,
            n_labeled=n_labeled,
            roc_auc=roc_auc,
            brier_score=brier,
            ece=ece,
            alerts=alerts or [],
            sufficient_data=sufficient_data,
        )

    def test_has_degradation_false_when_no_alerts(self) -> None:
        assert self._make_report(alerts=[]).has_degradation() is False

    def test_has_degradation_true_when_alerts(self) -> None:
        assert self._make_report(alerts=["ROC-AUC 0.60 < threshold 0.70"]).has_degradation() is True

    def test_to_mlflow_metrics_required_keys(self) -> None:
        report = self._make_report()
        m = report.to_mlflow_metrics()
        for key in (
            "monitor/n_predictions",
            "monitor/n_labeled",
            "monitor/sufficient_data",
            "monitor/n_alerts",
        ):
            assert key in m, f"Missing key: {key}"

    def test_to_mlflow_metrics_includes_roc_auc_when_set(self) -> None:
        assert "monitor/roc_auc" in self._make_report(roc_auc=0.75).to_mlflow_metrics()

    def test_to_mlflow_metrics_excludes_roc_auc_when_none(self) -> None:
        assert "monitor/roc_auc" not in self._make_report(roc_auc=None).to_mlflow_metrics()

    def test_to_mlflow_metrics_includes_brier_when_set(self) -> None:
        assert "monitor/brier_score" in self._make_report(brier=0.18).to_mlflow_metrics()

    def test_to_mlflow_metrics_excludes_brier_when_none(self) -> None:
        assert "monitor/brier_score" not in self._make_report(brier=None).to_mlflow_metrics()

    def test_to_mlflow_metrics_includes_ece_when_set(self) -> None:
        assert "monitor/ece" in self._make_report(ece=0.07).to_mlflow_metrics()

    def test_to_mlflow_metrics_excludes_ece_when_none(self) -> None:
        assert "monitor/ece" not in self._make_report(ece=None).to_mlflow_metrics()

    def test_to_mlflow_metrics_all_values_are_float(self) -> None:
        report = self._make_report()
        for k, v in report.to_mlflow_metrics().items():
            assert isinstance(v, float), f"{k} is not float"

    def test_n_alerts_equals_alert_list_length(self) -> None:
        alerts = ["alert1", "alert2", "alert3"]
        m = self._make_report(alerts=alerts).to_mlflow_metrics()
        assert m["monitor/n_alerts"] == 3.0

    def test_sufficient_data_false_when_insufficient(self) -> None:
        report = self._make_report(sufficient_data=False, roc_auc=None, brier=None, ece=None)
        assert report.sufficient_data is False
        assert report.has_degradation() is False


# ── Alert threshold logic ──────────────────────────────────────────────────────


class TestAlertThresholds:
    """Verify threshold constants are honoured by hand-constructing reports."""

    def test_roc_auc_threshold_is_0_70(self) -> None:
        from ml.monitoring.model_monitor import _ALERT_THRESHOLDS

        assert _ALERT_THRESHOLDS["roc_auc"] == 0.70

    def test_brier_threshold_is_0_25(self) -> None:
        from ml.monitoring.model_monitor import _ALERT_THRESHOLDS

        assert _ALERT_THRESHOLDS["brier_score"] == 0.25

    def test_ece_threshold_is_0_10(self) -> None:
        from ml.monitoring.model_monitor import _ALERT_THRESHOLDS

        assert _ALERT_THRESHOLDS["ece"] == 0.10

    def test_min_samples_is_30(self) -> None:
        from ml.monitoring.model_monitor import _MIN_SAMPLES_FOR_EVALUATION

        assert _MIN_SAMPLES_FOR_EVALUATION == 30


# ── DriftDetector: _ks_test and _chi2_test unit tests ─────────────────────────


class TestStatisticalTests:
    def test_ks_test_same_distribution_high_p(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 300)
        b = rng.normal(0, 1, 300)
        _, p = DriftDetector._ks_test(a, b)
        assert p > 0.05

    def test_ks_test_different_distribution_low_p(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 300)
        b = rng.normal(5, 1, 300)  # 5 sigma shift
        _, p = DriftDetector._ks_test(a, b)
        assert p < 0.001

    def test_ks_test_returns_two_floats(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        stat, p = DriftDetector._ks_test(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        assert isinstance(stat, float)
        assert isinstance(p, float)

    def test_chi2_test_same_dist_high_p(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = pd.Series(["A"] * 100 + ["B"] * 100)
        cur = pd.Series(["A"] * 95 + ["B"] * 105)
        _, p, name = DriftDetector._chi2_test(ref, cur)
        assert p > 0.05
        assert name == "chi2_ks"

    def test_chi2_test_different_dist_low_p(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = pd.Series(["A"] * 1 + ["B"] * 999)
        cur = pd.Series(["A"] * 999 + ["B"] * 1)
        _, p, _ = DriftDetector._chi2_test(ref, cur)
        assert p < 0.001


# ── DriftDetector: _EXCLUDE_COLS guard ────────────────────────────────────────


class TestExcludeCols:
    def test_patient_id_excluded(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = pd.DataFrame(
            {"patient_id": range(100), "x": np.random.default_rng(0).normal(0, 1, 100)}
        )
        det = DriftDetector(reference_df=ref)
        report = det.run(
            pd.DataFrame(
                {"patient_id": range(100), "x": np.random.default_rng(1).normal(5, 1, 100)}
            )
        )
        features = [r.feature for r in report.feature_results]
        assert "patient_id" not in features

    def test_feature_date_excluded(self) -> None:
        from ml.monitoring.drift_detector import DriftDetector

        ref = pd.DataFrame(
            {"feature_date": ["2024-01-01"] * 100, "x": np.random.default_rng(0).normal(0, 1, 100)}
        )
        det = DriftDetector(reference_df=ref)
        report = det.run(
            pd.DataFrame(
                {
                    "feature_date": ["2025-01-01"] * 100,
                    "x": np.random.default_rng(1).normal(5, 1, 100),
                }
            )
        )
        features = [r.feature for r in report.feature_results]
        assert "feature_date" not in features
