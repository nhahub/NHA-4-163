"""Data and prediction drift detection using Evidently.

Compares the current production feature distribution against a reference
dataset (typically the training data distribution) to detect:

- **Data drift** — individual feature distributions have shifted
  (detected per-feature with statistical tests; Kolmogorov-Smirnov for
  continuous features, chi-squared for categoricals).
- **Dataset drift** — the share of drifted features exceeds a threshold
  (default: 50 % of features drifted → dataset is considered drifted).
- **Prediction drift** — the distribution of model output scores has
  shifted (indicates covariate or label shift).

Usage::

    from ml.monitoring.drift_detector import DriftDetector, DriftReport

    detector = DriftDetector(reference_df=train_df)
    report = detector.run(current_df=production_df)

    if report.dataset_drifted:
        trigger_retraining()

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_metrics(report.to_mlflow_metrics())
        mlflow.log_text(report.to_json(), "drift_report.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Features to exclude from drift analysis (identifiers / targets)
_EXCLUDE_COLS = frozenset({"patient_id", "feature_date", "label", "split"})

# Drift detection thresholds
_DATASET_DRIFT_THRESHOLD = 0.5  # >50% features drifted → dataset drift
_FEATURE_DRIFT_P_VALUE = 0.05  # p-value threshold for per-feature tests


@dataclass
class FeatureDriftResult:
    """Drift statistics for a single feature.

    Attributes:
        feature: Feature name.
        drift_detected: Whether drift was detected (p < threshold).
        statistic: Test statistic value.
        p_value: p-value from the statistical test.
        test_name: Name of the test applied (``ks`` or ``chi2``).
        reference_mean: Mean in the reference dataset (numeric features).
        current_mean: Mean in the current dataset (numeric features).
    """

    feature: str
    drift_detected: bool
    statistic: float
    p_value: float
    test_name: str
    reference_mean: float | None = None
    current_mean: float | None = None


@dataclass
class DriftReport:
    """Summary of a drift detection run.

    Attributes:
        dataset_drifted: True when the share of drifted features exceeds
            ``_DATASET_DRIFT_THRESHOLD``.
        drift_share: Fraction of features where drift was detected.
        n_features: Total number of features analysed.
        n_drifted: Number of features with detected drift.
        feature_results: Per-feature drift results.
        prediction_drift_detected: True when the prediction score
            distribution has drifted.
        prediction_p_value: KS p-value for prediction distribution drift.
        reference_n: Row count of the reference dataset.
        current_n: Row count of the current dataset.
    """

    dataset_drifted: bool
    drift_share: float
    n_features: int
    n_drifted: int
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    prediction_drift_detected: bool = False
    prediction_p_value: float | None = None
    reference_n: int = 0
    current_n: int = 0

    def to_mlflow_metrics(self) -> dict[str, float]:
        """Return flat dict of metrics suitable for ``mlflow.log_metrics``.

        Returns:
            Dict with keys like ``drift/dataset_drifted``,
            ``drift/drift_share``, etc.
        """
        metrics: dict[str, float] = {
            "drift/dataset_drifted": float(self.dataset_drifted),
            "drift/drift_share": self.drift_share,
            "drift/n_features": float(self.n_features),
            "drift/n_drifted": float(self.n_drifted),
            "drift/prediction_drifted": float(self.prediction_drift_detected),
        }
        if self.prediction_p_value is not None:
            metrics["drift/prediction_p_value"] = self.prediction_p_value
        for r in self.feature_results:
            metrics[f"drift/feature/{r.feature}/p_value"] = r.p_value
        return metrics

    def to_json(self) -> str:
        """Serialise the full report to a JSON string.

        Returns:
            JSON string with all drift statistics.
        """
        d = asdict(self)
        return json.dumps(d, indent=2, default=str)

    @property
    def top_drifted_features(self) -> list[FeatureDriftResult]:
        """Return drifted features sorted by ascending p-value (most drifted first)."""
        return sorted(
            [r for r in self.feature_results if r.drift_detected],
            key=lambda r: r.p_value,
        )


class DriftDetector:
    """Detects data and prediction drift between a reference and current dataset.

    Uses pure numpy/scipy statistical tests so Evidently is optional —
    the detector degrades gracefully when Evidently is unavailable and
    falls back to scipy KS / chi-squared tests.

    Attributes:
        reference_df: Reference feature DataFrame (training distribution).
        categorical_features: Columns to treat as categorical (chi-squared
            test).  Auto-detected when ``None``.
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        categorical_features: list[str] | None = None,
    ) -> None:
        """Initialise with a reference (training) dataset.

        Args:
            reference_df: DataFrame of feature vectors from the training set.
            categorical_features: Columns to treat as categorical.
                Inferred from dtype when ``None``.
        """
        self._ref = reference_df.copy()
        self._cat_features = set(categorical_features or [])
        if not categorical_features:
            for col in self._ref.columns:
                if col in _EXCLUDE_COLS:
                    continue
                if self._ref[col].dtype == object or str(self._ref[col].dtype) == "category":
                    self._cat_features.add(col)

    def run(
        self,
        current_df: pd.DataFrame,
        prediction_col: str | None = None,
    ) -> DriftReport:
        """Run drift detection between the reference and current datasets.

        Args:
            current_df: DataFrame of current production feature vectors.
            prediction_col: Optional column name containing model scores.
                When provided, prediction distribution drift is also checked.

        Returns:
            ``DriftReport`` with per-feature and dataset-level results.
        """
        feature_cols = [
            c
            for c in self._ref.columns
            if c not in _EXCLUDE_COLS and c in current_df.columns and c != prediction_col
        ]

        results: list[FeatureDriftResult] = []
        for col in feature_cols:
            ref_vals = self._ref[col].dropna()
            cur_vals = current_df[col].dropna()
            if len(ref_vals) < 5 or len(cur_vals) < 5:
                continue
            result = self._test_feature(col, ref_vals, cur_vals)
            results.append(result)

        n_drifted = sum(1 for r in results if r.drift_detected)
        drift_share = n_drifted / len(results) if results else 0.0
        dataset_drifted = drift_share > _DATASET_DRIFT_THRESHOLD

        pred_drifted = False
        pred_p = None
        if (
            prediction_col
            and prediction_col in self._ref.columns
            and prediction_col in current_df.columns
        ):
            ref_pred = self._ref[prediction_col].dropna().values
            cur_pred = current_df[prediction_col].dropna().values
            if len(ref_pred) >= 5 and len(cur_pred) >= 5:
                stat, pred_p = self._ks_test(ref_pred, cur_pred)
                pred_drifted = pred_p < _FEATURE_DRIFT_P_VALUE

        log.info(
            "Drift report: %d/%d features drifted (%.1f%%), dataset_drifted=%s",
            n_drifted,
            len(results),
            drift_share * 100,
            dataset_drifted,
        )

        return DriftReport(
            dataset_drifted=dataset_drifted,
            drift_share=drift_share,
            n_features=len(results),
            n_drifted=n_drifted,
            feature_results=results,
            prediction_drift_detected=pred_drifted,
            prediction_p_value=pred_p,
            reference_n=len(self._ref),
            current_n=len(current_df),
        )

    def _test_feature(
        self,
        col: str,
        ref: pd.Series,
        cur: pd.Series,
    ) -> FeatureDriftResult:
        if col in self._cat_features:
            stat, p, test = self._chi2_test(ref, cur)
            return FeatureDriftResult(
                feature=col,
                drift_detected=p < _FEATURE_DRIFT_P_VALUE,
                statistic=stat,
                p_value=p,
                test_name=test,
            )
        ref_arr = ref.astype(float).values
        cur_arr = cur.astype(float).values
        stat, p = self._ks_test(ref_arr, cur_arr)
        return FeatureDriftResult(
            feature=col,
            drift_detected=p < _FEATURE_DRIFT_P_VALUE,
            statistic=float(stat),
            p_value=float(p),
            test_name="ks",
            reference_mean=float(np.mean(ref_arr)),
            current_mean=float(np.mean(cur_arr)),
        )

    @staticmethod
    def _ks_test(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
        """Two-sample Kolmogorov-Smirnov test."""
        from scipy import stats

        result = stats.ks_2samp(ref, cur)
        return float(result.statistic), float(result.pvalue)

    @staticmethod
    def _chi2_test(
        ref: pd.Series,
        cur: pd.Series,
    ) -> tuple[float, float, str]:
        """Chi-squared test on category frequencies."""
        from scipy import stats

        cats = sorted(set(ref.unique()) | set(cur.unique()))
        ref_counts = ref.value_counts().reindex(cats, fill_value=0)
        cur_counts = cur.value_counts().reindex(cats, fill_value=0)
        table = np.vstack([ref_counts.values, cur_counts.values])
        result = stats.chi2_contingency(table, correction=False)
        return float(result.statistic), float(result.pvalue), "chi2_ks"


def load_reference_dataset(
    delta_base: str,
    feature_date: str,
    spark: Any = None,
) -> pd.DataFrame:
    """Load the reference feature dataset from the Delta feature store.

    Args:
        delta_base: S3/MinIO base path for Delta tables (e.g.,
            ``s3a://healthcare-delta``).
        feature_date: ISO-8601 date of the training feature snapshot.
        spark: Optional SparkSession.  Creates a local session when ``None``.

    Returns:
        Pandas DataFrame of the reference feature vectors.
    """
    from pyspark.sql import SparkSession

    if spark is None:
        spark = SparkSession.builder.appName("drift-reference-loader").getOrCreate()

    path = f"{delta_base}/features/feature_vector/"
    df = spark.read.format("delta").load(path).filter(f"feature_date = '{feature_date}'").toPandas()
    log.info("Loaded reference dataset: %d rows from %s", len(df), path)
    return df
