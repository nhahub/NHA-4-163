"""Post-hoc probability calibration for ML models.

Wraps sklearn's calibration utilities to provide a uniform interface for
both XGBoost and GNN models.

Method selection guide
----------------------
- **Platt (sigmoid)**: fast, works well when the base model's scores are
  monotone but poorly scaled (common in XGBoost with class imbalance).
  Use when calibration set is small (< 1 000 samples).
- **Isotonic**: non-parametric, corrects any monotone distortion.  Requires
  at least ~1 000 calibration samples to avoid overfitting.  Preferred when
  the reliability diagram shows non-sigmoid deviations.

CLAUDE.md requirement: every model MUST pass Brier score + reliability
diagram checks before being registered.  This module supports that check.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)


class CalibrationMethod(StrEnum):
    """Calibration method choices."""

    SIGMOID = "sigmoid"  # Platt scaling
    ISOTONIC = "isotonic"  # Isotonic regression


class _PredictProbaWrapper:
    """Thin wrapper around a predict_proba callable."""

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._fn = predict_fn

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities for compatibility helpers."""
        return self._fn(X).reshape(-1)


class CalibratedModel:
    """Calibrated probability estimator wrapping any predict_proba callable.

    Attributes:
        method: Which calibration method was applied.
        brier_before: Brier score on calibration data before calibration.
        brier_after: Brier score on calibration data after calibration.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        method: CalibrationMethod,
        brier_before: float,
        brier_after: float,
    ) -> None:
        self._predict_fn = predict_fn
        self.method = method
        self.brier_before = brier_before
        self.brier_after = brier_after

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated positive-class probabilities.

        Args:
            X: Feature matrix (n_samples × n_features).

        Returns:
            1-D array of calibrated probabilities in [0, 1].
        """
        pos = self._predict_fn(X).reshape(-1)
        return np.column_stack([1.0 - pos, pos])


def calibrate(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: CalibrationMethod = CalibrationMethod.SIGMOID,
) -> CalibratedModel:
    """Fit a post-hoc calibrator on held-out calibration data.

    Args:
        predict_fn: Function mapping feature matrix → positive-class
            probabilities (1-D array).  Must not have been trained on
            ``X_cal``/``y_cal`` — use a held-out calibration split.
        X_cal: Calibration feature matrix.
        y_cal: Binary calibration labels (0/1).
        method: ``CalibrationMethod.SIGMOID`` (Platt) or
            ``CalibrationMethod.ISOTONIC``.

    Returns:
        A ``CalibratedModel`` whose ``predict_proba`` returns calibrated
        probabilities.

    Raises:
        ValueError: If ``X_cal`` is empty or ``y_cal`` has only one class.
    """
    from sklearn.metrics import brier_score_loss

    if X_cal.shape[0] == 0:
        raise ValueError("Calibration set is empty")
    if len(np.unique(y_cal)) < 2:
        raise ValueError("Calibration set must contain both positive and negative examples")

    raw_proba = predict_fn(X_cal)
    brier_before = float(brier_score_loss(y_cal, raw_proba))

    wrapper = _PredictProbaWrapper(predict_fn)
    raw_proba = wrapper.predict(X_cal)

    if method is CalibrationMethod.SIGMOID:
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(raw_proba.reshape(-1, 1), y_cal)

        def calibrated_predict_fn(X: np.ndarray) -> np.ndarray:
            scores = wrapper.predict(X).reshape(-1, 1)
            return calibrator.predict_proba(scores)[:, 1]

    else:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_proba, y_cal)

        def calibrated_predict_fn(X: np.ndarray) -> np.ndarray:
            return calibrator.predict(wrapper.predict(X))

    cal_proba = calibrated_predict_fn(X_cal)
    brier_after = float(brier_score_loss(y_cal, cal_proba))

    log.info(
        "Calibration (%s): Brier %.4f → %.4f (Δ=%.4f)",
        method.value,
        brier_before,
        brier_after,
        brier_after - brier_before,
    )
    return CalibratedModel(calibrated_predict_fn, method, brier_before, brier_after)
