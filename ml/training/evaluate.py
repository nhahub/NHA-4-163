"""Model evaluation metrics and reliability analysis.

Computes the full evaluation suite required before a model can be
registered (per CLAUDE.md: "Never mark a model as ready without
calibration — use Brier score + reliability diagrams").

Metrics computed
----------------
- ROC-AUC (area under the receiver-operating curve)
- PR-AUC  (area under the precision-recall curve)
- Brier score (mean squared probability error)
- ECE (Expected Calibration Error) — weighted average bin-wise |accuracy - confidence|
- Threshold metrics at 0.5: precision, recall, F1, specificity
- Reliability diagram data (fraction_positive, mean_predicted) for 10 bins

All values are logged to MLflow by the training scripts; this module
only computes, not logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


@dataclass
class ThresholdMetrics:
    """Binary classification metrics at a fixed decision threshold.

    Attributes:
        threshold: Decision threshold used.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN)  (a.k.a. sensitivity / TPR).
        f1: Harmonic mean of precision and recall.
        specificity: TN / (TN + FP)  (a.k.a. TNR).
        accuracy: (TP + TN) / total.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    specificity: float
    accuracy: float


@dataclass
class EvaluationResult:
    """Full evaluation result for a binary classifier.

    Attributes:
        roc_auc: Area under ROC curve.
        pr_auc: Area under precision-recall curve (more informative for imbalance).
        brier_score: Mean squared calibration error; lower is better.
        ece: Expected Calibration Error (0 = perfectly calibrated).
        threshold_metrics: Metrics at threshold=0.5.
        calibration_fraction_pos: Per-bin fraction of positives (reliability diagram y).
        calibration_mean_pred: Per-bin mean predicted probability (reliability diagram x).
        n_samples: Total evaluation sample count.
        n_positive: Count of positive labels.
    """

    roc_auc: float
    pr_auc: float
    brier_score: float
    ece: float
    threshold_metrics: ThresholdMetrics
    calibration_fraction_pos: list[float]
    calibration_mean_pred: list[float]
    n_samples: int
    n_positive: int

    def to_mlflow_metrics(self) -> dict[str, float]:
        """Return flat dict of all scalar metrics for MLflow ``log_metrics``.

        Returns:
            Dict mapping metric name to float value.
        """
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "precision_at_0.5": self.threshold_metrics.precision,
            "recall_at_0.5": self.threshold_metrics.recall,
            "f1_at_0.5": self.threshold_metrics.f1,
            "specificity_at_0.5": self.threshold_metrics.specificity,
            "accuracy_at_0.5": self.threshold_metrics.accuracy,
        }


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Divides predicted probabilities into ``n_bins`` equal-width bins and
    computes the weighted average of |mean_predicted - fraction_positive|.

    Args:
        y_true: Binary true labels.
        y_proba: Predicted positive-class probabilities.
        n_bins: Number of calibration bins.

    Returns:
        ECE value in [0, 1]; 0 = perfect calibration.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_proba[mask].mean())
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / n if n > 0 else 0.0


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    n_calibration_bins: int = 10,
) -> EvaluationResult:
    """Compute the full evaluation suite for a binary classifier.

    Args:
        y_true: Binary true labels (0/1), shape (n_samples,).
        y_proba: Predicted positive-class probabilities, shape (n_samples,).
        threshold: Decision threshold for threshold-based metrics.
        n_calibration_bins: Number of bins for reliability diagram and ECE.

    Returns:
        ``EvaluationResult`` with all computed metrics.

    Raises:
        ValueError: If ``y_true`` contains only one class (AUC undefined).
    """
    if len(np.unique(y_true)) < 2:
        raise ValueError("Evaluation set must contain both positive and negative examples")

    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = float(tp + tn) / float(len(y_true))

    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_calibration_bins)

    result = EvaluationResult(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        brier_score=float(brier_score_loss(y_true, y_proba)),
        ece=expected_calibration_error(y_true, y_proba, n_calibration_bins),
        threshold_metrics=ThresholdMetrics(
            threshold=threshold,
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            specificity=specificity,
            accuracy=accuracy,
        ),
        calibration_fraction_pos=frac_pos.tolist(),
        calibration_mean_pred=mean_pred.tolist(),
        n_samples=int(len(y_true)),
        n_positive=int(y_true.sum()),
    )

    log.info(
        "Eval — ROC-AUC: %.4f  PR-AUC: %.4f  Brier: %.4f  ECE: %.4f",
        result.roc_auc,
        result.pr_auc,
        result.brier_score,
        result.ece,
    )
    return result
