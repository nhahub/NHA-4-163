"""Algorithmic fairness analysis for binary classification.

Computes group-wise metrics and fairness gap measures for sensitive
attributes (age group, gender).  Results are stored as MLflow artifacts
so fairness is auditable before a model is promoted to production.

Fairness definitions used
-------------------------
- **Statistical parity difference**: P(ŷ=1 | A=a) − P(ŷ=1 | A=b).
  A value close to 0 means the model predicts positive outcomes at
  similar rates across groups.  NOT a clinical goal — prevalence differs
  across groups by biology — but flags severe prediction bias.

- **Equal opportunity difference**: TPR(A=a) − TPR(A=b).
  Measures whether the model identifies hereditary risk equally well
  across demographic groups.  The most clinically relevant fairness metric
  here because missing a high-risk patient is the worst failure mode.

- **Predictive equality difference**: FPR(A=a) − FPR(A=b).
  Measures whether healthy patients are incorrectly flagged at the same
  rate across groups.

All differences are reported as max_group − min_group (unsigned gap).

References: Hardt et al. "Equality of Opportunity in Supervised Learning" (2016).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class GroupMetrics:
    """Metrics for a single demographic subgroup.

    Attributes:
        group_value: The sensitive attribute value for this group.
        n: Sample count.
        n_positive: Count of true positives.
        prevalence: Fraction of true positives.
        predicted_positive_rate: Fraction predicted positive (at threshold).
        tpr: True positive rate (recall / sensitivity).
        fpr: False positive rate.
        brier_score: Mean squared probability error within the group.
        roc_auc: ROC-AUC within the group (None if only one class present).
    """

    group_value: str
    n: int
    n_positive: int
    prevalence: float
    predicted_positive_rate: float
    tpr: float
    fpr: float
    brier_score: float
    roc_auc: float | None


@dataclass
class FairnessReport:
    """Fairness analysis across demographic subgroups.

    Attributes:
        sensitive_column: Name of the sensitive feature column.
        group_metrics: Dict mapping group value to its GroupMetrics.
        statistical_parity_gap: Max − min predicted_positive_rate across groups.
        equal_opportunity_gap: Max − min TPR across groups.
        predictive_equality_gap: Max − min FPR across groups.
        brier_gap: Max − min Brier score across groups.
    """

    sensitive_column: str
    group_metrics: dict[str, GroupMetrics]
    statistical_parity_gap: float
    equal_opportunity_gap: float
    predictive_equality_gap: float
    brier_gap: float

    def to_mlflow_metrics(self, prefix: str = "") -> dict[str, float]:
        """Return flat dict of fairness gap metrics for MLflow.

        Args:
            prefix: Optional prefix appended to each key.

        Returns:
            Dict mapping metric name to float value.
        """
        pfx = f"{prefix}_" if prefix else ""
        return {
            f"{pfx}statistical_parity_gap": self.statistical_parity_gap,
            f"{pfx}equal_opportunity_gap": self.equal_opportunity_gap,
            f"{pfx}predictive_equality_gap": self.predictive_equality_gap,
            f"{pfx}brier_gap": self.brier_gap,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Serialise group metrics to a DataFrame for CSV/artifact logging.

        Returns:
            One row per group with all GroupMetrics fields as columns.
        """
        from dataclasses import asdict

        rows = [asdict(gm) for gm in self.group_metrics.values()]
        return pd.DataFrame(rows)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    group_value: str,
) -> GroupMetrics:
    """Compute metrics for a single group subset.

    Args:
        y_true: True binary labels for this group.
        y_pred: Hard predictions (0/1) for this group.
        y_proba: Positive-class probabilities for this group.
        group_value: Human-readable group identifier.

    Returns:
        ``GroupMetrics`` for the group.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    n = len(y_true)
    n_pos = int(y_true.sum())
    n - n_pos

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    tpr = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    ppr = float((y_pred == 1).sum()) / n if n > 0 else 0.0
    brier = float(brier_score_loss(y_true, y_proba)) if n > 0 else 0.0

    auc: float | None = None
    if len(np.unique(y_true)) >= 2:
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            auc = None

    return GroupMetrics(
        group_value=group_value,
        n=n,
        n_positive=n_pos,
        prevalence=float(n_pos) / n if n > 0 else 0.0,
        predicted_positive_rate=ppr,
        tpr=tpr,
        fpr=fpr,
        brier_score=brier,
        roc_auc=auc,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def compute_fairness_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sensitive_series: pd.Series,
    threshold: float = 0.5,
    min_group_size: int = 30,
) -> FairnessReport:
    """Compute per-group fairness metrics for a sensitive attribute.

    Groups with fewer than ``min_group_size`` samples are excluded from
    gap calculations (insufficient power for meaningful comparison).

    Args:
        y_true: Binary true labels, shape (n_samples,).
        y_proba: Positive-class probabilities, shape (n_samples,).
        sensitive_series: Categorical sensitive attribute values aligned
            with y_true (e.g. ``df["age_group"]`` or ``df["gender"]``).
        threshold: Decision threshold for hard predictions.
        min_group_size: Minimum group size to include in gap analysis.

    Returns:
        ``FairnessReport`` for the given sensitive attribute.
    """
    y_pred = (y_proba >= threshold).astype(int)
    col_name = sensitive_series.name or "sensitive_attribute"

    groups = sensitive_series.unique()
    group_results: dict[str, GroupMetrics] = {}
    for grp in sorted(str(g) for g in groups):
        mask = (sensitive_series == grp).values
        if mask.sum() < min_group_size:
            log.warning(
                "Group '%s'='%s' has only %d samples — excluded from gap calculations",
                col_name,
                grp,
                mask.sum(),
            )
            continue
        gm = _group_metrics(y_true[mask], y_pred[mask], y_proba[mask], grp)
        group_results[grp] = gm
        log.info(
            "Fairness [%s=%s]: n=%d  PPR=%.3f  TPR=%.3f  FPR=%.3f  Brier=%.4f",
            col_name,
            grp,
            gm.n,
            gm.predicted_positive_rate,
            gm.tpr,
            gm.fpr,
            gm.brier_score,
        )

    def _gap(values: list[float]) -> float:
        return max(values) - min(values) if len(values) >= 2 else 0.0

    pprs = [gm.predicted_positive_rate for gm in group_results.values()]
    tprs = [gm.tpr for gm in group_results.values()]
    fprs = [gm.fpr for gm in group_results.values()]
    briers = [gm.brier_score for gm in group_results.values()]

    return FairnessReport(
        sensitive_column=col_name,
        group_metrics=group_results,
        statistical_parity_gap=_gap(pprs),
        equal_opportunity_gap=_gap(tprs),
        predictive_equality_gap=_gap(fprs),
        brier_gap=_gap(briers),
    )
