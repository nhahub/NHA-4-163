"""XGBoost hereditary disease risk model.

Wraps ``xgboost.XGBClassifier`` with:
- Dataclass-driven hyperparameter configuration
- Auto-computed ``scale_pos_weight`` for class-imbalanced data
- SHAP global feature importances (via TreeExplainer)
- MLflow-compatible ``log_model`` integration

The model produces calibrated probabilities only after
``ml.models.calibration`` has been applied; raw XGBoost scores
are used internally for early stopping and HPO.

Split discipline: callers MUST split by patient_id before passing data
here.  Never split by row — family members share graph features and
row-level splits cause data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xgboost as xgb

log = logging.getLogger(__name__)


@dataclass
class XGBConfig:
    """Hyperparameter configuration for the XGBoost model.

    Attributes:
        n_estimators: Maximum number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage used in updates (eta).
        subsample: Fraction of training rows sampled per tree.
        colsample_bytree: Fraction of features sampled per tree.
        min_child_weight: Minimum sum of instance weight in a leaf.
        gamma: Minimum loss reduction required to make a further partition.
        scale_pos_weight: Weight ratio for imbalanced classes.  ``None``
            means auto-compute from training label distribution.
        early_stopping_rounds: Stop when val metric does not improve.
        eval_metric: XGBoost eval metric (``aucpr`` preferred for imbalance).
        random_state: Seed for reproducibility.
        n_jobs: Parallel threads (-1 = all cores).
    """

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    scale_pos_weight: float | None = None  # None → auto
    early_stopping_rounds: int = 50
    eval_metric: str = "aucpr"
    random_state: int = 42
    n_jobs: int = -1


class HeredityXGBModel:
    """XGBoost binary classifier for hereditary disease risk.

    Attributes:
        config: Hyperparameter configuration.
        feature_names: Column names in the order the model was trained on.
    """

    def __init__(self, config: XGBConfig | None = None) -> None:
        """Initialise the model with an optional config.

        Args:
            config: ``XGBConfig`` instance.  Defaults to ``XGBConfig()``
                if not provided.
        """
        self.config: XGBConfig = config or XGBConfig()
        self._model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Train the XGBoost classifier.

        Args:
            X_train: Training feature matrix (n_samples × n_features).
            y_train: Binary training labels (0/1).
            X_val: Validation feature matrix for early stopping.
            y_val: Binary validation labels.
            feature_names: Ordered column names matching X_train columns.

        Raises:
            ValueError: If X_train has zero rows.
        """
        if X_train.shape[0] == 0:
            raise ValueError("Training set is empty")

        self.feature_names = list(feature_names)

        # Auto-compute scale_pos_weight from label distribution
        spw = self.config.scale_pos_weight
        if spw is None:
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            spw = float(n_neg) / max(n_pos, 1)
            log.info("Auto scale_pos_weight=%.2f  (neg=%d, pos=%d)", spw, n_neg, n_pos)

        self._model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            scale_pos_weight=spw,
            eval_metric=self.config.eval_metric,
            early_stopping_rounds=self.config.early_stopping_rounds,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            verbosity=0,
        )
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iter = getattr(self._model, "best_iteration", self.config.n_estimators)
        log.info("XGBoost training complete — best_iteration=%d", best_iter)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the positive class.

        Args:
            X: Feature matrix (n_samples × n_features).

        Returns:
            1-D array of shape (n_samples,) with values in [0, 1].

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted — call fit() first")
        return self._model.predict_proba(X)[:, 1]

    # ── Interpretability ──────────────────────────────────────────────────────

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for the given samples using TreeExplainer.

        Args:
            X: Feature matrix (n_samples × n_features).

        Returns:
            SHAP value matrix of shape (n_samples × n_features).

        Raises:
            RuntimeError: If the model is not fitted.
            ImportError: If the ``shap`` package is not installed.
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        try:
            import shap
        except ImportError as exc:
            raise ImportError("Install 'shap' for SHAP explanations") from exc

        explainer = shap.TreeExplainer(self._model)
        return explainer.shap_values(X)

    @property
    def feature_importances(self) -> dict[str, float]:
        """Return gain-based feature importances as a {name: score} dict.

        Returns:
            Dict mapping feature name to normalized importance score.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        scores = self._model.feature_importances_
        total = scores.sum()
        if total == 0:
            return {name: 0.0 for name in self.feature_names}
        normalized = scores / total
        return dict(zip(self.feature_names, normalized.tolist()))

    def params_dict(self) -> dict[str, Any]:
        """Return config as a flat dict suitable for MLflow log_params.

        Returns:
            Dict of hyperparameter names to values.
        """
        from dataclasses import asdict

        return {k: str(v) for k, v in asdict(self.config).items()}
