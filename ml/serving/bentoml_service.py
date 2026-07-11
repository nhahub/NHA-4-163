"""BentoML service for standalone model deployment.

Wraps the MLflow-registered XGBoost hereditary-risk model as a BentoML
``Service`` so it can be containerised with ``bentoml build`` and deployed
independently of the FastAPI API (e.g., to BentoCloud or a dedicated GPU
node for larger GNN models in Phase 8).

Usage
-----
1. Save the model from MLflow into the BentoML model store::

       python -m ml.serving.bentoml_service save

2. Build a Bento::

       bentoml build

3. Containerise and serve::

       bentoml containerize hereditary_risk_svc:latest
       docker run -p 3000:3000 hereditary_risk_svc:latest

Environment variables
---------------------
MLFLOW_TRACKING_URI   MLflow tracking server (required for ``save`` step)
MODEL_NAME            Registered model name (default: hereditary-risk-xgboost)
MODEL_STAGE           Model stage (default: Staging)
BENTO_MODEL_TAG       Override the BentoML model tag used at serve time
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "hereditary-risk-xgboost"
_DEFAULT_STAGE = "Staging"

# BentoML model tag written by ``save_model_to_bentoml()``
_BENTO_MODEL_NAME = "hereditary_risk_xgboost"


def save_model_to_bentoml(
    tracking_uri: str,
    model_name: str = _DEFAULT_MODEL_NAME,
    stage: str = _DEFAULT_STAGE,
) -> str:
    """Pull a model from the MLflow registry and store it in BentoML's model store.

    Args:
        tracking_uri: MLflow tracking server URI.
        model_name: Registered model name.
        stage: Model stage (Staging, Production, etc.).

    Returns:
        BentoML model tag string (``name:version``).

    Raises:
        RuntimeError: If the model cannot be loaded from MLflow.
    """
    import bentoml
    import mlflow
    import mlflow.xgboost

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{stage}"

    xgb_model = mlflow.xgboost.load_model(model_uri)

    # Retrieve feature names from the MLflow registry
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise RuntimeError(f"No version found for {model_name}/{stage}")

    mv = versions[0]
    try:
        feat_names: list[str] = list(xgb_model.feature_names_in_)
    except AttributeError:
        run_data = client.get_run(mv.run_id).data
        feat_names = run_data.tags.get("feature_columns", "").split(",") or []

    saved = bentoml.xgboost.save_model(
        _BENTO_MODEL_NAME,
        xgb_model,
        signatures={"predict_proba": {"batchable": True}},
        labels={"stage": stage, "mlflow_version": mv.version, "mlflow_run_id": mv.run_id},
        custom_objects={"feature_names": feat_names},
    )
    return str(saved.tag)


# ── BentoML Service definition ────────────────────────────────────────────────

try:
    import bentoml
    from bentoml.io import JSON

    _BENTO_MODEL_TAG = os.environ.get("BENTO_MODEL_TAG", _BENTO_MODEL_NAME)
    _runner = bentoml.xgboost.get(_BENTO_MODEL_TAG).to_runner()

    svc = bentoml.Service("hereditary_risk_svc", runners=[_runner])

    @svc.api(input=JSON(), output=JSON())
    async def predict(body: dict[str, Any]) -> dict[str, Any]:
        """BentoML inference endpoint.

        Accepts a JSON object with ``patient_id``, ``features`` (dict of
        feature name → value), and optional ``include_shap`` flag.

        Returns:
            JSON with ``risk_score`` (float), ``risk_tier`` (str), and
            optional ``top_risk_factors`` list.

        Args:
            body: Input dict with ``features`` key mapping feature names
                to numeric values.
        """
        import shap

        features: dict[str, Any] = body.get("features", {})

        # Retrieve ordered feature names stored in the BentoML model store
        bento_model = bentoml.xgboost.get(_BENTO_MODEL_TAG)
        feat_names: list[str] = bento_model.custom_objects.get("feature_names", [])

        row = np.array(
            [[float(features.get(name) or 0.0) for name in feat_names]],
            dtype=np.float32,
        )

        proba = await _runner.predict_proba.async_run(row)
        risk_score = float(proba[0, 1])

        # Risk tier
        if risk_score < 0.25:
            tier = "low"
        elif risk_score < 0.50:
            tier = "moderate"
        elif risk_score < 0.75:
            tier = "high"
        else:
            tier = "very_high"

        result: dict[str, Any] = {"risk_score": risk_score, "risk_tier": tier}

        if body.get("include_shap", False):
            try:
                xgb_model = bento_model.load_model()
                explainer = shap.TreeExplainer(xgb_model)
                sv = explainer.shap_values(row)[0]
                top_n: int = int(body.get("top_n_factors", 5))
                pairs = sorted(
                    zip(feat_names, sv.tolist(), strict=False),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:top_n]
                result["top_risk_factors"] = [
                    {
                        "feature": name,
                        "raw_value": float(features.get(name) or 0.0),
                        "shap_value": float(val),
                        "direction": "increases_risk" if val > 0 else "decreases_risk",
                    }
                    for name, val in pairs
                ]
            except Exception as exc:
                # SHAP is optional — serve the prediction without factors.
                log.debug("SHAP computation failed: %s", exc)

        return result

except ImportError:
    # BentoML not installed — this module is import-safe so other code can
    # call save_model_to_bentoml() without the server running
    svc = None  # type: ignore[assignment]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "save":
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        model_name = os.environ.get("MODEL_NAME", _DEFAULT_MODEL_NAME)
        stage = os.environ.get("MODEL_STAGE", _DEFAULT_STAGE)
        tag = save_model_to_bentoml(tracking_uri, model_name, stage)
        print(f"Saved model to BentoML store: {tag}")  # noqa: T201 — CLI output
    else:
        print("Usage: python -m ml.serving.bentoml_service save")  # noqa: T201 — CLI output
        sys.exit(1)
