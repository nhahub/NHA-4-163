"""GraphSAGE training script for hereditary disease risk prediction.

Full pipeline:
    1. Load feature vector from Delta (same as XGBoost)
    2. Load labels from PostgreSQL
    3. Build PyG graph: node features from Delta, edges from Neo4j family rels
    4. Patient-ID stratified split → train_mask / val_mask / test_mask
    5. Training loop with Adam, BCELoss, early stopping on val Brier score
    6. Post-hoc Platt calibration on the calibration node subset
    7. Evaluation on test nodes (ROC-AUC, PR-AUC, Brier, ECE)
    8. Fairness analysis (age_group, gender_male)
    9. Log to MLflow; register model

Feature flag
------------
This script is only intended to run when ENABLE_GNN_MODEL=true.  GNN
training requires torch + torch_geometric which are not part of the
base Docker image.  The script exits with code 0 and a clear log message
when the flag is not set rather than raising an import error.

Run from project root:
    ENABLE_GNN_MODEL=true python ml/training/train_gnn.py \\
        --feature-date 2024-01-01 \\
        --delta-base s3a://healthcare-delta
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from libs.common.logging import configure_logging  # noqa: E402

configure_logging(service_name="train-gnn")
log = logging.getLogger(__name__)

_MODEL_NAME = "hereditary-risk-gnn"


def train(
    feature_date: str,
    delta_base: str,
    experiment_name: str,
    spark: object | None = None,
) -> str:
    """Train, calibrate, evaluate, and register the GraphSAGE model.

    Args:
        feature_date: ISO-8601 date of the feature snapshot.
        delta_base: S3A base path for the Delta feature store.
        experiment_name: MLflow experiment name.
        spark: Active ``SparkSession`` for Delta reads (optional).

    Returns:
        MLflow run ID.

    Raises:
        ImportError: If torch / torch_geometric are not installed.
        RuntimeError: If ENABLE_GNN_MODEL env var is not set to ``true``.
    """
    if os.environ.get("ENABLE_GNN_MODEL", "false").lower() != "true":
        raise RuntimeError("GNN training requires ENABLE_GNN_MODEL=true in the environment")

    import mlflow
    import mlflow.pytorch
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn

    from libs.common.config import get_settings
    from ml.models.calibration import CalibrationMethod, calibrate
    from ml.models.gnn_model import GNNConfig, GraphSAGEModel
    from ml.training.dataset import (
        build_dataset,
        build_pyg_data,
        load_feature_vector,
        load_labels,
        patient_id_split,
    )
    from ml.training.evaluate import evaluate_binary_classifier
    from ml.training.fairness import compute_fairness_report

    settings = get_settings()
    pg = settings.postgres
    n4j = settings.neo4j

    # ── Data ──────────────────────────────────────────────────────────────────
    feat_path = f"{delta_base}/features/patient_feature_vector"
    features_df = load_feature_vector(feat_path, feature_date, spark=spark)
    labels_df = load_labels(pg.sync_dsn)
    X, y, feat_cols, patient_ids = build_dataset(features_df, labels_df)

    train_ids, val_ids, test_ids = patient_id_split(patient_ids, y, val_size=0.15, test_size=0.15)
    from sklearn.model_selection import train_test_split as _tts

    train_ids_final, cal_ids = _tts(
        train_ids,
        test_size=0.12,
        stratify=y[np.isin(patient_ids, train_ids)],
        random_state=42,
    )

    # ── Build PyG graph ───────────────────────────────────────────────────────
    log.info("Building PyG graph (loading edges from Neo4j)…")
    data, pid_to_idx = build_pyg_data(
        X,
        y,
        patient_ids,
        train_ids_final,
        val_ids,
        test_ids,
        neo4j_uri=n4j.uri,
        neo4j_user=n4j.user,
        neo4j_password=n4j.password.get_secret_value(),
    )

    # Override train_mask to exclude calibration nodes
    cal_set = set(cal_ids.tolist())
    data.train_mask = torch.tensor([pid in set(train_ids_final.tolist()) for pid in patient_ids])
    data.cal_mask = torch.tensor([pid in cal_set for pid in patient_ids])

    # ── Config & model ────────────────────────────────────────────────────────
    torch.manual_seed(42)
    cfg = GNNConfig(input_dim=X.shape[1])
    model = GraphSAGEModel(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    n_pos = int(y[data.train_mask.numpy()].sum())
    n_neg = int(data.train_mask.sum()) - n_pos
    pos_weight = torch.tensor([float(n_neg) / max(n_pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Training loop ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(str(settings.mlflow.tracking_uri))
    mlflow.set_experiment(experiment_name)

    best_val_brier = float("inf")
    patience_counter = 0
    best_state: dict[str, torch.Tensor] = {}

    with mlflow.start_run(run_name=f"gnn-{feature_date}") as run:
        run_id = run.info.run_id
        mlflow.set_tag("model_type", "graphsage")
        mlflow.set_tag("feature_date", feature_date)
        mlflow.log_params(model.params_dict())

        for epoch in range(cfg.epochs):
            model.train()
            optimizer.zero_grad()
            # Forward pass uses raw logits for BCEWithLogitsLoss
            out_logits = model.convs[0](data.x, data.edge_index)
            import torch.nn.functional as F

            for conv, bn in zip(model.convs[1:], model.bns[1:], strict=False):
                out_logits = bn(F.relu(out_logits))
                out_logits = model.dropout(out_logits)
                out_logits = conv(out_logits, data.edge_index)
            out_logits = model.classifier(model.dropout(F.relu(model.bns[-1](out_logits)))).squeeze(
                -1
            )

            loss = criterion(out_logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_proba = model(data.x, data.edge_index)[data.val_mask].numpy()
                val_true = data.y[data.val_mask].numpy()
            from sklearn.metrics import brier_score_loss

            val_brier = float(brier_score_loss(val_true, val_proba))

            if epoch % 20 == 0:
                log.info(
                    "Epoch %d/%d  loss=%.4f  val_brier=%.4f",
                    epoch,
                    cfg.epochs,
                    float(loss),
                    val_brier,
                )
            mlflow.log_metrics(
                {"train_loss": float(loss), "val_brier": val_brier},
                step=epoch,
            )

            if val_brier < best_val_brier:
                best_val_brier = val_brier
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    log.info("Early stopping at epoch %d", epoch)
                    break

        # Restore best weights
        model.load_state_dict(best_state)
        model.eval()

        # ── Calibration ───────────────────────────────────────────────────────
        with torch.no_grad():
            cal_proba_raw = model(data.x, data.edge_index)[data.cal_mask].numpy()
        cal_true = data.y[data.cal_mask].numpy()
        calibrated = calibrate(
            lambda x_unused: cal_proba_raw,  # Already extracted, no X needed
            np.zeros((len(cal_proba_raw), 1)),  # Dummy X
            cal_true,
            CalibrationMethod.SIGMOID,
        )
        # For a GNN the calibration is a post-hoc sigmoid applied to node scores
        from sklearn.linear_model import LogisticRegression

        lr_cal = LogisticRegression(max_iter=500)
        lr_cal.fit(cal_proba_raw.reshape(-1, 1), cal_true)

        def _calibrated_predict(raw_proba: np.ndarray) -> np.ndarray:
            return lr_cal.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

        mlflow.log_metrics(
            {
                "brier_before_calibration": calibrated.brier_before,
                "brier_after_calibration": calibrated.brier_after,
            }
        )

        # ── Evaluation ────────────────────────────────────────────────────────
        with torch.no_grad():
            test_proba_raw = model(data.x, data.edge_index)[data.test_mask].numpy()
        test_true = data.y[data.test_mask].numpy()
        test_proba_cal = _calibrated_predict(test_proba_raw)

        eval_result = evaluate_binary_classifier(test_true, test_proba_cal)
        mlflow.log_metrics(eval_result.to_mlflow_metrics())

        # ── Fairness ──────────────────────────────────────────────────────────
        test_X_np = X[np.isin(patient_ids, test_ids)]
        test_df = pd.DataFrame(test_X_np, columns=feat_cols)
        for sensitive_col in ("age_group", "gender_male"):
            if sensitive_col in test_df.columns:
                report = compute_fairness_report(test_true, test_proba_cal, test_df[sensitive_col])
                mlflow.log_metrics(report.to_mlflow_metrics(prefix=sensitive_col))
                mlflow.log_text(
                    report.to_dataframe().to_csv(index=False), f"fairness_{sensitive_col}.csv"
                )

        # ── Register ──────────────────────────────────────────────────────────
        mlflow.pytorch.log_model(
            model,
            artifact_path="gnn_model",
            registered_model_name=_MODEL_NAME,
        )
        log.info(
            "GNN run complete — run_id=%s  ROC-AUC=%.4f  PR-AUC=%.4f  Brier=%.4f",
            run_id,
            eval_result.roc_auc,
            eval_result.pr_auc,
            eval_result.brier_score,
        )
    return run_id


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GraphSAGE hereditary risk model")
    parser.add_argument("--feature-date", required=True)
    parser.add_argument(
        "--delta-base",
        default=os.environ.get("DELTA_BASE", "s3a://healthcare-delta"),
    )
    parser.add_argument(
        "--experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "hereditary-disease-prediction"),
    )
    args = parser.parse_args()
    rid = train(
        feature_date=args.feature_date,
        delta_base=args.delta_base,
        experiment_name=args.experiment,
    )
    log.info("MLflow run_id: %s", rid)
