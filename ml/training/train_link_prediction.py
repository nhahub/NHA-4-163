"""Train the GraphSAGE pedigree link-prediction model (Tier 6, feature #22).

Pipeline:
    1. Generate synthetic three-generation pedigrees (there is no shipped
       labelled pedigree corpus) — see :mod:`ml.models.pedigree_graph`.
    2. Union them into one graph; build the node-feature matrix.
    3. Split the typed positive edges into train/val/test; message passing uses
       *train* edges only so val/test edges are never leaked into the encoder.
    4. Sample within-family non-edges as the negative ("no_edge") class.
    5. Train encoder + edge decoder with class-weighted cross-entropy, early
       stopping on validation loss.
    6. Evaluate (accuracy, macro one-vs-rest ROC-AUC) and persist the model to a
       local artifact ({state_dict, config}) that the API loads at inference.
       MLflow logging is best-effort (skipped if no tracking server).

Feature flag: honours ``ENABLE_GNN_MODEL`` for parity with
:mod:`ml.training.train_gnn`, but the default artifact path lets the API load a
model without any MLflow/registry dependency.

Run from the project root:
    python ml/training/train_link_prediction.py --families 400 --epochs 200
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

log = logging.getLogger(__name__)

# Default location the API inference service loads from.
DEFAULT_ARTIFACT_PATH = os.path.join(_PROJECT_ROOT, "ml", "artifacts", "pedigree_gnn.pt")


def train(
    n_families: int = 400,
    seed: int = 42,
    epochs: int | None = None,
    out_path: str = DEFAULT_ARTIFACT_PATH,
    use_mlflow: bool = False,
) -> dict[str, float]:
    """Train and persist the pedigree link-prediction model.

    Args:
        n_families: Number of synthetic families to generate.
        seed: RNG seed for reproducibility.
        epochs: Override the config's max epochs.
        out_path: Where to write the model artifact.
        use_mlflow: If True, attempt MLflow logging (best-effort).

    Returns:
        A dict of final evaluation metrics.

    Raises:
        ImportError: If torch / torch_geometric are unavailable.
    """
    import random

    import torch
    import torch.nn as nn

    from ml.models.gnn_link_prediction import (
        CLASS_TO_IDX,
        EDGE_CLASSES,
        GNNLinkConfig,
        GraphSAGELinkPredictor,
    )
    from ml.models.pedigree_graph import (
        NODE_FEATURE_DIM,
        derive_relationships,
        generate_families,
        node_feature_vector,
    )

    torch.manual_seed(seed)
    rng = random.Random(seed)

    # ── Build the unioned graph (star per family = inference topology) ─────────
    families = generate_families(n_families, seed=seed)
    node_ids: list[str] = []
    idx_of: dict[str, int] = {}
    features: list[list[float]] = []

    mp: list[list[int]] = [[], []]  # proband-star message edges
    positives: list[tuple[int, int, int]] = []  # relative↔relative, typed
    negatives_pool: list[tuple[int, int]] = []  # relative↔relative, unrelated

    for fam in families:
        fam_ids = [node.node_id for node in fam.nodes]
        for node in fam.nodes:
            idx_of[node.node_id] = len(node_ids)
            node_ids.append(node.node_id)
            features.append(node_feature_vector(node))

        pid = fam.proband_id
        pi = idx_of[pid]
        # Star: proband ↔ every other member (mirrors FamilyMemberHistory).
        for other in fam_ids:
            if other == pid:
                continue
            oi = idx_of[other]
            mp[0] += [pi, oi]
            mp[1] += [oi, pi]

        rels = derive_relationships(fam_ids, fam.parent_edges, fam.spouse_edges)
        # Predict only relative↔relative edges (proband-incident ones are known).
        for i in range(len(fam_ids)):
            for j in range(i + 1, len(fam_ids)):
                a, b = fam_ids[i], fam_ids[j]
                if a == pid or b == pid:
                    continue
                cat = rels.get(frozenset((a, b)))
                if cat is not None:
                    positives.append((idx_of[a], idx_of[b], CLASS_TO_IDX[cat]))
                else:
                    negatives_pool.append((idx_of[a], idx_of[b]))

    x = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(mp, dtype=torch.long)

    rng.shuffle(positives)
    n = len(positives)
    n_test = int(n * 0.15)
    n_val = int(n * 0.15)
    test_pos = positives[:n_test]
    val_pos = positives[n_test : n_test + n_val]
    train_pos = positives[n_test + n_val :]

    cfg = GNNLinkConfig(input_dim=NODE_FEATURE_DIM, epochs=epochs or 200)

    def sample_negatives(
        pos: list[tuple[int, int, int]], ratio: float
    ) -> list[tuple[int, int, int]]:
        """Draw relative↔relative non-edges labelled ``no_edge``."""
        target = int(len(pos) * ratio)
        if not negatives_pool:
            return []
        chosen = (
            negatives_pool if target >= len(negatives_pool) else rng.sample(negatives_pool, target)
        )
        return [(a, b, CLASS_TO_IDX["no_edge"]) for a, b in chosen]

    def make_batch(
        pos: list[tuple[int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neg = sample_negatives(pos, cfg.neg_ratio)
        allp = pos + neg
        pairs = torch.tensor([[a for a, _, _ in allp], [b for _, b, _ in allp]], dtype=torch.long)
        labels = torch.tensor([c for _, _, c in allp], dtype=torch.long)
        return pairs, labels

    val_pairs, val_labels = make_batch(val_pos)

    # Class weights: down-weight the dominant no_edge class.
    counts = [1] * cfg.num_classes
    for _, _, c in train_pos:
        counts[c] += 1
    counts[CLASS_TO_IDX["no_edge"]] += int(len(train_pos) * cfg.neg_ratio)
    total = sum(counts)
    weights = torch.tensor([total / (cfg.num_classes * c) for c in counts], dtype=torch.float32)

    model = GraphSAGELinkPredictor(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val = float("inf")
    best_state: dict = {}
    patience = 0

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        pairs, labels = make_batch(train_pos)  # resample negatives each epoch
        logits = model(x, edge_index, pairs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x, edge_index, val_pairs), val_labels))
        if epoch % 25 == 0:
            log.info(
                "epoch %d/%d loss=%.4f val_loss=%.4f", epoch, cfg.epochs, loss.item(), val_loss
            )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                log.info("early stopping at epoch %d", epoch)
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    test_pairs, test_labels = make_batch(test_pos)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x, edge_index, test_pairs), dim=-1)
    preds = probs.argmax(dim=-1)
    accuracy = float((preds == test_labels).float().mean())

    try:
        from sklearn.metrics import f1_score, roc_auc_score

        macro_f1 = float(
            f1_score(
                test_labels.numpy(),
                preds.numpy(),
                labels=list(range(cfg.num_classes)),
                average="macro",
                zero_division=0,
            )
        )
        macro_auc = float(
            roc_auc_score(
                test_labels.numpy(),
                probs.numpy(),
                multi_class="ovr",
                average="macro",
                labels=list(range(cfg.num_classes)),
            )
        )
    except Exception as exc:  # pragma: no cover - metric best-effort
        log.warning("metric computation skipped: %s", exc)
        macro_f1 = macro_auc = float("nan")

    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_auc": round(macro_auc, 4),
        "val_loss": round(best_val, 4),
        "n_positives": float(n),
    }
    log.info("Link-prediction eval: %s", metrics)

    # ── Persist artifact ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    from dataclasses import asdict

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "edge_classes": list(EDGE_CLASSES),
            "feature_dim": NODE_FEATURE_DIM,
            "metrics": metrics,
        },
        out_path,
    )
    log.info("Saved pedigree GNN to %s", out_path)

    if use_mlflow:  # pragma: no cover - requires a tracking server
        try:
            import mlflow
            import mlflow.pytorch

            from libs.common.config import get_settings

            mlflow.set_tracking_uri(str(get_settings().mlflow.tracking_uri))
            mlflow.set_experiment("pedigree-link-prediction")
            with mlflow.start_run(run_name="pedigree-gnn"):
                mlflow.log_params(model.params_dict())
                mlflow.log_metrics(metrics)
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="pedigree_gnn",
                    registered_model_name="hereditary-pedigree-gnn",
                )
        except Exception as exc:
            log.warning("MLflow logging skipped: %s", exc)

    return metrics


if __name__ == "__main__":
    from libs.common.logging import configure_logging

    configure_logging(service_name="train-link-prediction")

    parser = argparse.ArgumentParser(description="Train pedigree GNN link predictor")
    parser.add_argument("--families", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out", default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    train(
        n_families=args.families,
        seed=args.seed,
        epochs=args.epochs,
        out_path=args.out,
        use_mlflow=args.mlflow,
    )
