"""Trained-GNN pedigree link prediction (Tier 6, feature #22).

Inference wrapper around the GraphSAGE link predictor trained by
:mod:`ml.training.train_link_prediction`. Given a proband's family subgraph
(nodes + known relationship edges), it encodes node embeddings and scores every
candidate non-edge, returning the highest-probability missing relationships.

This is the *trained-GNN* replacement for the deterministic structural predictor
in :mod:`services.api.services.pedigree_service`. The router uses this when a
trained model is available and falls back to the structural predictor otherwise,
so the endpoint always works (and unit tests run without torch installed).

Availability is gated on both ``torch``/``torch_geometric`` being importable and
a model artifact existing at ``PEDIGREE_GNN_PATH`` (default:
``ml/artifacts/pedigree_gnn.pt``). Loading is lazy and cached.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from ml.models.pedigree_graph import (
    PedigreeNode,
    node_feature_vector,
)
from services.api.services.pedigree_service import SuggestedLink

log = logging.getLogger(__name__)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    _TORCH_AVAILABLE = False

# This file is at <root>/services/api/services/ — four levels below the root.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_DEFAULT_PATH = os.path.join(_PROJECT_ROOT, "ml", "artifacts", "pedigree_gnn.pt")

# Relationships that are symmetric (order of source/target is irrelevant).
_SYMMETRIC = {"sibling", "spouse"}

_model_lock = threading.Lock()
_model_cache: dict[str, dict[str, Any]] = {}


def _artifact_path() -> str:
    return os.environ.get("PEDIGREE_GNN_PATH", _DEFAULT_PATH)


def gnn_available() -> bool:
    """Return whether a trained GNN can be used (torch present + artifact on disk)."""
    if os.environ.get("ENABLE_PEDIGREE_GNN", "true").lower() == "false":
        return False
    return _TORCH_AVAILABLE and os.path.isfile(_artifact_path())


def _load_model() -> dict[str, Any]:
    """Load and cache the trained model + metadata (thread-safe)."""
    path = _artifact_path()
    with _model_lock:
        cached = _model_cache.get(path)
        if cached is not None:
            return cached

        from ml.models.gnn_link_prediction import GNNLinkConfig, GraphSAGELinkPredictor

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = GNNLinkConfig(**ckpt["config"])
        model = GraphSAGELinkPredictor(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        bundle = {"model": model, "classes": ckpt["edge_classes"]}
        _model_cache[path] = bundle
        return bundle


def _direction(rel: str, gen_u: int, gen_v: int, u: str, v: str) -> tuple[str, str]:
    """Resolve (source, target) for a predicted relationship using generations.

    For directional relationships the older generation is the source
    (parent/grandparent of, or aunt/uncle of, the younger node). Symmetric
    relationships keep the given order.
    """
    if rel in _SYMMETRIC:
        return u, v
    # parent / grandparent / aunt_uncle: higher generation → source.
    return (u, v) if gen_u >= gen_v else (v, u)


def suggest_links_gnn(
    nodes: list[PedigreeNode],
    known_pairs: list[tuple[str, str]],
    max_suggestions: int = 25,
    threshold: float = 0.5,
) -> list[SuggestedLink]:
    """Predict missing pedigree edges with the trained GNN.

    Args:
        nodes: All family members (including the proband) as pedigree nodes.
        known_pairs: Already-recorded undirected relationships (used both for
            message passing and to exclude from suggestions).
        max_suggestions: Cap on returned suggestions.
        threshold: Minimum predicted probability for a suggestion.

    Returns:
        Ranked :class:`SuggestedLink` objects (highest probability first).
    """
    if len(nodes) < 2 or not known_pairs:
        return []

    bundle = _load_model()
    model = bundle["model"]
    classes: list[str] = bundle["classes"]

    idx_of = {n.node_id: i for i, n in enumerate(nodes)}
    gen_of = {n.node_id: n.generation for n in nodes}
    x = torch.tensor([node_feature_vector(n) for n in nodes], dtype=torch.float32)

    known: set[frozenset[str]] = set()
    mp: list[list[int]] = [[], []]
    for u, v in known_pairs:
        if u in idx_of and v in idx_of:
            known.add(frozenset((u, v)))
            mp[0] += [idx_of[u], idx_of[v]]
            mp[1] += [idx_of[v], idx_of[u]]
    if not mp[0]:
        return []
    edge_index = torch.tensor(mp, dtype=torch.long)

    # Candidate pairs = all unordered node pairs not already recorded.
    ids = [n.node_id for n in nodes]
    cand_pairs: list[tuple[str, str]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if frozenset((ids[i], ids[j])) not in known:
                cand_pairs.append((ids[i], ids[j]))
    if not cand_pairs:
        return []

    pair_index = torch.tensor(
        [[idx_of[u] for u, _ in cand_pairs], [idx_of[v] for _, v in cand_pairs]],
        dtype=torch.long,
    )

    no_edge_idx = classes.index("no_edge")
    with torch.no_grad():
        z = model.encode(x, edge_index)
        probs = torch.softmax(model.decode(z, pair_index), dim=-1)

    suggestions: list[SuggestedLink] = []
    for (u, v), row in zip(cand_pairs, probs, strict=False):
        cls = int(row.argmax())
        if cls == no_edge_idx:
            continue
        p = float(row[cls])
        if p < threshold:
            continue
        rel = classes[cls]
        src, tgt = _direction(rel, gen_of.get(u, 0), gen_of.get(v, 0), u, v)
        suggestions.append(
            SuggestedLink(
                source=src,
                target=tgt,
                relationship=rel,
                confidence=round(p, 4),
                support=1,
                rationale=(
                    f"GraphSAGE link predictor: P({rel})={p:.2f} from learned "
                    f"embeddings of the family graph."
                ),
            )
        )

    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    return suggestions[:max_suggestions]
