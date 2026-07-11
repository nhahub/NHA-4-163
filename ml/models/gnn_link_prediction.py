"""GraphSAGE link-prediction / relationship-typing model for pedigree completion.

Backs feature #22 (Tier 6). Where :mod:`ml.models.gnn_model` is a node-level
risk *classifier*, this is an *edge-level* model: it learns node embeddings over
a family graph and a decoder that, for any pair of family members, predicts the
relationship category between them (or "no edge"). Suggesting the highest-scoring
non-edges completes an incomplete pedigree.

Architecture
------------
Encoder (inductive GraphSAGE, so new patients need no retraining)::

    x → SAGEConv(in→hidden) → BatchNorm → ReLU → Dropout
      → SAGEConv(hidden→emb) → embeddings z

Edge decoder (a pair is described by a symmetric interaction vector so the score
does not depend on argument order)::

    pair(z_u, z_v) = [z_u + z_v , |z_u − z_v| , z_u * z_v]
    logits = MLP(pair) → softmax over {no_edge, parent, sibling, spouse,
                                       grandparent, aunt_uncle}

``parent`` is directional, so at inference both orderings are scored and the
higher-probability direction is reported.

Environment gate
----------------
Requires ``torch`` and ``torch_geometric``; importing without them sets
``_TORCH_AVAILABLE = False`` and constructing the model raises ``ImportError``
with an install hint (same pattern as :mod:`ml.models.gnn_model`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from torch_geometric.nn import SAGEConv

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    _TORCH_AVAILABLE = False


# Relationship categories the decoder can emit. Index 0 is the negative class.
# These match services.api.services.inheritance_service.categorise_relationship
# (spouse included; directional labels other than ``parent`` are collapsed onto
# their canonical form during graph construction).
EDGE_CLASSES: tuple[str, ...] = (
    "no_edge",
    "parent",
    "sibling",
    "spouse",
    "grandparent",
    "aunt_uncle",
)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(EDGE_CLASSES)}


@dataclass
class GNNLinkConfig:
    """Architecture / training hyperparameters for the link predictor.

    Attributes:
        input_dim: Node feature width.
        hidden_dim: Width of the hidden GraphSAGE layer.
        embed_dim: Node embedding width produced by the encoder.
        decoder_hidden: Hidden width of the edge-decoder MLP.
        num_classes: Number of edge classes (see :data:`EDGE_CLASSES`).
        dropout: Dropout probability.
        learning_rate: Adam learning rate.
        weight_decay: L2 regularisation coefficient.
        epochs: Maximum training epochs.
        patience: Early-stopping patience on validation loss.
        neg_ratio: Negative ("no_edge") samples drawn per positive edge.
        random_state: Seed for reproducibility.
    """

    input_dim: int = 8  # == ml.models.pedigree_graph.NODE_FEATURE_DIM
    hidden_dim: int = 64
    embed_dim: int = 32
    decoder_hidden: int = 64
    num_classes: int = len(EDGE_CLASSES)
    dropout: float = 0.3
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 25
    neg_ratio: float = 2.0
    random_state: int = 42


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch and torch_geometric are required for the GNN link predictor. "
            "Install with: pip install torch torch_geometric"
        )


class GraphSAGELinkPredictor(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """GraphSAGE encoder + edge-type decoder for pedigree link prediction."""

    def __init__(self, config: GNNLinkConfig) -> None:
        """Initialise the model.

        Args:
            config: :class:`GNNLinkConfig` with architecture parameters.

        Raises:
            ImportError: If torch / torch_geometric are not installed.
        """
        _require_torch()
        super().__init__()
        self.config = config

        self.conv1 = SAGEConv(config.input_dim, config.hidden_dim)
        self.bn1 = nn.BatchNorm1d(config.hidden_dim)
        self.conv2 = SAGEConv(config.hidden_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Pair interaction vector is 3 × embed_dim (sum, abs-diff, hadamard).
        self.decoder = nn.Sequential(
            nn.Linear(3 * config.embed_dim, config.decoder_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden, config.num_classes),
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Encode node features + graph structure into node embeddings.

        Args:
            x: Node feature matrix (num_nodes × input_dim).
            edge_index: Undirected edge index (2 × num_edges).

        Returns:
            Node embeddings (num_nodes × embed_dim).
        """
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = self.dropout(F.relu(h))
        h = self.conv2(h, edge_index)
        return h

    def decode(self, z: Tensor, pairs: Tensor) -> Tensor:
        """Score candidate pairs into per-class logits.

        Args:
            z: Node embeddings (num_nodes × embed_dim).
            pairs: Long tensor of shape (2 × num_pairs) of node indices (u, v).

        Returns:
            Logits of shape (num_pairs × num_classes).
        """
        z_u = z[pairs[0]]
        z_v = z[pairs[1]]
        interaction = torch.cat([z_u + z_v, (z_u - z_v).abs(), z_u * z_v], dim=-1)
        return self.decoder(interaction)

    def forward(self, x: Tensor, edge_index: Tensor, pairs: Tensor) -> Tensor:
        """Encode then decode; returns per-class logits for ``pairs``."""
        z = self.encode(x, edge_index)
        return self.decode(z, pairs)

    def params_dict(self) -> dict[str, str]:
        """Return config as a flat string dict for MLflow ``log_params``."""
        from dataclasses import asdict

        return {k: str(v) for k, v in asdict(self.config).items()}
