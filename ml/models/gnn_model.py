"""GraphSAGE model for hereditary disease risk prediction.

Architecture
------------
Input (node features)
  → GraphSAGEConv(input_dim → hidden_dim) → BatchNorm → ReLU → Dropout
  → GraphSAGEConv(hidden_dim → hidden_dim) → BatchNorm → ReLU → Dropout
  → Linear(hidden_dim → 1) → Sigmoid

Design rationale
----------------
- **GraphSAGE over GCN**: inductive — predicts risk for new patient nodes
  without retraining the full model.  Suitable for continuous data ingestion.
- **Mean aggregator**: robust to variable-degree family networks (some patients
  have many documented relatives; others have none).
- **BatchNorm**: stabilises training with heterogeneous family sizes.
- **Node-level task**: each patient node receives its own risk score.  Family
  member nodes contribute via neighbourhood aggregation.

Environment gate
----------------
This module requires ``torch`` and ``torch_geometric``.  It is only loaded
when ``ENABLE_GNN_MODEL=true`` (the flag is checked in the training script).
Importing without the packages installed raises ``ImportError`` with a clear
install instruction.
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
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class GNNConfig:
    """Architecture and training hyperparameters for GraphSAGE.

    Attributes:
        input_dim: Number of input node features (must match feature matrix width).
        hidden_dim: Width of hidden GraphSAGE layers.
        num_layers: Number of GraphSAGE convolutional layers (≥ 2).
        dropout: Dropout probability applied after each hidden layer.
        learning_rate: Adam optimiser learning rate.
        weight_decay: L2 regularisation coefficient.
        epochs: Maximum training epochs.
        patience: Early-stopping patience (epochs without val improvement).
        random_state: Seed for reproducibility.
    """

    input_dim: int = 30
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20
    random_state: int = 42


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch and torch_geometric are required for GNN training. "
            "Install with: pip install torch torch_geometric"
        )


class GraphSAGEModel(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """GraphSAGE binary node classifier for hereditary disease risk.

    Attributes:
        config: GNN architecture configuration.
        convs: List of GraphSAGEConv layers.
        bns: List of BatchNorm1d layers (one per hidden layer).
        classifier: Final linear projection to scalar logit.
        dropout: Dropout layer (shared across hidden layers).
    """

    def __init__(self, config: GNNConfig) -> None:
        """Initialise GraphSAGE model.

        Args:
            config: ``GNNConfig`` with architecture parameters.

        Raises:
            ImportError: If torch / torch_geometric are not installed.
        """
        _require_torch()
        super().__init__()
        self.config = config

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_dim = config.input_dim
        for _ in range(config.num_layers):
            self.convs.append(SAGEConv(in_dim, config.hidden_dim))
            self.bns.append(nn.BatchNorm1d(config.hidden_dim))
            in_dim = config.hidden_dim

        self.classifier = nn.Linear(config.hidden_dim, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Compute node-level risk probability estimates.

        Args:
            x: Node feature matrix of shape (num_nodes × input_dim).
            edge_index: Edge index tensor of shape (2 × num_edges).

        Returns:
            1-D tensor of shape (num_nodes,) with probabilities in [0, 1].
        """
        for conv, bn in zip(self.convs, self.bns, strict=False):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return torch.sigmoid(self.classifier(x)).squeeze(-1)

    def params_dict(self) -> dict[str, str]:
        """Return config as a flat dict for MLflow log_params.

        Returns:
            Dict of hyperparameter names to string values.
        """
        from dataclasses import asdict

        return {k: str(v) for k, v in asdict(self.config).items()}
