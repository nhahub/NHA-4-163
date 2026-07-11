"""Unit tests for the trained-GNN pedigree link predictor (Tier 6, #22).

Torch-gated: skipped entirely when torch / torch_geometric are not installed
(the API falls back to the structural predictor in that case). A tiny model is
trained into a temp file so the tests do not depend on any committed artifact.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


from ml.models.gnn_link_prediction import (  # noqa: E402
    EDGE_CLASSES,
    GNNLinkConfig,
    GraphSAGELinkPredictor,
)
from ml.models.pedigree_graph import NODE_FEATURE_DIM, PedigreeNode  # noqa: E402


class TestModel:
    def test_encode_decode_shapes(self) -> None:
        cfg = GNNLinkConfig(input_dim=NODE_FEATURE_DIM, hidden_dim=8, embed_dim=4)
        model = GraphSAGELinkPredictor(cfg)
        model.eval()
        x = torch.randn(5, NODE_FEATURE_DIM)
        edge_index = torch.tensor([[0, 1, 2], [1, 0, 0]], dtype=torch.long)
        z = model.encode(x, edge_index)
        assert z.shape == (5, 4)
        pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        logits = model.decode(z, pairs)
        assert logits.shape == (2, cfg.num_classes)

    def test_params_dict_stringified(self) -> None:
        model = GraphSAGELinkPredictor(GNNLinkConfig())
        params = model.params_dict()
        assert all(isinstance(v, str) for v in params.values())


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory) -> str:
    """Train a small model into a temp artifact for inference tests."""
    from ml.training.train_link_prediction import train

    out = str(tmp_path_factory.mktemp("gnn") / "pedigree_gnn.pt")
    metrics = train(n_families=60, seed=7, epochs=60, out_path=out, use_mlflow=False)
    # A functional model should discriminate classes well above chance.
    assert metrics["macro_auc"] > 0.8
    return out


class TestInference:
    def _nodes_and_known(self):
        nodes = [
            PedigreeNode("proband", "female", 0, False, 1.0, is_proband=True),
            PedigreeNode("mom", "female", 1, False, 0.5),
            PedigreeNode("dad", "male", 1, False, 0.5),
            PedigreeNode("sib", "male", 0, True, 0.5),
            PedigreeNode("child", "male", -1, False, 0.5),
        ]
        known = [
            ("proband", "mom"),
            ("proband", "dad"),
            ("proband", "sib"),
            ("proband", "child"),
        ]
        return nodes, known

    def test_suggestions_are_sensible(self, trained_model_path, monkeypatch) -> None:
        monkeypatch.setenv("PEDIGREE_GNN_PATH", trained_model_path)
        monkeypatch.setenv("ENABLE_PEDIGREE_GNN", "true")
        import services.api.services.gnn_pedigree_service as svc

        importlib.reload(svc)
        assert svc.gnn_available()

        nodes, known = self._nodes_and_known()
        suggestions = svc.suggest_links_gnn(nodes, known, threshold=0.4)
        assert suggestions

        known_sets = {frozenset(p) for p in known}
        for s in suggestions:
            assert s.relationship in EDGE_CLASSES
            assert s.relationship != "no_edge"
            assert 0.0 <= s.confidence <= 1.0
            # Never re-suggests an already-known edge.
            assert frozenset((s.source, s.target)) not in known_sets

        # Co-parents mom & dad should be recovered as spouses.
        pairs = {(frozenset((s.source, s.target)), s.relationship) for s in suggestions}
        assert (frozenset(("mom", "dad")), "spouse") in pairs

    def test_ranked_by_confidence(self, trained_model_path, monkeypatch) -> None:
        monkeypatch.setenv("PEDIGREE_GNN_PATH", trained_model_path)
        import services.api.services.gnn_pedigree_service as svc

        importlib.reload(svc)
        nodes, known = self._nodes_and_known()
        suggestions = svc.suggest_links_gnn(nodes, known, threshold=0.0)
        confs = [s.confidence for s in suggestions]
        assert confs == sorted(confs, reverse=True)

    def test_empty_when_no_known_edges(self, trained_model_path, monkeypatch) -> None:
        monkeypatch.setenv("PEDIGREE_GNN_PATH", trained_model_path)
        import services.api.services.gnn_pedigree_service as svc

        importlib.reload(svc)
        nodes, _ = self._nodes_and_known()
        assert svc.suggest_links_gnn(nodes, [], threshold=0.4) == []


class TestAvailabilityGate:
    def test_disabled_by_env(self, trained_model_path, monkeypatch) -> None:
        monkeypatch.setenv("PEDIGREE_GNN_PATH", trained_model_path)
        monkeypatch.setenv("ENABLE_PEDIGREE_GNN", "false")
        import services.api.services.gnn_pedigree_service as svc

        importlib.reload(svc)
        assert svc.gnn_available() is False

    def test_missing_artifact(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PEDIGREE_GNN_PATH", str(tmp_path / "nope.pt"))
        monkeypatch.setenv("ENABLE_PEDIGREE_GNN", "true")
        import services.api.services.gnn_pedigree_service as svc

        importlib.reload(svc)
        assert svc.gnn_available() is False
