"""Unit tests for pedigree graph construction & features (Tier 6, #22).

Pure — no torch required.
"""

from __future__ import annotations

from ml.models.pedigree_graph import (
    CATEGORY_GENERATION,
    NODE_FEATURE_DIM,
    PedigreeNode,
    derive_relationships,
    generate_families,
    node_feature_vector,
)


class TestNodeFeatures:
    def test_vector_width_is_constant(self) -> None:
        node = PedigreeNode("p", "female", 0, False, 1.0, is_proband=True)
        assert len(node_feature_vector(node)) == NODE_FEATURE_DIM

    def test_sex_one_hot(self) -> None:
        male = node_feature_vector(PedigreeNode("a", "male", 1, False, 0.5))
        female = node_feature_vector(PedigreeNode("b", "female", 1, False, 0.5))
        assert male[0] == 1.0 and male[1] == 0.0
        assert female[0] == 0.0 and female[1] == 1.0

    def test_generation_sign_flags(self) -> None:
        ancestor = node_feature_vector(PedigreeNode("a", None, 2, False, 0.25))
        descendant = node_feature_vector(PedigreeNode("d", None, -1, False, 0.5))
        assert ancestor[4] == 1.0 and ancestor[5] == 0.0
        assert descendant[4] == 0.0 and descendant[5] == 1.0

    def test_degree_clamped(self) -> None:
        v = node_feature_vector(PedigreeNode("x", None, 0, False, 5.0))
        assert v[7] == 1.0


class TestDeriveRelationships:
    def _family(self):
        node_ids = ["gm", "gf", "mom", "dad", "uncle", "proband", "sib"]
        parent_edges = [
            ("gm", "mom"),
            ("gf", "mom"),
            ("gm", "uncle"),
            ("gf", "uncle"),
            ("mom", "proband"),
            ("dad", "proband"),
            ("mom", "sib"),
            ("dad", "sib"),
        ]
        spouse_edges = [("gm", "gf"), ("mom", "dad")]
        return node_ids, parent_edges, spouse_edges

    def test_core_relationships(self) -> None:
        rels = derive_relationships(*self._family())
        assert rels[frozenset({"mom", "dad"})] == "spouse"
        assert rels[frozenset({"gm", "mom"})] == "parent"
        assert rels[frozenset({"proband", "sib"})] == "sibling"
        assert rels[frozenset({"mom", "uncle"})] == "sibling"
        assert rels[frozenset({"gm", "proband"})] == "grandparent"
        assert rels[frozenset({"uncle", "proband"})] == "aunt_uncle"

    def test_unrelated_pair_absent(self) -> None:
        # dad's parents are not in the family → dad has no link to gm.
        rels = derive_relationships(*self._family())
        assert frozenset({"dad", "gm"}) not in rels


class TestGenerateFamilies:
    def test_deterministic_and_has_proband(self) -> None:
        fams_a = generate_families(5, seed=1)
        fams_b = generate_families(5, seed=1)
        assert len(fams_a) == 5
        for fam in fams_a:
            assert fam.proband_id
            assert any(n.is_proband for n in fam.nodes)
        # Reproducible node counts for a fixed seed.
        assert [len(f.nodes) for f in fams_a] == [len(f.nodes) for f in fams_b]

    def test_generation_map_covers_categories(self) -> None:
        for cat in ("parent", "child", "sibling", "grandparent", "aunt_uncle"):
            assert cat in CATEGORY_GENERATION
