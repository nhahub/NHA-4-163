"""Unit tests for pedigree link prediction / completion (Tier 6)."""

from __future__ import annotations

from services.api.services.pedigree_service import KnownEdge, suggest_links


def _rel(suggestions, source, target):
    """Return the relationship suggested between two nodes (unordered), or None."""
    for s in suggestions:
        if {s.source, s.target} == {source, target}:
            return s
    return None


class TestSuggestLinks:
    def test_grandparent_inferred_through_parent_chain(self) -> None:
        # GM is parent of P; P is parent of C  ⇒  GM is grandparent of C.
        edges = [
            KnownEdge("GM", "P", "parent"),
            KnownEdge("P", "C", "parent"),
        ]
        suggestions = suggest_links(edges)
        gp = _rel(suggestions, "GM", "C")
        assert gp is not None
        assert gp.relationship == "grandparent"
        assert gp.source == "GM" and gp.target == "C"

    def test_shared_parent_implies_siblings(self) -> None:
        # A parent of two children who are siblings of each other.
        edges = [
            KnownEdge("P", "S1", "sibling"),
            KnownEdge("P", "S2", "sibling"),
        ]
        suggestions = suggest_links(edges)
        sib = _rel(suggestions, "S1", "S2")
        assert sib is not None
        assert sib.relationship == "sibling"

    def test_aunt_uncle_inferred(self) -> None:
        # P's sibling SIB, and P is parent of C  ⇒  SIB is aunt/uncle of C.
        edges = [
            KnownEdge("P", "SIB", "sibling"),
            KnownEdge("P", "C", "parent"),
        ]
        suggestions = suggest_links(edges)
        au = _rel(suggestions, "SIB", "C")
        assert au is not None
        assert au.relationship == "aunt_uncle"
        assert au.source == "SIB" and au.target == "C"

    def test_co_parents_suggested_as_spouses(self) -> None:
        edges = [
            KnownEdge("MOM", "C", "parent"),
            KnownEdge("DAD", "C", "parent"),
        ]
        suggestions = suggest_links(edges)
        sp = _rel(suggestions, "MOM", "DAD")
        assert sp is not None
        assert sp.relationship == "spouse"

    def test_known_edges_not_re_suggested(self) -> None:
        edges = [
            KnownEdge("P", "S1", "sibling"),
            KnownEdge("P", "S2", "sibling"),
            KnownEdge("S1", "S2", "sibling"),  # already recorded
        ]
        suggestions = suggest_links(edges)
        assert _rel(suggestions, "S1", "S2") is None

    def test_more_paths_raise_confidence(self) -> None:
        # Two independent parents both linking S1 and S2 as siblings.
        single = suggest_links([KnownEdge("MOM", "S1", "parent"), KnownEdge("MOM", "S2", "parent")])
        double = suggest_links(
            [
                KnownEdge("MOM", "S1", "parent"),
                KnownEdge("MOM", "S2", "parent"),
                KnownEdge("DAD", "S1", "parent"),
                KnownEdge("DAD", "S2", "parent"),
            ]
        )
        s1 = _rel(single, "S1", "S2")
        s2 = _rel(double, "S1", "S2")
        assert s1 and s2
        assert s2.support > s1.support
        assert s2.confidence > s1.confidence

    def test_empty_input_returns_nothing(self) -> None:
        assert suggest_links([]) == []

    def test_self_and_unknown_edges_ignored(self) -> None:
        edges = [
            KnownEdge("P", "P", "sibling"),  # self loop
            KnownEdge("P", "X", "cousin"),  # unknown relationship
        ]
        assert suggest_links(edges) == []

    def test_results_sorted_by_confidence(self) -> None:
        edges = [
            KnownEdge("GM", "P", "parent"),
            KnownEdge("P", "C1", "parent"),
            KnownEdge("P", "C2", "parent"),
        ]
        suggestions = suggest_links(edges)
        confidences = [s.confidence for s in suggestions]
        assert confidences == sorted(confidences, reverse=True)

    def test_max_suggestions_respected(self) -> None:
        edges = [KnownEdge("P", f"S{i}", "sibling") for i in range(6)]
        suggestions = suggest_links(edges, max_suggestions=3)
        assert len(suggestions) <= 3
