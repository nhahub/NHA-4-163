"""Pedigree link prediction / completion (Tier 6 — ML Trust & Decision Support).

Suggests likely-missing family edges so clinicians can complete incomplete
pedigrees — the structural task the feature roadmap frames as "GNN link
prediction". A true graph neural network needs a trained embedding model and
PyTorch Geometric; here we ship a **transparent structural link-predictor**
instead: it composes known relationships with a small relationship-algebra and
scores candidates by how many independent paths support them (a common-neighbour
style signal). This is deterministic, dependency-free and fully explainable —
every suggestion carries the path that produced it — in the same spirit as the
rest of the service layer. It is deliberately swappable for a learned GNN
without changing the router or schema.

Relationships are categorical (``parent``, ``child``, ``sibling``, ``spouse``,
``grandparent``, ``grandchild``, ``aunt_uncle``, ``nibling``), matching
:func:`services.api.services.inheritance_service.categorise_relationship`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

# ── Relationship algebra ──────────────────────────────────────────────────────

# Inverse of each directed relationship (used to traverse edges both ways).
_INVERSE: dict[str, str] = {
    "parent": "child",
    "child": "parent",
    "sibling": "sibling",
    "spouse": "spouse",
    "grandparent": "grandchild",
    "grandchild": "grandparent",
    "aunt_uncle": "nibling",
    "nibling": "aunt_uncle",
}

# Composition: (a REL1 b) then (b REL2 c) implies (a RESULT c).
# Only *canonical-direction* results are emitted (parent/grandparent/aunt_uncle/
# sibling/spouse); their inverses (child/grandchild/nibling) describe the same
# real-world edge and would only duplicate a suggestion, so they are omitted.
_COMPOSE: dict[tuple[str, str], str] = {
    ("parent", "parent"): "grandparent",  # grandparent through an intermediate parent
    ("parent", "sibling"): "parent",  # a parent of your sibling is your parent
    ("sibling", "parent"): "aunt_uncle",  # your sibling is your child's aunt/uncle
    ("sibling", "sibling"): "sibling",  # a sibling of your sibling is your sibling
    ("child", "parent"): "sibling",  # two children of the same parent are siblings
}

# Base confidence for each inferred relationship from a single supporting path.
_BASE_CONFIDENCE: dict[str, float] = {
    "grandparent": 0.85,
    "parent": 0.70,
    "aunt_uncle": 0.75,
    "sibling": 0.60,
    "spouse": 0.50,
}

# Symmetric relationships are stored/deduplicated on an unordered pair.
_SYMMETRIC = {"sibling", "spouse"}


@dataclass(frozen=True)
class KnownEdge:
    """A recorded, directed relationship: ``source`` is ``relationship`` of ``target``."""

    source: str
    target: str
    relationship: str


@dataclass(frozen=True)
class SuggestedLink:
    """A predicted missing edge with its explanation."""

    source: str
    target: str
    relationship: str
    confidence: float
    support: int
    rationale: str


def _pair_key(a: str, b: str, rel: str) -> tuple[Any, ...]:
    """Deduplication key: unordered for symmetric relationships, else directed."""
    if rel in _SYMMETRIC:
        return (rel, frozenset((a, b)))
    return (rel, a, b)


def suggest_links(edges: list[KnownEdge], max_suggestions: int = 25) -> list[SuggestedLink]:
    """Predict likely-missing pedigree edges from the known relationships.

    Args:
        edges: Known directed relationships among family members.
        max_suggestions: Cap on the number of ranked suggestions returned.

    Returns:
        Ranked :class:`SuggestedLink` objects (highest confidence first). Edges
        already present (in either direction for symmetric relationships) are
        never suggested.
    """
    # Build a traversable adjacency including inverse edges, and remember the
    # full set of known relationships for de-duplication.
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    known: set[tuple[Any, ...]] = set()
    parents_of: dict[str, set[str]] = defaultdict(set)

    for e in edges:
        if not e.source or not e.target or e.source == e.target:
            continue
        rel = e.relationship
        if rel not in _INVERSE:
            continue
        adjacency[e.source].append((e.target, rel))
        adjacency[e.target].append((e.source, _INVERSE[rel]))
        known.add(_pair_key(e.source, e.target, rel))
        known.add(_pair_key(e.target, e.source, _INVERSE[rel]))
        if rel == "parent":
            parents_of[e.target].add(e.source)
        elif rel == "child":
            parents_of[e.source].add(e.target)

    # Candidate (key) → {"rel", "src", "tgt", intermediates:set, paths:list}.
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _add(src: str, tgt: str, rel: str, intermediate: str, why: str) -> None:
        if src == tgt or rel not in _BASE_CONFIDENCE:
            return
        if _pair_key(src, tgt, rel) in known:
            return
        key = _pair_key(src, tgt, rel)
        entry = candidates.setdefault(
            key,
            {"rel": rel, "src": src, "tgt": tgt, "inter": set(), "paths": []},
        )
        if intermediate not in entry["inter"]:
            entry["inter"].add(intermediate)
            entry["paths"].append(why)

    # 1. Two-hop relationship composition (common-neighbour traversal).
    for a in list(adjacency):
        for b, r1 in adjacency[a]:
            for c, r2 in adjacency.get(b, []):
                if c == a:
                    continue
                inferred = _COMPOSE.get((r1, r2))
                if inferred is None or inferred not in _BASE_CONFIDENCE:
                    continue
                _add(
                    a,
                    c,
                    inferred,
                    intermediate=b,
                    why=f"{a} is {r1} of {b}, {b} is {r2} of {c}",
                )

    # 2. Co-parents of a shared child are spouses/partners.
    for child, parents in parents_of.items():
        plist = sorted(parents)
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                _add(
                    plist[i],
                    plist[j],
                    "spouse",
                    intermediate=child,
                    why=f"{plist[i]} and {plist[j]} are both parents of {child}",
                )

    suggestions: list[SuggestedLink] = []
    for entry in candidates.values():
        support = len(entry["inter"])
        base = _BASE_CONFIDENCE[entry["rel"]]
        # Independent supporting paths compound the confidence (capped).
        confidence = min(0.99, 1.0 - (1.0 - base) ** support)
        rationale = "; ".join(entry["paths"][:3])
        if support > 3:
            rationale += f"; (+{support - 3} more paths)"
        suggestions.append(
            SuggestedLink(
                source=entry["src"],
                target=entry["tgt"],
                relationship=entry["rel"],
                confidence=round(confidence, 4),
                support=support,
                rationale=rationale,
            )
        )

    suggestions.sort(key=lambda s: (s.confidence, s.support), reverse=True)
    return suggestions[:max_suggestions]
