"""Pedigree graph construction & node features for the GNN link predictor.

Pure Python / stdlib only (no torch, no numpy) so it can be imported by both the
training script (:mod:`ml.training.train_link_prediction`) and the API inference
service (:mod:`services.api.services.gnn_pedigree_service`) without pulling heavy
dependencies, and unit-tested on its own. It defines:

* the canonical node-feature encoding (identical at train and inference time),
* a synthetic pedigree generator (fundamental parent/spouse edges only), and
* :func:`derive_relationships`, which expands the fundamentals into every
  pairwise relationship label the model predicts.

A node's feature vector is derived purely from its *role* relative to the
proband — sex, generation offset, affected status and coefficient of
relatedness — so the same encoding is produced whether a node comes from a
synthetic family or from ``FamilyMemberHistory`` rows at inference.

Training/inference topology match
---------------------------------
At inference the only recorded edges are proband→relative (``FamilyMemberHistory``
is proband-centric), so the message-passing graph is a *star* centred on the
proband. Training mirrors this: message passing uses the proband star, and the
decoder is trained to recover the *relative↔relative* relationships (siblings,
grandparent, aunt/uncle, co-parent spouses) that are not directly recorded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

NODE_FEATURE_DIM = 8

# Relationship category → generation offset from the proband (proband = 0).
CATEGORY_GENERATION: dict[str, int] = {
    "self": 0,
    "parent": 1,
    "child": -1,
    "sibling": 0,
    "spouse": 0,
    "grandparent": 2,
    "grandchild": -2,
    "aunt_uncle": 1,
    "nibling": -1,
    "cousin": 0,
}


@dataclass
class PedigreeNode:
    """A person in a pedigree, described by their role relative to the proband.

    Attributes:
        node_id: Stable identifier (string).
        sex: ``"male"`` / ``"female"`` / ``None``.
        generation: Signed generation offset from the proband (parent = +1).
        affected: Whether the person has the condition of interest.
        degree: Coefficient of relatedness to the proband in [0, 1]
            (proband = 1.0, first-degree = 0.5, spouse = 0.0).
        is_proband: True for the index patient.
    """

    node_id: str
    sex: str | None
    generation: int
    affected: bool
    degree: float
    is_proband: bool = False


def node_feature_vector(node: PedigreeNode) -> list[float]:
    """Return the canonical fixed-width feature vector for a node.

    Args:
        node: The pedigree node.

    Returns:
        A list of :data:`NODE_FEATURE_DIM` floats.
    """
    sex = (node.sex or "").lower()
    gen = float(node.generation)
    return [
        1.0 if sex == "male" else 0.0,
        1.0 if sex == "female" else 0.0,
        gen / 2.0,
        abs(gen) / 2.0,
        1.0 if node.generation > 0 else 0.0,
        1.0 if node.generation < 0 else 0.0,
        1.0 if node.affected else 0.0,
        max(0.0, min(1.0, float(node.degree))),
    ]


@dataclass
class SyntheticFamily:
    """A generated family described by its fundamental edges.

    ``parent_edges`` are ``(parent_id, child_id)`` and ``spouse_edges`` are
    unordered ``(a_id, b_id)``. All other relationships are derived from these
    by :func:`derive_relationships`. ``proband_id`` identifies the index node.
    """

    nodes: list[PedigreeNode]
    parent_edges: list[tuple[str, str]] = field(default_factory=list)
    spouse_edges: list[tuple[str, str]] = field(default_factory=list)
    proband_id: str = ""


def derive_relationships(
    node_ids: list[str],
    parent_edges: list[tuple[str, str]],
    spouse_edges: list[tuple[str, str]],
) -> dict[frozenset, str]:
    """Expand fundamental parent/spouse edges into all pairwise relationships.

    Args:
        node_ids: All member ids.
        parent_edges: ``(parent, child)`` directed edges.
        spouse_edges: unordered spouse pairs.

    Returns:
        Mapping of ``frozenset({a, b})`` → category for every pair that has one
        of ``parent`` / ``sibling`` / ``spouse`` / ``grandparent`` /
        ``aunt_uncle``. Pairs absent from the mapping have no modelled
        relationship (i.e. a legitimate ``no_edge``).
    """
    parents: dict[str, set[str]] = {n: set() for n in node_ids}
    children: dict[str, set[str]] = {n: set() for n in node_ids}
    for p, c in parent_edges:
        parents[c].add(p)
        children[p].add(c)
    spouses: dict[str, set[str]] = {n: set() for n in node_ids}
    for a, b in spouse_edges:
        spouses[a].add(b)
        spouses[b].add(a)

    rels: dict[frozenset, str] = {}
    n = len(node_ids)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = node_ids[i], node_ids[j]
            key = frozenset((a, b))
            # parent / child
            if b in parents[a] or a in parents[b]:
                rels[key] = "parent"
                continue
            # spouse
            if b in spouses[a]:
                rels[key] = "spouse"
                continue
            # sibling: share ≥1 parent
            if parents[a] & parents[b]:
                rels[key] = "sibling"
                continue
            # grandparent: a is parent of a parent of b (or vice versa)
            gp_a = {gp for p in parents[a] for gp in parents[p]}
            gp_b = {gp for p in parents[b] for gp in parents[p]}
            if b in gp_a or a in gp_b:
                rels[key] = "grandparent"
                continue
            # aunt/uncle: a is a sibling of a parent of b (or vice versa)
            sib_of_parent_b = any(parents[p] & parents[a] for p in parents[b] if p != a)
            sib_of_parent_a = any(parents[p] & parents[b] for p in parents[a] if p != b)
            if sib_of_parent_b or sib_of_parent_a:
                rels[key] = "aunt_uncle"
                continue
    return rels


def _sex(rng: random.Random) -> str:
    return rng.choice(["male", "female"])


def generate_family(rng: random.Random, prefix: str) -> SyntheticFamily:
    """Generate one random three-generation pedigree (fundamentals only).

    Args:
        rng: Seeded random generator.
        prefix: Unique node-id prefix for this family.

    Returns:
        A :class:`SyntheticFamily` with parent/spouse edges and a proband.
    """
    nodes: list[PedigreeNode] = []
    parent_edges: list[tuple[str, str]] = []
    spouse_edges: list[tuple[str, str]] = []
    affected_prob = rng.uniform(0.05, 0.4)

    def add(
        node_id: str, sex: str | None, generation: int, degree: float, is_proband: bool = False
    ) -> str:
        nid = f"{prefix}:{node_id}"
        nodes.append(
            PedigreeNode(
                node_id=nid,
                sex=sex,
                generation=generation,
                affected=rng.random() < affected_prob,
                degree=degree,
                is_proband=is_proband,
            )
        )
        return nid

    # Generation +1: parents.
    mom = add("mom", "female", 1, 0.5)
    dad = add("dad", "male", 1, 0.5)
    spouse_edges.append((mom, dad))

    # Generation +2: grandparents (parents of mom/dad), each set optional.
    mom_gps: list[str] = []
    if rng.random() < 0.85:
        gm = add("gm_m", "female", 2, 0.25)
        gf = add("gf_m", "male", 2, 0.25)
        spouse_edges.append((gm, gf))
        parent_edges += [(gm, mom), (gf, mom)]
        mom_gps = [gm, gf]
    dad_gps: list[str] = []
    if rng.random() < 0.85:
        gm2 = add("gm_p", "female", 2, 0.25)
        gf2 = add("gf_p", "male", 2, 0.25)
        spouse_edges.append((gm2, gf2))
        parent_edges += [(gm2, dad), (gf2, dad)]
        dad_gps = [gm2, gf2]

    # Aunts / uncles: extra children of a grandparent set (⇒ siblings of a parent).
    for i in range(rng.randint(0, 2)):
        if mom_gps:
            au = add(f"aunt_m{i}", _sex(rng), 1, 0.25)
            parent_edges += [(mom_gps[0], au), (mom_gps[1], au)]
    for i in range(rng.randint(0, 2)):
        if dad_gps:
            au = add(f"aunt_p{i}", _sex(rng), 1, 0.25)
            parent_edges += [(dad_gps[0], au), (dad_gps[1], au)]

    # Proband + siblings (generation 0), children of mom & dad.
    proband = add("proband", _sex(rng), 0, 1.0, is_proband=True)
    parent_edges += [(mom, proband), (dad, proband)]
    for i in range(rng.randint(0, 3)):
        sib = add(f"sib{i}", _sex(rng), 0, 0.5)
        parent_edges += [(mom, sib), (dad, sib)]

    # Proband's spouse + children (generation -1).
    if rng.random() < 0.7:
        spouse = add("spouse", _sex(rng), 0, 0.0)
        spouse_edges.append((proband, spouse))
        for i in range(rng.randint(0, 3)):
            child = add(f"child{i}", _sex(rng), -1, 0.5)
            parent_edges += [(proband, child), (spouse, child)]

    return SyntheticFamily(
        nodes=nodes,
        parent_edges=parent_edges,
        spouse_edges=spouse_edges,
        proband_id=proband,
    )


def generate_families(n: int, seed: int = 42) -> list[SyntheticFamily]:
    """Generate ``n`` synthetic families with a fixed seed."""
    rng = random.Random(seed)
    return [generate_family(rng, prefix=f"f{i}") for i in range(n)]
