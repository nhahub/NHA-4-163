"""Pedigree link-prediction endpoints (Tier 6 — ML Trust & Decision Support).

GET /patients/{id}/pedigree/suggestions — likely-missing family edges

Builds a relationship graph from the patient's ``FamilyMemberHistory`` records
and suggests edges that would complete the pedigree. When a trained GraphSAGE
link-prediction model is available (see
:mod:`services.api.services.gnn_pedigree_service`) it is used; otherwise the
endpoint falls back to the deterministic structural predictor in
:mod:`services.api.services.pedigree_service`. The ``method`` field reports which
produced the suggestions.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from libs.common.models.family_member_history import FamilyMemberHistory
from libs.common.models.patient import Patient
from ml.models.pedigree_graph import CATEGORY_GENERATION, PedigreeNode
from services.api.db import DbSession
from services.api.schemas.pedigree_schemas import (
    PedigreeSuggestionsResponse,
    SuggestedLinkSchema,
)
from services.api.services.gnn_pedigree_service import gnn_available, suggest_links_gnn
from services.api.services.inheritance_service import categorise_relationship
from services.api.services.pedigree_service import KnownEdge, suggest_links

log = logging.getLogger(__name__)

router = APIRouter(tags=["decision-support"])

_PROBAND = "proband"

# Categories that anchor a usable graph edge (others carry no composition rule).
_EDGEABLE = {
    "parent",
    "child",
    "sibling",
    "grandparent",
    "grandchild",
    "aunt_uncle",
    "nibling",
    "spouse",
}


# Map a proband→relative category onto a directed KnownEdge (source REL target)
# for the structural predictor. ``P`` is the proband, ``R`` the relative.
def _edge_for(category: str, proband: str, relative: str) -> KnownEdge | None:
    if category == "parent":
        return KnownEdge(relative, proband, "parent")
    if category == "child":
        return KnownEdge(proband, relative, "parent")
    if category == "sibling":
        return KnownEdge(proband, relative, "sibling")
    if category == "grandparent":
        return KnownEdge(relative, proband, "grandparent")
    if category == "grandchild":
        return KnownEdge(proband, relative, "grandparent")
    if category == "aunt_uncle":
        return KnownEdge(relative, proband, "aunt_uncle")
    if category == "nibling":
        return KnownEdge(proband, relative, "aunt_uncle")
    if category == "spouse":
        return KnownEdge(proband, relative, "spouse")
    return None  # cousin / unknown — no composition rule


@router.get(
    "/patients/{patient_id}/pedigree/suggestions",
    response_model=PedigreeSuggestionsResponse,
    summary="Suggest likely-missing pedigree edges",
)
async def get_pedigree_suggestions(
    patient_id: uuid.UUID,
    db: DbSession,
    max_suggestions: int = Query(default=25, ge=1, le=100),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
) -> PedigreeSuggestionsResponse:
    """Suggest edges that would complete the patient's pedigree.

    Uses the trained GraphSAGE link predictor when available, otherwise the
    structural fallback.

    Args:
        patient_id: Proband patient UUID.
        db: Async database session.
        max_suggestions: Maximum number of ranked suggestions.
        threshold: Minimum predicted probability (GNN path only).

    Returns:
        Ranked suggested links with human-readable labels and rationale.

    Raises:
        HTTPException 404: Patient not found.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    members = (
        (
            await db.execute(
                select(FamilyMemberHistory).where(FamilyMemberHistory.patient_id == patient_id)
            )
        )
        .scalars()
        .all()
    )

    labels: dict[str, str] = {_PROBAND: "Proband (this patient)"}
    edges: list[KnownEdge] = []  # structural representation
    nodes: list[PedigreeNode] = [  # GNN representation
        PedigreeNode(
            node_id=_PROBAND,
            sex=(patient.gender.value if patient.gender is not None else None),
            generation=0,
            affected=False,
            degree=1.0,
            is_proband=True,
        )
    ]
    known_pairs: list[tuple[str, str]] = []

    for m in members:
        node = str(m.related_patient_id) if m.related_patient_id else f"fmh:{m.id}"
        labels[node] = m.relationship_display or m.relationship_code
        category = categorise_relationship(m.relationship_code)
        if category not in _EDGEABLE:
            continue

        edge = _edge_for(category, _PROBAND, node)
        if edge is not None:
            edges.append(edge)

        nodes.append(
            PedigreeNode(
                node_id=node,
                sex=m.sex,
                generation=CATEGORY_GENERATION.get(category, 0),
                affected=bool(m.conditions),
                degree=(
                    float(m.degree_of_relatedness) if m.degree_of_relatedness is not None else 0.5
                ),
            )
        )
        known_pairs.append((_PROBAND, node))

    if gnn_available():
        method = "gnn"
        suggestions = suggest_links_gnn(
            nodes, known_pairs, max_suggestions=max_suggestions, threshold=threshold
        )
    else:
        method = "structural"
        suggestions = suggest_links(edges, max_suggestions=max_suggestions)

    def _label(node: str) -> str:
        return labels.get(node, node)

    def _relabel(text: str) -> str:
        # Replace internal node ids with human labels (longest ids first so
        # "fmh:<uuid>" is substituted before any bare uuid substring).
        for node in sorted(labels, key=len, reverse=True):
            text = text.replace(node, labels[node])
        return text

    note = None
    if not known_pairs:
        note = "No usable family relationships recorded — add relatives to enable suggestions."

    log.info(
        "Pedigree suggestions: patient=%s method=%s edges=%d suggestions=%d",
        patient_id,
        method,
        len(known_pairs),
        len(suggestions),
    )
    return PedigreeSuggestionsResponse(
        patient_id=patient_id,
        method=method,
        known_edges=len(known_pairs),
        members=len(members),
        suggestions=[
            SuggestedLinkSchema(
                source=_label(s.source),
                target=_label(s.target),
                relationship=s.relationship,
                confidence=s.confidence,
                support=s.support,
                rationale=_relabel(s.rationale),
            )
            for s in suggestions
        ],
        note=note,
    )
