"""Response schemas for pedigree link prediction / completion (Tier 6)."""

from __future__ import annotations

import uuid

from pydantic import BaseModel


class SuggestedLinkSchema(BaseModel):
    """A predicted missing pedigree edge with its explanation."""

    source: str
    target: str
    relationship: str
    confidence: float
    support: int
    rationale: str


class PedigreeSuggestionsResponse(BaseModel):
    """Ranked suggested edges to complete a patient's pedigree."""

    patient_id: uuid.UUID
    method: str  # "gnn" (trained GraphSAGE) or "structural" (fallback)
    known_edges: int
    members: int
    suggestions: list[SuggestedLinkSchema]
    note: str | None = None
