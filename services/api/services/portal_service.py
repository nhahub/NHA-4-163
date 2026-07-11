"""Patient-portal presentation logic (Tier 7).

Pure helpers that shape internal records into the read-only, patient-friendly
views returned by the portal endpoints.  Two concerns live here:

* Translating a raw risk score into a lay-friendly tier + guidance (so the
  patient sees "moderate risk — discuss screening with your clinician" rather
  than a bare probability).
* De-identifying family-history rows for the patient's own pedigree view —
  relatives are PHI, so only relationship and affected-status are exposed,
  never a relative's name or contact details.

Keeping these pure makes the patient-facing wording unit-testable and free of
DB/HTTP coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Lay-friendly banding of a calibrated risk probability.
_LOW_MAX = 0.33
_MODERATE_MAX = 0.67


@dataclass(frozen=True)
class RiskProfileView:
    """A patient-friendly rendering of a risk prediction."""

    risk_score: float | None
    risk_tier: str
    band: str
    guidance: str
    model_version: str | None
    predicted_at: str | None


def lay_band(score: float) -> str:
    """Bucket a risk probability into a lay-friendly band.

    Args:
        score: Calibrated probability in [0, 1].

    Returns:
        One of ``"low"``, ``"moderate"``, ``"high"``.
    """
    if score < _LOW_MAX:
        return "low"
    if score < _MODERATE_MAX:
        return "moderate"
    return "high"


def _guidance_for_band(band: str) -> str:
    """Return patient-facing guidance text for a risk band."""
    return {
        "low": (
            "Your estimated hereditary risk is low. Continue routine preventive "
            "care and let your care team know of any new family history."
        ),
        "moderate": (
            "Your estimated hereditary risk is moderate. Consider discussing "
            "screening options and family history with your clinician."
        ),
        "high": (
            "Your estimated hereditary risk is high. We recommend discussing "
            "genetic counselling and enhanced screening with your care team."
        ),
    }[band]


def build_risk_profile(
    latest: Any | None,
) -> RiskProfileView:
    """Build a patient-friendly risk profile from a PredictionLog row.

    Args:
        latest: The most recent ``PredictionLog`` for the patient, or ``None``
            when no prediction exists yet.

    Returns:
        A :class:`RiskProfileView`.  When ``latest`` is ``None`` the profile
        reports an ``"unknown"`` band with neutral guidance.
    """
    if latest is None:
        return RiskProfileView(
            risk_score=None,
            risk_tier="unknown",
            band="unknown",
            guidance=(
                "No hereditary risk assessment is available yet. Your care team "
                "can run one during your next visit."
            ),
            model_version=None,
            predicted_at=None,
        )

    score = float(latest.risk_score)
    band = lay_band(score)
    predicted_at = (
        latest.predicted_at.isoformat()
        if getattr(latest, "predicted_at", None) is not None
        else None
    )
    return RiskProfileView(
        risk_score=round(score, 4),
        risk_tier=latest.risk_tier,
        band=band,
        guidance=_guidance_for_band(band),
        model_version=getattr(latest, "model_version", None),
        predicted_at=predicted_at,
    )


def deidentify_family_member(
    fmh: Any,
) -> dict[str, Any]:
    """Render a family-history row for the patient's own pedigree view.

    Exposes relationship, sex, affected status, and relatedness only — never
    the relative's name, contact details, or date of birth.

    Args:
        fmh: A ``FamilyMemberHistory`` ORM row.

    Returns:
        A PHI-safe dict describing the relative.
    """
    conditions = fmh.conditions or []
    affected = len(conditions) > 0
    condition_names: list[str] = []
    for cond in conditions:
        if isinstance(cond, dict):
            code = cond.get("code") or {}
            display = code.get("display") if isinstance(code, dict) else None
            if display:
                condition_names.append(display)

    degree = float(fmh.degree_of_relatedness) if fmh.degree_of_relatedness is not None else None
    return {
        "relationship_code": fmh.relationship_code,
        "relationship_display": fmh.relationship_display,
        "sex": fmh.sex,
        "degree_of_relatedness": degree,
        "affected": affected,
        "conditions": condition_names,
    }
