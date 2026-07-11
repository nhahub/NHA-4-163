"""What-If hereditary-risk simulator (Tier 6 — ML Trust & Decision Support).

Lets a clinician toggle a patient's clinical/family factors and see how the
hereditary-risk estimate moves — *without writing anything to the database*.
It pairs with the SHAP explanations from the production model: where SHAP
explains a single prediction, this shows how the same prediction *responds* to
counterfactual changes ("what if this relative were affected?", "what if we
add this hereditary diagnosis?").

Design (same deterministic, dependency-free philosophy as
:mod:`services.api.services.differential_service` and
:mod:`services.api.services.prs_service`):

* Risk is a **transparent logistic model** over a small set of interpretable
  factors.  Each factor contributes an additive term in log-odds space, so a
  contribution is directly comparable to a SHAP value (log-odds units) and the
  whole thing is auditable — there is no opaque tree ensemble here.
* The coefficients are illustrative and calibrated so the demo behaves
  sensibly; they are **not** trained clinical weights.  The module is
  deliberately swappable: the real system would seed the baseline factors from
  ``feature_service`` and let this recompute counterfactuals in-memory.

The point of the tier is *trust*: the numbers are explainable and the deltas
are reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ── Factor catalogue ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RiskFactor:
    """One interpretable input to the transparent logistic risk model.

    Attributes:
        key: Stable identifier used in requests.
        display: Human-readable label.
        coefficient: Per-unit weight in log-odds space.
        default: Baseline value assumed when the factor is not supplied.
        unit: Short description of what one unit means.
        kind: ``count`` (non-negative integer), ``rate`` (a [0, 1] fraction) or
            ``binary`` (0/1 flag).
    """

    key: str
    display: str
    coefficient: float
    default: float
    unit: str
    kind: str


# Intercept chosen so an "average" patient (all defaults) sits at a low-moderate
# baseline risk.
_INTERCEPT = -2.30

RISK_FACTORS: dict[str, RiskFactor] = {
    "age_years": RiskFactor(
        key="age_years",
        display="Age",
        coefficient=0.018,
        default=45.0,
        unit="years",
        kind="count",
    ),
    "hereditary_condition_count": RiskFactor(
        key="hereditary_condition_count",
        display="Hereditary conditions on record",
        coefficient=0.90,
        default=0.0,
        unit="conditions",
        kind="count",
    ),
    "comorbidity_count": RiskFactor(
        key="comorbidity_count",
        display="Total comorbidities",
        coefficient=0.20,
        default=1.0,
        unit="conditions",
        kind="count",
    ),
    "affected_first_degree_relatives": RiskFactor(
        key="affected_first_degree_relatives",
        display="Affected first-degree relatives",
        coefficient=0.85,
        default=0.0,
        unit="relatives",
        kind="count",
    ),
    "affected_second_degree_relatives": RiskFactor(
        key="affected_second_degree_relatives",
        display="Affected second-degree relatives",
        coefficient=0.35,
        default=0.0,
        unit="relatives",
        kind="count",
    ),
    "family_risk_prevalence": RiskFactor(
        key="family_risk_prevalence",
        display="Family disease prevalence",
        coefficient=1.60,
        default=0.0,
        unit="fraction of relatives affected",
        kind="rate",
    ),
    "active_medication_count": RiskFactor(
        key="active_medication_count",
        display="Active medications",
        coefficient=0.08,
        default=1.0,
        unit="medications",
        kind="count",
    ),
    "carrier_variant_present": RiskFactor(
        key="carrier_variant_present",
        display="Pathogenic carrier variant present",
        coefficient=1.20,
        default=0.0,
        unit="flag",
        kind="binary",
    ),
}


def _sigmoid(x: float) -> float:
    """Logistic function."""
    return 1.0 / (1.0 + math.exp(-x))


def _coerce(factor: RiskFactor, value: float) -> float:
    """Clamp a raw value into the range implied by the factor's ``kind``."""
    v = float(value)
    if factor.kind == "rate":
        return min(max(v, 0.0), 1.0)
    if factor.kind == "binary":
        return 1.0 if v >= 0.5 else 0.0
    return max(v, 0.0)  # count


@dataclass(frozen=True)
class FactorContribution:
    """A single factor's contribution to the log-odds, in SHAP-like form."""

    key: str
    display: str
    value: float
    log_odds_contribution: float
    delta_from_baseline: float


@dataclass(frozen=True)
class WhatIfResult:
    """Baseline vs simulated risk with per-factor explanations."""

    baseline_risk: float
    simulated_risk: float
    risk_delta: float
    baseline_factors: dict[str, float]
    simulated_factors: dict[str, float]
    contributions: list[FactorContribution] = field(default_factory=list)
    interpretation: str = ""


def _risk_from_factors(factors: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Return (risk, per-factor log-odds terms) for a full factor mapping."""
    log_odds = _INTERCEPT
    terms: dict[str, float] = {}
    for key, factor in RISK_FACTORS.items():
        value = _coerce(factor, factors.get(key, factor.default))
        term = factor.coefficient * value
        terms[key] = term
        log_odds += term
    return _sigmoid(log_odds), terms


def _normalise(factors: dict[str, float] | None) -> dict[str, float]:
    """Fill in defaults and drop unknown keys, returning a full factor map."""
    factors = factors or {}
    return {key: _coerce(f, factors.get(key, f.default)) for key, f in RISK_FACTORS.items()}


def simulate(
    baseline: dict[str, float] | None,
    modifications: dict[str, float] | None,
) -> WhatIfResult:
    """Recompute risk under a set of counterfactual factor changes.

    Args:
        baseline: Current patient factors (missing factors use their defaults).
            Unknown keys are ignored.
        modifications: Overrides applied on top of the baseline to form the
            "what-if" scenario. Unknown keys are ignored.

    Returns:
        A :class:`WhatIfResult` comparing baseline and simulated risk with a
        per-factor breakdown of what moved the log-odds.
    """
    base_factors = _normalise(baseline)
    sim_factors = dict(base_factors)
    for key, value in (modifications or {}).items():
        if key in RISK_FACTORS:
            sim_factors[key] = _coerce(RISK_FACTORS[key], value)

    baseline_risk, base_terms = _risk_from_factors(base_factors)
    simulated_risk, sim_terms = _risk_from_factors(sim_factors)

    contributions = [
        FactorContribution(
            key=key,
            display=RISK_FACTORS[key].display,
            value=round(sim_factors[key], 4),
            log_odds_contribution=round(sim_terms[key], 4),
            delta_from_baseline=round(sim_terms[key] - base_terms[key], 4),
        )
        for key in RISK_FACTORS
    ]
    # Most-influential changes first, then largest absolute contribution.
    contributions.sort(
        key=lambda c: (abs(c.delta_from_baseline), abs(c.log_odds_contribution)),
        reverse=True,
    )

    delta = simulated_risk - baseline_risk
    if abs(delta) < 1e-4:
        verb = "does not change"
    elif delta > 0:
        verb = f"increases by {delta * 100:.1f} percentage points"
    else:
        verb = f"decreases by {abs(delta) * 100:.1f} percentage points"

    return WhatIfResult(
        baseline_risk=round(baseline_risk, 4),
        simulated_risk=round(simulated_risk, 4),
        risk_delta=round(delta, 4),
        baseline_factors={k: round(v, 4) for k, v in base_factors.items()},
        simulated_factors={k: round(v, 4) for k, v in sim_factors.items()},
        contributions=contributions,
        interpretation=(
            f"Simulated hereditary risk {verb} "
            f"(from {baseline_risk:.1%} to {simulated_risk:.1%})."
        ),
    )
