"""Model monitoring & fairness metrics (Tier 6 — ML Trust & Decision Support).

Pure, dependency-free statistics used to keep a regulated PHI model honest:

* **Drift** — Population Stability Index (PSI) between a reference and a current
  score distribution. PSI is the standard model-monitoring drift metric; the
  usual thresholds are <0.10 (stable), 0.10–0.25 (moderate shift) and >0.25
  (significant shift).
* **Fairness** — risk-score parity across demographic subgroups (sex / age
  band / ethnicity / race). Reports each group's mean score, the *disparate
  impact ratio* (min mean ÷ max mean; the "four-fifths rule" flags <0.80) and
  the *statistical parity difference* (max − min mean).
* **Calibration** — Brier score and a reliability table, for when observed
  outcomes are available.

Everything here operates on plain Python numbers so it can be unit-tested
without a database, MLflow, numpy or scipy — mirroring the rest of the
deterministic service layer.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

# ── Drift (Population Stability Index) ────────────────────────────────────────


@dataclass(frozen=True)
class DriftBin:
    """One PSI bucket over the [0, 1] score range."""

    lower: float
    upper: float
    reference_pct: float
    current_pct: float
    contribution: float


@dataclass(frozen=True)
class DriftResult:
    """Population-stability-index drift summary between two score samples."""

    psi: float
    verdict: str
    reference_count: int
    current_count: int
    bins: list[DriftBin] = field(default_factory=list)


def _histogram(scores: list[float], edges: list[float]) -> list[float]:
    """Return the fraction of ``scores`` falling in each [edge, edge) bucket."""
    counts = [0] * (len(edges) - 1)
    n = len(scores)
    if n == 0:
        return [0.0] * (len(edges) - 1)
    for s in scores:
        s = min(max(float(s), 0.0), 1.0)
        # Find bucket; last bucket is closed on the right.
        for i in range(len(edges) - 1):
            if s < edges[i + 1] or i == len(edges) - 2:
                counts[i] += 1
                break
    return [c / n for c in counts]


def population_stability_index(
    reference: list[float], current: list[float], bins: int = 10
) -> DriftResult:
    """Compute the PSI between a reference and a current score distribution.

    Args:
        reference: Baseline risk scores in [0, 1] (e.g. a trusted window).
        current: Recent risk scores in [0, 1].
        bins: Number of equal-width buckets over [0, 1].

    Returns:
        A :class:`DriftResult`. With no data on either side the PSI is 0.0.
    """
    edges = [i / bins for i in range(bins + 1)]
    ref_pct = _histogram(reference, edges)
    cur_pct = _histogram(current, edges)

    eps = 1e-6  # avoid log(0) / division by zero on empty buckets
    out_bins: list[DriftBin] = []
    psi = 0.0
    for i in range(bins):
        r = max(ref_pct[i], eps)
        c = max(cur_pct[i], eps)
        contribution = (c - r) * math.log(c / r)
        psi += contribution
        out_bins.append(
            DriftBin(
                lower=round(edges[i], 4),
                upper=round(edges[i + 1], 4),
                reference_pct=round(ref_pct[i], 4),
                current_pct=round(cur_pct[i], 4),
                contribution=round(contribution, 5),
            )
        )

    if not reference or not current:
        psi = 0.0

    if psi < 0.10:
        verdict = "stable"
    elif psi < 0.25:
        verdict = "moderate_shift"
    else:
        verdict = "significant_shift"

    return DriftResult(
        psi=round(psi, 4),
        verdict=verdict,
        reference_count=len(reference),
        current_count=len(current),
        bins=out_bins,
    )


# ── Fairness (subgroup parity) ────────────────────────────────────────────────


@dataclass(frozen=True)
class GroupStat:
    """Risk-score summary for one demographic subgroup."""

    group: str
    count: int
    mean_score: float
    high_risk_rate: float


@dataclass(frozen=True)
class FairnessResult:
    """Subgroup parity summary across one demographic attribute."""

    attribute: str
    groups: list[GroupStat]
    disparate_impact_ratio: float
    statistical_parity_difference: float
    passes_four_fifths: bool
    interpretation: str


def fairness_report(
    attribute: str,
    scores_by_group: dict[str, list[float]],
    high_risk_threshold: float = 0.5,
    min_group_size: int = 1,
) -> FairnessResult:
    """Assess risk-score parity across subgroups of one attribute.

    Args:
        attribute: Name of the demographic attribute (e.g. ``sex``).
        scores_by_group: Mapping of group label → risk scores in [0, 1].
        high_risk_threshold: Score at/above which a patient is "high risk";
            used for the per-group high-risk rate.
        min_group_size: Groups smaller than this are excluded (too few samples
            to be meaningful, and to avoid re-identifying tiny cohorts).

    Returns:
        A :class:`FairnessResult`. ``disparate_impact_ratio`` is min/max of the
        group mean scores; ``passes_four_fifths`` is that ratio ≥ 0.80.
    """
    stats: list[GroupStat] = []
    for group, scores in scores_by_group.items():
        clean = [min(max(float(s), 0.0), 1.0) for s in scores]
        if len(clean) < min_group_size or not clean:
            continue
        mean = sum(clean) / len(clean)
        high = sum(1 for s in clean if s >= high_risk_threshold) / len(clean)
        stats.append(
            GroupStat(
                group=group,
                count=len(clean),
                mean_score=round(mean, 4),
                high_risk_rate=round(high, 4),
            )
        )

    stats.sort(key=lambda g: g.mean_score, reverse=True)

    means = [g.mean_score for g in stats]
    if len(means) >= 2 and max(means) > 0:
        di = min(means) / max(means)
        spd = max(means) - min(means)
    else:
        di = 1.0
        spd = 0.0
    passes = di >= 0.80

    if len(stats) < 2:
        interp = "Not enough subgroups to assess parity."
    elif passes:
        interp = (
            f"No material disparity across {attribute}: disparate-impact ratio "
            f"{di:.2f} (≥ 0.80 four-fifths rule)."
        )
    else:
        interp = (
            f"Potential disparity across {attribute}: disparate-impact ratio "
            f"{di:.2f} is below the 0.80 four-fifths threshold — review before "
            f"release."
        )

    return FairnessResult(
        attribute=attribute,
        groups=stats,
        disparate_impact_ratio=round(di, 4),
        statistical_parity_difference=round(spd, 4),
        passes_four_fifths=passes,
        interpretation=interp,
    )


# ── Calibration (Brier + reliability) ─────────────────────────────────────────


@dataclass(frozen=True)
class ReliabilityBin:
    """One reliability-diagram bucket."""

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float


@dataclass(frozen=True)
class CalibrationResult:
    """Brier score and reliability table for predicted vs observed outcomes."""

    brier_score: float
    sample_size: int
    bins: list[ReliabilityBin] = field(default_factory=list)


def calibration_report(
    predicted: list[float], observed: list[int], bins: int = 10
) -> CalibrationResult:
    """Compute the Brier score and a reliability table.

    Args:
        predicted: Predicted probabilities in [0, 1].
        observed: Binary outcomes (0/1) aligned with ``predicted``.
        bins: Number of equal-width probability buckets.

    Returns:
        A :class:`CalibrationResult`.

    Raises:
        ValueError: If the inputs differ in length.
    """
    if len(predicted) != len(observed):
        raise ValueError("predicted and observed must be the same length")
    n = len(predicted)
    if n == 0:
        return CalibrationResult(brier_score=0.0, sample_size=0, bins=[])

    brier = sum((p - y) ** 2 for p, y in zip(predicted, observed, strict=False)) / n

    edges = [i / bins for i in range(bins + 1)]
    grouped: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for p, y in zip(predicted, observed, strict=False):
        p = min(max(float(p), 0.0), 1.0)
        for i in range(bins):
            if p < edges[i + 1] or i == bins - 1:
                grouped[i].append((p, int(y)))
                break

    out_bins: list[ReliabilityBin] = []
    for i in range(bins):
        pairs = grouped.get(i, [])
        if not pairs:
            continue
        mean_pred = sum(p for p, _ in pairs) / len(pairs)
        obs_rate = sum(y for _, y in pairs) / len(pairs)
        out_bins.append(
            ReliabilityBin(
                lower=round(edges[i], 4),
                upper=round(edges[i + 1], 4),
                count=len(pairs),
                mean_predicted=round(mean_pred, 4),
                observed_rate=round(obs_rate, 4),
            )
        )

    return CalibrationResult(brier_score=round(brier, 4), sample_size=n, bins=out_bins)
