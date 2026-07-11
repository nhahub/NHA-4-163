"""Polygenic Risk Score (PRS) integration (Tier 5 — Genetics & Genomics).

Combines a published, curated PRS panel with the platform's XGBoost/GNN output
to produce a blended risk estimate for a common, complex disease.

Pipeline (all pure, dependency-free — uses only the stdlib ``math``):

1. A patient's risk-allele *dosages* (0/1/2 copies per SNP) are scored against a
   panel of per-allele effect sizes (log-odds ``beta``) → a raw PRS.
2. The raw PRS is standardised against the panel's population mean/SD → a
   z-score and percentile (normal CDF via ``math.erf``).
3. The z-score is turned into an odds ratio versus the population mean, then an
   absolute PRS-implied risk using the panel's baseline prevalence.
4. The PRS-implied risk is blended with the ML model's risk in log-odds space
   (a weighted logit average), so the two independent signals reinforce.

The panels are small and transparent (mirroring
:mod:`services.api.services.differential_service`), not a live PGS Catalog call.
Effect sizes are illustrative and calibrated so the demo behaves sensibly; they
are **not** clinically validated weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PRSPanel:
    """A curated polygenic risk panel for one disease."""

    key: str
    display: str
    condition_code: str
    baseline_prevalence: float
    mean: float
    sd: float
    # rsID → per-risk-allele effect size (natural-log odds ratio).
    weights: dict[str, float]


PRS_PANELS: dict[str, PRSPanel] = {
    "coronary_artery_disease": PRSPanel(
        key="coronary_artery_disease",
        display="Coronary artery disease",
        condition_code="I25",
        baseline_prevalence=0.06,
        mean=0.0,
        sd=0.60,
        weights={
            "rs10757278": 0.29,  # 9p21
            "rs1333049": 0.24,  # 9p21
            "rs17465637": 0.15,  # MIA3
            "rs6725887": 0.17,  # WDR12
            "rs9349379": 0.14,  # PHACTR1
        },
    ),
    "type_2_diabetes": PRSPanel(
        key="type_2_diabetes",
        display="Type 2 diabetes",
        condition_code="E11",
        baseline_prevalence=0.10,
        mean=0.0,
        sd=0.55,
        weights={
            "rs7903146": 0.34,  # TCF7L2
            "rs1801282": 0.14,  # PPARG
            "rs5219": 0.15,  # KCNJ11
            "rs13266634": 0.12,  # SLC30A8
            "rs4402960": 0.11,  # IGF2BP2
        },
    ),
    "breast_cancer": PRSPanel(
        key="breast_cancer",
        display="Breast cancer",
        condition_code="C50",
        baseline_prevalence=0.12,
        mean=0.0,
        sd=0.50,
        weights={
            "rs2981582": 0.26,  # FGFR2
            "rs3803662": 0.20,  # TOX3
            "rs889312": 0.13,  # MAP3K1
            "rs13281615": 0.10,  # 8q24
        },
    ),
    "alzheimer_disease": PRSPanel(
        key="alzheimer_disease",
        display="Late-onset Alzheimer disease",
        condition_code="G30",
        baseline_prevalence=0.10,
        mean=0.0,
        sd=0.70,
        weights={
            "rs429358": 1.10,  # APOE e4
            "rs7412": -0.40,  # APOE e2 (protective)
            "rs75932628": 0.80,  # TREM2
            "rs6656401": 0.15,  # CR1
        },
    ),
}


def _normal_cdf(z: float) -> float:
    """Standard-normal cumulative distribution via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _logit(p: float) -> float:
    """Log-odds of a probability, clamped away from 0/1 for numerical safety."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    """Logistic function."""
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class PRSResult:
    """The computed PRS and its blend with the ML model."""

    disease: str
    display: str
    raw_score: float
    z_score: float
    percentile: float
    odds_ratio: float
    prs_absolute_risk: float
    ml_risk: float | None
    blended_risk: float
    snps_used: int
    snps_available: int
    interpretation: str


def compute_prs(
    disease: str,
    dosages: dict[str, int],
    ml_risk: float | None = None,
    prs_weight: float = 0.4,
) -> PRSResult:
    """Compute a PRS from risk-allele dosages and blend it with the ML risk.

    Args:
        disease: Panel key (see :data:`PRS_PANELS`).
        dosages: Mapping of rsID → risk-allele copies (0, 1, or 2). rsIDs not in
            the panel are ignored; panel SNPs missing from the mapping are
            treated as dosage 0 (population reference).
        ml_risk: The ML model's calibrated risk in [0, 1], or ``None`` if no
            prediction is available (then the blend is the PRS risk alone).
        prs_weight: Weight of the PRS signal in the log-odds blend, in [0, 1].
            The ML signal gets ``1 - prs_weight``.

    Returns:
        A :class:`PRSResult`.

    Raises:
        ValueError: If ``disease`` is not a known panel.
    """
    panel = PRS_PANELS.get(disease)
    if panel is None:
        raise ValueError(f"Unknown PRS panel: {disease!r}")

    raw = 0.0
    used = 0
    for rsid, beta in panel.weights.items():
        dose = dosages.get(rsid, 0)
        if dose:
            used += 1
        raw += dose * beta

    z = (raw - panel.mean) / panel.sd if panel.sd else 0.0
    percentile = round(_normal_cdf(z) * 100.0, 1)
    odds_ratio = round(math.exp(raw - panel.mean), 3)

    base = panel.baseline_prevalence
    base_odds = base / (1 - base)
    ind_odds = base_odds * math.exp(raw - panel.mean)
    prs_risk = ind_odds / (1 + ind_odds)

    if ml_risk is None:
        blended = prs_risk
    else:
        w = min(max(prs_weight, 0.0), 1.0)
        blended = _sigmoid((1 - w) * _logit(ml_risk) + w * _logit(prs_risk))

    if percentile >= 90:
        band = "high genetic risk (top decile)"
    elif percentile >= 75:
        band = "above-average genetic risk"
    elif percentile <= 10:
        band = "low genetic risk (bottom decile)"
    else:
        band = "average genetic risk"

    return PRSResult(
        disease=panel.key,
        display=panel.display,
        raw_score=round(raw, 4),
        z_score=round(z, 3),
        percentile=percentile,
        odds_ratio=odds_ratio,
        prs_absolute_risk=round(prs_risk, 4),
        ml_risk=round(ml_risk, 4) if ml_risk is not None else None,
        blended_risk=round(blended, 4),
        snps_used=used,
        snps_available=len(panel.weights),
        interpretation=(
            f"{panel.display}: {band} " f"(PRS {percentile:.0f}th percentile, OR {odds_ratio:.2f})."
        ),
    )
