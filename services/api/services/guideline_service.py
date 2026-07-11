"""Guideline-based screening recommendations (Tier 6 — ML Trust & Decision Support).

Turns a bare risk score into *actionable next steps* by mapping the patient's
risk, age, sex and recorded conditions onto established clinical screening
guidelines (NCCN hereditary cancer, USPSTF, ACC/AHA). A number alone ("62%
risk") is not clinically useful; "meets NCCN criteria for BRCA testing —
consider referral to genetics" is.

Same curated, deterministic, dependency-free approach as
:mod:`services.api.services.differential_service`: a small, transparent rule
base rather than an opaque recommender. Each rule states its source and
rationale so the advice is auditable. The thresholds below are representative
of published guidance but are **illustrative** — a production deployment would
version this catalogue against the current guideline releases.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# ── Recommendation model ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class GuidelineRecommendation:
    """One actionable screening/next-step recommendation."""

    guideline_id: str
    source: str
    title: str
    recommendation: str
    urgency: str  # "routine" | "soon" | "urgent"
    rationale: str


@dataclass(frozen=True)
class PatientContext:
    """The clinical facts a rule may condition on."""

    age: int | None
    sex: str | None  # "male" | "female" | None
    risk_score: float
    condition_codes: frozenset[str]  # ICD-10 prefixes present on the patient
    has_hereditary_condition: bool
    affected_first_degree_relatives: int


@dataclass(frozen=True)
class _Rule:
    """A guideline rule: a predicate plus the recommendation it emits."""

    predicate: Callable[[PatientContext], bool]
    recommendation: GuidelineRecommendation


def _has(ctx: PatientContext, *prefixes: str) -> bool:
    """True if the patient has any condition whose code starts with a prefix."""
    return any(code.startswith(p) for code in ctx.condition_codes for p in prefixes)


# ── Rule catalogue ────────────────────────────────────────────────────────────

_RULES: list[_Rule] = [
    # Hereditary breast/ovarian cancer — NCCN genetic testing criteria.
    _Rule(
        predicate=lambda c: (
            (c.sex == "female" or _has(c, "C50"))
            and (c.risk_score >= 0.5 or c.affected_first_degree_relatives >= 1)
            and _has(c, "C50", "C56", "Z80")
        ),
        recommendation=GuidelineRecommendation(
            guideline_id="nccn-hboc-testing",
            source="NCCN Genetic/Familial High-Risk Assessment: Breast, Ovarian",
            title="Consider BRCA1/2 germline testing",
            recommendation=(
                "Refer to genetic counselling and offer BRCA1/2 (and panel) "
                "testing; if positive, begin enhanced surveillance (annual "
                "breast MRI + mammography) and discuss risk-reducing options."
            ),
            urgency="soon",
            rationale=(
                "Personal/family history of breast or ovarian cancer with "
                "elevated hereditary risk meets NCCN testing criteria."
            ),
        ),
    ),
    # Lynch syndrome — colorectal / endometrial.
    _Rule(
        predicate=lambda c: (
            (c.risk_score >= 0.5 or c.affected_first_degree_relatives >= 1)
            and _has(c, "C18", "C19", "C20", "C54", "Z80")
        ),
        recommendation=GuidelineRecommendation(
            guideline_id="nccn-lynch-testing",
            source="NCCN Genetic/Familial High-Risk Assessment: Colorectal",
            title="Evaluate for Lynch syndrome",
            recommendation=(
                "Order tumour MMR/MSI testing and refer for germline Lynch "
                "panel; if confirmed, start colonoscopy every 1–2 years from "
                "age 20–25 (or 2–5 years before earliest family diagnosis)."
            ),
            urgency="soon",
            rationale=(
                "Colorectal/endometrial cancer with a hereditary risk signal "
                "meets NCCN criteria for Lynch-syndrome evaluation."
            ),
        ),
    ),
    # Average-risk colorectal screening — USPSTF age threshold.
    _Rule(
        predicate=lambda c: c.age is not None
        and 45 <= c.age <= 75
        and not _has(c, "C18", "C19", "C20"),
        recommendation=GuidelineRecommendation(
            guideline_id="uspstf-crc-45",
            source="USPSTF Colorectal Cancer Screening (Grade A/B)",
            title="Colorectal cancer screening due",
            recommendation=(
                "Offer colorectal cancer screening (colonoscopy every 10 years "
                "or annual FIT). Start earlier and screen more often if risk is "
                "elevated or a first-degree relative was affected."
            ),
            urgency="routine",
            rationale="Adults aged 45–75 warrant colorectal screening (USPSTF).",
        ),
    ),
    # Early colorectal screening for high hereditary risk under 45.
    _Rule(
        predicate=lambda c: c.age is not None
        and c.age < 45
        and (c.risk_score >= 0.6 or c.affected_first_degree_relatives >= 1)
        and _has(c, "C18", "C19", "C20", "Z80"),
        recommendation=GuidelineRecommendation(
            guideline_id="nccn-crc-early",
            source="NCCN Colorectal Cancer Screening (increased risk)",
            title="Begin colorectal screening before age 45",
            recommendation=(
                "Begin colonoscopy now given elevated hereditary risk — "
                "typically 10 years before the earliest family diagnosis or at "
                "age 40, whichever is sooner."
            ),
            urgency="soon",
            rationale=(
                "A first-degree relative with colorectal cancer or high "
                "hereditary risk warrants earlier-than-average screening."
            ),
        ),
    ),
    # Mammography — USPSTF.
    _Rule(
        predicate=lambda c: c.sex == "female" and c.age is not None and 40 <= c.age <= 74,
        recommendation=GuidelineRecommendation(
            guideline_id="uspstf-mammo-40",
            source="USPSTF Breast Cancer Screening (Grade B)",
            title="Biennial mammography",
            recommendation=(
                "Offer screening mammography every 2 years. Consider annual "
                "imaging with breast MRI if hereditary risk is high."
            ),
            urgency="routine",
            rationale="Women aged 40–74 warrant biennial mammography (USPSTF).",
        ),
    ),
    # Lipid / cardiovascular — familial hypercholesterolaemia signal.
    _Rule(
        predicate=lambda c: (c.risk_score >= 0.5 or c.affected_first_degree_relatives >= 1)
        and _has(c, "E78", "I21", "I25"),
        recommendation=GuidelineRecommendation(
            guideline_id="acc-fh-lipids",
            source="ACC/AHA & NLA Familial Hypercholesterolaemia guidance",
            title="Screen for familial hypercholesterolaemia",
            recommendation=(
                "Obtain a fasting lipid panel; if LDL-C is markedly elevated, "
                "evaluate for familial hypercholesterolaemia and cascade-screen "
                "first-degree relatives."
            ),
            urgency="soon",
            rationale=(
                "Premature cardiovascular disease or dyslipidaemia with a "
                "family signal suggests an inherited lipid disorder."
            ),
        ),
    ),
    # Type 2 diabetes screening — USPSTF.
    _Rule(
        predicate=lambda c: c.age is not None and 35 <= c.age <= 70 and not _has(c, "E11"),
        recommendation=GuidelineRecommendation(
            guideline_id="uspstf-t2dm-35",
            source="USPSTF Prediabetes and Type 2 Diabetes Screening (Grade B)",
            title="Screen for type 2 diabetes",
            recommendation="Screen with fasting glucose or HbA1c every 3 years.",
            urgency="routine",
            rationale="Adults aged 35–70 warrant periodic diabetes screening.",
        ),
    ),
]

# High overall risk with no more specific rule still deserves a referral.
_GENERIC_HIGH_RISK = GuidelineRecommendation(
    guideline_id="genetics-referral",
    source="Clinical genetics referral (general)",
    title="Refer to clinical genetics",
    recommendation=(
        "Overall hereditary risk is high without a specific screening rule "
        "matched — refer for formal genetic counselling and a detailed "
        "three-generation pedigree review."
    ),
    urgency="soon",
    rationale="High model risk with an incomplete guideline match.",
)

_URGENCY_RANK = {"urgent": 0, "soon": 1, "routine": 2}


def recommend(ctx: PatientContext) -> list[GuidelineRecommendation]:
    """Return guideline-based recommendations for a patient context.

    Args:
        ctx: The patient's clinical facts.

    Returns:
        Matching recommendations, most urgent first. When overall risk is high
        (≥ 0.75) but nothing specific matched, a generic genetics referral is
        appended so the clinician is never left with only a number.
    """
    matched = [r.recommendation for r in _RULES if _safe(r.predicate, ctx)]

    if not matched and ctx.risk_score >= 0.75:
        matched.append(_GENERIC_HIGH_RISK)

    matched.sort(key=lambda r: _URGENCY_RANK.get(r.urgency, 3))
    return matched


def _safe(predicate: Callable[[PatientContext], bool], ctx: PatientContext) -> bool:
    """Evaluate a rule predicate, treating any error as a non-match."""
    try:
        return bool(predicate(ctx))
    except Exception:  # pragma: no cover - defensive; rules are simple
        return False


def catalogue() -> list[GuidelineRecommendation]:
    """Return every recommendation in the rule base (for documentation/UI)."""
    return [r.recommendation for r in _RULES] + [_GENERIC_HIGH_RISK]
