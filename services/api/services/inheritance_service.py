"""Mendelian inheritance risk calculator (Tier 5 — Genetics & Genomics).

Given a pedigree and an affected proband, this module computes carrier and
affected probabilities for each relative using single-locus Mendelian
transmission rules, keyed on the inheritance mode (autosomal dominant/recessive,
X-linked recessive/dominant, mitochondrial) plus a disease penetrance and a
population carrier frequency for the "other" allele.

Design goals
------------
* **Deterministic and fully explainable** — every probability comes with a
  human-readable ``basis`` string.  This complements (rather than competes with)
  the ML risk model: a clinician can see exactly *why* a number was produced.
* **Dependency-free** — pure Python, unit-testable without a database, in the
  same spirit as :mod:`services.api.services.differential_service`.
* **Reuses existing data** — ``degree_of_relatedness`` already stored on
  :class:`~libs.common.models.family_member_history.FamilyMemberHistory` is used
  as the transmission probability for dominant alleles; relationship codes drive
  the exact rules for recessive and sex-linked modes.

Modelling assumptions (documented so the numbers are defensible)
----------------------------------------------------------------
* Single affected proband, single disease locus, no consanguinity loops.
* Dominant: the proband is an obligate heterozygote ``Aa``.
* Recessive: the proband is an affected homozygote ``aa``; matings into the
  family draw the second allele from the general population at
  ``carrier_frequency``.
* Penetrance is the probability an at-risk genotype actually manifests disease.

These are textbook genetic-counselling approximations, not a substitute for a
formal Bayesian pedigree analysis (which would require the full genotyped
graph).  The canonical first-degree results (sibling of an ``aa`` proband: 1/4
affected, 1/2 carrier; child of an affected dominant case: 1/2) are exact.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Inheritance model catalogue ──────────────────────────────────────────────


@dataclass(frozen=True)
class InheritanceModel:
    """A supported inheritance mode with clinically reasonable defaults."""

    key: str
    display: str
    default_penetrance: float
    default_carrier_frequency: float
    sex_linked: bool
    description: str


INHERITANCE_MODELS: dict[str, InheritanceModel] = {
    "autosomal_dominant": InheritanceModel(
        key="autosomal_dominant",
        display="Autosomal dominant",
        default_penetrance=0.80,
        default_carrier_frequency=0.001,
        sex_linked=False,
        description=(
            "One pathogenic allele is sufficient. Each child of an affected "
            "heterozygote has a 50% chance of inheriting it (e.g. BRCA1/2, "
            "Huntington disease, familial hypercholesterolaemia)."
        ),
    ),
    "autosomal_recessive": InheritanceModel(
        key="autosomal_recessive",
        display="Autosomal recessive",
        default_penetrance=1.0,
        default_carrier_frequency=0.02,
        sex_linked=False,
        description=(
            "Two pathogenic alleles are required to be affected. Parents of an "
            "affected child are obligate carriers; siblings have a 25% risk "
            "(e.g. cystic fibrosis, sickle-cell disease, haemochromatosis)."
        ),
    ),
    "x_linked_recessive": InheritanceModel(
        key="x_linked_recessive",
        display="X-linked recessive",
        default_penetrance=1.0,
        default_carrier_frequency=0.001,
        sex_linked=True,
        description=(
            "Gene on the X chromosome; males (hemizygous) are affected with a "
            "single allele, females are usually unaffected carriers (e.g. "
            "haemophilia A/B, Duchenne muscular dystrophy)."
        ),
    ),
    "x_linked_dominant": InheritanceModel(
        key="x_linked_dominant",
        display="X-linked dominant",
        default_penetrance=0.9,
        default_carrier_frequency=0.0005,
        sex_linked=True,
        description=(
            "One X-linked allele causes disease in both sexes. An affected "
            "father transmits to all daughters and no sons (e.g. X-linked "
            "hypophosphataemia)."
        ),
    ),
    "mitochondrial": InheritanceModel(
        key="mitochondrial",
        display="Mitochondrial (maternal)",
        default_penetrance=0.6,
        default_carrier_frequency=0.0,
        sex_linked=True,
        description=(
            "Transmitted exclusively through the maternal line; an affected "
            "mother passes the variant to all children, an affected father to "
            "none (e.g. MELAS, Leber hereditary optic neuropathy)."
        ),
    ),
}


# ── Relationship categorisation ──────────────────────────────────────────────

# HL7 v3 FamilyMember codes → coarse genetic category.
_PARENT_CODES = {"MTH", "FTH", "NMTH", "NFTH", "PRN", "NPRN", "MOM", "DAD", "PARENT"}
_CHILD_CODES = {"SON", "DAU", "CHILD", "NCHILD", "DAUC", "SONC", "STPCHLD"}
_SIBLING_CODES = {"SIB", "BRO", "SIS", "NBRO", "NSIS", "HSIB", "HBRO", "HSIS", "SIBLING"}
_GRANDPARENT_CODES = {"GRPRN", "GRMTH", "GRFTH", "MGRMTH", "MGRFTH", "PGRMTH", "PGRFTH"}
_GRANDCHILD_CODES = {"GRNDCHILD", "GRNDSON", "GRNDDAU"}
_AUNT_UNCLE_CODES = {"AUNT", "UNCLE", "MAUNT", "PAUNT", "MUNCLE", "PUNCLE"}
_NIBLING_CODES = {"NIECE", "NEPHEW", "NIENEPH"}
_COUSIN_CODES = {"COUSN", "COUSIN"}
_SPOUSE_CODES = {"SPS", "HUSB", "WIFE", "DOMPART", "SIGOTHR"}

_FEMALE_CODES = {
    "MTH",
    "NMTH",
    "MOM",
    "SIS",
    "NSIS",
    "HSIS",
    "DAU",
    "DAUC",
    "AUNT",
    "MAUNT",
    "PAUNT",
    "NIECE",
    "GRMTH",
    "MGRMTH",
    "PGRMTH",
    "GRNDDAU",
    "WIFE",
}
_MALE_CODES = {
    "FTH",
    "NFTH",
    "DAD",
    "BRO",
    "NBRO",
    "HBRO",
    "SON",
    "SONC",
    "UNCLE",
    "MUNCLE",
    "PUNCLE",
    "NEPHEW",
    "GRFTH",
    "MGRFTH",
    "PGRFTH",
    "GRNDSON",
    "HUSB",
}


def categorise_relationship(relationship_code: str) -> str:
    """Map an HL7 v3 family-member code to a genetic category.

    Args:
        relationship_code: HL7 v3 FamilyMember code (case-insensitive).

    Returns:
        One of ``parent``, ``child``, ``sibling``, ``grandparent``,
        ``grandchild``, ``aunt_uncle``, ``nibling``, ``cousin``, ``spouse`` or
        ``unknown``.
    """
    code = (relationship_code or "").strip().upper()
    if code in _PARENT_CODES:
        return "parent"
    if code in _CHILD_CODES:
        return "child"
    if code in _SIBLING_CODES:
        return "sibling"
    if code in _GRANDPARENT_CODES:
        return "grandparent"
    if code in _GRANDCHILD_CODES:
        return "grandchild"
    if code in _AUNT_UNCLE_CODES:
        return "aunt_uncle"
    if code in _NIBLING_CODES:
        return "nibling"
    if code in _COUSIN_CODES:
        return "cousin"
    if code in _SPOUSE_CODES:
        return "spouse"
    return "unknown"


def infer_sex(relationship_code: str, relative_sex: str | None) -> str | None:
    """Return ``male``/``female``/``None`` from an explicit sex or the code."""
    if relative_sex:
        s = relative_sex.strip().lower()
        if s in ("male", "m"):
            return "male"
        if s in ("female", "f"):
            return "female"
    code = (relationship_code or "").strip().upper()
    if code in _FEMALE_CODES:
        return "female"
    if code in _MALE_CODES:
        return "male"
    return None


# ── Result type ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RelativeRisk:
    """Computed Mendelian risk for one relative of the proband."""

    carrier_probability: float
    affected_probability: float
    basis: str


def _clamp(value: float) -> float:
    """Clamp a probability into [0, 1] and round for stable output."""
    return round(max(0.0, min(1.0, value)), 4)


# ── Per-mode risk rules ──────────────────────────────────────────────────────


def _dominant_risk(
    category: str, degree: float, penetrance: float, carrier_freq: float
) -> RelativeRisk:
    """Autosomal-dominant transmission from a heterozygous affected proband."""
    if category == "spouse" or degree <= 0.0:
        p = carrier_freq
        return RelativeRisk(
            _clamp(p),
            _clamp(p * penetrance),
            "Not a blood relative — background population carrier risk only.",
        )
    # For a dominant allele the probability a relative carries it equals the
    # coefficient of relationship (0.5 for first-degree, 0.25 second, ...).
    carrier = degree
    return RelativeRisk(
        _clamp(carrier),
        _clamp(carrier * penetrance),
        (
            f"Dominant allele shared with probability {carrier:.3f} "
            f"(coefficient of relationship); expressed at {penetrance:.0%} "
            f"penetrance."
        ),
    )


def _recessive_risk(
    category: str, degree: float, penetrance: float, carrier_freq: float
) -> RelativeRisk:
    """Autosomal-recessive risk relative to an affected (aa) proband."""
    if category == "parent":
        return RelativeRisk(
            _clamp(1.0),
            _clamp(carrier_freq * penetrance),
            "Parent of an affected child is an obligate heterozygous carrier.",
        )
    if category == "sibling":
        # Both parents Aa → offspring 1/4 aa, 1/2 Aa, 1/4 AA.
        return RelativeRisk(
            _clamp(0.5),
            _clamp(0.25 * penetrance),
            "Full sibling of an affected child: 25% affected, 50% carrier "
            "(both parents obligate carriers).",
        )
    if category == "child":
        # Child inherits one 'a' for certain; second allele from partner.
        return RelativeRisk(
            _clamp(1.0 - carrier_freq),
            _clamp(carrier_freq * penetrance),
            "Child of an affected individual is an obligate carrier; affected "
            f"only if the other parent also transmits (≈{carrier_freq:.1%}).",
        )
    if category in ("grandparent", "aunt_uncle", "grandchild", "nibling"):
        # Second-degree: carrier risk roughly halves relative to obligate.
        return RelativeRisk(
            _clamp(0.5),
            _clamp(carrier_freq * penetrance),
            "Second-degree relative: ~50% carrier probability by descent from "
            "an obligate carrier.",
        )
    if category == "cousin":
        return RelativeRisk(
            _clamp(0.25),
            _clamp(carrier_freq * penetrance),
            "Third-degree relative (first cousin): ~25% carrier probability.",
        )
    if category == "spouse" or degree <= 0.0:
        return RelativeRisk(
            _clamp(2 * carrier_freq),
            _clamp(carrier_freq * carrier_freq * penetrance),
            "Not a blood relative — background population carrier risk only.",
        )
    # Fallback keyed on degree of relatedness.
    return RelativeRisk(
        _clamp(degree),
        _clamp(carrier_freq * penetrance),
        f"Carrier probability approximated from coefficient of relationship " f"({degree:.3f}).",
    )


def _x_linked_recessive_risk(
    category: str, sex: str | None, degree: float, penetrance: float
) -> RelativeRisk:
    """X-linked recessive risk (assumes the affected proband is male)."""
    if category == "parent":
        if sex == "female":
            return RelativeRisk(
                _clamp(1.0),
                _clamp(0.0),
                "Mother of an affected male is an obligate carrier.",
            )
        return RelativeRisk(
            _clamp(0.0),
            _clamp(0.0),
            "Father does not transmit an X allele to an affected son.",
        )
    if category == "sibling":
        if sex == "male":
            return RelativeRisk(
                _clamp(0.0),
                _clamp(0.5 * penetrance),
                "Brother has a 50% chance of being affected (carrier mother).",
            )
        if sex == "female":
            return RelativeRisk(
                _clamp(0.5),
                _clamp(0.0),
                "Sister has a 50% chance of being a carrier (carrier mother).",
            )
        return RelativeRisk(
            _clamp(0.25),
            _clamp(0.25 * penetrance),
            "Sibling (sex unknown): averaged carrier/affected risk.",
        )
    if category == "child":
        if sex == "female":
            return RelativeRisk(
                _clamp(1.0),
                _clamp(0.0),
                "Daughter of an affected male is an obligate carrier.",
            )
        if sex == "male":
            return RelativeRisk(
                _clamp(0.0),
                _clamp(0.0),
                "Son of an affected male inherits the Y — not at risk.",
            )
        return RelativeRisk(
            _clamp(0.5),
            _clamp(0.0),
            "Child (sex unknown): daughters obligate carriers, sons unaffected.",
        )
    if category in ("grandchild", "nibling", "aunt_uncle", "grandparent"):
        return RelativeRisk(
            _clamp(0.25),
            _clamp(0.125 * penetrance),
            "Second-degree X-linked risk, halved along the maternal line.",
        )
    return RelativeRisk(
        _clamp(0.0),
        _clamp(0.0),
        "No established X-linked transmission path to this relative.",
    )


def _x_linked_dominant_risk(
    category: str, sex: str | None, degree: float, penetrance: float
) -> RelativeRisk:
    """X-linked dominant risk (transmission depends on proband and relative sex)."""
    if category == "child":
        if sex == "female":
            return RelativeRisk(
                _clamp(1.0),
                _clamp(1.0 * penetrance),
                "Daughter of an affected male inherits his X — affected.",
            )
        if sex == "male":
            return RelativeRisk(
                _clamp(0.0),
                _clamp(0.0),
                "Son of an affected male inherits the Y — not affected.",
            )
        return RelativeRisk(
            _clamp(0.5),
            _clamp(0.5 * penetrance),
            "Child (parent/sex unknown): averaged X-linked dominant risk.",
        )
    if category in ("parent", "sibling"):
        return RelativeRisk(
            _clamp(0.5),
            _clamp(0.5 * penetrance),
            "First-degree relative: ~50% X-linked dominant transmission.",
        )
    if degree > 0:
        return RelativeRisk(
            _clamp(degree),
            _clamp(degree * penetrance),
            f"X-linked dominant risk approximated from coefficient " f"({degree:.3f}).",
        )
    return RelativeRisk(_clamp(0.0), _clamp(0.0), "No transmission path.")


def _mitochondrial_risk(category: str, sex: str | None, penetrance: float) -> RelativeRisk:
    """Mitochondrial (maternal) transmission risk."""
    if category == "child":
        # Depends on the proband's sex, which we do not have here; assume the
        # maternal line is intact when the proband is female is handled by the
        # caller. Report the maximal maternal-line risk with an explicit basis.
        return RelativeRisk(
            _clamp(1.0),
            _clamp(1.0 * penetrance),
            "Maternally transmitted: all children of an affected mother inherit "
            "the variant (0% from an affected father).",
        )
    if category == "sibling":
        return RelativeRisk(
            _clamp(1.0),
            _clamp(1.0 * penetrance),
            "Shares the maternal lineage: same mitochondrial variant expected.",
        )
    if category == "parent" and sex == "female":
        return RelativeRisk(
            _clamp(1.0),
            _clamp(1.0 * penetrance),
            "Mother is the maternal source of the mitochondrial variant.",
        )
    return RelativeRisk(
        _clamp(0.0),
        _clamp(0.0),
        "Outside the maternal lineage — not at mitochondrial risk.",
    )


def compute_relative_risk(
    mode: str,
    relationship_code: str,
    degree_of_relatedness: float | None,
    relative_sex: str | None = None,
    penetrance: float | None = None,
    carrier_frequency: float | None = None,
) -> RelativeRisk:
    """Compute carrier/affected probabilities for one relative of the proband.

    Args:
        mode: Inheritance mode key (see :data:`INHERITANCE_MODELS`).
        relationship_code: HL7 v3 family-member code of the relative.
        degree_of_relatedness: Wright coefficient (0.5 first-degree, ...). Used
            for dominant transmission and as a fallback; when ``None`` it is
            inferred from the relationship category.
        relative_sex: ``male``/``female`` if known (needed for sex-linked modes).
        penetrance: Override disease penetrance; defaults to the model default.
        carrier_frequency: Override population carrier frequency for the other
            allele; defaults to the model default.

    Returns:
        A :class:`RelativeRisk`.

    Raises:
        ValueError: If ``mode`` is not a supported inheritance model.
    """
    model = INHERITANCE_MODELS.get(mode)
    if model is None:
        raise ValueError(f"Unsupported inheritance mode: {mode!r}")

    pen = model.default_penetrance if penetrance is None else penetrance
    cf = model.default_carrier_frequency if carrier_frequency is None else carrier_frequency
    category = categorise_relationship(relationship_code)
    sex = infer_sex(relationship_code, relative_sex)

    # Derive a default coefficient from the category when not supplied.
    if degree_of_relatedness is None:
        degree_of_relatedness = _DEFAULT_DEGREE.get(category, 0.0)
    degree = max(0.0, min(1.0, float(degree_of_relatedness)))

    if mode == "autosomal_dominant":
        return _dominant_risk(category, degree, pen, cf)
    if mode == "autosomal_recessive":
        return _recessive_risk(category, degree, pen, cf)
    if mode == "x_linked_recessive":
        return _x_linked_recessive_risk(category, sex, degree, pen)
    if mode == "x_linked_dominant":
        return _x_linked_dominant_risk(category, sex, degree, pen)
    if mode == "mitochondrial":
        return _mitochondrial_risk(category, sex, pen)
    raise ValueError(f"Unsupported inheritance mode: {mode!r}")  # pragma: no cover


_DEFAULT_DEGREE: dict[str, float] = {
    "parent": 0.5,
    "child": 0.5,
    "sibling": 0.5,
    "grandparent": 0.25,
    "grandchild": 0.25,
    "aunt_uncle": 0.25,
    "nibling": 0.25,
    "cousin": 0.125,
    "spouse": 0.0,
    "unknown": 0.0,
}
