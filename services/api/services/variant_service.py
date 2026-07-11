"""Variant parsing and annotation (Tier 5 — Genetics & Genomics).

Two responsibilities, both dependency-free and unit-testable:

1. :func:`parse_vcf` — parse the text of a minimal VCF 4.x file into raw variant
   dicts (CHROM/POS/REF/ALT/ID plus a best-effort GENE from the INFO column).
2. :func:`annotate_variant` — classify a variant against a curated ClinVar/OMIM
   knowledge base of well-established pathogenic variants, returning a
   pathogenicity classification, disease association, and inheritance mode.

The curated KB mirrors the approach used elsewhere in the system
(:mod:`services.api.services.differential_service`): a small, transparent,
offline table rather than a live ClinVar API call.  Annotation matches on rsID
first, then on ``gene`` (a known pathogenic gene without a specific rsID is
treated as ``uncertain_significance`` unless the gene is a well-known
high-penetrance gene, in which case a gene-level default applies).
"""

from __future__ import annotations

from dataclasses import dataclass

from libs.common.models.genetic_test import ClinicalSignificance

# ── Curated ClinVar/OMIM knowledge base ──────────────────────────────────────


@dataclass(frozen=True)
class VariantAnnotation:
    """The clinical interpretation attached to a variant."""

    gene: str | None
    clinical_significance: ClinicalSignificance
    condition_code: str | None
    condition_display: str | None
    inheritance_mode: str | None
    clinvar_id: str | None


# rsID → annotation for specific, well-characterised variants.
_RSID_KB: dict[str, VariantAnnotation] = {
    # HFE p.C282Y — hereditary haemochromatosis (autosomal recessive).
    "rs1800562": VariantAnnotation(
        "HFE",
        ClinicalSignificance.PATHOGENIC,
        "E83.110",
        "Hereditary haemochromatosis",
        "autosomal_recessive",
        "VCV000009749",
    ),
    # HBB p.E6V — sickle-cell disease (autosomal recessive).
    "rs334": VariantAnnotation(
        "HBB",
        ClinicalSignificance.PATHOGENIC,
        "D57.1",
        "Sickle-cell disease",
        "autosomal_recessive",
        "VCV000015333",
    ),
    # CFTR F508del — cystic fibrosis (autosomal recessive).
    "rs113993960": VariantAnnotation(
        "CFTR",
        ClinicalSignificance.PATHOGENIC,
        "E84.9",
        "Cystic fibrosis",
        "autosomal_recessive",
        "VCV000007105",
    ),
    # APOE e4 — Alzheimer risk allele (risk factor, not Mendelian).
    "rs429358": VariantAnnotation(
        "APOE",
        ClinicalSignificance.UNCERTAIN,
        "G30.9",
        "Late-onset Alzheimer disease (risk allele)",
        None,
        "VCV000017848",
    ),
    # LDLR — familial hypercholesterolaemia (autosomal dominant).
    "rs121908025": VariantAnnotation(
        "LDLR",
        ClinicalSignificance.PATHOGENIC,
        "E78.01",
        "Familial hypercholesterolaemia",
        "autosomal_dominant",
        "VCV000003714",
    ),
    # F8 intron-22 inversion proxy — haemophilia A (X-linked recessive).
    "rs28937869": VariantAnnotation(
        "F8",
        ClinicalSignificance.PATHOGENIC,
        "D66",
        "Haemophilia A",
        "x_linked_recessive",
        "VCV000010012",
    ),
}

# Gene → default annotation for high-penetrance genes where any reported
# (non-benign) variant is clinically actionable pending review.
_GENE_KB: dict[str, VariantAnnotation] = {
    "BRCA1": VariantAnnotation(
        "BRCA1",
        ClinicalSignificance.LIKELY_PATHOGENIC,
        "C50.9",
        "Hereditary breast and ovarian cancer",
        "autosomal_dominant",
        None,
    ),
    "BRCA2": VariantAnnotation(
        "BRCA2",
        ClinicalSignificance.LIKELY_PATHOGENIC,
        "C50.9",
        "Hereditary breast and ovarian cancer",
        "autosomal_dominant",
        None,
    ),
    "MLH1": VariantAnnotation(
        "MLH1",
        ClinicalSignificance.LIKELY_PATHOGENIC,
        "C18.9",
        "Lynch syndrome",
        "autosomal_dominant",
        None,
    ),
    "MSH2": VariantAnnotation(
        "MSH2",
        ClinicalSignificance.LIKELY_PATHOGENIC,
        "C18.9",
        "Lynch syndrome",
        "autosomal_dominant",
        None,
    ),
    "MSH6": VariantAnnotation(
        "MSH6",
        ClinicalSignificance.LIKELY_PATHOGENIC,
        "C18.9",
        "Lynch syndrome",
        "autosomal_dominant",
        None,
    ),
    "APC": VariantAnnotation(
        "APC",
        ClinicalSignificance.LIKELY_PATHOGENIC,
        "D12.6",
        "Familial adenomatous polyposis",
        "autosomal_dominant",
        None,
    ),
    "HTT": VariantAnnotation(
        "HTT",
        ClinicalSignificance.PATHOGENIC,
        "G10",
        "Huntington disease",
        "autosomal_dominant",
        None,
    ),
    "DMD": VariantAnnotation(
        "DMD",
        ClinicalSignificance.LIKELY_PATHOGENIC,
        "G71.0",
        "Duchenne muscular dystrophy",
        "x_linked_recessive",
        None,
    ),
}

_UNKNOWN = VariantAnnotation(None, ClinicalSignificance.UNCERTAIN, None, None, None, None)


def annotate_variant(gene: str | None = None, rs_id: str | None = None) -> VariantAnnotation:
    """Classify a variant against the curated ClinVar/OMIM knowledge base.

    Matching precedence: specific rsID → gene-level default → uncertain.

    Args:
        gene: HGNC gene symbol (case-insensitive), if known.
        rs_id: dbSNP rsID (e.g. ``rs334``), if known.

    Returns:
        A :class:`VariantAnnotation`; an unknown variant is classified as
        ``uncertain_significance`` with no disease association.
    """
    if rs_id:
        hit = _RSID_KB.get(rs_id.strip().lower())
        if hit is not None:
            return hit
    if gene:
        hit = _GENE_KB.get(gene.strip().upper())
        if hit is not None:
            return hit
    return _UNKNOWN


@dataclass(frozen=True)
class ParsedVariant:
    """A raw variant parsed from a VCF record (pre-annotation)."""

    chromosome: str | None
    position: int | None
    rs_id: str | None
    ref_allele: str | None
    alt_allele: str | None
    gene: str | None
    zygosity: str


def _extract_info(info: str) -> dict[str, str]:
    """Parse a VCF INFO column (``KEY=VALUE;FLAG;...``) into a dict."""
    out: dict[str, str] = {}
    for field in info.split(";"):
        if not field:
            continue
        if "=" in field:
            key, _, value = field.partition("=")
            out[key.strip().upper()] = value.strip()
        else:
            out[field.strip().upper()] = ""
    return out


def _zygosity_from_genotype(gt: str) -> str:
    """Map a VCF GT (e.g. ``0/1``, ``1|1``) to a zygosity label."""
    alleles = gt.replace("|", "/").split("/")
    alleles = [a for a in alleles if a not in ("", ".")]
    if not alleles:
        return "unknown"
    non_ref = [a for a in alleles if a != "0"]
    if not non_ref:
        return "unknown"
    if len(alleles) == 1:
        return "hemizygous"
    if all(a == alleles[0] for a in alleles):
        return "homozygous"
    return "heterozygous"


def parse_vcf(text: str, max_variants: int = 1000) -> list[ParsedVariant]:
    """Parse the text of a minimal VCF 4.x file into raw variant records.

    Only the first sample's genotype (if present) is used to infer zygosity.
    ``GENE`` is read from the INFO column when present.

    Args:
        text: Full VCF file contents.
        max_variants: Safety cap on the number of records parsed.

    Returns:
        A list of :class:`ParsedVariant` (never annotated — call
        :func:`annotate_variant` separately).
    """
    variants: list[ParsedVariant] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            # Tolerate whitespace-separated files.
            cols = line.split()
        if len(cols) < 5:
            continue
        chrom, pos, vid, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]
        info = _extract_info(cols[7]) if len(cols) > 7 else {}

        rs_id = vid if vid.lower().startswith("rs") else info.get("RS")
        if rs_id and not rs_id.lower().startswith("rs"):
            rs_id = f"rs{rs_id}"

        zygosity = "unknown"
        if len(cols) >= 10 and "GT" in cols[8].split(":"):
            gt_index = cols[8].split(":").index("GT")
            sample_fields = cols[9].split(":")
            if gt_index < len(sample_fields):
                zygosity = _zygosity_from_genotype(sample_fields[gt_index])

        try:
            position = int(pos)
        except ValueError:
            position = None

        variants.append(
            ParsedVariant(
                chromosome=chrom or None,
                position=position,
                rs_id=rs_id or None,
                ref_allele=ref or None,
                alt_allele=alt or None,
                gene=info.get("GENE") or None,
                zygosity=zygosity,
            )
        )
        if len(variants) >= max_variants:
            break
    return variants


_SIGNIFICANCE_RANK = {
    ClinicalSignificance.PATHOGENIC: 5,
    ClinicalSignificance.LIKELY_PATHOGENIC: 4,
    ClinicalSignificance.UNCERTAIN: 3,
    ClinicalSignificance.LIKELY_BENIGN: 2,
    ClinicalSignificance.BENIGN: 1,
}


def summarise_significance(
    significances: list[ClinicalSignificance],
) -> ClinicalSignificance:
    """Return the most clinically significant classification in a set."""
    if not significances:
        return ClinicalSignificance.UNCERTAIN
    return max(significances, key=lambda s: _SIGNIFICANCE_RANK[s])


def is_pathogenic(significance: ClinicalSignificance) -> bool:
    """True for pathogenic / likely-pathogenic classifications."""
    return significance in (
        ClinicalSignificance.PATHOGENIC,
        ClinicalSignificance.LIKELY_PATHOGENIC,
    )
