"""Unit tests for VCF parsing and variant annotation (Tier 5)."""

from __future__ import annotations

from libs.common.models.genetic_test import ClinicalSignificance
from services.api.services.variant_service import (
    annotate_variant,
    is_pathogenic,
    parse_vcf,
    summarise_significance,
)


class TestAnnotation:
    def test_known_rsid_is_pathogenic(self) -> None:
        ann = annotate_variant(rs_id="rs334")
        assert ann.gene == "HBB"
        assert ann.clinical_significance == ClinicalSignificance.PATHOGENIC
        assert ann.inheritance_mode == "autosomal_recessive"

    def test_rsid_case_insensitive(self) -> None:
        assert annotate_variant(rs_id="RS334").gene == "HBB"

    def test_high_penetrance_gene_default(self) -> None:
        ann = annotate_variant(gene="BRCA1")
        assert ann.clinical_significance == ClinicalSignificance.LIKELY_PATHOGENIC
        assert ann.inheritance_mode == "autosomal_dominant"

    def test_rsid_takes_precedence_over_gene(self) -> None:
        ann = annotate_variant(gene="BRCA1", rs_id="rs334")
        assert ann.gene == "HBB"  # rsID match wins

    def test_unknown_variant_is_uncertain(self) -> None:
        ann = annotate_variant(gene="ZZZ9", rs_id="rs00000000")
        assert ann.clinical_significance == ClinicalSignificance.UNCERTAIN
        assert ann.condition_code is None


class TestVCFParsing:
    def test_parses_basic_record_with_gene_and_genotype(self) -> None:
        vcf = (
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
            "11\t5248232\trs334\tT\tA\t.\tPASS\tGENE=HBB\tGT\t0/1\n"
        )
        variants = parse_vcf(vcf)
        assert len(variants) == 1
        v = variants[0]
        assert v.rs_id == "rs334"
        assert v.gene == "HBB"
        assert v.chromosome == "11"
        assert v.position == 5248232
        assert v.zygosity == "heterozygous"

    def test_homozygous_genotype(self) -> None:
        vcf = (
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
            "6\t26093141\trs1800562\tG\tA\t.\tPASS\tGENE=HFE\tGT\t1/1\n"
        )
        assert parse_vcf(vcf)[0].zygosity == "homozygous"

    def test_skips_headers_and_blanks(self) -> None:
        vcf = "##header\n\n#CHROM\tPOS\tID\tREF\tALT\n1\t100\t.\tA\tG\n"
        variants = parse_vcf(vcf)
        assert len(variants) == 1
        assert variants[0].rs_id is None

    def test_rs_from_info_when_id_missing(self) -> None:
        vcf = (
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "1\t100\t.\tA\tG\t.\tPASS\tRS=429358;GENE=APOE\n"
        )
        v = parse_vcf(vcf)[0]
        assert v.rs_id == "rs429358"
        assert v.gene == "APOE"

    def test_max_variants_cap(self) -> None:
        lines = ["#CHROM\tPOS\tID\tREF\tALT"]
        lines += [f"1\t{i}\t.\tA\tG" for i in range(10)]
        assert len(parse_vcf("\n".join(lines), max_variants=3)) == 3

    def test_end_to_end_parse_then_annotate(self) -> None:
        vcf = (
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
            "11\t5248232\trs334\tT\tA\t.\tPASS\t.\tGT\t1/1\n"
        )
        v = parse_vcf(vcf)[0]
        ann = annotate_variant(gene=v.gene, rs_id=v.rs_id)
        assert is_pathogenic(ann.clinical_significance)


class TestSignificanceSummary:
    def test_returns_most_significant(self) -> None:
        result = summarise_significance(
            [
                ClinicalSignificance.BENIGN,
                ClinicalSignificance.PATHOGENIC,
                ClinicalSignificance.UNCERTAIN,
            ]
        )
        assert result == ClinicalSignificance.PATHOGENIC

    def test_empty_defaults_uncertain(self) -> None:
        assert summarise_significance([]) == ClinicalSignificance.UNCERTAIN

    def test_is_pathogenic_predicate(self) -> None:
        assert is_pathogenic(ClinicalSignificance.LIKELY_PATHOGENIC)
        assert not is_pathogenic(ClinicalSignificance.BENIGN)
