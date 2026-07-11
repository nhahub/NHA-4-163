"""Request/response schemas for genetic test ingestion (Tier 5)."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

_TEST_TYPE_PATTERN = r"^(panel|wes|wgs|single_gene|karyotype|vcf_upload)$"


class VariantInput(BaseModel):
    """A single structured variant in a genetic-test upload."""

    model_config = ConfigDict(str_strip_whitespace=True)

    gene: str | None = Field(default=None, max_length=50)
    rs_id: str | None = Field(default=None, max_length=30)
    chromosome: str | None = Field(default=None, max_length=5)
    position: int | None = Field(default=None, ge=0)
    ref_allele: str | None = Field(default=None, max_length=255)
    alt_allele: str | None = Field(default=None, max_length=255)
    hgvs: str | None = Field(default=None, max_length=255)
    zygosity: str = Field(
        default="unknown",
        pattern=r"^(heterozygous|homozygous|hemizygous|unknown)$",
    )

    @model_validator(mode="after")
    def _require_identifier(self) -> VariantInput:
        """Each variant must carry at least a gene or an rsID to be annotatable."""
        if not self.gene and not self.rs_id:
            raise ValueError("variant requires at least one of 'gene' or 'rs_id'")
        return self


class GeneticTestCreate(BaseModel):
    """Request body for POST /patients/{id}/genetic-tests.

    Provide *either* a list of structured ``variants`` *or* raw ``vcf_content``.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    test_type: str = Field(default="panel", pattern=_TEST_TYPE_PATTERN)
    lab_name: str | None = Field(default=None, max_length=255)
    method: str | None = Field(default=None, max_length=255)
    performed_date: date | None = None
    source_filename: str | None = Field(default=None, max_length=255)
    variants: list[VariantInput] | None = Field(default=None, max_length=1000)
    vcf_content: str | None = Field(default=None, max_length=2_000_000)

    @model_validator(mode="after")
    def _require_payload(self) -> GeneticTestCreate:
        """Require exactly one of ``variants`` or ``vcf_content``."""
        has_variants = bool(self.variants)
        has_vcf = bool(self.vcf_content and self.vcf_content.strip())
        if has_variants == has_vcf:
            raise ValueError("provide exactly one of 'variants' or 'vcf_content'")
        return self


class VariantResponse(BaseModel):
    """An annotated variant returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    gene: str | None = None
    rs_id: str | None = None
    chromosome: str | None = None
    position: int | None = None
    ref_allele: str | None = None
    alt_allele: str | None = None
    hgvs: str | None = None
    zygosity: str
    clinical_significance: str
    condition_code: str | None = None
    condition_display: str | None = None
    inheritance_mode: str | None = None
    clinvar_id: str | None = None


class GeneticTestResponse(BaseModel):
    """A genetic test/report with its annotated variants."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    organization_id: uuid.UUID | None = None
    test_type: str
    status: str
    lab_name: str | None = None
    method: str | None = None
    performed_date: date | None = None
    source_filename: str | None = None
    overall_interpretation: str | None = None
    variant_count: int
    pathogenic_count: int
    created_at: datetime | None = None
    variants: list[VariantResponse] = []
