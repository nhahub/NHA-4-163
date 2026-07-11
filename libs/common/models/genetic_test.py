"""Genetic test & variant ORM models (Tier 5 — Genetics & Genomics).

Captures the results of a genetic test (panel, WES/WGS, single-gene, or an
uploaded VCF) and the individual variants it reported, each annotated with a
pathogenicity classification and disease association from a curated
ClinVar/OMIM knowledge base (see
:mod:`services.api.services.variant_service`).

These feed two downstream consumers:
* the ML pipeline's ``is_hereditary`` signal, and
* the Mendelian calculator (a confirmed pathogenic variant establishes the
  proband as affected/carrier for a specific gene).

PHI note: genomic data is highly identifying.  Variant rows store coordinates,
gene, and classification — treat the whole record as PHI; all access is audited.

FHIR analogue: loosely maps to the ``MolecularSequence`` / genomics
``Observation`` profiles, but stored in a purpose-built shape here.
"""

from __future__ import annotations

import enum
import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, Enum, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from libs.common.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from libs.common.models.patient import Patient


class GeneticTestType(enum.StrEnum):
    """The kind of genetic test performed."""

    PANEL = "panel"
    WES = "wes"  # whole-exome sequencing
    WGS = "wgs"  # whole-genome sequencing
    SINGLE_GENE = "single_gene"
    KARYOTYPE = "karyotype"
    VCF_UPLOAD = "vcf_upload"


class GeneticTestStatus(enum.StrEnum):
    """Report status."""

    PRELIMINARY = "preliminary"
    FINAL = "final"
    AMENDED = "amended"
    ENTERED_IN_ERROR = "entered-in-error"


class ClinicalSignificance(enum.StrEnum):
    """ACMG/ClinVar 5-tier pathogenicity classification."""

    PATHOGENIC = "pathogenic"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    UNCERTAIN = "uncertain_significance"
    LIKELY_BENIGN = "likely_benign"
    BENIGN = "benign"


class Zygosity(enum.StrEnum):
    """Allelic state of a variant."""

    HETEROZYGOUS = "heterozygous"
    HOMOZYGOUS = "homozygous"
    HEMIZYGOUS = "hemizygous"
    UNKNOWN = "unknown"


class GeneticTest(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One genetic test/report for a patient."""

    __tablename__ = "genetic_test"

    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organization.id", ondelete="CASCADE"),
        index=True,
    )

    test_type: Mapped[GeneticTestType] = mapped_column(
        Enum(GeneticTestType, name="genetic_test_type"), nullable=False, index=True
    )
    status: Mapped[GeneticTestStatus] = mapped_column(
        Enum(GeneticTestStatus, name="genetic_test_status"),
        nullable=False,
        default=GeneticTestStatus.FINAL,
    )

    lab_name: Mapped[str | None] = mapped_column(String(255))
    method: Mapped[str | None] = mapped_column(String(255))
    performed_date: Mapped[date | None] = mapped_column(Date)
    source_filename: Mapped[str | None] = mapped_column(String(255))

    # Overall report interpretation (worst variant significance, summarised).
    overall_interpretation: Mapped[str | None] = mapped_column(String(500))
    variant_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pathogenic_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    variants: Mapped[list[Variant]] = relationship(
        back_populates="genetic_test", cascade="all, delete-orphan"
    )
    patient: Mapped[Patient] = relationship()


class Variant(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One annotated variant reported by a genetic test."""

    __tablename__ = "variant"

    genetic_test_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("genetic_test.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalised for efficient per-patient variant queries.
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Locus ─────────────────────────────────────────────────────────────────
    gene: Mapped[str | None] = mapped_column(String(50), index=True)
    chromosome: Mapped[str | None] = mapped_column(String(5))
    position: Mapped[int | None] = mapped_column(Integer)
    ref_allele: Mapped[str | None] = mapped_column(String(255))
    alt_allele: Mapped[str | None] = mapped_column(String(255))
    rs_id: Mapped[str | None] = mapped_column(String(30), index=True)
    hgvs: Mapped[str | None] = mapped_column(String(255))

    # ── Annotation ────────────────────────────────────────────────────────────
    zygosity: Mapped[Zygosity] = mapped_column(
        Enum(Zygosity, name="variant_zygosity"),
        nullable=False,
        default=Zygosity.UNKNOWN,
    )
    clinical_significance: Mapped[ClinicalSignificance] = mapped_column(
        Enum(ClinicalSignificance, name="variant_clinical_significance"),
        nullable=False,
        default=ClinicalSignificance.UNCERTAIN,
        index=True,
    )
    condition_code: Mapped[str | None] = mapped_column(String(50))
    condition_display: Mapped[str | None] = mapped_column(String(255))
    inheritance_mode: Mapped[str | None] = mapped_column(String(40))
    clinvar_id: Mapped[str | None] = mapped_column(String(50))

    genetic_test: Mapped[GeneticTest] = relationship(back_populates="variants")
