"""Genetic test ingestion endpoints (Tier 5 — Genetics & Genomics).

POST /patients/{id}/genetic-tests — upload a VCF or structured variant report
GET  /patients/{id}/genetic-tests — list annotated variants + pathogenicity

Uploaded variants are annotated against the curated ClinVar/OMIM knowledge base
(:mod:`services.api.services.variant_service`).  A confirmed pathogenic variant
sets ``is_hereditary`` on any matching active condition, feeding the ML pipeline
and the Mendelian calculator.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from libs.common.models.condition import Condition
from libs.common.models.genetic_test import (
    GeneticTest,
    GeneticTestStatus,
    GeneticTestType,
    Variant,
    Zygosity,
)
from libs.common.models.patient import Patient
from services.api.db import DbSession
from services.api.schemas.genetics_schemas import (
    GeneticTestCreate,
    GeneticTestResponse,
)
from services.api.services.variant_service import (
    annotate_variant,
    is_pathogenic,
    parse_vcf,
    summarise_significance,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["genetics"])


def _build_variant(
    patient_id: uuid.UUID,
    test_id: uuid.UUID,
    *,
    gene: str | None,
    rs_id: str | None,
    chromosome: str | None,
    position: int | None,
    ref_allele: str | None,
    alt_allele: str | None,
    hgvs: str | None,
    zygosity: str,
) -> Variant:
    """Annotate a raw variant and build a persistable :class:`Variant`."""
    ann = annotate_variant(gene=gene, rs_id=rs_id)
    return Variant(
        genetic_test_id=test_id,
        patient_id=patient_id,
        gene=gene or ann.gene,
        rs_id=rs_id,
        chromosome=chromosome,
        position=position,
        ref_allele=ref_allele,
        alt_allele=alt_allele,
        hgvs=hgvs,
        zygosity=Zygosity(zygosity),
        clinical_significance=ann.clinical_significance,
        condition_code=ann.condition_code,
        condition_display=ann.condition_display,
        inheritance_mode=ann.inheritance_mode,
        clinvar_id=ann.clinvar_id,
    )


@router.post(
    "/patients/{patient_id}/genetic-tests",
    response_model=GeneticTestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a VCF or structured variant report",
)
async def create_genetic_test(
    patient_id: uuid.UUID, body: GeneticTestCreate, db: DbSession
) -> GeneticTestResponse:
    """Ingest a genetic test, annotate its variants, and persist the report.

    Args:
        patient_id: Patient UUID the test belongs to.
        body: Either a list of structured variants or raw VCF text.
        db: Async database session.

    Returns:
        The stored test with all annotated variants.

    Raises:
        HTTPException 404: Patient not found.
        HTTPException 422: Neither variants nor parseable VCF content supplied.
    """
    patient = await db.get(Patient, patient_id)
    if patient is None or patient.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Patient not found")

    test = GeneticTest(
        patient_id=patient_id,
        organization_id=patient.organization_id,
        test_type=GeneticTestType(body.test_type),
        status=GeneticTestStatus.FINAL,
        lab_name=body.lab_name,
        method=body.method,
        performed_date=body.performed_date,
        source_filename=body.source_filename,
    )
    db.add(test)
    await db.flush()  # assign test.id

    variants: list[Variant] = []
    if body.variants:
        for vi in body.variants:
            variants.append(
                _build_variant(
                    patient_id,
                    test.id,
                    gene=vi.gene,
                    rs_id=vi.rs_id,
                    chromosome=vi.chromosome,
                    position=vi.position,
                    ref_allele=vi.ref_allele,
                    alt_allele=vi.alt_allele,
                    hgvs=vi.hgvs,
                    zygosity=vi.zygosity,
                )
            )
    else:
        parsed = parse_vcf(body.vcf_content or "")
        if not parsed:
            raise HTTPException(status_code=422, detail="No variant records found in VCF content")
        for p in parsed:
            variants.append(
                _build_variant(
                    patient_id,
                    test.id,
                    gene=p.gene,
                    rs_id=p.rs_id,
                    chromosome=p.chromosome,
                    position=p.position,
                    ref_allele=p.ref_allele,
                    alt_allele=p.alt_allele,
                    hgvs=None,
                    zygosity=p.zygosity,
                )
            )

    for v in variants:
        db.add(v)

    significances = [v.clinical_significance for v in variants]
    overall = summarise_significance(significances)
    pathogenic = [v for v in variants if is_pathogenic(v.clinical_significance)]

    test.variant_count = len(variants)
    test.pathogenic_count = len(pathogenic)
    test.overall_interpretation = (
        f"{len(pathogenic)} pathogenic/likely-pathogenic variant(s); "
        f"overall classification: {overall.value.replace('_', ' ')}."
    )

    # Back-propagate to conditions: mark matching active diagnoses hereditary.
    for v in pathogenic:
        if not v.condition_code:
            continue
        await db.execute(
            update(Condition)
            .where(
                Condition.patient_id == patient_id,
                Condition.code == v.condition_code,
                Condition.is_hereditary.is_(False),
            )
            .values(is_hereditary=True)
        )

    await db.flush()

    loaded = (
        await db.execute(
            select(GeneticTest)
            .where(GeneticTest.id == test.id)
            .options(selectinload(GeneticTest.variants))
        )
    ).scalar_one()
    log.info(
        "Genetic test ingested: patient=%s test=%s variants=%d pathogenic=%d",
        patient_id,
        test.id,
        len(variants),
        len(pathogenic),
    )
    return GeneticTestResponse.model_validate(loaded)


@router.get(
    "/patients/{patient_id}/genetic-tests",
    response_model=list[GeneticTestResponse],
    summary="List a patient's genetic tests and annotated variants",
)
async def list_genetic_tests(patient_id: uuid.UUID, db: DbSession) -> list[GeneticTestResponse]:
    """List a patient's genetic tests, newest first, with annotated variants.

    Args:
        patient_id: Patient UUID.
        db: Async database session.

    Returns:
        Genetic tests with their variants.
    """
    tests = (
        (
            await db.execute(
                select(GeneticTest)
                .where(GeneticTest.patient_id == patient_id)
                .options(selectinload(GeneticTest.variants))
                .order_by(GeneticTest.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
    return [GeneticTestResponse.model_validate(t) for t in tests]
