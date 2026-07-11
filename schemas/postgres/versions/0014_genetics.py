"""0014 — Genetic tests and annotated variants.

Revision: 0014
Parent:   0013 (cascade screening)

Creates the ``genetic_test`` and ``variant`` tables plus their enum types.
A genetic test captures a report (panel/WES/WGS/single-gene/VCF); each variant
carries a ClinVar/OMIM-derived pathogenicity classification and disease
association.

PHI note: genomic data is highly identifying — treat both tables as PHI.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None

_TEST_TYPE = sa.Enum(
    "panel",
    "wes",
    "wgs",
    "single_gene",
    "karyotype",
    "vcf_upload",
    name="genetic_test_type",
)
_TEST_STATUS = sa.Enum(
    "preliminary", "final", "amended", "entered-in-error", name="genetic_test_status"
)
_SIGNIFICANCE = sa.Enum(
    "pathogenic",
    "likely_pathogenic",
    "uncertain_significance",
    "likely_benign",
    "benign",
    name="variant_clinical_significance",
)
_ZYGOSITY = sa.Enum("heterozygous", "homozygous", "hemizygous", "unknown", name="variant_zygosity")


def upgrade() -> None:
    op.create_table(
        "genetic_test",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patient.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organization.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("test_type", _TEST_TYPE, nullable=False),
        sa.Column("status", _TEST_STATUS, nullable=False, server_default="final"),
        sa.Column("lab_name", sa.String(255)),
        sa.Column("method", sa.String(255)),
        sa.Column("performed_date", sa.Date),
        sa.Column("source_filename", sa.String(255)),
        sa.Column("overall_interpretation", sa.String(500)),
        sa.Column("variant_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("pathogenic_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index("ix_genetic_test_patient_id", "genetic_test", ["patient_id"])
    op.create_index("ix_genetic_test_organization_id", "genetic_test", ["organization_id"])
    op.create_index("ix_genetic_test_test_type", "genetic_test", ["test_type"])

    op.create_table(
        "variant",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "genetic_test_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("genetic_test.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patient.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("gene", sa.String(50)),
        sa.Column("chromosome", sa.String(5)),
        sa.Column("position", sa.Integer),
        sa.Column("ref_allele", sa.String(255)),
        sa.Column("alt_allele", sa.String(255)),
        sa.Column("rs_id", sa.String(30)),
        sa.Column("hgvs", sa.String(255)),
        sa.Column("zygosity", _ZYGOSITY, nullable=False, server_default="unknown"),
        sa.Column(
            "clinical_significance",
            _SIGNIFICANCE,
            nullable=False,
            server_default="uncertain_significance",
        ),
        sa.Column("condition_code", sa.String(50)),
        sa.Column("condition_display", sa.String(255)),
        sa.Column("inheritance_mode", sa.String(40)),
        sa.Column("clinvar_id", sa.String(50)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    for col in ("genetic_test_id", "patient_id", "gene", "rs_id", "clinical_significance"):
        op.create_index(f"ix_variant_{col}", "variant", [col])


def downgrade() -> None:
    op.drop_table("variant")
    op.drop_table("genetic_test")
    bind = op.get_bind()
    _ZYGOSITY.drop(bind, checkfirst=True)
    _SIGNIFICANCE.drop(bind, checkfirst=True)
    _TEST_STATUS.drop(bind, checkfirst=True)
    _TEST_TYPE.drop(bind, checkfirst=True)
