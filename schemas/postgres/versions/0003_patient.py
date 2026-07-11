"""Create patient table.

PHI columns (name, dob, address, contact) are created as plain VARCHAR here.
Phase 7 migration will wrap them in pgcrypto / application-level envelope
encryption.  A TODO comment marks each PHI column for the Phase 7 engineer.

Revision ID: m0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "m0003"
down_revision: str | None = "m0002"
branch_labels = None
depends_on = None

_GENDER_ENUM = postgresql.ENUM(
    "male",
    "female",
    "other",
    "unknown",
    name="administrative_gender",
    create_type=False,
)


def upgrade() -> None:
    _GENDER_ENUM.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "patient",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
            nullable=False,
        ),
        # ── External identifiers ─────────────────────────────────────────────
        sa.Column("external_id", sa.String(255), unique=True),  # [PHI]
        sa.Column("identifier_system", sa.String(255)),
        # ── Name [PHI — encrypt in Phase 7] ──────────────────────────────────
        sa.Column("family_name", sa.String(255)),
        sa.Column("given_name", sa.String(255)),
        sa.Column("middle_name", sa.String(255)),
        # ── Demographics ─────────────────────────────────────────────────────
        sa.Column("date_of_birth", sa.Date),  # [PHI]
        sa.Column("gender", _GENDER_ENUM),
        sa.Column("ethnicity", sa.String(100)),
        sa.Column("race", sa.String(100)),
        # ── Deceased ─────────────────────────────────────────────────────────
        sa.Column("deceased", sa.Boolean, server_default="false", nullable=False),
        sa.Column("deceased_date", sa.Date),
        # ── Contact [PHI — encrypt in Phase 7] ───────────────────────────────
        sa.Column("phone", sa.String(50)),
        sa.Column("email", sa.String(255)),
        # ── Address [PHI — encrypt in Phase 7] ───────────────────────────────
        sa.Column("address_line", sa.String(500)),
        sa.Column("city", sa.String(255)),
        sa.Column("state", sa.String(100)),
        sa.Column("postal_code", sa.String(20)),
        sa.Column("country", sa.String(100), server_default="US"),
        # ── Communication ────────────────────────────────────────────────────
        sa.Column("language", sa.String(10), server_default="en"),
        # ── Consent ──────────────────────────────────────────────────────────
        sa.Column("research_consent", sa.Boolean, server_default="false", nullable=False),
        sa.Column("research_consent_date", sa.DateTime(timezone=True)),
        # ── Cross-system refs ─────────────────────────────────────────────────
        sa.Column("neo4j_node_id", sa.String(255)),
        # ── Audit ─────────────────────────────────────────────────────────────
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
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("created_by", sa.String(255)),
        sa.Column("updated_by", sa.String(255)),
    )

    op.create_index("ix_patient_external_id", "patient", ["external_id"])
    op.create_index("ix_patient_date_of_birth", "patient", ["date_of_birth"])
    op.create_index("ix_patient_gender", "patient", ["gender"])
    op.create_index("ix_patient_neo4j_node_id", "patient", ["neo4j_node_id"])

    # Trigram index for fuzzy family_name search (de-identified analytics).
    op.execute(
        "CREATE INDEX ix_patient_family_name_trgm ON patient "
        "USING gin (family_name gin_trgm_ops);"
    )

    op.execute("""
        CREATE TRIGGER trg_patient_updated_at
        BEFORE UPDATE ON patient
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_patient_updated_at ON patient;")
    op.drop_index("ix_patient_family_name_trgm", table_name="patient")
    op.drop_table("patient")
    _GENDER_ENUM.drop(op.get_bind(), checkfirst=True)
