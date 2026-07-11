"""0011 — Organizations (multi-tenant) and Row Level Security.

Revision: 0011
Parent:   0010 (service accounts + RLS on patients)

Changes
-------
1. ``organization`` table — tenant metadata + SHA-256 API-key hash (never
   plaintext).
2. ``patient.organization_id`` — nullable FK scoping each patient to a tenant.
   Nullable so pre-existing single-tenant data is unaffected.
3. RLS policy on ``patient`` keyed off the ``app.current_org`` GUC — a
   defence-in-depth layer so a connection scoped to one tenant cannot read
   another tenant's rows even if the application filter is bypassed. The
   ``healthcare_app`` role (BYPASSRLS) continues to enforce scoping at the API
   layer; this policy protects direct/least-privilege connections.

PHI note: ``organization`` holds no patient data — only tenant name/slug and an
API-key hash.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "organization",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column("api_key_hash", sa.String(64)),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.text("true")),
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
    )
    op.create_unique_constraint("uq_organization_slug", "organization", ["slug"])
    op.create_unique_constraint("uq_organization_api_key_hash", "organization", ["api_key_hash"])
    op.create_index("ix_organization_slug", "organization", ["slug"])
    op.create_index("ix_organization_api_key_hash", "organization", ["api_key_hash"])

    # Patient → organization scoping.
    op.add_column(
        "patient",
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organization.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_patient_organization_id", "patient", ["organization_id"])

    # ── Row Level Security: tenant isolation on patient ───────────────────────
    op.execute("ALTER TABLE patient ENABLE ROW LEVEL SECURITY;")
    op.execute("""
        CREATE POLICY patient_org_isolation ON patient
        USING (
            organization_id IS NULL
            OR current_setting('app.current_org', true) IS NULL
            OR organization_id = current_setting('app.current_org', true)::uuid
        );
        """)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS patient_org_isolation ON patient;")
    op.execute("ALTER TABLE patient DISABLE ROW LEVEL SECURITY;")
    op.drop_index("ix_patient_organization_id", table_name="patient")
    op.drop_column("patient", "organization_id")
    op.drop_index("ix_organization_api_key_hash", table_name="organization")
    op.drop_index("ix_organization_slug", table_name="organization")
    op.drop_table("organization")
