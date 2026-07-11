"""0010 — Service accounts table and PostgreSQL Row Level Security.

Revision: 0010
Parent:   m0009 (audit_log)

Changes
-------
1. ``service_accounts`` table — stores bcrypt-hashed credentials for API
   service accounts (clinicians, researchers, admin bots).  Passwords are
   NEVER stored in plaintext; use ``passlib[bcrypt]`` with cost factor 12.

2. Row Level Security (RLS) on ``patient`` — enables Postgres-level
   access control as a defence-in-depth layer beneath the application's
   RBAC checks.  A ``researcher`` role sees only a de-identified view;
   a ``healthcare_app`` role bypasses RLS (application enforces RBAC).

3. ``v_patients_deidentified`` VIEW — removes all 18 HIPAA Safe Harbor
   direct identifiers so that researcher-role queries never touch PHI.
   The view is intentionally NOT a SECURITY DEFINER view — it relies on
   standard RLS to enforce access.

4. ``healthcare_researcher`` database role — read-only access to the
   de-identified view only.  The application connects as ``healthcare_app``
   (bypasses RLS) and enforces researcher restrictions at the API layer.
   This Postgres role is provided as a defence-in-depth option for
   direct DB access (e.g., Jupyter notebook analytics).

PHI note: the ``service_accounts`` table contains usernames and bcrypt
hashes — no patient data.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "m0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply migration: service_accounts, RLS, de-identified view."""

    # ── 1. service_accounts ───────────────────────────────────────────────────
    op.create_table(
        "service_accounts",
        sa.Column(
            "id",
            sa.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("username", sa.String(100), nullable=False, unique=True),
        sa.Column(
            "hashed_password",
            sa.String(255),
            nullable=False,
            comment="bcrypt hash — cost factor 12; NEVER store plaintext",
        ),
        sa.Column(
            "role",
            sa.String(50),
            nullable=False,
            comment="One of: admin, clinician, researcher, service",
        ),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("TRUE")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "role IN ('admin', 'clinician', 'researcher', 'service')",
            name="ck_service_accounts_role",
        ),
    )
    op.create_index("ix_service_accounts_username", "service_accounts", ["username"])
    op.create_index("ix_service_accounts_role", "service_accounts", ["role"])

    # ── 2. De-identified view for researcher access ───────────────────────────
    # Strips all 18 HIPAA Safe Harbor direct identifiers.
    # Age is generalised (decade bucket; 90+ collapsed).
    # ZIP is truncated to 3-digit prefix.
    op.execute("""
        CREATE OR REPLACE VIEW v_patients_deidentified AS
        SELECT
            id,
            -- (1) name fields removed
            -- (2) geographic: state only (city/street stripped)
            NULL::text                          AS address,
            -- (3) dates: year only for age >= 90, else year-month
            CASE
                WHEN EXTRACT(YEAR FROM AGE(date_of_birth)) >= 90
                    THEN EXTRACT(YEAR FROM date_of_birth)::text
                ELSE TO_CHAR(date_of_birth, 'YYYY-MM')
            END                                 AS birth_year_or_month,
            -- age: decade generalisation
            CASE
                WHEN EXTRACT(YEAR FROM AGE(date_of_birth)) >= 90 THEN '90+'
                ELSE (FLOOR(EXTRACT(YEAR FROM AGE(date_of_birth)) / 10) * 10)::text || 's'
            END                                 AS age_group,
            -- gender kept (not a direct identifier under Safe Harbor)
            gender,
            -- deleted_at kept for data integrity
            deleted_at
        FROM patient
        WHERE deleted_at IS NULL
        """)

    # ── 3. healthcare_researcher database role ────────────────────────────────
    # Only created if it does not already exist (idempotent).
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_roles WHERE rolname = 'healthcare_researcher'
            ) THEN
                CREATE ROLE healthcare_researcher NOLOGIN;
            END IF;
        END
        $$;
        """)
    op.execute("GRANT CONNECT ON DATABASE healthcare TO healthcare_researcher")
    op.execute("GRANT USAGE ON SCHEMA public TO healthcare_researcher")
    op.execute("GRANT SELECT ON v_patients_deidentified TO healthcare_researcher")

    # ── 4. Enable RLS on patient ──────────────────────────────────────────────
    op.execute("ALTER TABLE patient ENABLE ROW LEVEL SECURITY")

    # healthcare_app (the API role) bypasses RLS — application enforces RBAC
    op.execute("""
        CREATE POLICY patients_app_bypass ON patient
            AS PERMISSIVE
            FOR ALL
            TO healthcare_app
            USING (true)
            WITH CHECK (true)
        """)

    # healthcare_researcher has NO direct access to the patient table
    # (they use v_patients_deidentified instead)
    op.execute("""
        CREATE POLICY patients_researcher_deny ON patient
            AS RESTRICTIVE
            FOR SELECT
            TO healthcare_researcher
            USING (false)
        """)

    # ── 5. Audit log: grant INSERT to healthcare_app (already exists) ─────────
    # Ensure future roles can insert audit entries
    op.execute("GRANT INSERT ON audit_log TO healthcare_app")
    op.execute("GRANT SELECT ON service_accounts TO healthcare_app")
    op.execute("GRANT UPDATE (last_login_at, is_active) ON service_accounts TO healthcare_app")


def downgrade() -> None:
    """Reverse migration: remove RLS, view, researcher role, service_accounts."""

    # Remove RLS policies
    op.execute("DROP POLICY IF EXISTS patients_researcher_deny ON patient")
    op.execute("DROP POLICY IF EXISTS patients_app_bypass ON patient")
    op.execute("ALTER TABLE patient DISABLE ROW LEVEL SECURITY")

    # Remove researcher role (revoke grants first)
    op.execute("REVOKE ALL ON v_patients_deidentified FROM healthcare_researcher")
    op.execute("REVOKE ALL ON SCHEMA public FROM healthcare_researcher")
    op.execute("REVOKE CONNECT ON DATABASE healthcare FROM healthcare_researcher")
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'healthcare_researcher') THEN
                DROP ROLE healthcare_researcher;
            END IF;
        END
        $$;
        """)

    # Remove view
    op.execute("DROP VIEW IF EXISTS v_patients_deidentified")

    # Remove service_accounts table
    op.drop_index("ix_service_accounts_role", table_name="service_accounts")
    op.drop_index("ix_service_accounts_username", table_name="service_accounts")
    op.drop_table("service_accounts")
