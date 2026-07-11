"""Baseline: PostgreSQL extensions and shared utilities.

Installs extensions needed by all subsequent migrations:
- pgcrypto:   gen_random_uuid(), pgp_sym_encrypt() (Phase 7 field encryption)
- pg_trgm:    trigram similarity indexes for fuzzy name search
- btree_gin:  GIN indexes on scalar types (useful for JSONB + text combinations)

Also creates the ``updated_at`` trigger function used by every table to keep
the column current on bulk UPDATE statements that bypass the ORM layer.

Revision ID: m0001
"""

from __future__ import annotations

from alembic import op

revision: str = "m0001"
down_revision: str | None = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gin;")

    # Trigger function: automatically update `updated_at` on row modification.
    # Used by every table that has an `updated_at` column.
    op.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$;
        """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS set_updated_at();")
    # Extensions are not dropped on downgrade — other objects may depend on them.
