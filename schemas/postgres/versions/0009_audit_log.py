"""Create audit_log table with append-only enforcement.

The append-only guarantee is enforced by a PostgreSQL trigger that raises
an exception on any UPDATE or DELETE attempt.  In production, the database
role used by application services must also be denied UPDATE/DELETE via
GRANT (Phase 7 RLS policy).

Partitioned by month on ``occurred_at`` for efficient long-term retention
queries and time-based archival to S3.

Revision ID: m0009
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "m0009"
down_revision: str | None = "m0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("actor_id", sa.String(255), nullable=False),
        sa.Column("actor_type", sa.String(50), nullable=False),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("resource_type", sa.String(100), nullable=False),
        sa.Column("resource_id", sa.String(255)),
        sa.Column("service_name", sa.String(100)),
        sa.Column("user_agent", sa.String(500)),
        sa.Column("ip_address", postgresql.INET),
        sa.Column("outcome", sa.String(50), nullable=False),
        sa.Column("outcome_detail", sa.String(1000)),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
    )

    for col in (
        "actor_id",
        "actor_type",
        "action",
        "resource_type",
        "resource_id",
        "service_name",
        "outcome",
        "occurred_at",
    ):
        op.create_index(f"ix_audit_log_{col}", "audit_log", [col])

    # Trigger: block any UPDATE or DELETE — audit records are immutable.
    op.execute("""
        CREATE OR REPLACE FUNCTION enforce_audit_log_immutability()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION
                'audit_log rows are immutable: UPDATE and DELETE are not permitted. '
                'Attempted operation: % on row id=%',
                TG_OP, OLD.id;
        END;
        $$;
        """)
    op.execute("""
        CREATE TRIGGER trg_audit_log_immutable
        BEFORE UPDATE OR DELETE ON audit_log
        FOR EACH ROW EXECUTE FUNCTION enforce_audit_log_immutability();
        """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_audit_log_immutable ON audit_log;")
    op.execute("DROP FUNCTION IF EXISTS enforce_audit_log_immutability();")
    op.drop_table("audit_log")
