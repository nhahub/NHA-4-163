"""Create physician table.

Physician has no FK dependencies so it is created before patient, encounter,
condition, and medication_request which all reference it.

Revision ID: m0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "m0002"
down_revision: str | None = "m0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "physician",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("npi", sa.String(10), nullable=False),
        sa.Column("family_name", sa.String(255)),
        sa.Column("given_name", sa.String(255)),
        sa.Column("specialty", sa.String(255)),
        sa.Column("specialty_code", sa.String(20)),
        sa.Column("neo4j_node_id", sa.String(255)),
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
        sa.Column("created_by", sa.String(255)),
        sa.Column("updated_by", sa.String(255)),
    )
    op.create_unique_constraint("uq_physician_npi", "physician", ["npi"])
    op.create_index("ix_physician_npi", "physician", ["npi"])
    op.create_index("ix_physician_specialty_code", "physician", ["specialty_code"])

    op.execute("""
        CREATE TRIGGER trg_physician_updated_at
        BEFORE UPDATE ON physician
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_physician_updated_at ON physician;")
    op.drop_table("physician")
