"""Create family_member_history table.

Revision ID: m0008
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "m0008"
down_revision: str | None = "m0007"
branch_labels = None
depends_on = None

_FMH_STATUS = postgresql.ENUM(
    "partial",
    "completed",
    "entered-in-error",
    "health-unknown",
    name="family_member_history_status",
    create_type=False,
)


def upgrade() -> None:
    _FMH_STATUS.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "family_member_history",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patient.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "related_patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patient.id", ondelete="SET NULL"),
        ),
        sa.Column("status", _FMH_STATUS, nullable=False),
        sa.Column("relationship_code", sa.String(50), nullable=False),
        sa.Column("relationship_display", sa.String(100)),
        sa.Column(
            "degree_of_relatedness",
            sa.Numeric(precision=5, scale=4),
            sa.CheckConstraint(
                "degree_of_relatedness >= 0 AND degree_of_relatedness <= 1",
                name="ck_fmh_degree_range",
            ),
        ),
        sa.Column("sex", sa.String(20)),
        sa.Column("born_date", sa.Date),
        sa.Column("deceased", sa.Boolean),
        sa.Column("deceased_age_years", sa.Integer),
        sa.Column("deceased_date", sa.Date),
        sa.Column("conditions", postgresql.JSONB, server_default="[]"),
        sa.Column("neo4j_synced", sa.Boolean, server_default="false", nullable=False),
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

    op.create_index("ix_fmh_patient_id", "family_member_history", ["patient_id"])
    op.create_index("ix_fmh_related_patient_id", "family_member_history", ["related_patient_id"])
    op.create_index("ix_fmh_relationship_code", "family_member_history", ["relationship_code"])
    op.create_index("ix_fmh_neo4j_synced", "family_member_history", ["neo4j_synced"])

    # GIN index on conditions JSONB — supports queries like:
    # WHERE conditions @> '[{"code": {"code": "I10"}}]'
    op.execute(
        "CREATE INDEX ix_fmh_conditions_gin ON family_member_history "
        "USING gin (conditions jsonb_path_ops);"
    )

    op.execute("""
        CREATE TRIGGER trg_fmh_updated_at
        BEFORE UPDATE ON family_member_history
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_fmh_updated_at ON family_member_history;")
    op.drop_index("ix_fmh_conditions_gin", table_name="family_member_history")
    op.drop_table("family_member_history")
    _FMH_STATUS.drop(op.get_bind(), checkfirst=True)
