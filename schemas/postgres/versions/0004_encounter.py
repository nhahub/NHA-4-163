"""Create encounter and encounter_participant tables.

Revision ID: m0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "m0004"
down_revision: str | None = "m0003"
branch_labels = None
depends_on = None

_STATUS_ENUM = postgresql.ENUM(
    "planned",
    "arrived",
    "triaged",
    "in-progress",
    "onleave",
    "finished",
    "cancelled",
    "entered-in-error",
    "unknown",
    name="encounter_status",
    create_type=False,
)


def upgrade() -> None:
    _STATUS_ENUM.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "encounter",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patient.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", _STATUS_ENUM, nullable=False),
        sa.Column("encounter_class", sa.String(20)),
        sa.Column("type_code", sa.String(100)),
        sa.Column("type_display", sa.String(255)),
        sa.Column("service_type", sa.String(255)),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True)),
        sa.Column("facility_name", sa.String(255)),
        sa.Column("facility_id", sa.String(255)),
        sa.Column("resource_json", postgresql.JSONB),
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
    op.create_index("ix_encounter_patient_id", "encounter", ["patient_id"])
    op.create_index("ix_encounter_status", "encounter", ["status"])
    op.create_index("ix_encounter_period_start", "encounter", ["period_start"])
    op.create_index("ix_encounter_class", "encounter", ["encounter_class"])

    # Many-to-many: encounter ↔ physician participants.
    op.create_table(
        "encounter_participant",
        sa.Column(
            "encounter_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("encounter.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "physician_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("physician.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
    )

    op.execute("""
        CREATE TRIGGER trg_encounter_updated_at
        BEFORE UPDATE ON encounter
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_encounter_updated_at ON encounter;")
    op.drop_table("encounter_participant")
    op.drop_table("encounter")
    _STATUS_ENUM.drop(op.get_bind(), checkfirst=True)
