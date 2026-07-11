"""Create observation table.

Revision ID: m0006
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "m0006"
down_revision: str | None = "m0005"
branch_labels = None
depends_on = None

_OBS_STATUS = postgresql.ENUM(
    "registered",
    "preliminary",
    "final",
    "amended",
    "corrected",
    "cancelled",
    "entered-in-error",
    "unknown",
    name="observation_status",
    create_type=False,
)


def upgrade() -> None:
    _OBS_STATUS.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "observation",
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
            "encounter_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("encounter.id", ondelete="SET NULL"),
        ),
        sa.Column("status", _OBS_STATUS, nullable=False),
        sa.Column("category", sa.String(100)),
        sa.Column("code_system", sa.String(255), nullable=False),
        sa.Column("code", sa.String(50), nullable=False),
        sa.Column("code_display", sa.String(500)),
        sa.Column("effective_datetime", sa.DateTime(timezone=True), nullable=False),
        # Polymorphic value columns
        sa.Column("value_quantity", sa.Numeric(precision=18, scale=6)),
        sa.Column("value_unit", sa.String(50)),
        sa.Column("value_unit_system", sa.String(255)),
        sa.Column("value_string", sa.String(500)),
        sa.Column("value_boolean", sa.Boolean),
        sa.Column("value_codeable_code", sa.String(50)),
        sa.Column("value_codeable_display", sa.String(500)),
        # Reference range
        sa.Column("ref_range_low", sa.Numeric(precision=18, scale=6)),
        sa.Column("ref_range_high", sa.Numeric(precision=18, scale=6)),
        sa.Column("ref_range_text", sa.String(255)),
        sa.Column("interpretation", sa.String(10)),
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

    for col in ("patient_id", "encounter_id", "status", "category", "code", "effective_datetime"):
        op.create_index(f"ix_observation_{col}", "observation", [col])

    # Composite: patient + code + effective — time-series feature extraction pattern.
    op.create_index(
        "ix_observation_patient_code_dt",
        "observation",
        ["patient_id", "code", "effective_datetime"],
    )

    op.execute("""
        CREATE TRIGGER trg_observation_updated_at
        BEFORE UPDATE ON observation
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_observation_updated_at ON observation;")
    op.drop_table("observation")
    _OBS_STATUS.drop(op.get_bind(), checkfirst=True)
