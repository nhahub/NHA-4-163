"""Create medication_request table.

Revision ID: m0007
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "m0007"
down_revision: str | None = "m0006"
branch_labels = None
depends_on = None

_MR_STATUS = postgresql.ENUM(
    "active",
    "on-hold",
    "cancelled",
    "completed",
    "entered-in-error",
    "stopped",
    "draft",
    "unknown",
    name="medication_request_status",
    create_type=False,
)
_MR_INTENT = postgresql.ENUM(
    "proposal",
    "plan",
    "order",
    "original-order",
    "reflex-order",
    "filler-order",
    "instance-order",
    "option",
    name="medication_request_intent",
    create_type=False,
)


def upgrade() -> None:
    for enum in (_MR_STATUS, _MR_INTENT):
        enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "medication_request",
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
        sa.Column(
            "requester_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("physician.id", ondelete="SET NULL"),
        ),
        sa.Column("status", _MR_STATUS, nullable=False),
        sa.Column("intent", _MR_INTENT, nullable=False),
        sa.Column("medication_code_system", sa.String(255)),
        sa.Column("medication_code", sa.String(50), nullable=False),
        sa.Column("medication_display", sa.String(500)),
        sa.Column("dosage_text", sa.String(500)),
        sa.Column("dosage_timing", sa.String(255)),
        sa.Column("dosage_route", sa.String(100)),
        sa.Column("dose_quantity", sa.Numeric(precision=10, scale=3)),
        sa.Column("dose_unit", sa.String(50)),
        sa.Column("dispense_quantity", sa.Numeric(precision=10, scale=3)),
        sa.Column("dispense_unit", sa.String(50)),
        sa.Column("number_of_repeats", sa.Integer),
        sa.Column("authored_on", sa.DateTime(timezone=True), nullable=False),
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

    for col in (
        "patient_id",
        "encounter_id",
        "requester_id",
        "status",
        "medication_code",
        "authored_on",
    ):
        op.create_index(f"ix_medication_request_{col}", "medication_request", [col])

    # Composite: patient + med code — prescription history feature extraction.
    op.create_index(
        "ix_medication_request_patient_code",
        "medication_request",
        ["patient_id", "medication_code"],
    )

    op.execute("""
        CREATE TRIGGER trg_medication_request_updated_at
        BEFORE UPDATE ON medication_request
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_medication_request_updated_at ON medication_request;")
    op.drop_table("medication_request")
    for enum in (_MR_INTENT, _MR_STATUS):
        enum.drop(op.get_bind(), checkfirst=True)
