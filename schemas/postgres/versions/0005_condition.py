"""Create condition table.

Revision ID: m0005
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "m0005"
down_revision: str | None = "m0004"
branch_labels = None
depends_on = None

_CLINICAL_STATUS = postgresql.ENUM(
    "active",
    "recurrence",
    "relapse",
    "inactive",
    "remission",
    "resolved",
    name="clinical_status",
    create_type=False,
)
_VERIFICATION_STATUS = postgresql.ENUM(
    "unconfirmed",
    "provisional",
    "differential",
    "confirmed",
    "refuted",
    "entered-in-error",
    name="verification_status",
    create_type=False,
)
_SEVERITY = postgresql.ENUM(
    "severe",
    "moderate",
    "mild",
    name="condition_severity",
    create_type=False,
)


def upgrade() -> None:
    for enum in (_CLINICAL_STATUS, _VERIFICATION_STATUS, _SEVERITY):
        enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "condition",
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
            "recorder_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("physician.id", ondelete="SET NULL"),
        ),
        sa.Column("clinical_status", _CLINICAL_STATUS, nullable=False),
        sa.Column("verification_status", _VERIFICATION_STATUS),
        sa.Column("severity", _SEVERITY),
        sa.Column("code_system", sa.String(255), nullable=False),
        sa.Column("code", sa.String(50), nullable=False),
        sa.Column("code_display", sa.String(500)),
        sa.Column("code_text", sa.String(500)),
        sa.Column("onset_datetime", sa.DateTime(timezone=True)),
        sa.Column("onset_age_years", sa.Integer),
        sa.Column("abatement_datetime", sa.DateTime(timezone=True)),
        sa.Column("is_hereditary", sa.Boolean, server_default="false", nullable=False),
        sa.Column("family_history_flag", sa.Boolean, server_default="false", nullable=False),
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
        "recorder_id",
        "clinical_status",
        "verification_status",
        "code",
        "onset_datetime",
        "is_hereditary",
        "family_history_flag",
    ):
        op.create_index(f"ix_condition_{col}", "condition", [col])

    # Composite: patient + code — the most common query pattern for ML feature extraction.
    op.create_index("ix_condition_patient_code", "condition", ["patient_id", "code"])

    op.execute("""
        CREATE TRIGGER trg_condition_updated_at
        BEFORE UPDATE ON condition
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_condition_updated_at ON condition;")
    op.drop_table("condition")
    for enum in (_SEVERITY, _VERIFICATION_STATUS, _CLINICAL_STATUS):
        enum.drop(op.get_bind(), checkfirst=True)
