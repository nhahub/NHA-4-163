"""0013 — Cascade screening (proband → at-risk relatives workflow).

Revision: 0013
Parent:   0012 (notifications)

Creates the ``cascade_screening`` and ``cascade_task`` tables plus their enum
types.  A screening run captures the inheritance context used to rank a
proband's blood relatives; each task tracks one relative's outreach lifecycle.

PHI note: these tables store relationship metadata, probabilities, and status
only — never relative names or contact details in the clear.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None

_PRIORITY = sa.Enum("high", "medium", "low", name="cascade_priority")
_TASK_STATUS = sa.Enum(
    "pending",
    "contacted",
    "scheduled",
    "screened",
    "declined",
    "completed",
    name="cascade_task_status",
)


def upgrade() -> None:
    op.create_table(
        "cascade_screening",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "proband_patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patient.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organization.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("condition_code", sa.String(50)),
        sa.Column("condition_display", sa.String(255)),
        sa.Column("inheritance_mode", sa.String(40), nullable=False),
        sa.Column("penetrance", sa.Numeric(precision=4, scale=3)),
        sa.Column("task_count", sa.Integer, nullable=False, server_default="0"),
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
    )
    op.create_index(
        "ix_cascade_screening_proband_patient_id",
        "cascade_screening",
        ["proband_patient_id"],
    )
    op.create_index(
        "ix_cascade_screening_organization_id",
        "cascade_screening",
        ["organization_id"],
    )
    op.create_index("ix_cascade_screening_condition_code", "cascade_screening", ["condition_code"])

    op.create_table(
        "cascade_task",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "screening_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cascade_screening.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "family_member_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("family_member_history.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "related_patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patient.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("relationship_code", sa.String(50), nullable=False),
        sa.Column("relationship_display", sa.String(100)),
        sa.Column("degree_of_relatedness", sa.Numeric(precision=5, scale=4)),
        sa.Column("priority", _PRIORITY, nullable=False),
        sa.Column("priority_score", sa.Numeric(precision=6, scale=4), nullable=False),
        sa.Column("carrier_probability", sa.Numeric(precision=6, scale=5)),
        sa.Column("affected_probability", sa.Numeric(precision=6, scale=5)),
        sa.Column("status", _TASK_STATUS, nullable=False, server_default="pending"),
        sa.Column("recommended_action", sa.String(500)),
        sa.Column("notes", sa.String(1000)),
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
    )
    for col in ("screening_id", "family_member_id", "related_patient_id", "priority", "status"):
        op.create_index(f"ix_cascade_task_{col}", "cascade_task", [col])


def downgrade() -> None:
    op.drop_table("cascade_task")
    op.drop_table("cascade_screening")
    bind = op.get_bind()
    _TASK_STATUS.drop(bind, checkfirst=True)
    _PRIORITY.drop(bind, checkfirst=True)
