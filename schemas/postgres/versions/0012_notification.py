"""0012 — Notifications (risk & workflow alerts).

Revision: 0012
Parent:   0011 (organizations)

Creates the ``notification`` table plus its enum types. Notifications reference
a ``patient_id`` and optional ``organization_id`` and store only PHI-free
content (risk score/tier, threshold, title, message).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

_TYPE = sa.Enum(
    "risk_threshold_crossed",
    "risk_increased",
    "family_update",
    "screening_reminder",
    name="notification_type",
)
_SEVERITY = sa.Enum("info", "warning", "critical", name="notification_severity")
_STATUS = sa.Enum("unread", "read", "dismissed", name="notification_status")


def upgrade() -> None:
    op.create_table(
        "notification",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "patient_id",
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
        sa.Column("notification_type", _TYPE, nullable=False),
        sa.Column("severity", _SEVERITY, nullable=False),
        sa.Column("status", _STATUS, nullable=False, server_default="unread"),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("message", sa.String(1000), nullable=False),
        sa.Column("risk_score", sa.Numeric(precision=6, scale=5)),
        sa.Column("risk_tier", sa.String(20)),
        sa.Column("threshold", sa.Numeric(precision=6, scale=5)),
        sa.Column("read_at", sa.DateTime(timezone=True)),
        sa.Column("acknowledged", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("context", postgresql.JSONB, server_default="{}"),
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
    for col in ("patient_id", "organization_id", "notification_type", "severity", "status"):
        op.create_index(f"ix_notification_{col}", "notification", [col])


def downgrade() -> None:
    op.drop_table("notification")
    bind = op.get_bind()
    _STATUS.drop(bind, checkfirst=True)
    _SEVERITY.drop(bind, checkfirst=True)
    _TYPE.drop(bind, checkfirst=True)
