"""0015 — Granular patient consent records (Tier 7).

Revision: 0015
Parent:   0014 (genetic tests)

Creates the append-only ``consent_record`` table plus its enum types.  Each row
is one consent decision (grant/deny/withdraw) for a patient + scope; the
effective state for a scope is the most recent row.  Enforced at the research
export layer.

PHI note: the table references a ``patient_id`` and stores scope/decision
metadata and PHI-free notes only.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None

_SCOPE = sa.Enum(
    "research",
    "data_sharing",
    "treatment",
    "family_contact",
    "genetic_testing",
    "marketing",
    name="consent_scope",
)
_STATUS = sa.Enum("granted", "denied", "withdrawn", name="consent_status")
_METHOD = sa.Enum("written", "verbal", "electronic", "portal", name="consent_method")


def upgrade() -> None:
    op.create_table(
        "consent_record",
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
        sa.Column("scope", _SCOPE, nullable=False),
        sa.Column("status", _STATUS, nullable=False),
        sa.Column("method", _METHOD, nullable=True),
        sa.Column("granted_at", sa.DateTime(timezone=True)),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("withdrawn_at", sa.DateTime(timezone=True)),
        sa.Column("policy_version", sa.String(50)),
        sa.Column("notes", sa.String(1000)),
        sa.Column("created_by", sa.String(255)),
        sa.Column("updated_by", sa.String(255)),
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
    for col in ("patient_id", "organization_id", "scope", "status"):
        op.create_index(f"ix_consent_record_{col}", "consent_record", [col])


def downgrade() -> None:
    op.drop_table("consent_record")
    bind = op.get_bind()
    _METHOD.drop(bind, checkfirst=True)
    _STATUS.drop(bind, checkfirst=True)
    _SCOPE.drop(bind, checkfirst=True)
