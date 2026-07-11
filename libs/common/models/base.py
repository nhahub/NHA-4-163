"""SQLAlchemy 2.0 declarative base and shared mixins.

All ORM models inherit from ``Base``.  Mixins are composed in, not inherited
from a deep hierarchy, keeping each model self-documenting.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Project-wide declarative base.

    Alembic's ``env.py`` imports this to drive autogenerate comparison.
    """

    pass


class UUIDPrimaryKeyMixin:
    """UUID primary key generated server-side by PostgreSQL gen_random_uuid().

    Using server-side generation means the DB is the single source of truth
    for IDs, preventing client-generated UUID collisions in multi-writer setups.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )


class TimestampMixin:
    """Append-only audit timestamps.  ``updated_at`` uses a DB trigger (see
    migration 0001) rather than SQLAlchemy ``onupdate`` so that bulk UPDATE
    statements also keep the column current without hitting the ORM layer.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """Logical delete via ``deleted_at`` timestamp.

    Queries that should exclude soft-deleted rows must add
    ``WHERE deleted_at IS NULL`` explicitly or use a view.
    """

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )


class ActorMixin:
    """Track which service or user last wrote this row.

    ``created_by`` / ``updated_by`` store the OAuth2 subject (``sub`` claim)
    of the actor.  They are set by the application layer, never by the DB.
    Never populate these with raw user-supplied strings without validation.
    """

    created_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
