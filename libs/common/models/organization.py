"""Organization ORM model — multi-tenant boundary (Tier 4).

An ``Organization`` represents a healthcare tenant (hospital, clinic network,
research group).  Patient records are scoped to an organization via
``Patient.organization_id`` so that one deployment can serve multiple isolated
tenants.  Tenants authenticate machine-to-machine requests with an API key; the
key is stored only as a SHA-256 hash, never in plaintext.

Isolation is enforced at two layers:
  1. Application layer — org-scoped dependencies stamp/filter ``organization_id``.
  2. Database layer — Row Level Security policies (migration 0012) provide
     defence-in-depth so a compromised connection cannot cross tenants.

PHI note: this table holds no patient data — only tenant metadata and a key hash.
"""

from __future__ import annotations

import hashlib
import secrets

from sqlalchemy import Boolean, String, text
from sqlalchemy.orm import Mapped, mapped_column

from libs.common.models.base import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin


def hash_api_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest used to store/look up an API key.

    Args:
        raw_key: The plaintext API key.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key and its stored hash.

    Returns:
        A ``(raw_key, key_hash)`` tuple. The raw key is shown to the caller
        exactly once; only the hash is persisted.
    """
    raw = f"hc_{secrets.token_urlsafe(32)}"
    return raw, hash_api_key(raw)


class Organization(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
    """A tenant that owns a partition of patient records."""

    __tablename__ = "organization"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # URL-safe short identifier, unique per deployment.
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)

    # SHA-256 hash of the tenant's API key (never the plaintext key).
    api_key_hash: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)

    active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=text("true"), nullable=False
    )
