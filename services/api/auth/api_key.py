"""Organization API-key authentication for multi-tenant access (Tier 4).

Tenants authenticate machine-to-machine requests with an ``X-API-Key`` header.
The key is looked up by its SHA-256 hash (never stored in plaintext) against the
``organization`` table.  Two dependencies are provided:

- ``CurrentOrgDep``          — requires a valid, active organization key (401 otherwise).
- ``OptionalOrgDep``         — resolves the org if a key is present, else ``None``
                               (lets endpoints support both single- and multi-tenant use).
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select

from libs.common.models.organization import Organization, hash_api_key
from services.api.db import DbSession

log = logging.getLogger(__name__)


async def _lookup_org(db: DbSession, api_key: str) -> Organization | None:
    """Return the active organization owning ``api_key``, or ``None``."""
    key_hash = hash_api_key(api_key)
    result = await db.execute(
        select(Organization).where(
            Organization.api_key_hash == key_hash,
            Organization.active.is_(True),
            Organization.deleted_at.is_(None),
        )
    )
    return result.scalars().first()


async def get_optional_org(
    db: DbSession,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Organization | None:
    """Resolve the organization from ``X-API-Key`` if provided, else ``None``.

    Args:
        db: Async database session.
        x_api_key: The tenant API key header (optional).

    Returns:
        The matching active :class:`Organization`, or ``None`` if no key was
        supplied.

    Raises:
        HTTPException 401: If a key is supplied but does not match an active org.
    """
    if not x_api_key:
        return None
    org = await _lookup_org(db, x_api_key)
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive organization API key",
        )
    return org


async def get_current_org(
    db: DbSession,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Organization:
    """Require a valid, active organization API key.

    Args:
        db: Async database session.
        x_api_key: The tenant API key header.

    Returns:
        The authenticated :class:`Organization`.

    Raises:
        HTTPException 401: If the key is missing, invalid, or inactive.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    org = await _lookup_org(db, x_api_key)
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive organization API key",
        )
    return org


CurrentOrgDep = Annotated[Organization, Depends(get_current_org)]
OptionalOrgDep = Annotated[Organization | None, Depends(get_optional_org)]
