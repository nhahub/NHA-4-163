"""Async SQLAlchemy database engine and session management.

Provides a FastAPI dependency ``DbSession`` that yields an ``AsyncSession``
per-request with automatic commit/rollback.  The engine is initialised once
at import time using the same Postgres DSN from ``libs.common.config``.

Usage in a route::

    from services.api.db import DbSession

    @router.post("/patients")
    async def create_patient(body: PatientCreate, db: DbSession):
        ...
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from libs.common.config import get_settings

log = logging.getLogger(__name__)

_settings = get_settings()

# Build async DSN from sync DSN: postgresql://... → postgresql+asyncpg://...
_sync_dsn = _settings.postgres.sync_dsn
_async_dsn = _sync_dsn.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    _async_dsn,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

AsyncSessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def _get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield an async database session with automatic commit/rollback.

    Commits on success, rolls back on exception, always closes.

    Yields:
        AsyncSession bound to the shared engine.
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


DbSession = Annotated[AsyncSession, Depends(_get_db_session)]
