"""FastAPI dependency injection providers.

Application-scoped singletons (model, Redis client) are initialised in
the lifespan context manager in ``main.py`` and stored on
``request.app.state``.  Route handlers receive them via these typed
``Annotated`` dependency functions.

Auth dependencies (``CurrentUserDep``, ``require_permission``) are
re-exported here so route handlers import from a single location.

Usage in a route::

    from services.api.deps import ModelDep, CacheDep, CurrentUserDep
    from services.api.auth.rbac import Permission, require_permission

    @router.post("/predict/hereditary-risk")
    async def predict(
        body: PredictHeredityRiskRequest,
        model: ModelDep,
        cache: CacheDep,
        user: Annotated[UserClaims, Depends(require_permission(Permission.PREDICT_RISK))],
    ) -> HeredityRiskResponse:
        ...
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from libs.common.config import Settings, get_settings
from services.api.auth.models import UserClaims
from services.api.auth.rbac import CurrentUserDep, Permission, require_permission
from services.api.services.cache_service import CacheService
from services.api.services.model_service import ModelService

# ── Settings ──────────────────────────────────────────────────────────────────


def _get_settings() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(_get_settings)]


# ── Model service ─────────────────────────────────────────────────────────────


def _get_model(request: Request) -> ModelService:
    """Return the app-scoped ModelService instance.

    Args:
        request: Current FastAPI request (injects app.state).

    Returns:
        The pre-loaded ModelService.

    Raises:
        RuntimeError: If the lifespan initialisation was skipped.
    """
    svc: ModelService = request.app.state.model_service
    if not svc.is_loaded:
        raise RuntimeError("ModelService is not initialised")
    return svc


ModelDep = Annotated[ModelService, Depends(_get_model)]


# ── Cache service ─────────────────────────────────────────────────────────────


def _get_cache(request: Request) -> CacheService:
    """Return the app-scoped CacheService instance.

    Args:
        request: Current FastAPI request.

    Returns:
        The pre-connected CacheService.
    """
    return request.app.state.cache_service  # type: ignore[no-any-return]


CacheDep = Annotated[CacheService, Depends(_get_cache)]


# ── Auth re-exports ───────────────────────────────────────────────────────────
# These are imported from auth.rbac and re-exported here so route handlers
# only need to import from services.api.deps.

__all__ = [
    "CacheDep",
    "CurrentUserDep",
    "ModelDep",
    "Permission",
    "SettingsDep",
    "UserClaims",
    "require_permission",
]
