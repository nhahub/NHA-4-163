"""JWT token creation and verification.

Tokens are signed with HMAC-SHA256 (HS256) using the secret configured
in ``JWT_SECRET_KEY``.  Each token carries:

- ``sub``  — username (service account ID)
- ``role`` — RBAC role string
- ``jti``  — UUID4 token ID (supports future revocation via Redis blocklist)
- ``iat``  — issued-at timestamp (UTC)
- ``exp``  — expiry timestamp (UTC)

Token revocation
----------------
``jti`` is not currently checked against a Redis blocklist (Phase 7 scope).
Add a ``POST /auth/revoke`` endpoint + Redis SET check in Phase 8.

PHI note: tokens contain NO patient data — only the service account
identifier and role.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException, status

from services.api.auth.models import UserClaims

log = logging.getLogger(__name__)


def _jwt_settings() -> Any:
    """Lazy-load JWTSettings to avoid circular imports at module load time."""
    from libs.common.config import get_settings

    return get_settings().jwt


def create_access_token(user_id: str, role: str) -> tuple[str, int]:
    """Create a signed JWT access token.

    Args:
        user_id: Subject identifier (service account username).
        role: RBAC role string (``admin``, ``clinician``, etc.).

    Returns:
        ``(token_string, expires_in_seconds)`` tuple.

    Raises:
        RuntimeError: If JWT signing fails unexpectedly.
    """
    import jwt as pyjwt

    settings = _jwt_settings()
    now = datetime.now(tz=UTC)
    expire_seconds = settings.expire_minutes * 60
    exp = now + timedelta(seconds=expire_seconds)

    payload: dict[str, Any] = {
        "sub": user_id,
        "role": role,
        "jti": str(uuid.uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }

    token: str = pyjwt.encode(
        payload,
        settings.secret_key.get_secret_value(),
        algorithm=settings.algorithm,
    )
    return token, expire_seconds


def verify_token(token: str) -> UserClaims:
    """Decode and validate a JWT Bearer token.

    Args:
        token: Raw JWT string from the ``Authorization: Bearer`` header.

    Returns:
        ``UserClaims`` with the validated payload fields.

    Raises:
        HTTPException 401: If the token is expired, malformed, or has an
            invalid signature.
    """
    import jwt as pyjwt

    settings = _jwt_settings()
    try:
        payload: dict[str, Any] = pyjwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.algorithm],
            options={"require": ["sub", "role", "jti", "exp", "iat"]},
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired — please re-authenticate",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except pyjwt.InvalidTokenError as exc:
        log.debug("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or malformed token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return UserClaims(
        user_id=payload["sub"],
        role=payload["role"],
        jti=payload["jti"],
    )
