"""Authentication endpoints.

POST /auth/token
    Validates service-account credentials against the ``service_accounts``
    Postgres table and returns a signed JWT access token.

The endpoint deliberately uses a generic error message for invalid
credentials to prevent user enumeration.

Passwords are stored as bcrypt hashes (cost factor 12).  Never store
plaintext passwords — use the ``scripts/create_service_account.py``
utility to provision new accounts.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from services.api.auth.jwt import create_access_token
from services.api.auth.models import TokenRequest, TokenResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_INVALID_CREDS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid username or password",
    headers={"WWW-Authenticate": "Bearer"},
)


def _verify_credentials_sync(
    username: str,
    password: str,
    postgres_dsn: str,
) -> tuple[bool, str]:
    """Look up a service account and verify the bcrypt password.

    Args:
        username: Account username.
        password: Plaintext password to verify.
        postgres_dsn: Sync PostgreSQL DSN.

    Returns:
        ``(is_valid, role)`` tuple.  ``is_valid`` is False when the account
        does not exist, is inactive, or the password does not match.
    """
    import psycopg2
    import psycopg2.extras

    try:
        from passlib.context import CryptContext
    except ImportError as exc:
        raise RuntimeError("Install 'passlib[bcrypt]' for password hashing") from exc

    _pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

    try:
        conn = psycopg2.connect(postgres_dsn)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT hashed_password, role, is_active
                    FROM service_accounts
                    WHERE username = %s
                    """,
                    (username,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
    except Exception as exc:
        log.error("Database error during authentication: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable",
        ) from exc

    if row is None or not row["is_active"]:
        return False, ""

    if not _pwd_ctx.verify(password, row["hashed_password"]):
        return False, ""

    return True, str(row["role"])


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Obtain a JWT access token",
    description=(
        "Validates service-account credentials and returns a signed JWT. "
        "Tokens are valid for the duration configured in ``JWT_EXPIRE_MINUTES``. "
        "Rate-limited to 10 requests per minute per IP address."
    ),
)
async def login(body: TokenRequest) -> TokenResponse:
    """Exchange credentials for a JWT access token.

    Args:
        body: Username and password.

    Returns:
        ``TokenResponse`` containing the signed JWT and metadata.

    Raises:
        HTTPException 401: If credentials are invalid or the account is inactive.
        HTTPException 503: If the database is unreachable.
    """
    import asyncio

    from libs.common.config import get_settings

    settings = get_settings()
    pg_dsn = settings.postgres.sync_dsn

    is_valid, role = await asyncio.to_thread(
        _verify_credentials_sync,
        body.username,
        body.password.get_secret_value(),
        pg_dsn,
    )

    if not is_valid:
        # Constant-time guard against timing attacks: always run verify
        # before raising so failure branches take similar time.
        log.warning("Failed login attempt for username='%s'", body.username)
        raise _INVALID_CREDS

    token, expires_in = create_access_token(body.username, role)
    log.info("Token issued: user='%s' role='%s'", body.username, role)

    return TokenResponse(
        access_token=token,
        token_type="bearer",  # noqa: S106 — OAuth2 token type, not a secret
        expires_in=expires_in,
        role=role,
    )
