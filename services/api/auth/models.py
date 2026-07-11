"""Pydantic models for authentication request / response bodies.

These models carry no PHI — they represent service account credentials
and issued token metadata only.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class TokenRequest(BaseModel):
    """Credentials submitted to POST /auth/token.

    Follows the OAuth 2.0 Resource Owner Password Credentials grant so
    that the OpenAPI UI can generate a usable Authorize dialog.
    """

    model_config = ConfigDict(frozen=True)

    username: str = Field(..., min_length=1, max_length=100)
    password: SecretStr = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """JWT access token returned by POST /auth/token."""

    model_config = ConfigDict(frozen=True)

    access_token: str
    token_type: str = "bearer"  # noqa: S105 — OAuth2 token type, not a secret
    expires_in: int = Field(..., description="Token lifetime in seconds")
    role: str = Field(..., description="Role granted to this token")


class UserClaims(BaseModel):
    """Validated JWT payload embedded in every authenticated request.

    Extracted from the Bearer token by ``verify_token()`` and injected
    into route handlers via the ``CurrentUserDep`` dependency.

    Attributes:
        user_id: Username / subject from the ``sub`` claim.
        role: Role string (``admin``, ``clinician``, ``researcher``,
            ``service``).
        jti: JWT ID — unique per token, used for revocation checks.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str
    role: str
    jti: str
