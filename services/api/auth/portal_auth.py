"""Patient-scoped authentication for the SMART on FHIR patient portal (Tier 7).

The patient portal issues **patient-scoped** access tokens distinct from the
staff/service JWTs handled by :mod:`services.api.auth.jwt`.  A portal token
carries a SMART launch context — the ``patient`` claim binds the token to a
single patient id and the ``scope`` claim is restricted to read-only patient
access (``patient/*.read``).  Portal endpoints resolve the caller's own patient
id from this claim, so a patient can only ever read their own record.

This is a pragmatic SMART-on-FHIR subset (mirroring the FHIR tier's approach):
we model the standalone-launch token context and scope enforcement, not the
full OAuth2 authorization-code + PKCE handshake, which requires an external
identity provider.

PHI note: the token embeds only a patient UUID and scope string — never names
or clinical data.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger(__name__)

# SMART scope granted to portal tokens: read-only access to the launch patient.
PATIENT_SCOPE = "patient/*.read"
_PATIENT_ROLE = "patient"
_TOKEN_TTL_SECONDS = 3600  # 1 hour, matching a typical SMART session.

_bearer = HTTPBearer(auto_error=True)


@dataclass(frozen=True)
class PatientContext:
    """SMART launch context extracted from a validated portal token.

    Attributes:
        patient_id: The patient the token is scoped to (own record).
        scope: The granted SMART scope string.
        jti: Unique token id (supports future revocation).
    """

    patient_id: uuid.UUID
    scope: str
    jti: str


def _jwt_settings() -> Any:
    """Lazy-load JWT settings to avoid import-time config coupling."""
    from libs.common.config import get_settings

    return get_settings().jwt


def create_patient_token(patient_id: uuid.UUID) -> tuple[str, int]:
    """Mint a read-only, patient-scoped SMART access token.

    Args:
        patient_id: The patient to bind the token to (launch context).

    Returns:
        ``(token_string, expires_in_seconds)``.
    """
    import jwt as pyjwt

    settings = _jwt_settings()
    now = datetime.now(tz=UTC)
    exp = now + timedelta(seconds=_TOKEN_TTL_SECONDS)
    payload: dict[str, Any] = {
        "sub": str(patient_id),
        "role": _PATIENT_ROLE,
        "patient": str(patient_id),
        "scope": PATIENT_SCOPE,
        "jti": str(uuid.uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    token: str = pyjwt.encode(
        payload,
        settings.secret_key.get_secret_value(),
        algorithm=settings.algorithm,
    )
    return token, _TOKEN_TTL_SECONDS


def verify_patient_token(token: str) -> PatientContext:
    """Decode and validate a patient-scoped portal token.

    Args:
        token: Raw JWT from the ``Authorization: Bearer`` header.

    Returns:
        The :class:`PatientContext` for the token.

    Raises:
        HTTPException 401: Token expired, malformed, wrong role, or missing the
            ``patient`` launch context.
    """
    import jwt as pyjwt

    settings = _jwt_settings()
    try:
        payload: dict[str, Any] = pyjwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.algorithm],
            options={"require": ["sub", "role", "patient", "scope", "exp"]},
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Portal session has expired — please re-launch.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except pyjwt.InvalidTokenError as exc:
        log.debug("Portal token validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or malformed portal token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if payload.get("role") != _PATIENT_ROLE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not a patient-scoped portal token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        patient_id = uuid.UUID(str(payload["patient"]))
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Portal token missing a valid patient launch context",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return PatientContext(
        patient_id=patient_id,
        scope=str(payload.get("scope", "")),
        jti=str(payload.get("jti", "")),
    )


def _get_current_patient(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(_bearer)],
) -> PatientContext:
    """FastAPI dependency: resolve the patient from a Bearer portal token."""
    return verify_patient_token(credentials.credentials)


CurrentPatientDep = Annotated[PatientContext, Depends(_get_current_patient)]
