"""Role-Based Access Control (RBAC) for the Healthcare Prediction API.

Roles
-----
``admin``       Full access — manage users, view audit log, all predictions.
``clinician``   Clinical staff — predict risk, view family profiles, read patient.
``researcher``  Data scientists — predict risk on de-identified data only;
                no family profile access (contains first-degree relative PHI).
``service``     Machine-to-machine — predict + read only (no write/mutation);
                used by internal services and EHR integrations.

Permission matrix
-----------------
Permission                admin   clinician   researcher  service
predict:risk               ✓         ✓           ✓          ✓
predict:symptom            ✓         ✓                      ✓
read:family_profile        ✓         ✓                      ✓
read:patient               ✓         ✓                      ✓
view:audit_log             ✓
manage:users               ✓

Usage in a route::

    from services.api.auth.rbac import Permission, require_permission

    @router.get("/patient/{id}/family-risk-profile")
    async def get_profile(
        patient_id: uuid.UUID,
        _: Annotated[UserClaims, Depends(require_permission(Permission.READ_FAMILY_PROFILE))],
    ) -> FamilyRiskProfileResponse:
        ...
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.api.auth.jwt import verify_token
from services.api.auth.models import UserClaims

_bearer = HTTPBearer(auto_error=True)


class Role(StrEnum):
    """RBAC roles available in the system."""

    ADMIN = "admin"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    SERVICE = "service"


class Permission(StrEnum):
    """Fine-grained permissions checked on each protected endpoint."""

    PREDICT_RISK = "predict:risk"
    PREDICT_SYMPTOM = "predict:symptom"
    READ_FAMILY_PROFILE = "read:family_profile"
    READ_PATIENT = "read:patient"
    WRITE_PATIENT = "write:patient"
    WRITE_CLINICAL = "write:clinical"
    WRITE_ENCOUNTER = "write:encounter"
    RUN_BATCH_SCREEN = "run:batch_screen"
    VIEW_AUDIT_LOG = "view:audit_log"
    MANAGE_USERS = "manage:users"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.ADMIN: frozenset(Permission),  # all permissions
    Role.CLINICIAN: frozenset(
        {
            Permission.PREDICT_RISK,
            Permission.PREDICT_SYMPTOM,
            Permission.READ_FAMILY_PROFILE,
            Permission.READ_PATIENT,
            Permission.WRITE_PATIENT,
            Permission.WRITE_CLINICAL,
            Permission.WRITE_ENCOUNTER,
            Permission.RUN_BATCH_SCREEN,
        }
    ),
    Role.RESEARCHER: frozenset(
        {
            Permission.PREDICT_RISK,
            Permission.READ_PATIENT,
        }
    ),
    Role.SERVICE: frozenset(
        {
            Permission.PREDICT_RISK,
            Permission.PREDICT_SYMPTOM,
            Permission.READ_FAMILY_PROFILE,
            Permission.READ_PATIENT,
        }
    ),
}


def _get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(_bearer)],
) -> UserClaims:
    """Extract and validate the Bearer token from the Authorization header.

    Args:
        credentials: Injected by FastAPI's HTTPBearer security scheme.

    Returns:
        Validated ``UserClaims`` for the authenticated user.

    Raises:
        HTTPException 401: If the token is missing, expired, or invalid.
    """
    return verify_token(credentials.credentials)


CurrentUserDep = Annotated[UserClaims, Depends(_get_current_user)]


def require_permission(permission: Permission) -> Any:
    """FastAPI dependency factory that enforces a single permission.

    Creates a dependency that:
    1. Validates the Bearer token (via ``_get_current_user``).
    2. Checks that the token's role has the requested permission.

    Args:
        permission: The permission required to access the endpoint.

    Returns:
        A FastAPI ``Depends``-compatible callable yielding ``UserClaims``.

    Raises:
        HTTPException 401: Token missing or invalid.
        HTTPException 403: Role lacks the required permission.

    Example::

        @router.post("/predict/hereditary-risk")
        async def predict(
            body: PredictHeredityRiskRequest,
            user: Annotated[UserClaims, Depends(require_permission(Permission.PREDICT_RISK))],
        ) -> HeredityRiskResponse:
            ...
    """

    def _check(user: CurrentUserDep) -> UserClaims:
        try:
            role = Role(user.role)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Unknown role '{user.role}'",
            ) from exc
        if permission not in ROLE_PERMISSIONS.get(role, frozenset()):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{user.role}' does not have permission '{permission.value}'. "
                    "Contact your administrator."
                ),
            )
        return user

    return Depends(_check)
