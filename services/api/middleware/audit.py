"""HIPAA-compliant request audit logging middleware.

Logs every API request to the PostgreSQL ``audit_log`` table.  Only
metadata is recorded — request/response bodies are NEVER logged because
they may contain PHI.

Logged fields (matches migration 0009_audit_log schema)
--------------------------------------------------------
- ``actor_id``       Username from JWT ``sub`` claim; ``"anonymous"`` if
                     unauthenticated.
- ``actor_type``     ``"service_account"`` | ``"anonymous"``
- ``action``         HTTP method (e.g., ``"POST"``)
- ``resource_type``  Derived from URL path (``"prediction"``, ``"patient"``,
                     ``"auth"``, ``"other"``)
- ``resource_id``    Patient UUID extracted from path params only (never body)
- ``service_name``   Always ``"healthcare-api"``
- ``user_agent``     ``User-Agent`` request header
- ``ip_address``     Client IP from ``X-Forwarded-For`` or direct connection
- ``outcome``        ``"success"`` | ``"client_error"`` | ``"server_error"``
- ``outcome_detail`` HTTP status code as string
- ``metadata``       JSON: {request_id, latency_ms, path, query_params}

The write is fire-and-forget (``asyncio.create_task``) so a slow or
failed audit write never blocks the API response.

PHI guarantee: resource_id is extracted from path parameters only;
the request body (which may contain PHI) is never read by this middleware.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger(__name__)

_SERVICE_NAME = "healthcare-api"

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)

_EXCLUDED_PATHS = {"/health", "/ready", "/metrics", "/docs", "/openapi.json", "/redoc"}


def _resource_type(path: str) -> str:
    """Map a URL path to a resource type label for the audit log.

    Args:
        path: URL path string.

    Returns:
        One of ``"prediction"``, ``"patient"``, ``"auth"``, ``"other"``.
    """
    if path.startswith("/predict/"):
        return "prediction"
    if path.startswith("/patient/"):
        return "patient"
    if path.startswith("/auth/"):
        return "auth"
    return "other"


def _outcome(status_code: int) -> str:
    """Map an HTTP status code to an outcome label.

    Args:
        status_code: HTTP response status code.

    Returns:
        ``"success"``, ``"client_error"``, or ``"server_error"``.
    """
    if status_code < 400:
        return "success"
    if status_code < 500:
        return "client_error"
    return "server_error"


def _extract_actor(request: Request) -> tuple[str, str]:
    """Extract actor_id and actor_type from the JWT in the Authorization header.

    Decodes the token without verifying the signature here (verification
    already happened in the route handler's dependency).  If no token is
    present or decoding fails, returns ``("anonymous", "anonymous")``.

    Args:
        request: Incoming FastAPI request.

    Returns:
        ``(actor_id, actor_type)`` strings.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        return "anonymous", "anonymous"
    try:
        import jwt as pyjwt

        token = auth[7:]
        payload = pyjwt.decode(token, options={"verify_signature": False})
        return str(payload.get("sub", "anonymous")), "service_account"
    except Exception:
        return "anonymous", "anonymous"


def _extract_resource_id(request: Request) -> str | None:
    """Extract a patient UUID from request path parameters only.

    Never reads the request body to avoid PHI exposure.

    Args:
        request: Incoming FastAPI request.

    Returns:
        UUID string, or ``None`` if not found.
    """
    pid = request.path_params.get("patient_id")
    if pid:
        return str(pid)
    match = _UUID_RE.search(request.url.path)
    return match.group(0) if match else None


def _get_client_ip(request: Request) -> str:
    """Return the best available client IP.

    Args:
        request: Incoming request.

    Returns:
        IP string.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


async def _write_audit_log(
    dsn: str,
    actor_id: str,
    actor_type: str,
    action: str,
    resource_type: str,
    resource_id: str | None,
    user_agent: str,
    ip_address: str,
    outcome: str,
    outcome_detail: str,
    metadata: dict[str, object],
) -> None:
    """Insert one immutable row into the ``audit_log`` table.

    Runs in a thread pool (called via ``asyncio.to_thread`` from the
    fire-and-forget task) so it does not block the event loop.

    Args:
        dsn: PostgreSQL sync DSN.
        actor_id: Subject from the JWT (or ``"anonymous"``).
        actor_type: ``"service_account"`` or ``"anonymous"``.
        action: HTTP method string.
        resource_type: High-level resource category.
        resource_id: Patient UUID string or ``None``.
        user_agent: ``User-Agent`` header value.
        ip_address: Client IP string.
        outcome: ``"success"``, ``"client_error"``, or ``"server_error"``.
        outcome_detail: HTTP status code as string.
        metadata: JSON-serialisable supplemental metadata.
    """
    try:
        import psycopg2

        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_log (
                        actor_id, actor_type, action,
                        resource_type, resource_id, service_name,
                        user_agent, ip_address,
                        outcome, outcome_detail, metadata
                    ) VALUES (
                        %s, %s, %s,
                        %s, %s::uuid, %s,
                        %s, %s::inet,
                        %s, %s, %s::jsonb
                    )
                    """,
                    (
                        actor_id,
                        actor_type,
                        action,
                        resource_type,
                        resource_id,
                        _SERVICE_NAME,
                        user_agent[:500] if user_agent else "",
                        ip_address,
                        outcome,
                        outcome_detail,
                        json.dumps(metadata),
                    ),
                )
                conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        log.warning("Audit log write failed: %s", exc)


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that writes a HIPAA audit log entry per request.

    Attributes:
        dsn: PostgreSQL connection string for the ``audit_log`` table.
    """

    def __init__(self, app: object, dsn: str) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.dsn = dsn

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process a request, then schedule a fire-and-forget audit write.

        Args:
            request: Incoming request.
            call_next: Next middleware / handler in the chain.

        Returns:
            Response from the downstream handler.
        """
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.monotonic()

        response = await call_next(request)

        latency_ms = round((time.monotonic() - start) * 1000, 2)
        actor_id, actor_type = _extract_actor(request)
        resource_id = _extract_resource_id(request)
        ip = _get_client_ip(request)

        metadata: dict[str, object] = {
            "request_id": request_id,
            "latency_ms": latency_ms,
            "path": request.url.path,
            "query": str(request.url.query) if request.url.query else "",
        }

        asyncio.create_task(
            asyncio.to_thread(
                _write_audit_log_sync,
                self.dsn,
                actor_id,
                actor_type,
                request.method,
                _resource_type(request.url.path),
                resource_id,
                request.headers.get("User-Agent", ""),
                ip,
                _outcome(response.status_code),
                str(response.status_code),
                metadata,
            )
        )

        response.headers["X-Request-ID"] = request_id
        return response


def _write_audit_log_sync(
    dsn: str,
    actor_id: str,
    actor_type: str,
    action: str,
    resource_type: str,
    resource_id: str | None,
    user_agent: str,
    ip_address: str,
    outcome: str,
    outcome_detail: str,
    metadata: dict[str, object],
) -> None:
    """Synchronous audit insert — called via asyncio.to_thread.

    Identical signature to ``_write_audit_log`` but synchronous so it
    can be dispatched to the thread pool without nesting coroutines.
    """
    try:
        import psycopg2

        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_log (
                        actor_id, actor_type, action,
                        resource_type, resource_id, service_name,
                        user_agent, ip_address,
                        outcome, outcome_detail, metadata
                    ) VALUES (
                        %s, %s, %s,
                        %s, %s::uuid, %s,
                        %s, %s::inet,
                        %s, %s, %s::jsonb
                    )
                    """,
                    (
                        actor_id,
                        actor_type,
                        action,
                        resource_type,
                        resource_id,
                        _SERVICE_NAME,
                        user_agent[:500] if user_agent else "",
                        ip_address,
                        outcome,
                        outcome_detail,
                        json.dumps(metadata),
                    ),
                )
                conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        log.warning("Audit log write failed: %s", exc)
