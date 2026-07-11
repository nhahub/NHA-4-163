"""Redis-backed fixed-window rate limiting middleware.

Enforces per-route request limits to protect against brute-force and
abuse.  Two strategies are used depending on the path:

- **Auth endpoints** (``/auth/token``): rate-limited **per client IP**
  to prevent credential stuffing (10 requests / 60 s per IP).
- **API endpoints** (``/predict/``, ``/patient/``): rate-limited **per
  authenticated user** (extracted from the ``Authorization`` header's
  JWT subject) with a higher limit (60 requests / 60 s).

Algorithm: fixed-window counter with Redis ``INCR`` + ``EXPIRE``.
The window key rolls over every ``window_seconds`` aligned to the UTC
epoch (e.g., minute boundaries for 60-second windows).

Response headers
----------------
``X-RateLimit-Limit``      Maximum requests allowed in the window.
``X-RateLimit-Remaining``  Requests remaining in the current window.
``X-RateLimit-Reset``      UTC epoch second when the window resets.

On limit exceeded, returns ``HTTP 429 Too Many Requests`` with a
``Retry-After`` header indicating seconds until the window resets.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger(__name__)

# Each rule is: path prefix, max requests, window in seconds.
_RATE_RULES: list[tuple[str, int, int]] = [
    ("/auth/token", 10, 60),  # brute-force guard — per IP
    ("/predict/", 60, 60),  # prediction endpoints — per user
    ("/patient/", 30, 60),  # patient endpoints — per user
]

_EXCLUDED_PATHS = {"/health", "/ready", "/metrics", "/docs", "/openapi.json", "/redoc"}


def _match_rule(path: str) -> tuple[int, int] | None:
    """Return (max_requests, window_seconds) for the first matching rule.

    Args:
        path: URL path string.

    Returns:
        ``(limit, window)`` tuple, or ``None`` if no rule matches.
    """
    for prefix, limit, window in _RATE_RULES:
        if path.startswith(prefix):
            return limit, window
    return None


def _identifier(request: Request, use_ip: bool) -> str:
    """Build the rate-limit bucket key for this request.

    Auth endpoints use the client IP; all other endpoints use the JWT
    subject extracted from the ``Authorization`` header (falls back to IP
    when no token is present).

    Args:
        request: Incoming FastAPI request.
        use_ip: Force IP-based identification (for auth paths).

    Returns:
        Identifier string used in the Redis key.
    """
    if not use_ip:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:]
            try:
                import jwt as pyjwt

                from libs.common.config import get_settings

                settings = get_settings().jwt
                payload = pyjwt.decode(
                    token,
                    settings.secret_key.get_secret_value(),
                    algorithms=[settings.algorithm],
                )
                sub: str = payload.get("sub", "")
                if sub:
                    return f"user:{sub}"
            except Exception as exc:
                # Token parsing failed; fall through to IP-based identification.
                log.debug("Rate-limit token parse failed: %s", exc)

    forwarded = request.headers.get("X-Forwarded-For")
    ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "unknown")
    )
    return f"ip:{ip}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces per-path rate limits via Redis.

    Attributes:
        _redis: redis.asyncio client stored on ``app.state.cache_service``.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Check rate limit before passing request to the next handler.

        Args:
            request: Incoming request.
            call_next: Next middleware / route handler.

        Returns:
            The downstream response, or a 429 response if limit exceeded.
        """
        path = request.url.path

        if path in _EXCLUDED_PATHS:
            return await call_next(request)

        rule = _match_rule(path)
        if rule is None:
            return await call_next(request)

        limit, window = rule
        use_ip = path.startswith("/auth/")
        identifier = _identifier(request, use_ip=use_ip)

        # Fixed-window key: aligned to wall-clock window boundaries
        window_start = int(time.time()) // window
        redis_key = f"rl:{identifier}:{path.split('/')[1]}:{window_start}"
        reset_at = (window_start + 1) * window

        try:
            cache = request.app.state.cache_service
            redis = cache._client  # access underlying redis.asyncio client

            count: int = await redis.incr(redis_key)
            if count == 1:
                await redis.expire(redis_key, window)

            remaining = max(0, limit - count)

            if count > limit:
                retry_after = reset_at - int(time.time())
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded — please slow down."},
                    headers={
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset_at),
                        "Retry-After": str(max(retry_after, 1)),
                    },
                )
        except Exception as exc:
            # Never let a Redis failure block a legitimate request
            log.warning("Rate limit check failed (Redis error): %s", exc)
            return await call_next(request)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)
        return response
