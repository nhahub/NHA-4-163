"""Prometheus metrics instrumentation middleware.

Instruments every HTTP request with counters and latency histograms.
Additional domain-specific metrics (prediction scores, cache operations,
model load status) are exposed as module-level singletons so route
handlers can import and increment them directly.

Metrics exposed
---------------
healthcare_http_requests_total{method, path, status}
    Counter — total requests by HTTP method, normalised path template,
    and response status code.

healthcare_http_request_duration_seconds{method, path}
    Histogram — end-to-end request latency in seconds.

healthcare_prediction_risk_score
    Histogram — distribution of hereditary risk scores from the model.
    Populated by the predictions router after each successful inference.

healthcare_cache_operations_total{result}
    Counter — cache ``hit`` / ``miss`` per prediction request.

healthcare_model_loaded
    Gauge — 1.0 when the XGBoost model is loaded, 0.0 otherwise.
    Updated by the model service on load / unload.

healthcare_rate_limit_rejected_total{path}
    Counter — number of requests rejected by the rate limiter.

Path normalisation
------------------
Raw paths containing UUIDs (``/patient/uuid.../family-risk-profile``)
are normalised to their FastAPI route template
(``/patient/{patient_id}/family-risk-profile``) using Starlette's
``request.scope["route"]`` attribute.  This prevents unbounded label
cardinality in Prometheus.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

# ── Metric definitions (module-level singletons) ─────────────────────────────

try:
    from prometheus_client import Counter, Gauge, Histogram

    HTTP_REQUESTS = Counter(
        "healthcare_http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )

    HTTP_LATENCY = Histogram(
        "healthcare_http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "path"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    PREDICTION_SCORE = Histogram(
        "healthcare_prediction_risk_score",
        "Distribution of hereditary risk prediction scores",
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    CACHE_OPERATIONS = Counter(
        "healthcare_cache_operations_total",
        "Cache hit/miss count",
        ["result"],  # "hit" | "miss"
    )

    MODEL_LOADED = Gauge(
        "healthcare_model_loaded",
        "Whether the XGBoost model is currently loaded (1 = yes)",
    )

    RATE_LIMIT_REJECTED = Counter(
        "healthcare_rate_limit_rejected_total",
        "Requests rejected by the rate limiter",
        ["path"],
    )

    _PROMETHEUS_AVAILABLE = True

except ImportError:
    _PROMETHEUS_AVAILABLE = False

    # Stubs so callers that import these names don't crash
    class _Stub:
        def labels(self, **_: object) -> _Stub:
            return self

        def inc(self, _: float = 1) -> None:
            pass

        def observe(self, _: float) -> None:
            pass

        def set(self, _: float) -> None:
            pass

    HTTP_REQUESTS = _Stub()  # type: ignore[assignment]
    HTTP_LATENCY = _Stub()  # type: ignore[assignment]
    PREDICTION_SCORE = _Stub()  # type: ignore[assignment]
    CACHE_OPERATIONS = _Stub()  # type: ignore[assignment]
    MODEL_LOADED = _Stub()  # type: ignore[assignment]
    RATE_LIMIT_REJECTED = _Stub()  # type: ignore[assignment]


_EXCLUDED_PATHS = {"/metrics", "/health", "/ready", "/docs", "/openapi.json", "/redoc"}


def _normalise_path(request: Request) -> str:
    """Return the FastAPI route template for the request path.

    Avoids high cardinality from UUID path parameters by using the route
    pattern (``/patient/{patient_id}/family-risk-profile``) instead of
    the raw URL.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Normalised path string.
    """
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", request.url.path)
    return request.url.path


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that records Prometheus metrics per request.

    Silently skips metric recording when ``prometheus_client`` is not
    installed so the API stays functional in environments without it.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Record latency and request count, then pass to next handler.

        Args:
            request: Incoming request.
            call_next: Next middleware / route handler.

        Returns:
            Downstream response.
        """
        if not _PROMETHEUS_AVAILABLE or request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        path = _normalise_path(request)
        method = request.method
        start = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start
        status = str(response.status_code)

        HTTP_REQUESTS.labels(method=method, path=path, status=status).inc()
        HTTP_LATENCY.labels(method=method, path=path).observe(duration)

        return response
