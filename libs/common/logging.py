"""Structured JSON logging configuration for all services.

Call ``configure_logging()`` once at application startup. After that, every
``logging.getLogger(__name__)`` call in any module will emit JSON lines with
PHI automatically redacted.

Structured fields are emitted as a flat JSON object so Loki/ELK can index them
without additional parsing rules.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from libs.common.phi import PhiRedactingFilter


class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    _DEFAULT_FIELDS: tuple[str, ...] = (
        "name",
        "levelname",
        "pathname",
        "lineno",
        "funcName",
        "thread",
        "process",
    )

    def format(self, record: logging.LogRecord) -> str:
        """Serialize ``record`` to a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string terminated by ``\\n``.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # Exception info
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.exc_text:
            payload["exc_text"] = record.exc_text
        if record.stack_info:
            payload["stack_info"] = record.stack_info

        # Extra fields attached via logging.getLogger().info("msg", extra={...})
        skip = {
            "message",
            "msg",
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "id",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
        for key, value in record.__dict__.items():
            if key not in skip and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    service_name: str = "healthcare",
    enable_phi_redaction: bool = True,
) -> None:
    """Configure root logger with JSON formatting and PHI redaction.

    Should be called exactly once at application entry point (FastAPI startup,
    Spark driver main, Airflow DAG top-level, etc.).

    Args:
        level: Log level string — DEBUG, INFO, WARNING, ERROR, CRITICAL.
        service_name: Identifies this service in log aggregation queries.
        enable_phi_redaction: Attach ``PhiRedactingFilter`` to every handler.
            Set ``False`` only in unit tests that assert on exact log content.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any existing handlers (avoid duplicate output in reload scenarios)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())

    if enable_phi_redaction:
        handler.addFilter(PhiRedactingFilter())

    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "boto3", "botocore", "s3transfer", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(service_name).info(
        "Logging configured",
        extra={"service": service_name, "log_level": level},
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Thin wrapper around ``logging.getLogger`` so callers don't need to import
    the standard library directly.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A ``logging.Logger`` instance.
    """
    return logging.getLogger(name)
