"""PHI (Protected Health Information) redaction utilities.

Provides a ``redact_phi`` function and a ``PhiRedactingFilter`` for Python's
logging framework. Plug the filter in at application startup so no code path
can accidentally emit PHI to logs.

Medical validation note: the regex patterns here catch common PHI formats but
are not exhaustive. Any new field type that could identify a patient must be
added here AND reviewed by a compliance officer before going to production.
"""

from __future__ import annotations

import logging
import re
from typing import Any

# ---------------------------------------------------------------------------
# Redaction patterns — order matters: more specific patterns first
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, str]] = [
    # Social Security Number
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN-REDACTED]"),
    # US phone numbers
    (r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE-REDACTED]"),
    # Email addresses
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL-REDACTED]"),
    # Date of birth patterns (YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY)
    (r"\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\b", "[DOB-REDACTED]"),
    (r"\b\d{2}[-/]\d{2}[-/](?:19|20)\d{2}\b", "[DOB-REDACTED]"),
    # US ZIP codes (5 or 9 digit)
    (r"\b\d{5}(?:-\d{4})?\b", "[ZIP-REDACTED]"),
    # Patient ID / MRN — UUIDs
    (
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "[ID-REDACTED]",
    ),
    # IP addresses (could identify patient in some contexts)
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP-REDACTED]"),
    # Credit card numbers (Luhn-format 13-19 digits, spaced or dashed)
    (r"\b(?:\d[ -]?){13,19}\b", "[CC-REDACTED]"),
]

_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in _PATTERNS
]


def redact_phi(text: str) -> str:
    """Replace known PHI patterns in ``text`` with placeholder tokens.

    Args:
        text: Raw string that may contain PHI.

    Returns:
        A copy of ``text`` with PHI patterns replaced by redaction tokens.
        Returns an empty string if ``text`` is empty.
    """
    if not text:
        return text
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text


def redact_dict(data: dict[str, Any], phi_keys: frozenset[str] | None = None) -> dict[str, Any]:
    """Recursively redact PHI from a dictionary (e.g., a log record's extra fields).

    Keys listed in ``phi_keys`` have their values replaced wholesale with
    ``[REDACTED]``. All string values are also run through ``redact_phi``.

    Args:
        data: Dictionary that may contain PHI values.
        phi_keys: Set of key names whose values must be fully redacted regardless
            of content. Defaults to a built-in set of common PHI field names.

    Returns:
        A new dict with PHI redacted. Does not mutate the original.
    """
    if phi_keys is None:
        phi_keys = _DEFAULT_PHI_KEYS

    result: dict[str, Any] = {}
    for key, value in data.items():
        if key in phi_keys:
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = redact_dict(value, phi_keys)
        elif isinstance(value, list):
            result[key] = [
                (
                    redact_dict(item, phi_keys)
                    if isinstance(item, dict)
                    else redact_phi(str(item)) if isinstance(item, str) else item
                )
                for item in value
            ]
        elif isinstance(value, str):
            result[key] = redact_phi(value)
        else:
            result[key] = value
    return result


_DEFAULT_PHI_KEYS: frozenset[str] = frozenset(
    {
        "patient_id",
        "patient_name",
        "first_name",
        "last_name",
        "full_name",
        "date_of_birth",
        "dob",
        "ssn",
        "social_security_number",
        "address",
        "street_address",
        "city",
        "state",
        "zip_code",
        "postal_code",
        "phone",
        "phone_number",
        "mobile",
        "email",
        "email_address",
        "medical_record_number",
        "mrn",
        "national_id",
        "passport_number",
        "insurance_id",
        "insurance_number",
        "account_number",
        "diagnosis",
        "condition",
        "medication",
        "prescription",
        "ip_address",
        "device_id",
    }
)


class PhiRedactingFilter(logging.Filter):
    """Logging filter that redacts PHI from every log record.

    Attach to any ``logging.Handler`` or ``logging.Logger``.

    Example::

        import logging
        from libs.common.phi import PhiRedactingFilter

        handler = logging.StreamHandler()
        handler.addFilter(PhiRedactingFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact PHI from the log record message and all string attributes.

        Args:
            record: The log record to filter in-place.

        Returns:
            Always ``True`` — the record is modified but never suppressed.
        """
        record.msg = redact_phi(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_dict(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    redact_phi(str(arg)) if isinstance(arg, str) else arg for arg in record.args
                )
        return True
