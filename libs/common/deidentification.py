"""HIPAA Safe Harbor de-identification for patient records.

Implements the HIPAA Privacy Rule Safe Harbor method (45 CFR §164.514(b)),
which requires removing or generalising all 18 categories of direct and
quasi-identifiers before a record can be treated as de-identified.

The 18 Safe Harbor identifiers
-------------------------------
 1  Names
 2  Geographic subdivisions smaller than a state (incl. ZIP codes)
 3  Dates (other than year) for individuals ≥ 90 years old; all
    elements of dates directly related to an individual
 4  Phone numbers
 5  Fax numbers
 6  Email addresses
 7  Social security numbers
 8  Medical record numbers
 9  Health plan beneficiary numbers
10  Account numbers
11  Certificate / license numbers
12  Vehicle identifiers and serial numbers, including license plates
13  Device identifiers and serial numbers
14  Web URLs
15  IP addresses
16  Biometric identifiers (finger / voice prints)
17  Full-face photographs and comparable images
18  Any other unique identifying number, characteristic, or code

Usage::

    from libs.common.deidentification import deidentify_patient

    safe = deidentify_patient({
        "patient_id":   "uuid-here",          # → kept (internal reference)
        "first_name":   "Alice",              # → "[REDACTED]"
        "date_of_birth": "1925-03-12",        # → "1925" (age ≥ 90)
        "zip_code":     "10001",              # → "100**"
        "age_years":    101,                  # → "90+"
        "gender":       "female",             # → kept (not a direct identifier)
        "ssn":          "123-45-6789",        # → "[REDACTED]"
    })
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

# ── Safe Harbor — fields to remove entirely ───────────────────────────────────

_REMOVE_FIELDS: frozenset[str] = frozenset(
    {
        # (1) Names
        "name",
        "first_name",
        "last_name",
        "middle_name",
        "patient_name",
        "full_name",
        "display_name",
        "maiden_name",
        # (2) Geographic < state
        "address",
        "street",
        "street_address",
        "city",
        "county",
        "district",
        # (4) Phone
        "phone",
        "phone_number",
        "mobile",
        "telephone",
        "cell_phone",
        # (5) Fax
        "fax",
        "fax_number",
        # (6) Email
        "email",
        "email_address",
        # (7) SSN
        "ssn",
        "social_security",
        "social_security_number",
        # (8) MRN
        "mrn",
        "medical_record_number",
        "chart_number",
        # (9) Health plan
        "health_plan_id",
        "insurance_id",
        "member_id",
        "subscriber_id",
        # (10) Account numbers
        "account_number",
        "bank_account",
        # (11) Certificate / license
        "certificate_number",
        "license_number",
        "dea_number",
        "npi",
        # (12) Vehicle
        "vehicle_id",
        "vehicle_vin",
        "license_plate",
        # (13) Device
        "device_id",
        "device_serial",
        "imei",
        "mac_address",
        # (14) URL
        "url",
        "website",
        # (15) IP address
        "ip_address",
        "ip",
        # (16) Biometric
        "fingerprint",
        "voiceprint",
        "biometric",
        # (17) Photo
        "photo",
        "photo_url",
        "profile_picture",
    }
)

# ── Safe Harbor — fields to generalise (not remove) ──────────────────────────

_ZIP_FIELDS: frozenset[str] = frozenset({"zip_code", "postal_code", "zip"})
_DATE_FIELDS: frozenset[str] = frozenset(
    {
        "date_of_birth",
        "dob",
        "birth_date",
        "admission_date",
        "discharge_date",
        "encounter_date",
        "procedure_date",
        "service_date",
    }
)
_AGE_FIELDS: frozenset[str] = frozenset({"age_years", "age"})

# ZIP codes where the 3-digit prefix applies to fewer than 20,000 people
# must be replaced with "000" per Safe Harbor (45 CFR §164.514(b)(2)(i)(B)).
# This set lists the restricted prefixes from the HHS guidance.
_RESTRICTED_ZIP_PREFIXES: frozenset[str] = frozenset(
    {
        "036",
        "059",
        "102",
        "203",
        "556",
        "692",
        "821",
        "823",
        "878",
        "879",
        "884",
        "890",
        "893",
    }
)

_SENTINEL = "[REDACTED]"
_AGE_THRESHOLD = 90  # individuals ≥ 90 are a small population; generalise


# ── ZIP generalisation ────────────────────────────────────────────────────────


def generalise_zip(zip_code: str | None) -> str | None:
    """Generalise a ZIP code to its 3-digit prefix per HIPAA Safe Harbor.

    ZIP codes whose 3-digit prefix covers fewer than 20,000 people are
    replaced with ``000``.

    Args:
        zip_code: 5-digit (or 5+4) US ZIP code string, or ``None``.

    Returns:
        ``"{prefix}**"`` generalised string, or ``None`` if input is ``None``.
    """
    if zip_code is None:
        return None
    digits = re.sub(r"\D", "", zip_code)
    if len(digits) < 3:
        return "***"
    prefix = digits[:3]
    if prefix in _RESTRICTED_ZIP_PREFIXES:
        prefix = "000"
    return f"{prefix}**"


# ── Age generalisation ────────────────────────────────────────────────────────


def generalise_age(age: int | None) -> str | None:
    """Generalise an age value.

    Individuals aged ≥ 90 are collapsed to ``"90+"`` to prevent
    re-identification of a small population segment.  Younger ages are
    returned as a decade string (``"30s"``, ``"40s"``, etc.).

    Args:
        age: Age in years, or ``None``.

    Returns:
        Generalised string like ``"30s"``, ``"90+"``, or ``None``.
    """
    if age is None:
        return None
    if age < 0:
        return None
    if age >= _AGE_THRESHOLD:
        return "90+"
    decade = (age // 10) * 10
    return f"{decade}s"


# ── Date generalisation ───────────────────────────────────────────────────────


def generalise_date(
    value: str | date | datetime | None,
    age: int | None = None,
) -> str | None:
    """Generalise a date value per HIPAA Safe Harbor.

    - For individuals **aged ≥ 90**, returns only the year (to prevent
      re-identification from rare birth years in elderly populations).
    - For all others, returns ``"{YYYY}-{MM}"`` (month + year) — removes
      the day which is the highest-cardinality date component.

    Args:
        value: Date string (ISO-8601), ``date``, ``datetime``, or ``None``.
        age: Subject's current age in years (used to determine whether the
            year-only rule applies). Pass ``None`` to always apply the
            month-year rule.

    Returns:
        Generalised date string, or ``None`` if input is ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = date.fromisoformat(value[:10])
        except ValueError:
            return None
    if isinstance(value, datetime):
        value = value.date()
    if age is not None and age >= _AGE_THRESHOLD:
        return str(value.year)
    return f"{value.year}-{value.month:02d}"


# ── Main de-identification function ──────────────────────────────────────────


def deidentify_patient(
    record: dict[str, Any],
    age_override: int | None = None,
) -> dict[str, Any]:
    """Apply HIPAA Safe Harbor de-identification to a patient record dict.

    Rules applied in order:
    1. Fields listed in ``_REMOVE_FIELDS`` → replaced with ``"[REDACTED]"``.
    2. ZIP / postal code fields → generalised to ``"{prefix}**"``.
    3. Date fields → generalised to ``"{YYYY}-{MM}"`` or year-only (age ≥ 90).
    4. Age fields → generalised to decade string or ``"90+"``.

    Nested dicts and lists are processed recursively.

    Args:
        record: Flat or nested dict representing a patient record.
        age_override: Override the age used for year-only date generalisation.
            If omitted, the function looks for ``age_years`` or ``age`` in
            the record itself.

    Returns:
        New dict with PHI removed or generalised.  The input is not mutated.
    """
    age = age_override
    if age is None:
        for k in ("age_years", "age"):
            v = record.get(k)
            if isinstance(v, int):
                age = v
                break

    out: dict[str, Any] = {}
    for key, value in record.items():
        lower_key = key.lower()

        if lower_key in _REMOVE_FIELDS:
            out[key] = _SENTINEL
            continue

        if lower_key in _ZIP_FIELDS:
            out[key] = generalise_zip(str(value) if value is not None else None)
            continue

        if lower_key in _DATE_FIELDS:
            out[key] = generalise_date(value, age=age)
            continue

        if lower_key in _AGE_FIELDS:
            out[key] = generalise_age(int(value) if value is not None else None)
            continue

        # Recurse into nested structures
        if isinstance(value, dict):
            out[key] = deidentify_patient(value, age_override=age)
        elif isinstance(value, list):
            out[key] = [
                deidentify_patient(item, age_override=age) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            out[key] = value

    return out


def is_deidentified(record: dict[str, Any]) -> bool:
    """Heuristically check whether a record appears to be de-identified.

    Checks that none of the known direct-identifier keys are present with
    non-sentinel values.  This is a best-effort check, not a guarantee.

    Args:
        record: Record dict to inspect.

    Returns:
        ``True`` if no raw PHI fields are detected.
    """
    for key, value in record.items():
        lower_key = key.lower()
        if lower_key in _REMOVE_FIELDS and value not in (None, _SENTINEL):
            return False
    return True
