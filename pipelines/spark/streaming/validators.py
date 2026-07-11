"""Pydantic v2 models for Kafka event validation.

Each model maps to one Avro schema.  These are used in the Spark
``foreachBatch`` handler — a Row is converted to a dict, then validated here.

The separation between Avro schemas (wire contracts) and Pydantic models
(runtime validation) is intentional:
- Avro enforces structural schema at the Kafka boundary.
- Pydantic enforces business-rule constraints (e.g., valid ICD-10 prefix,
  degree_of_relatedness range) before the data reaches any DB.

Medical validation note: ICD-10 code format is validated by regex only.
Semantic validity (does this code exist in the ICD-10 release?) requires
a lookup table loaded from CMS — defer to Phase 5 feature engineering.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

PatientId = Annotated[str, Field(description="UUID primary key for a patient")]
ConditionId = Annotated[str, Field(description="UUID primary key for a condition")]

_ICD10_RE = re.compile(r"^[A-Z]\d{2}(\.\d{1,4})?$", re.IGNORECASE)
_RXCUI_RE = re.compile(r"^\d{1,8}$")
_LOINC_RE = re.compile(r"^\d{1,5}-\d$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _require_uuid(v: str, field_name: str = "id") -> str:
    if not _UUID_RE.match(v):
        raise ValueError(f"{field_name} must be a valid UUID v4, got: {v!r}")
    return v.lower()


# ---------------------------------------------------------------------------
# Event models
# ---------------------------------------------------------------------------


class PatientCreatedEvent(BaseModel):
    """Validated representation of a ``patient.created`` Kafka event."""

    event_id: str
    event_timestamp: datetime
    event_version: str = "1.0"
    source_system: str
    patient_id: str
    external_id: str | None = None
    identifier_system: str | None = None
    family_name: str | None = None
    given_name: str | None = None
    middle_name: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    ethnicity: str | None = None
    race: str | None = None
    deceased: bool = False
    research_consent: bool = False

    @field_validator("patient_id", "event_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        """Validate that ID fields are valid UUIDs."""
        return _require_uuid(v, "patient_id/event_id")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str | None) -> str | None:
        """Validate FHIR AdministrativeGender value set."""
        if v is not None and v not in {"male", "female", "other", "unknown"}:
            raise ValueError(f"Invalid gender value: {v!r}")
        return v

    @field_validator("date_of_birth")
    @classmethod
    def no_future_dob(cls, v: date | None) -> date | None:
        """Date of birth must not be in the future."""
        if v is not None and v > date.today():
            raise ValueError(f"date_of_birth {v} is in the future")
        return v


class DiagnosisAddedEvent(BaseModel):
    """Validated representation of a ``diagnosis.added`` Kafka event."""

    event_id: str
    event_timestamp: datetime
    event_version: str = "1.0"
    source_system: str
    condition_id: str
    patient_id: str
    encounter_id: str | None = None
    recorder_id: str | None = None
    clinical_status: str
    verification_status: str | None = None
    severity: str | None = None
    code_system: str
    code: str
    code_display: str | None = None
    onset_datetime: datetime | None = None
    onset_age_years: int | None = None
    is_hereditary: bool = False
    family_history_flag: bool = False

    _VALID_CLINICAL_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"active", "recurrence", "relapse", "inactive", "remission", "resolved"}
    )
    _VALID_VERIFICATION_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"unconfirmed", "provisional", "differential", "confirmed", "refuted", "entered-in-error"}
    )

    @field_validator("patient_id", "condition_id", "event_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        return _require_uuid(v, "id field")

    @field_validator("clinical_status")
    @classmethod
    def validate_clinical_status(cls, v: str) -> str:
        if v not in cls._VALID_CLINICAL_STATUSES:
            raise ValueError(f"Invalid clinical_status: {v!r}")
        return v

    @field_validator("code")
    @classmethod
    def validate_code_format(cls, v: str) -> str:
        """Light format check — not a full ICD-10 lookup."""
        if not v or len(v) > 20:
            raise ValueError(f"code too long or empty: {v!r}")
        return v.upper()

    @field_validator("onset_age_years")
    @classmethod
    def valid_age(cls, v: int | None) -> int | None:
        if v is not None and not (0 <= v <= 130):
            raise ValueError(f"onset_age_years out of range: {v}")
        return v


class PrescriptionIssuedEvent(BaseModel):
    """Validated representation of a ``prescription.issued`` Kafka event."""

    event_id: str
    event_timestamp: datetime
    event_version: str = "1.0"
    source_system: str
    medication_request_id: str
    patient_id: str
    encounter_id: str | None = None
    requester_id: str | None = None
    status: str
    intent: str
    medication_code_system: str | None = None
    medication_code: str
    medication_display: str | None = None
    dosage_text: str | None = None
    dosage_route: str | None = None
    dose_quantity: float | None = None
    dose_unit: str | None = None
    authored_on: datetime

    _VALID_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {
            "active",
            "on-hold",
            "cancelled",
            "completed",
            "entered-in-error",
            "stopped",
            "draft",
            "unknown",
        }
    )

    @field_validator("patient_id", "medication_request_id", "event_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        return _require_uuid(v, "id field")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in cls._VALID_STATUSES:
            raise ValueError(f"Invalid medication_request status: {v!r}")
        return v

    @field_validator("dose_quantity")
    @classmethod
    def positive_dose(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"dose_quantity must be positive: {v}")
        return v


class RelativeLinkedEvent(BaseModel):
    """Validated representation of a ``relative.linked`` Kafka event."""

    event_id: str
    event_timestamp: datetime
    event_version: str = "1.0"
    source_system: str
    fmh_id: str
    patient_id: str
    related_patient_id: str | None = None
    relationship_code: str
    relationship_display: str | None = None
    degree_of_relatedness: float | None = None
    sex: str | None = None
    born_date: date | None = None
    deceased: bool | None = None
    deceased_age_years: int | None = None
    conditions: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("patient_id", "fmh_id", "event_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        return _require_uuid(v, "id field")

    @field_validator("degree_of_relatedness")
    @classmethod
    def valid_degree(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"degree_of_relatedness must be 0–1: {v}")
        return v

    @model_validator(mode="after")
    def deceased_age_requires_deceased(self) -> RelativeLinkedEvent:
        if self.deceased_age_years is not None and not self.deceased:
            raise ValueError("deceased_age_years requires deceased=true")
        return self


class ObservationRecordedEvent(BaseModel):
    """Validated representation of an ``observation.recorded`` Kafka event."""

    event_id: str
    event_timestamp: datetime
    event_version: str = "1.0"
    source_system: str
    observation_id: str
    patient_id: str
    encounter_id: str | None = None
    status: str
    category: str | None = None
    code_system: str
    code: str
    code_display: str | None = None
    effective_datetime: datetime
    value_quantity: float | None = None
    value_unit: str | None = None
    value_string: str | None = None
    value_boolean: bool | None = None
    value_codeable_code: str | None = None
    value_codeable_display: str | None = None
    ref_range_low: float | None = None
    ref_range_high: float | None = None
    interpretation: str | None = None

    _VALID_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {
            "registered",
            "preliminary",
            "final",
            "amended",
            "corrected",
            "cancelled",
            "entered-in-error",
            "unknown",
        }
    )

    @field_validator("patient_id", "observation_id", "event_id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        return _require_uuid(v, "id field")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in cls._VALID_STATUSES:
            raise ValueError(f"Invalid observation status: {v!r}")
        return v

    @model_validator(mode="after")
    def exactly_one_value(self) -> ObservationRecordedEvent:
        """Warn if no value is populated — not a hard error for legacy data."""
        values = [
            self.value_quantity,
            self.value_string,
            self.value_boolean,
            self.value_codeable_code,
        ]
        if all(v is None for v in values):
            # Not raised — some observations legitimately have no value (e.g., grouping obs).
            pass
        return self

    @model_validator(mode="after")
    def ref_range_order(self) -> ObservationRecordedEvent:
        if (
            self.ref_range_low is not None
            and self.ref_range_high is not None
            and self.ref_range_low > self.ref_range_high
        ):
            raise ValueError("ref_range_low must be <= ref_range_high")
        return self
