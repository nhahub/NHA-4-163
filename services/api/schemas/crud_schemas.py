"""CRUD request/response schemas for Patient, Condition, Family, Medication.

All schemas use Pydantic v2 with strict validation.  PHI fields are present
in create/update requests (server-side encryption is handled by the service
layer) but are never exposed in list responses for researcher roles.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Pagination ────────────────────────────────────────────────────────────────

class PaginationParams(BaseModel):
    """Query parameters for paginated list endpoints."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int


# ── Patient ───────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    """Request body for POST /patients."""

    model_config = ConfigDict(str_strip_whitespace=True)

    given_name: str = Field(..., min_length=1, max_length=255)
    family_name: str = Field(..., min_length=1, max_length=255)
    middle_name: Optional[str] = Field(default=None, max_length=255)
    date_of_birth: date
    gender: str = Field(..., pattern=r"^(male|female|other|unknown)$")
    ethnicity: Optional[str] = Field(default=None, max_length=100)
    race: Optional[str] = Field(default=None, max_length=100)
    phone: Optional[str] = Field(default=None, max_length=50)
    email: Optional[str] = Field(default=None, max_length=255)
    address_line: Optional[str] = Field(default=None, max_length=500)
    city: Optional[str] = Field(default=None, max_length=255)
    state: Optional[str] = Field(default=None, max_length=100)
    postal_code: Optional[str] = Field(default=None, max_length=20)
    country: str = Field(default="US", max_length=100)
    language: str = Field(default="en", max_length=10)
    external_id: Optional[str] = Field(default=None, max_length=255)
    identifier_system: Optional[str] = Field(default=None, max_length=255)


class PatientUpdate(BaseModel):
    """Request body for PUT /patients/{id}.  All fields optional."""

    model_config = ConfigDict(str_strip_whitespace=True)

    given_name: Optional[str] = Field(default=None, max_length=255)
    family_name: Optional[str] = Field(default=None, max_length=255)
    middle_name: Optional[str] = Field(default=None, max_length=255)
    date_of_birth: Optional[date] = None
    gender: Optional[str] = Field(default=None, pattern=r"^(male|female|other|unknown)$")
    ethnicity: Optional[str] = Field(default=None, max_length=100)
    race: Optional[str] = Field(default=None, max_length=100)
    phone: Optional[str] = Field(default=None, max_length=50)
    email: Optional[str] = Field(default=None, max_length=255)
    address_line: Optional[str] = Field(default=None, max_length=500)
    city: Optional[str] = Field(default=None, max_length=255)
    state: Optional[str] = Field(default=None, max_length=100)
    postal_code: Optional[str] = Field(default=None, max_length=20)
    country: Optional[str] = Field(default=None, max_length=100)
    language: Optional[str] = Field(default=None, max_length=10)
    deceased: Optional[bool] = None
    deceased_date: Optional[date] = None
    research_consent: Optional[bool] = None


class PatientResponse(BaseModel):
    """Patient record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    external_id: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    middle_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    race: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address_line: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    deceased: bool = False
    research_consent: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PatientSummaryResponse(BaseModel):
    """Full clinical summary for a patient."""

    patient: PatientResponse
    conditions: list["ConditionResponse"] = []
    medications: list["MedicationResponse"] = []
    family_members: list["FamilyMemberResponse"] = []
    condition_count: int = 0
    active_medication_count: int = 0
    family_member_count: int = 0


# ── Condition ─────────────────────────────────────────────────────────────────

class ConditionCreate(BaseModel):
    """Request body for POST /patients/{id}/conditions."""

    model_config = ConfigDict(str_strip_whitespace=True)

    code: str = Field(..., min_length=1, max_length=50, description="ICD-10 or SNOMED code")
    code_system: str = Field(
        default="http://hl7.org/fhir/sid/icd-10",
        max_length=255,
        description="Coding system URI",
    )
    code_display: Optional[str] = Field(default=None, max_length=500)
    code_text: Optional[str] = Field(default=None, max_length=500)
    clinical_status: str = Field(
        default="active",
        pattern=r"^(active|confirmed|recurrence|relapse|inactive|remission|resolved)$",
    )
    verification_status: Optional[str] = Field(
        default="confirmed",
        pattern=r"^(unconfirmed|provisional|differential|confirmed|refuted|entered-in-error)$",
    )
    severity: Optional[str] = Field(
        default=None,
        pattern=r"^(severe|moderate|mild)$",
    )
    is_hereditary: bool = Field(default=False, description="Is this a hereditary condition?")
    onset_datetime: Optional[datetime] = None
    onset_age_years: Optional[int] = Field(default=None, ge=0, le=150)


class ConditionUpdate(BaseModel):
    """Request body for PUT /conditions/{id}."""

    clinical_status: Optional[str] = Field(
        default=None,
        pattern=r"^(active|confirmed|recurrence|relapse|inactive|remission|resolved)$",
    )
    verification_status: Optional[str] = Field(
        default=None,
        pattern=r"^(unconfirmed|provisional|differential|confirmed|refuted|entered-in-error)$",
    )
    severity: Optional[str] = Field(
        default=None,
        pattern=r"^(severe|moderate|mild)$",
    )
    is_hereditary: Optional[bool] = None
    abatement_datetime: Optional[datetime] = None


class ConditionResponse(BaseModel):
    """Condition record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    code: str
    code_system: str
    code_display: Optional[str] = None
    code_text: Optional[str] = None
    clinical_status: str
    verification_status: Optional[str] = None
    severity: Optional[str] = None
    is_hereditary: bool = False
    family_history_flag: bool = False
    onset_datetime: Optional[datetime] = None
    onset_age_years: Optional[int] = None
    abatement_datetime: Optional[datetime] = None
    created_at: Optional[datetime] = None


# ── Family Member ─────────────────────────────────────────────────────────────

class FamilyMemberCreate(BaseModel):
    """Request body for POST /patients/{id}/family."""

    model_config = ConfigDict(str_strip_whitespace=True)

    relationship_code: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="HL7 v3 code: MTH, FTH, SIB, GRPRN, etc.",
    )
    relationship_display: Optional[str] = Field(default=None, max_length=100)
    degree_of_relatedness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Wright coefficient: 0.5=1st degree, 0.25=2nd, 0.125=3rd",
    )
    related_patient_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Link to existing patient in system (if applicable)",
    )
    sex: Optional[str] = Field(default=None, max_length=20)
    born_date: Optional[date] = None
    deceased: Optional[bool] = None
    deceased_age_years: Optional[int] = Field(default=None, ge=0, le=150)
    conditions: Optional[list[dict]] = Field(
        default=None,
        description="FHIR-shaped condition objects: [{code, outcome, onset}]",
    )
    status: str = Field(default="completed", pattern=r"^(partial|completed|entered-in-error|health-unknown)$")


class FamilyMemberUpdate(BaseModel):
    """Request body for PUT /family/{id}."""

    relationship_code: Optional[str] = Field(default=None, max_length=50)
    relationship_display: Optional[str] = Field(default=None, max_length=100)
    degree_of_relatedness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    related_patient_id: Optional[uuid.UUID] = None
    deceased: Optional[bool] = None
    deceased_age_years: Optional[int] = Field(default=None, ge=0, le=150)
    conditions: Optional[list[dict]] = None
    status: Optional[str] = Field(
        default=None,
        pattern=r"^(partial|completed|entered-in-error|health-unknown)$",
    )


class FamilyMemberResponse(BaseModel):
    """Family member record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    related_patient_id: Optional[uuid.UUID] = None
    relationship_code: str
    relationship_display: Optional[str] = None
    degree_of_relatedness: Optional[float] = None
    sex: Optional[str] = None
    born_date: Optional[date] = None
    deceased: Optional[bool] = None
    deceased_age_years: Optional[int] = None
    conditions: Optional[list[dict]] = None
    status: str
    neo4j_synced: bool = False
    created_at: Optional[datetime] = None


# ── Medication ────────────────────────────────────────────────────────────────

class MedicationCreate(BaseModel):
    """Request body for POST /patients/{id}/medications."""

    model_config = ConfigDict(str_strip_whitespace=True)

    medication_code: str = Field(..., min_length=1, max_length=50, description="RxNorm RXCUI")
    medication_code_system: str = Field(
        default="http://www.nlm.nih.gov/research/umls/rxnorm",
        max_length=255,
    )
    medication_display: Optional[str] = Field(default=None, max_length=500)
    status: str = Field(
        default="active",
        pattern=r"^(active|on-hold|cancelled|completed|entered-in-error|stopped|draft|unknown)$",
    )
    intent: str = Field(
        default="order",
        pattern=r"^(proposal|plan|order|original-order|reflex-order|filler-order|instance-order|option)$",
    )
    dosage_text: Optional[str] = Field(default=None, max_length=500)
    dosage_timing: Optional[str] = Field(default=None, max_length=255)
    dosage_route: Optional[str] = Field(default=None, max_length=100)
    dose_quantity: Optional[float] = None
    dose_unit: Optional[str] = Field(default=None, max_length=50)
    authored_on: datetime = Field(default_factory=datetime.utcnow)


class MedicationUpdate(BaseModel):
    """Request body for PUT /medications/{id}."""

    status: Optional[str] = Field(
        default=None,
        pattern=r"^(active|on-hold|cancelled|completed|entered-in-error|stopped|draft|unknown)$",
    )
    dosage_text: Optional[str] = Field(default=None, max_length=500)
    dosage_timing: Optional[str] = Field(default=None, max_length=255)
    dose_quantity: Optional[float] = None
    dose_unit: Optional[str] = Field(default=None, max_length=50)


class MedicationResponse(BaseModel):
    """Medication record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    medication_code: str
    medication_code_system: Optional[str] = None
    medication_display: Optional[str] = None
    status: str
    intent: str
    dosage_text: Optional[str] = None
    dosage_timing: Optional[str] = None
    dosage_route: Optional[str] = None
    dose_quantity: Optional[float] = None
    dose_unit: Optional[str] = None
    authored_on: Optional[datetime] = None
    created_at: Optional[datetime] = None


# Forward reference resolution
PatientSummaryResponse.model_rebuild()
