"""CRUD request/response schemas for Patient, Condition, Family, Medication.

All schemas use Pydantic v2 with strict validation.  PHI fields are present
in create/update requests (server-side encryption is handled by the service
layer) but are never exposed in list responses for researcher roles.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

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
    middle_name: str | None = Field(default=None, max_length=255)
    date_of_birth: date
    gender: str = Field(..., pattern=r"^(male|female|other|unknown)$")
    ethnicity: str | None = Field(default=None, max_length=100)
    race: str | None = Field(default=None, max_length=100)
    phone: str | None = Field(default=None, max_length=50)
    email: str | None = Field(default=None, max_length=255)
    address_line: str | None = Field(default=None, max_length=500)
    city: str | None = Field(default=None, max_length=255)
    state: str | None = Field(default=None, max_length=100)
    postal_code: str | None = Field(default=None, max_length=20)
    country: str = Field(default="US", max_length=100)
    language: str = Field(default="en", max_length=10)
    external_id: str | None = Field(default=None, max_length=255)
    identifier_system: str | None = Field(default=None, max_length=255)


class PatientUpdate(BaseModel):
    """Request body for PUT /patients/{id}.  All fields optional."""

    model_config = ConfigDict(str_strip_whitespace=True)

    given_name: str | None = Field(default=None, max_length=255)
    family_name: str | None = Field(default=None, max_length=255)
    middle_name: str | None = Field(default=None, max_length=255)
    date_of_birth: date | None = None
    gender: str | None = Field(default=None, pattern=r"^(male|female|other|unknown)$")
    ethnicity: str | None = Field(default=None, max_length=100)
    race: str | None = Field(default=None, max_length=100)
    phone: str | None = Field(default=None, max_length=50)
    email: str | None = Field(default=None, max_length=255)
    address_line: str | None = Field(default=None, max_length=500)
    city: str | None = Field(default=None, max_length=255)
    state: str | None = Field(default=None, max_length=100)
    postal_code: str | None = Field(default=None, max_length=20)
    country: str | None = Field(default=None, max_length=100)
    language: str | None = Field(default=None, max_length=10)
    deceased: bool | None = None
    deceased_date: date | None = None
    research_consent: bool | None = None


class PatientResponse(BaseModel):
    """Patient record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    external_id: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    middle_name: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    ethnicity: str | None = None
    race: str | None = None
    phone: str | None = None
    email: str | None = None
    address_line: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country: str | None = None
    language: str | None = None
    deceased: bool = False
    research_consent: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PatientSummaryResponse(BaseModel):
    """Full clinical summary for a patient."""

    patient: PatientResponse
    conditions: list[ConditionResponse] = []
    medications: list[MedicationResponse] = []
    family_members: list[FamilyMemberResponse] = []
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
    code_display: str | None = Field(default=None, max_length=500)
    code_text: str | None = Field(default=None, max_length=500)
    clinical_status: str = Field(
        default="active",
        pattern=r"^(active|confirmed|recurrence|relapse|inactive|remission|resolved)$",
    )
    verification_status: str | None = Field(
        default="confirmed",
        pattern=r"^(unconfirmed|provisional|differential|confirmed|refuted|entered-in-error)$",
    )
    severity: str | None = Field(
        default=None,
        pattern=r"^(severe|moderate|mild)$",
    )
    is_hereditary: bool = Field(default=False, description="Is this a hereditary condition?")
    onset_datetime: datetime | None = None
    onset_age_years: int | None = Field(default=None, ge=0, le=150)


class ConditionUpdate(BaseModel):
    """Request body for PUT /conditions/{id}."""

    clinical_status: str | None = Field(
        default=None,
        pattern=r"^(active|confirmed|recurrence|relapse|inactive|remission|resolved)$",
    )
    verification_status: str | None = Field(
        default=None,
        pattern=r"^(unconfirmed|provisional|differential|confirmed|refuted|entered-in-error)$",
    )
    severity: str | None = Field(
        default=None,
        pattern=r"^(severe|moderate|mild)$",
    )
    is_hereditary: bool | None = None
    abatement_datetime: datetime | None = None


class ConditionResponse(BaseModel):
    """Condition record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    code: str
    code_system: str
    code_display: str | None = None
    code_text: str | None = None
    clinical_status: str
    verification_status: str | None = None
    severity: str | None = None
    is_hereditary: bool = False
    family_history_flag: bool = False
    onset_datetime: datetime | None = None
    onset_age_years: int | None = None
    abatement_datetime: datetime | None = None
    created_at: datetime | None = None


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
    relationship_display: str | None = Field(default=None, max_length=100)
    degree_of_relatedness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Wright coefficient: 0.5=1st degree, 0.25=2nd, 0.125=3rd",
    )
    related_patient_id: uuid.UUID | None = Field(
        default=None,
        description="Link to existing patient in system (if applicable)",
    )
    sex: str | None = Field(default=None, max_length=20)
    born_date: date | None = None
    deceased: bool | None = None
    deceased_age_years: int | None = Field(default=None, ge=0, le=150)
    conditions: list[dict[str, Any]] | None = Field(
        default=None,
        description="FHIR-shaped condition objects: [{code, outcome, onset}]",
    )
    status: str = Field(
        default="completed", pattern=r"^(partial|completed|entered-in-error|health-unknown)$"
    )


class FamilyMemberUpdate(BaseModel):
    """Request body for PUT /family/{id}."""

    relationship_code: str | None = Field(default=None, max_length=50)
    relationship_display: str | None = Field(default=None, max_length=100)
    degree_of_relatedness: float | None = Field(default=None, ge=0.0, le=1.0)
    related_patient_id: uuid.UUID | None = None
    deceased: bool | None = None
    deceased_age_years: int | None = Field(default=None, ge=0, le=150)
    conditions: list[dict[str, Any]] | None = None
    status: str | None = Field(
        default=None,
        pattern=r"^(partial|completed|entered-in-error|health-unknown)$",
    )


class FamilyMemberResponse(BaseModel):
    """Family member record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    related_patient_id: uuid.UUID | None = None
    relationship_code: str
    relationship_display: str | None = None
    degree_of_relatedness: float | None = None
    sex: str | None = None
    born_date: date | None = None
    deceased: bool | None = None
    deceased_age_years: int | None = None
    conditions: list[dict[str, Any]] | None = None
    status: str
    neo4j_synced: bool = False
    created_at: datetime | None = None


# ── Medication ────────────────────────────────────────────────────────────────


class MedicationCreate(BaseModel):
    """Request body for POST /patients/{id}/medications."""

    model_config = ConfigDict(str_strip_whitespace=True)

    medication_code: str = Field(..., min_length=1, max_length=50, description="RxNorm RXCUI")
    medication_code_system: str = Field(
        default="http://www.nlm.nih.gov/research/umls/rxnorm",
        max_length=255,
    )
    medication_display: str | None = Field(default=None, max_length=500)
    status: str = Field(
        default="active",
        pattern=r"^(active|on-hold|cancelled|completed|entered-in-error|stopped|draft|unknown)$",
    )
    intent: str = Field(
        default="order",
        pattern=r"^(proposal|plan|order|original-order|reflex-order|filler-order|instance-order|option)$",
    )
    dosage_text: str | None = Field(default=None, max_length=500)
    dosage_timing: str | None = Field(default=None, max_length=255)
    dosage_route: str | None = Field(default=None, max_length=100)
    dose_quantity: float | None = None
    dose_unit: str | None = Field(default=None, max_length=50)
    authored_on: datetime = Field(default_factory=datetime.utcnow)


class MedicationUpdate(BaseModel):
    """Request body for PUT /medications/{id}."""

    status: str | None = Field(
        default=None,
        pattern=r"^(active|on-hold|cancelled|completed|entered-in-error|stopped|draft|unknown)$",
    )
    dosage_text: str | None = Field(default=None, max_length=500)
    dosage_timing: str | None = Field(default=None, max_length=255)
    dose_quantity: float | None = None
    dose_unit: str | None = Field(default=None, max_length=50)


class MedicationResponse(BaseModel):
    """Medication record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    medication_code: str
    medication_code_system: str | None = None
    medication_display: str | None = None
    status: str
    intent: str
    dosage_text: str | None = None
    dosage_timing: str | None = None
    dosage_route: str | None = None
    dose_quantity: float | None = None
    dose_unit: str | None = None
    authored_on: datetime | None = None
    created_at: datetime | None = None


# ── Encounter ─────────────────────────────────────────────────────────────────


class EncounterCreate(BaseModel):
    """Request body for POST /patients/{id}/encounters."""

    model_config = ConfigDict(str_strip_whitespace=True)

    encounter_class: str = Field(
        default="AMB",
        pattern=r"^(AMB|IMP|EMER|HH|VR|SS)$",
        description="HL7 v3 ActCode: AMB=ambulatory, IMP=inpatient, EMER=emergency, HH=home health",
    )
    type_code: str | None = Field(default=None, max_length=100)
    type_display: str | None = Field(default=None, max_length=255)
    service_type: str | None = Field(default=None, max_length=255)
    facility_name: str | None = Field(default=None, max_length=255)
    facility_id: str | None = Field(default=None, max_length=255)


class EncounterUpdate(BaseModel):
    """Request body for PUT /encounters/{id}."""

    status: str | None = Field(
        default=None,
        pattern=r"^(planned|arrived|triaged|in-progress|onleave|finished|cancelled|entered-in-error|unknown)$",
    )
    encounter_class: str | None = Field(
        default=None,
        pattern=r"^(AMB|IMP|EMER|HH|VR|SS)$",
    )
    type_code: str | None = Field(default=None, max_length=100)
    type_display: str | None = Field(default=None, max_length=255)
    facility_name: str | None = Field(default=None, max_length=255)


class EncounterResponse(BaseModel):
    """Encounter record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    status: str
    encounter_class: str | None = None
    type_code: str | None = None
    type_display: str | None = None
    service_type: str | None = None
    facility_name: str | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    created_at: datetime | None = None


class EncounterDetailResponse(BaseModel):
    """Encounter with linked clinical data."""

    encounter: EncounterResponse
    conditions: list[ConditionResponse] = []
    observations: list[ObservationResponse] = []
    medications: list[MedicationResponse] = []


# ── Observation ───────────────────────────────────────────────────────────────


class ObservationCreate(BaseModel):
    """Request body for POST /patients/{id}/observations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    encounter_id: uuid.UUID | None = None
    code: str = Field(..., min_length=1, max_length=50, description="LOINC or SNOMED code")
    code_system: str = Field(
        default="http://loinc.org",
        max_length=255,
    )
    code_display: str | None = Field(default=None, max_length=500)
    category: str | None = Field(
        default="vital-signs",
        pattern=r"^(vital-signs|laboratory|imaging|exam|survey|social-history|activity)$",
    )
    status: str = Field(
        default="final",
        pattern=r"^(registered|preliminary|final|amended|corrected|cancelled|entered-in-error|unknown)$",
    )
    effective_datetime: datetime = Field(default_factory=datetime.utcnow)
    value_quantity: float | None = None
    value_unit: str | None = Field(default=None, max_length=50)
    value_string: str | None = Field(default=None, max_length=500)
    value_boolean: bool | None = None
    interpretation: str | None = Field(
        default=None,
        max_length=10,
        description="HL7 v3: H=high, L=low, N=normal, A=abnormal",
    )
    ref_range_low: float | None = None
    ref_range_high: float | None = None


class VitalsCreate(BaseModel):
    """Convenience schema for recording common vitals in one call."""

    model_config = ConfigDict(str_strip_whitespace=True)

    encounter_id: uuid.UUID | None = None
    effective_datetime: datetime = Field(default_factory=datetime.utcnow)
    systolic_bp: float | None = Field(default=None, ge=50, le=300, description="mmHg")
    diastolic_bp: float | None = Field(default=None, ge=20, le=200, description="mmHg")
    heart_rate: float | None = Field(default=None, ge=20, le=300, description="bpm")
    temperature: float | None = Field(default=None, ge=30, le=45, description="°C")
    spo2: float | None = Field(default=None, ge=50, le=100, description="%")
    weight: float | None = Field(default=None, ge=0.5, le=500, description="kg")
    height: float | None = Field(default=None, ge=20, le=300, description="cm")


class ObservationUpdate(BaseModel):
    """Request body for PUT /observations/{id}."""

    status: str | None = Field(
        default=None,
        pattern=r"^(registered|preliminary|final|amended|corrected|cancelled|entered-in-error|unknown)$",
    )
    value_quantity: float | None = None
    value_unit: str | None = Field(default=None, max_length=50)
    value_string: str | None = Field(default=None, max_length=500)
    interpretation: str | None = Field(default=None, max_length=10)


class ObservationResponse(BaseModel):
    """Observation record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    encounter_id: uuid.UUID | None = None
    status: str
    category: str | None = None
    code: str
    code_system: str
    code_display: str | None = None
    effective_datetime: datetime | None = None
    value_quantity: float | None = None
    value_unit: str | None = None
    value_string: str | None = None
    value_boolean: bool | None = None
    interpretation: str | None = None
    ref_range_low: float | None = None
    ref_range_high: float | None = None
    created_at: datetime | None = None


# ── Batch Screening ───────────────────────────────────────────────────────────


class BatchScreenRequest(BaseModel):
    """Request body for POST /predict/batch-screen."""

    patient_ids: list[uuid.UUID] | None = Field(
        default=None, max_length=500, description="Explicit patient UUIDs (max 500)"
    )
    filter_gender: str | None = Field(default=None, pattern=r"^(male|female|other|unknown)$")
    filter_min_age: int | None = Field(default=None, ge=0, le=150)
    filter_max_age: int | None = Field(default=None, ge=0, le=150)
    include_shap: bool = Field(default=False)
    top_n_factors: int = Field(default=3, ge=1, le=10)


class BatchScreenJobResponse(BaseModel):
    """Response for POST /predict/batch-screen (HTTP 202)."""

    job_id: str
    status: str  # pending | running | completed | failed
    total_patients: int
    progress: int = 0
    message: str = ""


class BatchScreenPatientResult(BaseModel):
    """Single patient result within a batch screening job."""

    patient_id: uuid.UUID
    risk_score: float
    risk_tier: str
    shap_factors: list[dict[str, Any]] | None = None


class BatchScreenResultResponse(BaseModel):
    """Response for GET /predict/batch-screen/{job_id}."""

    job_id: str
    status: str
    total_patients: int
    progress: int
    results: list[BatchScreenPatientResult] = []
    started_at: str | None = None
    completed_at: str | None = None
    message: str = ""


# ── Risk History ──────────────────────────────────────────────────────────────


class RiskHistoryEntry(BaseModel):
    """Single prediction log entry."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    risk_score: float
    risk_tier: str
    model_name: str
    model_version: str
    feature_date: str
    shap_top_factors: dict[str, Any] | None = None
    source: str
    predicted_at: datetime | None = None


class RiskTrendResponse(BaseModel):
    """Risk trend analysis for a patient."""

    patient_id: uuid.UUID
    current_score: float | None = None
    previous_score: float | None = None
    trend: str  # improving | worsening | stable | insufficient_data
    change_pct: float | None = None
    total_predictions: int
    history: list[RiskHistoryEntry] = []


# Forward reference resolution
PatientSummaryResponse.model_rebuild()
EncounterDetailResponse.model_rebuild()
