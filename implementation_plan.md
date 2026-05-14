# Tier 1 — Core CRUD Implementation Plan

## Goal

Add full Create/Read/Update/Delete operations for Patients, Conditions, Family Relationships, and Medications. This transforms the system from a read-only prediction demo into a functional clinical application.

## Proposed Changes

### Database Layer

#### [NEW] [db.py](file:///d:/Healthcare%20-%20Depi/services/api/db.py)

SQLAlchemy async engine + session factory. Uses the existing `POSTGRES_*` config from `libs/common/config.py`.

- `get_db_session()` — FastAPI dependency yielding an async `AsyncSession`
- `DbSession` — typed `Annotated` dependency (same pattern as `ModelDep`, `CacheDep`)
- Engine configured with connection pooling (`pool_size=5`, `max_overflow=10`)

---

### Pydantic Schemas (Request/Response Models)

#### [NEW] [crud_schemas.py](file:///d:/Healthcare%20-%20Depi/services/api/schemas/crud_schemas.py)

All CRUD request/response models in one file, grouped by entity:

**Patient schemas:**
- `PatientCreate` — required: `given_name`, `family_name`, `date_of_birth`, `gender`
- `PatientUpdate` — all fields optional (partial update)
- `PatientResponse` — full patient record (PHI fields redacted for researcher role)
- `PatientListResponse` — paginated list with total count

**Condition schemas:**
- `ConditionCreate` — required: `code`, `code_system`, `clinical_status`; optional: `is_hereditary`, `severity`, `onset_datetime`
- `ConditionUpdate` — status/severity updates
- `ConditionResponse` — single condition record

**Family schemas:**
- `FamilyMemberCreate` — required: `relationship_code`, `degree_of_relatedness`; optional: `related_patient_id`, `conditions` (JSON)
- `FamilyMemberUpdate` — update relationship details
- `FamilyMemberResponse` — family member record

**Medication schemas:**
- `MedicationCreate` — required: `medication_code`, `status`, `intent`, `authored_on`
- `MedicationUpdate` — status/dosage changes
- `MedicationResponse` — medication record

**Shared:**
- `PaginationParams` — `page`, `page_size` with defaults
- `PaginatedResponse[T]` — generic paginated wrapper

---

### API Routers

#### [NEW] [patient_crud.py](file:///d:/Healthcare%20-%20Depi/services/api/routers/patient_crud.py)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/patients` | Register a new patient |
| `GET` | `/patients` | List patients (paginated, searchable) |
| `GET` | `/patients/{id}` | Get patient by ID |
| `PUT` | `/patients/{id}` | Update patient |
| `DELETE` | `/patients/{id}` | Soft-delete patient |
| `GET` | `/patients/{id}/summary` | Full clinical summary |

#### [NEW] [conditions.py](file:///d:/Healthcare%20-%20Depi/services/api/routers/conditions.py)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/patients/{id}/conditions` | Add a condition/diagnosis |
| `GET` | `/patients/{id}/conditions` | List patient's conditions |
| `PUT` | `/conditions/{id}` | Update a condition |
| `DELETE` | `/conditions/{id}` | Remove a condition |

#### [NEW] [family.py](file:///d:/Healthcare%20-%20Depi/services/api/routers/family.py)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/patients/{id}/family` | Add a family member |
| `GET` | `/patients/{id}/family` | List family members |
| `PUT` | `/family/{id}` | Update a family member |
| `DELETE` | `/family/{id}` | Remove a family link |

#### [NEW] [medications.py](file:///d:/Healthcare%20-%20Depi/services/api/routers/medications.py)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/patients/{id}/medications` | Add a medication |
| `GET` | `/patients/{id}/medications` | List medications |
| `PUT` | `/medications/{id}` | Update a medication |
| `DELETE` | `/medications/{id}` | Remove a medication |

---

### Auth / RBAC Updates

#### [MODIFY] [rbac.py](file:///d:/Healthcare%20-%20Depi/services/api/auth/rbac.py)

Add new permissions:
- `WRITE_PATIENT` — create/update/delete patients (admin, clinician)
- `WRITE_CLINICAL` — create/update conditions, meds, family (admin, clinician)

---

### App Registration

#### [MODIFY] [main.py](file:///d:/Healthcare%20-%20Depi/services/api/main.py)

- Import and register the 4 new routers
- Add `PUT`, `DELETE` to CORS `allow_methods`
- Initialize DB engine in lifespan
- Bump version to `1.0.0`

---

### Streamlit UI

#### [NEW] [patient_management.py](file:///d:/Healthcare%20-%20Depi/services/streamlit/pages/patient_management.py)

New "👤 Patient Management" page with:
- Patient registration form
- Searchable patient table
- Condition entry with ICD-10 code input
- Family member linking
- Medication recording

#### [MODIFY] [app.py](file:///d:/Healthcare%20-%20Depi/services/streamlit/app.py)

- Add new page to sidebar navigation

---

### Dependencies

#### [NEW] [sqlalchemy[asyncio]](file:///d:/Healthcare%20-%20Depi/requirements.txt)

Add `sqlalchemy[asyncio]` and `asyncpg` to requirements for async database operations.

---

## Summary

| Type | Count | Files |
|---|---|---|
| New files | 7 | `db.py`, `crud_schemas.py`, `patient_crud.py`, `conditions.py`, `family.py`, `medications.py`, `patient_management.py` |
| Modified files | 3 | `main.py`, `rbac.py`, `app.py` |
| New dependencies | 2 | `asyncpg`, `sqlalchemy[asyncio]` |

## Verification Plan

### Automated Tests
- Start the Streamlit app and verify the new Patient Management page loads
- Test the API via the browser at `/docs` (Swagger UI)

### Manual Verification
- Create a patient → add conditions → add family members → add medications
- Verify the data appears in the dashboard and risk prediction pages
