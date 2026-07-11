# Tier 3 — Reporting & Compliance

This tier focuses on generating standardized clinical reports, ensuring FHIR R4 interoperability for EHR integrations, and providing robust bulk data import/export pipelines.

## User Review Required

> [!IMPORTANT]
> **PDF Generation Library**: I propose using `fpdf2` for generating the PDF Clinical Reports. It is lightweight, modern, and does not require external system binaries like wkhtmltopdf. Please confirm if this library is acceptable, or if you prefer `reportlab`.

> [!WARNING]
> **FHIR Strictness**: Our initial FHIR implementation will focus on core resource matching for `Patient`, `Condition`, and `Observation`. Let me know if you need full validation against external FHIR profiles, which would require heavier dependencies (like `fhir.resources`). I recommend standard Pydantic representations for this phase.

## Proposed Changes

---

### 1. Clinical Report Generation

We will build the infrastructure to export patient risk profiles as PDFs and monitor the system via a population dashboard.

#### [NEW] `services/api/routers/reports.py`
- `GET /patients/{id}/report/pdf`: Generates and returns a one-page PDF summary including patient demographics, risk score, current conditions, and top SHAP risk factors.

#### [NEW] `services/api/services/pdf_service.py`
- Service class utilizing `fpdf2` to construct the clinical report layout, headers, and tables.

#### [NEW] `services/streamlit/views/population_health.py`
- A new Population Health Dashboard in Streamlit.
- Visualizes aggregate risk distributions, screening rates, and demographic breakdowns.

#### [NEW] `services/streamlit/views/audit_viewer.py`
- A secure UI tab allowing Admins to query the `AuditLog` database table for compliance monitoring (tracking who accessed which PHI).

---

### 2. FHIR R4 Interoperability API

Exposing standard FHIR endpoints to allow EHR systems (like Epic/Cerner) to integrate with our prediction engine.

#### [NEW] `services/api/schemas/fhir_schemas.py`
- Pydantic models representing FHIR R4 standard structures for `Patient`, `Condition`, and `Bundle`.

#### [NEW] `services/api/routers/fhir.py`
- `GET /fhir/Patient/{id}`: Returns the patient in FHIR JSON format.
- `GET /fhir/Condition?patient={id}`: Returns a FHIR Bundle of the patient's conditions.
- `POST /fhir/Bundle`: Endpoint to ingest a transaction bundle containing a patient and their clinical data in one shot.

---

### 3. Data Import / Export

Tools for researchers and bulk data migrations.

#### [NEW] `services/api/routers/export.py`
- `GET /export/patients/deidentified`: Iterates through the database, strips PHI using our existing `libs.common.deidentification`, and streams a CSV/JSON file of research-ready data.

#### [NEW] `services/api/routers/import_data.py`
- `POST /import/csv`: Upload endpoint that accepts a CSV file of patients/conditions, validates rows, and bulk inserts them into the PostgreSQL database.

#### [MODIFY] `services/streamlit/app.py`
- Update the sidebar navigation to include the new "Population Health", "Audit Logs", and "Data Management" pages.

## Verification Plan

### Automated Tests
- The FHIR endpoints will be validated against strict JSON schema definitions to ensure R4 compliance.
- The `deidentified` export will be tested to guarantee absolutely no PHI (names, exact DOBs, raw UUIDs) leaks into the export payload.

### Manual Verification
- We will use the Streamlit interface to manually upload a test CSV file and verify the patients populate the system.
- We will generate a PDF report for a high-risk patient and verify the SHAP factors and formatting.
- We will retrieve a FHIR Bundle via the API and test it against a public FHIR validator.
