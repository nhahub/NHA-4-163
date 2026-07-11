"""Unit tests for libs/common/models/.

Tests run without a live database — they verify ORM model structure, enum
definitions, constraints, and relationship declarations.  No DB connection
is established; SQLAlchemy's metadata introspection is used instead.
"""

from __future__ import annotations

import sqlalchemy as sa

from libs.common.models import (
    AuditLog,
    Base,
    Condition,
    Encounter,
    FamilyMemberHistory,
    MedicationRequest,
    Observation,
    Patient,
    Physician,
)
from libs.common.models.condition import ClinicalStatus, ConditionSeverity, VerificationStatus
from libs.common.models.encounter import EncounterStatus
from libs.common.models.family_member_history import FamilyMemberHistoryStatus
from libs.common.models.medication_request import MedicationRequestIntent, MedicationRequestStatus
from libs.common.models.observation import ObservationStatus
from libs.common.models.patient import AdministrativeGender

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table(model: type) -> sa.Table:
    """Return the SQLAlchemy Table object for a model class."""
    return model.__table__  # type: ignore[attr-defined]


def _column_names(model: type) -> set[str]:
    return {c.name for c in _table(model).columns}


def _index_names(model: type) -> set[str]:
    return {i.name for i in _table(model).indexes}


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class TestBase:
    def test_all_models_registered(self) -> None:
        table_names = set(Base.metadata.tables.keys())
        expected = {
            "physician",
            "patient",
            "encounter",
            "encounter_participant",
            "condition",
            "observation",
            "medication_request",
            "family_member_history",
            "audit_log",
        }
        assert expected.issubset(table_names)


# ---------------------------------------------------------------------------
# Physician
# ---------------------------------------------------------------------------


class TestPhysicianModel:
    def test_table_name(self) -> None:
        assert Physician.__tablename__ == "physician"

    def test_required_columns_present(self) -> None:
        cols = _column_names(Physician)
        assert {"id", "npi", "created_at", "updated_at"}.issubset(cols)

    def test_npi_is_unique(self) -> None:
        npi_col = _table(Physician).c["npi"]
        assert npi_col.unique

    def test_npi_max_length(self) -> None:
        npi_col = _table(Physician).c["npi"]
        assert npi_col.type.length == 10


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------


class TestPatientModel:
    def test_table_name(self) -> None:
        assert Patient.__tablename__ == "patient"

    def test_phi_columns_present(self) -> None:
        cols = _column_names(Patient)
        phi_fields = {
            "family_name",
            "given_name",
            "date_of_birth",
            "phone",
            "email",
            "address_line",
            "postal_code",
        }
        assert phi_fields.issubset(cols)

    def test_soft_delete_column(self) -> None:
        assert "deleted_at" in _column_names(Patient)

    def test_gender_enum_values(self) -> None:
        assert set(AdministrativeGender) == {"male", "female", "other", "unknown"}

    def test_research_consent_defaults_false(self) -> None:
        col = _table(Patient).c["research_consent"]
        # server_default string contains 'false'
        assert "false" in str(col.server_default.arg).lower()

    def test_external_id_is_unique(self) -> None:
        col = _table(Patient).c["external_id"]
        assert col.unique

    def test_neo4j_node_id_present(self) -> None:
        assert "neo4j_node_id" in _column_names(Patient)

    def test_relationships_declared(self) -> None:
        rel_names = {r.key for r in Patient.__mapper__.relationships}
        assert {
            "conditions",
            "encounters",
            "observations",
            "medication_requests",
            "family_member_histories",
        }.issubset(rel_names)


# ---------------------------------------------------------------------------
# Encounter
# ---------------------------------------------------------------------------


class TestEncounterModel:
    def test_table_name(self) -> None:
        assert Encounter.__tablename__ == "encounter"

    def test_fk_to_patient(self) -> None:
        fks = {fk.target_fullname for fk in _table(Encounter).foreign_keys}
        assert "patient.id" in fks

    def test_status_enum_covers_fhir(self) -> None:
        fhir_statuses = {
            "planned",
            "arrived",
            "triaged",
            "in-progress",
            "onleave",
            "finished",
            "cancelled",
        }
        assert fhir_statuses.issubset({s.value for s in EncounterStatus})

    def test_encounter_participant_table_exists(self) -> None:
        assert "encounter_participant" in Base.metadata.tables

    def test_resource_json_is_jsonb(self) -> None:
        col = _table(Encounter).c["resource_json"]
        assert "JSONB" in type(col.type).__name__.upper()


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------


class TestConditionModel:
    def test_table_name(self) -> None:
        assert Condition.__tablename__ == "condition"

    def test_hereditary_extension_columns(self) -> None:
        cols = _column_names(Condition)
        assert {"is_hereditary", "family_history_flag"}.issubset(cols)

    def test_clinical_status_values(self) -> None:
        assert ClinicalStatus.CONFIRMED == "confirmed"
        assert ClinicalStatus.ACTIVE == "active"

    def test_verification_status_values(self) -> None:
        assert VerificationStatus.CONFIRMED == "confirmed"

    def test_severity_values(self) -> None:
        assert {s.value for s in ConditionSeverity} == {"severe", "moderate", "mild"}

    def test_fk_to_patient_and_encounter(self) -> None:
        fks = {fk.target_fullname for fk in _table(Condition).foreign_keys}
        assert {"patient.id", "encounter.id", "physician.id"}.issubset(fks)

    def test_code_columns_present(self) -> None:
        cols = _column_names(Condition)
        assert {"code_system", "code", "code_display"}.issubset(cols)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class TestObservationModel:
    def test_polymorphic_value_columns(self) -> None:
        cols = _column_names(Observation)
        assert {
            "value_quantity",
            "value_unit",
            "value_string",
            "value_boolean",
            "value_codeable_code",
        }.issubset(cols)

    def test_ref_range_columns(self) -> None:
        cols = _column_names(Observation)
        assert {"ref_range_low", "ref_range_high", "ref_range_text"}.issubset(cols)

    def test_observation_status_covers_fhir(self) -> None:
        fhir = {"registered", "preliminary", "final", "amended", "cancelled"}
        assert fhir.issubset({s.value for s in ObservationStatus})


# ---------------------------------------------------------------------------
# MedicationRequest
# ---------------------------------------------------------------------------


class TestMedicationRequestModel:
    def test_table_name(self) -> None:
        assert MedicationRequest.__tablename__ == "medication_request"

    def test_status_enum(self) -> None:
        assert MedicationRequestStatus.ACTIVE == "active"
        assert MedicationRequestStatus.COMPLETED == "completed"

    def test_intent_enum(self) -> None:
        assert MedicationRequestIntent.ORDER == "order"

    def test_medication_code_not_nullable(self) -> None:
        col = _table(MedicationRequest).c["medication_code"]
        assert not col.nullable

    def test_authored_on_not_nullable(self) -> None:
        col = _table(MedicationRequest).c["authored_on"]
        assert not col.nullable


# ---------------------------------------------------------------------------
# FamilyMemberHistory
# ---------------------------------------------------------------------------


class TestFamilyMemberHistoryModel:
    def test_table_name(self) -> None:
        assert FamilyMemberHistory.__tablename__ == "family_member_history"

    def test_degree_check_constraint_present(self) -> None:
        constraint_names = {c.name for c in _table(FamilyMemberHistory).constraints}
        assert "ck_fmh_degree_range" in constraint_names

    def test_two_fks_to_patient(self) -> None:
        fks = [
            fk
            for fk in _table(FamilyMemberHistory).foreign_keys
            if fk.target_fullname == "patient.id"
        ]
        assert len(fks) == 2, "Expected two FKs to patient (patient_id and related_patient_id)"

    def test_neo4j_synced_defaults_false(self) -> None:
        col = _table(FamilyMemberHistory).c["neo4j_synced"]
        assert "false" in str(col.server_default.arg).lower()

    def test_conditions_column_is_jsonb(self) -> None:
        col = _table(FamilyMemberHistory).c["conditions"]
        assert "JSONB" in type(col.type).__name__.upper()

    def test_status_enum_values(self) -> None:
        assert FamilyMemberHistoryStatus.COMPLETED == "completed"
        assert FamilyMemberHistoryStatus.HEALTH_UNKNOWN == "health-unknown"


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------


class TestAuditLogModel:
    def test_table_name(self) -> None:
        assert AuditLog.__tablename__ == "audit_log"

    def test_bigint_pk(self) -> None:
        col = _table(AuditLog).c["id"]
        assert isinstance(col.type, sa.BigInteger)

    def test_no_updated_at(self) -> None:
        # AuditLog is append-only — no update timestamps.
        assert "updated_at" not in _column_names(AuditLog)

    def test_required_audit_columns(self) -> None:
        cols = _column_names(AuditLog)
        assert {
            "actor_id",
            "actor_type",
            "action",
            "resource_type",
            "outcome",
            "occurred_at",
        }.issubset(cols)

    def test_ip_address_is_inet(self) -> None:
        from sqlalchemy.dialects.postgresql import INET

        col = _table(AuditLog).c["ip_address"]
        assert isinstance(col.type, INET)

    def test_metadata_column_name(self) -> None:
        # The ORM attribute is metadata_ but the column name is metadata.
        assert "metadata" in _column_names(AuditLog)
        assert "metadata_" not in _column_names(AuditLog)
