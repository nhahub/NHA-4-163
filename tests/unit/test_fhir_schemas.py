"""Unit tests for the FHIR R4 interoperability layer (Tier 3).

Covers:
- Resource conversion from ORM instances (Patient, Condition, Observation).
- Bundle searchset wrapping.
- Bundle ingestion helpers (_extract_patient, _extract_condition) and their
  robustness to partial/malformed FHIR input.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from libs.common.models.condition import ClinicalStatus, VerificationStatus
from libs.common.models.observation import ObservationStatus
from libs.common.models.patient import AdministrativeGender
from services.api.routers.fhir import _extract_condition, _extract_patient, _parse_fhir_gender
from services.api.schemas.fhir_schemas import (
    FHIRBundle,
    FHIRCondition,
    FHIRObservation,
    FHIRPatient,
)

_PID = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _fake_patient() -> SimpleNamespace:
    return SimpleNamespace(
        id=_PID,
        external_id="MRN-001",
        identifier_system="http://hospital.org/mrn",
        given_name="Alice",
        middle_name="Q",
        family_name="Smith",
        gender=AdministrativeGender.FEMALE,
        date_of_birth=date(1980, 5, 3),
        deceased=False,
        deleted_at=None,
    )


class TestFhirPatient:
    def test_basic_fields(self) -> None:
        fhir = FHIRPatient.from_orm_patient(_fake_patient())
        assert fhir.resourceType == "Patient"
        assert fhir.id == str(_PID)
        assert fhir.gender == "female"
        assert fhir.birthDate == "1980-05-03"
        assert fhir.active is True

    def test_name_includes_given_and_middle(self) -> None:
        fhir = FHIRPatient.from_orm_patient(_fake_patient())
        assert fhir.name[0].family == "Smith"
        assert fhir.name[0].given == ["Alice", "Q"]

    def test_identifier_uses_system(self) -> None:
        fhir = FHIRPatient.from_orm_patient(_fake_patient())
        assert fhir.identifier[0].value == "MRN-001"
        assert fhir.identifier[0].system == "http://hospital.org/mrn"

    def test_deleted_patient_is_inactive(self) -> None:
        p = _fake_patient()
        p.deleted_at = datetime.utcnow()
        assert FHIRPatient.from_orm_patient(p).active is False

    def test_none_gender(self) -> None:
        p = _fake_patient()
        p.gender = None
        assert FHIRPatient.from_orm_patient(p).gender is None


class TestFhirCondition:
    def _fake_condition(self) -> SimpleNamespace:
        return SimpleNamespace(
            id=uuid.uuid4(),
            patient_id=_PID,
            code="E11.9",
            code_system="http://hl7.org/fhir/sid/icd-10",
            code_display="Type 2 diabetes mellitus",
            code_text=None,
            clinical_status=ClinicalStatus.ACTIVE,
            verification_status=VerificationStatus.CONFIRMED,
            onset_datetime=datetime(2020, 1, 1, 12, 0),
        )

    def test_condition_code_and_subject(self) -> None:
        fhir = FHIRCondition.from_orm_condition(self._fake_condition())
        assert fhir.resourceType == "Condition"
        assert fhir.code.coding[0].code == "E11.9"
        assert fhir.subject.reference == f"Patient/{_PID}"
        assert fhir.clinicalStatus.coding[0].code == "active"

    def test_condition_without_verification(self) -> None:
        c = self._fake_condition()
        c.verification_status = None
        assert FHIRCondition.from_orm_condition(c).verificationStatus is None


class TestFhirObservation:
    def test_observation_quantity(self) -> None:
        obs = SimpleNamespace(
            id=uuid.uuid4(),
            patient_id=_PID,
            status=ObservationStatus.FINAL,
            category="vital-signs",
            code="8867-4",
            code_system="http://loinc.org",
            code_display="Heart rate",
            effective_datetime=datetime(2021, 6, 1, 9, 30),
            value_quantity=72.0,
            value_unit="beats/minute",
            value_unit_system="http://unitsofmeasure.org",
            value_string=None,
        )
        fhir = FHIRObservation.from_orm_observation(obs)
        assert fhir.status == "final"
        assert fhir.valueQuantity.value == 72.0
        assert fhir.category[0].coding[0].code == "vital-signs"


class TestBundle:
    def test_searchset_wraps_resources(self) -> None:
        conditions = [
            FHIRCondition(id="1", code=None, subject=None),
            FHIRCondition(id="2", code=None, subject=None),
        ]
        bundle = FHIRBundle.searchset(conditions)
        assert bundle.type == "searchset"
        assert bundle.total == 2
        assert len(bundle.entry) == 2


class TestIngestHelpers:
    def test_parse_gender_valid_and_invalid(self) -> None:
        assert _parse_fhir_gender("male") == AdministrativeGender.MALE
        assert _parse_fhir_gender("nonsense") == AdministrativeGender.UNKNOWN

    def test_extract_patient_from_fhir_dict(self) -> None:
        resource = {
            "resourceType": "Patient",
            "name": [{"family": "Doe", "given": ["John", "M"]}],
            "gender": "male",
            "birthDate": "1975-11-20",
            "identifier": [{"system": "sys", "value": "X1"}],
        }
        patient = _extract_patient(resource)
        assert patient.family_name == "Doe"
        assert patient.given_name == "John"
        assert patient.middle_name == "M"
        assert patient.gender == AdministrativeGender.MALE
        assert patient.date_of_birth == date(1975, 11, 20)
        assert patient.external_id == "X1"

    def test_extract_patient_tolerates_missing_fields(self) -> None:
        patient = _extract_patient({"resourceType": "Patient"})
        assert patient.gender == AdministrativeGender.UNKNOWN
        assert patient.date_of_birth is None
        assert patient.family_name is None

    def test_extract_condition_maps_code(self) -> None:
        resource = {
            "resourceType": "Condition",
            "code": {"coding": [{"system": "icd", "code": "C50.9", "display": "BrCa"}]},
            "clinicalStatus": {"coding": [{"code": "active"}]},
        }
        cond = _extract_condition(resource, _PID)
        assert cond.code == "C50.9"
        assert cond.code_system == "icd"
        assert cond.clinical_status == ClinicalStatus.ACTIVE
        assert cond.patient_id == _PID

    def test_extract_condition_without_coding_raises(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            _extract_condition({"resourceType": "Condition", "code": {}}, _PID)
        assert exc.value.status_code == 400
