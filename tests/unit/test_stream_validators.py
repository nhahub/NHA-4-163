"""Unit tests for pipelines/spark/streaming/validators.py.

No Spark or external services required — pure Pydantic validation tests.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from pipelines.spark.streaming.validators import (
    DiagnosisAddedEvent,
    ObservationRecordedEvent,
    PatientCreatedEvent,
    PrescriptionIssuedEvent,
    RelativeLinkedEvent,
)

_NOW = datetime.now(UTC)
_VALID_UUID = "550e8400-e29b-41d4-a716-446655440000"
_VALID_UUID2 = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


# ---------------------------------------------------------------------------
# PatientCreatedEvent
# ---------------------------------------------------------------------------


class TestPatientCreatedEvent:
    def _base(self, **overrides: object) -> dict:
        return {
            "event_id": _VALID_UUID,
            "event_timestamp": _NOW,
            "source_system": "test",
            "patient_id": _VALID_UUID,
            **overrides,
        }

    def test_valid_minimal(self) -> None:
        e = PatientCreatedEvent(**self._base())
        assert e.patient_id == _VALID_UUID.lower()

    def test_invalid_patient_id_not_uuid(self) -> None:
        with pytest.raises(Exception, match="UUID"):
            PatientCreatedEvent(**self._base(patient_id="not-a-uuid"))

    def test_invalid_gender_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PatientCreatedEvent(**self._base(gender="alien"))

    def test_valid_gender_values(self) -> None:
        for gender in ("male", "female", "other", "unknown"):
            e = PatientCreatedEvent(**self._base(gender=gender))
            assert e.gender == gender

    def test_future_dob_rejected(self) -> None:
        with pytest.raises(Exception, match="future"):
            PatientCreatedEvent(**self._base(date_of_birth=date(2099, 1, 1)))

    def test_past_dob_accepted(self) -> None:
        e = PatientCreatedEvent(**self._base(date_of_birth=date(1985, 6, 15)))
        assert e.date_of_birth == date(1985, 6, 15)

    def test_none_dob_accepted(self) -> None:
        e = PatientCreatedEvent(**self._base(date_of_birth=None))
        assert e.date_of_birth is None

    def test_research_consent_defaults_false(self) -> None:
        e = PatientCreatedEvent(**self._base())
        assert e.research_consent is False


# ---------------------------------------------------------------------------
# DiagnosisAddedEvent
# ---------------------------------------------------------------------------


class TestDiagnosisAddedEvent:
    def _base(self, **overrides: object) -> dict:
        return {
            "event_id": _VALID_UUID,
            "event_timestamp": _NOW,
            "source_system": "test",
            "condition_id": _VALID_UUID,
            "patient_id": _VALID_UUID2,
            "clinical_status": "active",
            "code_system": "http://hl7.org/fhir/sid/icd-10",
            "code": "I10",
            **overrides,
        }

    def test_valid_minimal(self) -> None:
        e = DiagnosisAddedEvent(**self._base())
        assert e.code == "I10"

    def test_invalid_clinical_status(self) -> None:
        with pytest.raises(ValidationError):
            DiagnosisAddedEvent(**self._base(clinical_status="cured"))

    def test_all_valid_clinical_statuses(self) -> None:
        for status in ("active", "recurrence", "relapse", "inactive", "remission", "resolved"):
            e = DiagnosisAddedEvent(**self._base(clinical_status=status))
            assert e.clinical_status == status

    def test_onset_age_out_of_range(self) -> None:
        with pytest.raises(Exception, match="range"):
            DiagnosisAddedEvent(**self._base(onset_age_years=200))

    def test_onset_age_zero_valid(self) -> None:
        e = DiagnosisAddedEvent(**self._base(onset_age_years=0))
        assert e.onset_age_years == 0

    def test_hereditary_defaults_false(self) -> None:
        e = DiagnosisAddedEvent(**self._base())
        assert e.is_hereditary is False


# ---------------------------------------------------------------------------
# PrescriptionIssuedEvent
# ---------------------------------------------------------------------------


class TestPrescriptionIssuedEvent:
    def _base(self, **overrides: object) -> dict:
        return {
            "event_id": _VALID_UUID,
            "event_timestamp": _NOW,
            "source_system": "test",
            "medication_request_id": _VALID_UUID,
            "patient_id": _VALID_UUID2,
            "status": "active",
            "intent": "order",
            "medication_code": "198440",
            "authored_on": _NOW,
            **overrides,
        }

    def test_valid_minimal(self) -> None:
        e = PrescriptionIssuedEvent(**self._base())
        assert e.medication_code == "198440"

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            PrescriptionIssuedEvent(**self._base(status="prescribed"))

    def test_negative_dose_quantity_rejected(self) -> None:
        with pytest.raises(Exception, match="positive"):
            PrescriptionIssuedEvent(**self._base(dose_quantity=-5.0))

    def test_zero_dose_quantity_rejected(self) -> None:
        with pytest.raises(Exception, match="positive"):
            PrescriptionIssuedEvent(**self._base(dose_quantity=0.0))

    def test_positive_dose_accepted(self) -> None:
        e = PrescriptionIssuedEvent(**self._base(dose_quantity=10.5))
        assert e.dose_quantity == 10.5


# ---------------------------------------------------------------------------
# RelativeLinkedEvent
# ---------------------------------------------------------------------------


class TestRelativeLinkedEvent:
    def _base(self, **overrides: object) -> dict:
        return {
            "event_id": _VALID_UUID,
            "event_timestamp": _NOW,
            "source_system": "test",
            "fmh_id": _VALID_UUID,
            "patient_id": _VALID_UUID2,
            "relationship_code": "MTH",
            **overrides,
        }

    def test_valid_minimal(self) -> None:
        e = RelativeLinkedEvent(**self._base())
        assert e.relationship_code == "MTH"

    def test_degree_out_of_range(self) -> None:
        with pytest.raises(Exception, match="0.1"):
            RelativeLinkedEvent(**self._base(degree_of_relatedness=1.5))

    def test_degree_boundary_values(self) -> None:
        for val in (0.0, 0.5, 1.0):
            e = RelativeLinkedEvent(**self._base(degree_of_relatedness=val))
            assert e.degree_of_relatedness == val

    def test_deceased_age_without_deceased_rejected(self) -> None:
        with pytest.raises(Exception, match="deceased"):
            RelativeLinkedEvent(**self._base(deceased_age_years=70, deceased=False))

    def test_deceased_age_with_deceased_accepted(self) -> None:
        e = RelativeLinkedEvent(**self._base(deceased_age_years=70, deceased=True))
        assert e.deceased_age_years == 70


# ---------------------------------------------------------------------------
# ObservationRecordedEvent
# ---------------------------------------------------------------------------


class TestObservationRecordedEvent:
    def _base(self, **overrides: object) -> dict:
        return {
            "event_id": _VALID_UUID,
            "event_timestamp": _NOW,
            "source_system": "test",
            "observation_id": _VALID_UUID,
            "patient_id": _VALID_UUID2,
            "status": "final",
            "code_system": "http://loinc.org",
            "code": "8302-2",
            "effective_datetime": _NOW,
            **overrides,
        }

    def test_valid_minimal(self) -> None:
        e = ObservationRecordedEvent(**self._base())
        assert e.code == "8302-2"

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            ObservationRecordedEvent(**self._base(status="done"))

    def test_ref_range_inverted_rejected(self) -> None:
        with pytest.raises(Exception, match="ref_range"):
            ObservationRecordedEvent(**self._base(ref_range_low=100.0, ref_range_high=50.0))

    def test_ref_range_equal_accepted(self) -> None:
        e = ObservationRecordedEvent(**self._base(ref_range_low=50.0, ref_range_high=50.0))
        assert e.ref_range_low == 50.0

    def test_all_value_types_none_accepted(self) -> None:
        e = ObservationRecordedEvent(**self._base())
        assert e.value_quantity is None
        assert e.value_string is None
