"""Unit tests for the de-identified research export (Tier 3).

The critical compliance property: the export must not leak PHI — no raw patient
UUIDs, names, exact dates of birth, or contact details.
"""

from __future__ import annotations

import uuid
from datetime import date

from libs.common.deidentification import is_deidentified
from services.api.routers import export as export_mod
from services.api.routers.export import _EXPORT_COLUMNS, _age_band, _pseudonym

# Direct/quasi identifiers that must never appear as export columns.
_FORBIDDEN_COLUMNS = {
    "id",
    "patient_id",
    "given_name",
    "family_name",
    "middle_name",
    "name",
    "date_of_birth",
    "dob",
    "phone",
    "email",
    "address_line",
    "city",
    "postal_code",
    "external_id",
}


class TestExportColumns:
    def test_no_phi_columns(self) -> None:
        assert _FORBIDDEN_COLUMNS.isdisjoint(set(_EXPORT_COLUMNS))

    def test_research_id_column_present(self) -> None:
        assert "research_id" in _EXPORT_COLUMNS


class TestPseudonym:
    def test_is_deterministic(self) -> None:
        pid = uuid.uuid4()
        assert _pseudonym(pid) == _pseudonym(pid)

    def test_differs_by_patient(self) -> None:
        assert _pseudonym(uuid.uuid4()) != _pseudonym(uuid.uuid4())

    def test_does_not_contain_raw_uuid(self) -> None:
        pid = uuid.uuid4()
        pseudo = _pseudonym(pid)
        assert str(pid) not in pseudo
        assert pseudo.startswith("R-")

    def test_salt_changes_output(self, monkeypatch) -> None:
        pid = uuid.uuid4()
        original = _pseudonym(pid)
        monkeypatch.setattr(export_mod, "_PSEUDONYM_SALT", "a-different-salt")
        assert _pseudonym(pid) != original


class TestAgeBand:
    def test_none_dob(self) -> None:
        assert _age_band(None) is None

    def test_generalises_to_decade(self) -> None:
        # A 45-ish year old should be reported as a decade band, never an exact age.
        band = _age_band(date(1980, 1, 1))
        assert band is not None
        assert band.endswith("s") or band == "90+"

    def test_elderly_collapsed(self) -> None:
        assert _age_band(date(1920, 1, 1)) == "90+"


class TestRowDeidentification:
    def test_synthesised_row_is_deidentified(self) -> None:
        # Build a row of the same shape _build_rows emits and confirm it passes
        # the Safe Harbor heuristic (no raw direct-identifier fields present).
        row = {
            "research_id": _pseudonym(uuid.uuid4()),
            "age_band": _age_band(date(1975, 3, 2)),
            "gender": "female",
            "ethnicity": "Hispanic",
            "race": None,
            "state": "FL",
            "deceased": False,
            "condition_count": 3,
            "hereditary_condition_count": 1,
            "active_medication_count": 2,
        }
        assert is_deidentified(row)
