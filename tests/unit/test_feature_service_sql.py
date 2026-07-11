"""Regression guard: the serving feature SQL must match the canonical ORM schema.

The live feature computation in ``services.api.services.feature_service`` issues
raw Postgres SQL (it deliberately avoids the ORM for latency).  That SQL once
drifted to a superseded *plural* draft schema (``FROM patients`` /
``conditions`` / ``medication_requests`` with an ``icd10_code`` column) that
does not exist in the canonical singular schema, so every live prediction would
have failed with ``UndefinedTable``/``UndefinedColumn``.

These tests parse the actual query strings and assert that every table and
selected column resolves against ``Base.metadata`` — the same metadata Alembic
and the ORM use — so the drift cannot silently return.
"""

from __future__ import annotations

import re

from libs.common.models import Base
from services.api.services import feature_service as fs

_METADATA = Base.metadata


def _from_table(sql: str) -> str:
    """Extract the single ``FROM <table>`` name from a simple SELECT."""
    match = re.search(r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql)
    assert match is not None, f"no FROM clause found in: {sql}"
    return match.group(1)


def _selected_source_columns(sql: str) -> list[str]:
    """Return the source column names of a simple ``SELECT ... FROM`` query.

    Handles ``<col> AS <alias>`` by taking ``<col>`` (the real column), which is
    how ``condition.code`` is exposed as ``icd10_code``.
    """
    select_body = re.search(r"\bSELECT\s+(.*?)\bFROM\b", sql, re.DOTALL | re.IGNORECASE)
    assert select_body is not None, f"no SELECT..FROM in: {sql}"
    columns: list[str] = []
    for raw in select_body.group(1).split(","):
        token = raw.strip()
        # Strip an "<expr> AS <alias>" tail, keep the source expression.
        source = re.split(r"\s+AS\s+", token, flags=re.IGNORECASE)[0].strip()
        columns.append(source)
    return columns


def _assert_query_matches_schema(sql: str, extra_columns: set[str]) -> None:
    """Assert a query's FROM table and its columns exist in the ORM metadata."""
    table_name = _from_table(sql)
    assert table_name in _METADATA.tables, (
        f"feature_service queries unknown table '{table_name}' "
        f"(available: {sorted(_METADATA.tables)})"
    )
    table_cols = set(_METADATA.tables[table_name].columns.keys())
    referenced = set(_selected_source_columns(sql)) | extra_columns
    missing = referenced - table_cols
    assert not missing, (
        f"feature_service SQL references columns {missing} that do not exist on "
        f"table '{table_name}' (has: {sorted(table_cols)})"
    )


class TestServingSqlMatchesSchema:
    def test_demographics_query(self) -> None:
        # WHERE uses id + deleted_at.
        _assert_query_matches_schema(fs._DEMOGRAPHICS_SQL, {"id", "deleted_at"})

    def test_conditions_query(self) -> None:
        # WHERE uses patient_id; SELECT aliases code AS icd10_code.
        _assert_query_matches_schema(fs._CONDITIONS_SQL, {"patient_id"})

    def test_medications_query(self) -> None:
        _assert_query_matches_schema(fs._MEDICATIONS_SQL, {"patient_id"})


class TestNoPluralSchemaRegression:
    def test_singular_tables_only(self) -> None:
        # The superseded plural draft tables must not exist in the ORM metadata
        # and must not reappear in the serving SQL.
        for stale in ("patients", "conditions", "medication_requests", "diagnoses"):
            assert stale not in _METADATA.tables
        combined = fs._DEMOGRAPHICS_SQL + fs._CONDITIONS_SQL + fs._MEDICATIONS_SQL
        assert _from_table(fs._DEMOGRAPHICS_SQL) == "patient"
        assert _from_table(fs._CONDITIONS_SQL) == "condition"
        assert _from_table(fs._MEDICATIONS_SQL) == "medication_request"
        # The non-existent icd10_code *column* must only appear as an alias.
        assert re.search(r"\bAS\s+icd10_code\b", combined) is not None
        assert "FROM conditions" not in combined


class TestEnumValuesAlignWithFilters:
    """The Python-side status filters must use real enum wire values."""

    def test_active_condition_statuses_are_valid(self) -> None:
        from libs.common.models.condition import ClinicalStatus

        valid = {s.value for s in ClinicalStatus}
        assert set(fs._ACTIVE_STATUSES) <= valid

    def test_stopped_medication_statuses_are_valid(self) -> None:
        from libs.common.models.medication_request import MedicationRequestStatus

        valid = {s.value for s in MedicationRequestStatus}
        assert set(fs._STOPPED_STATUSES) <= valid
