"""Great Expectations data quality suites for each ingestion path.

Usage in batch DAG or Spark foreachBatch::

    from libs.common.quality import validate_patient_dataframe, ValidationResult

    result = validate_patient_dataframe(df)
    if not result.success:
        for failure in result.failures:
            log.warning("Quality check failed", extra=failure)

Design decisions:
- Expectations are defined as pure Python (no YAML/JSON config files) so they
  are version-controlled alongside the code that uses them.
- Each function returns a ``ValidationResult`` dataclass rather than exposing
  GE internals to callers — this keeps the GE version pinned at one place.
- Severity levels: CRITICAL expectations block ingestion (must_be_true);
  WARNING expectations log but do not block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import great_expectations as gx
from great_expectations.data_context import EphemeralDataContext
from great_expectations.data_context.types.base import DataContextConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Summary of a GE validation run.

    Attributes:
        success: True if all CRITICAL expectations passed.
        failures: List of dicts describing failed expectations.
        statistics: Dict with counts (evaluated, successful, failed).
    """

    success: bool
    failures: list[dict[str, Any]]
    statistics: dict[str, int]


# ---------------------------------------------------------------------------
# GE context builder (ephemeral — no filesystem state)
# ---------------------------------------------------------------------------


def _build_context() -> EphemeralDataContext:
    """Build an in-memory GE data context with no filesystem dependencies.

    Returns:
        Ephemeral GE data context.
    """
    config = DataContextConfig(
        store_backend_defaults=gx.data_context.types.base.InMemoryStoreBackendDefaults()
    )
    return gx.get_context(project_config=config)


# ---------------------------------------------------------------------------
# Patient expectations
# ---------------------------------------------------------------------------


def _patient_suite(
    context: EphemeralDataContext,
) -> Any:
    suite_name = "patient_suite"
    suite = context.add_or_update_expectation_suite(suite_name)

    # CRITICAL: patient_id must be non-null UUID
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "patient_id"},
            meta={"severity": "critical"},
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={
                "column": "patient_id",
                "regex": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            },
            meta={"severity": "critical"},
        )
    )
    # CRITICAL: event_id non-null
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "event_id"},
            meta={"severity": "critical"},
        )
    )
    # WARNING: gender in allowed set
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "gender",
                "value_set": ["male", "female", "other", "unknown", None],
                "mostly": 0.99,
            },
            meta={"severity": "warning"},
        )
    )
    # WARNING: date_of_birth completeness >= 95%
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "date_of_birth", "mostly": 0.95},
            meta={"severity": "warning"},
        )
    )
    return suite


# ---------------------------------------------------------------------------
# Diagnosis expectations
# ---------------------------------------------------------------------------


def _diagnosis_suite(
    context: EphemeralDataContext,
) -> Any:
    suite_name = "diagnosis_suite"
    suite = context.add_or_update_expectation_suite(suite_name)

    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "patient_id"},
            meta={"severity": "critical"},
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "condition_id"},
            meta={"severity": "critical"},
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "code"},
            meta={"severity": "critical"},
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "clinical_status",
                "value_set": [
                    "active",
                    "recurrence",
                    "relapse",
                    "inactive",
                    "remission",
                    "resolved",
                ],
            },
            meta={"severity": "critical"},
        )
    )
    # WARNING: verified diagnoses >= 80% of batch
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "verification_status",
                "value_set": ["confirmed", "provisional", None],
                "mostly": 0.80,
            },
            meta={"severity": "warning"},
        )
    )
    return suite


# ---------------------------------------------------------------------------
# Observation expectations
# ---------------------------------------------------------------------------


def _observation_suite(
    context: EphemeralDataContext,
) -> Any:
    suite_name = "observation_suite"
    suite = context.add_or_update_expectation_suite(suite_name)

    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "patient_id"},
            meta={"severity": "critical"},
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "observation_id"},
            meta={"severity": "critical"},
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "effective_datetime"},
            meta={"severity": "critical"},
        )
    )
    suite.add_expectation(
        gx.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "status",
                "value_set": ["final", "amended", "corrected", "preliminary"],
                "mostly": 0.95,
            },
            meta={"severity": "warning"},
        )
    )
    return suite


# ---------------------------------------------------------------------------
# Generic validation runner
# ---------------------------------------------------------------------------


def _run_validation(
    data: list[dict[str, Any]],
    suite_builder: Any,
) -> ValidationResult:
    """Run GE validation over a list of record dicts.

    Args:
        data: List of record dicts to validate.
        suite_builder: Callable that takes a context and returns a suite.

    Returns:
        ``ValidationResult`` summarising the outcome.
    """
    if not data:
        return ValidationResult(success=True, failures=[], statistics={})

    try:
        import pandas as pd

        df = pd.DataFrame(data)
    except ImportError:
        log.warning("pandas not available — skipping GE validation")
        return ValidationResult(success=True, failures=[], statistics={})

    context = _build_context()
    suite = suite_builder(context)

    ds = context.sources.add_pandas("runtime_source")
    da = ds.add_dataframe_asset("runtime_asset")
    batch_request = da.build_batch_request(dataframe=df)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite=suite,
    )
    result = validator.validate()

    failures = []
    for r in result.results:
        if not r.success:
            failures.append(
                {
                    "expectation": r.expectation_config.expectation_type,
                    "column": r.expectation_config.kwargs.get("column", ""),
                    "severity": r.expectation_config.meta.get("severity", "warning"),
                    "partial_unexpected": r.result.get("partial_unexpected_list", [])[:5],
                }
            )

    critical_failures = [f for f in failures if f["severity"] == "critical"]
    stats = result.statistics or {}

    return ValidationResult(
        success=len(critical_failures) == 0,
        failures=failures,
        statistics={
            "evaluated": stats.get("evaluated_expectations", 0),
            "successful": stats.get("successful_expectations", 0),
            "failed": stats.get("unsuccessful_expectations", 0),
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_patient_records(records: list[dict[str, Any]]) -> ValidationResult:
    """Run patient data quality checks.

    Args:
        records: List of PatientCreated event dicts.

    Returns:
        ``ValidationResult`` — check ``success`` before ingesting.
    """
    return _run_validation(records, _patient_suite)


def validate_diagnosis_records(records: list[dict[str, Any]]) -> ValidationResult:
    """Run diagnosis data quality checks.

    Args:
        records: List of DiagnosisAdded event dicts.

    Returns:
        ``ValidationResult``.
    """
    return _run_validation(records, _diagnosis_suite)


def validate_observation_records(records: list[dict[str, Any]]) -> ValidationResult:
    """Run observation data quality checks.

    Args:
        records: List of ObservationRecorded event dicts.

    Returns:
        ``ValidationResult``.
    """
    return _run_validation(records, _observation_suite)
