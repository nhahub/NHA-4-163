"""Unit tests for the clinical PDF report service (Tier 3)."""

from __future__ import annotations

from datetime import date

from services.api.services.pdf_service import (
    ReportData,
    _compute_age,
    generate_patient_report,
)


def _sample_data(**overrides: object) -> ReportData:
    base = {
        "patient_id": "12345678-1234-5678-1234-567812345678",
        "full_name": "Alice Smith",
        "date_of_birth": date(1980, 5, 3),
        "gender": "female",
        "risk_score": 0.78,
        "risk_tier": "high",
        "conditions": [("E11.9", "Type 2 diabetes mellitus", "active")],
        "shap_factors": [
            {
                "feature": "family_risk_prevalence",
                "shap_value": 0.42,
                "direction": "increases_risk",
            },
        ],
    }
    base.update(overrides)
    return ReportData(**base)  # type: ignore[arg-type]


class TestGenerateReport:
    def test_returns_valid_pdf_bytes(self) -> None:
        pdf = generate_patient_report(_sample_data())
        assert isinstance(pdf, bytes)
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 500

    def test_handles_missing_risk_and_conditions(self) -> None:
        pdf = generate_patient_report(
            _sample_data(risk_score=None, risk_tier=None, conditions=[], shap_factors=[])
        )
        assert pdf.startswith(b"%PDF")

    def test_handles_missing_dob(self) -> None:
        pdf = generate_patient_report(_sample_data(date_of_birth=None))
        assert pdf.startswith(b"%PDF")


class TestComputeAge:
    def test_none_dob(self) -> None:
        assert _compute_age(None) is None

    def test_known_age(self) -> None:
        # Born in 2000 → age is current year minus 2000 (± birthday not yet passed)
        age = _compute_age(date(2000, 1, 1))
        assert age is not None and age >= 24
