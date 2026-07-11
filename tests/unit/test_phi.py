"""Unit tests for libs/common/phi.py.

All tests run without external services (pure Python).
PHI redaction is a security-critical feature — test coverage must be exhaustive.
"""

import logging

import pytest

from libs.common.phi import PhiRedactingFilter, redact_dict, redact_phi


class TestRedactPhi:
    """Tests for the redact_phi() string function."""

    # ── SSN ──────────────────────────────────────────────────────────────────
    def test_ssn_standard_format(self) -> None:
        assert redact_phi("SSN: 123-45-6789") == "SSN: [SSN-REDACTED]"

    def test_ssn_not_a_partial_match(self) -> None:
        # A 6-digit number should not match SSN pattern
        result = redact_phi("code: 123456")
        assert "SSN-REDACTED" not in result

    # ── Phone numbers ─────────────────────────────────────────────────────────
    def test_us_phone_dashed(self) -> None:
        assert "[PHONE-REDACTED]" in redact_phi("call 555-867-5309")

    def test_us_phone_dotted(self) -> None:
        assert "[PHONE-REDACTED]" in redact_phi("mobile: 555.867.5309")

    def test_us_phone_with_country_code(self) -> None:
        assert "[PHONE-REDACTED]" in redact_phi("+1-800-555-1234 for support")

    # ── Email ─────────────────────────────────────────────────────────────────
    def test_email_redacted(self) -> None:
        assert "[EMAIL-REDACTED]" in redact_phi("contact john.doe@hospital.org today")

    def test_email_in_log_context(self) -> None:
        result = redact_phi("user email is patient@clinic.com for record")
        assert "patient@clinic.com" not in result

    # ── Dates of birth ────────────────────────────────────────────────────────
    def test_dob_iso_format(self) -> None:
        assert "[DOB-REDACTED]" in redact_phi("born 1985-06-15")

    def test_dob_us_format(self) -> None:
        assert "[DOB-REDACTED]" in redact_phi("dob: 06/15/1985")

    # ── UUID ─────────────────────────────────────────────────────────────────
    def test_uuid_patient_id_redacted(self) -> None:
        uid = "550e8400-e29b-41d4-a716-446655440000"
        result = redact_phi(f"patient_id={uid}")
        assert uid not in result
        assert "[ID-REDACTED]" in result

    # ── ZIP codes ─────────────────────────────────────────────────────────────
    def test_zip_5digit(self) -> None:
        assert "[ZIP-REDACTED]" in redact_phi("address zip 90210")

    def test_zip_9digit(self) -> None:
        assert "[ZIP-REDACTED]" in redact_phi("zip+4: 90210-1234")

    # ── IP addresses ─────────────────────────────────────────────────────────
    def test_ip_address_redacted(self) -> None:
        assert "[IP-REDACTED]" in redact_phi("request from 192.168.1.1")

    # ── Clean strings ────────────────────────────────────────────────────────
    def test_clean_string_unchanged(self) -> None:
        msg = "Prediction model loaded successfully with 0 errors"
        assert redact_phi(msg) == msg

    def test_empty_string_unchanged(self) -> None:
        assert redact_phi("") == ""

    def test_multiple_phi_in_one_string(self) -> None:
        msg = "Patient 123-45-6789 called 555-867-5309"
        result = redact_phi(msg)
        assert "123-45-6789" not in result
        assert "555-867-5309" not in result
        assert "[SSN-REDACTED]" in result
        assert "[PHONE-REDACTED]" in result


class TestRedactDict:
    """Tests for the redact_dict() function."""

    def test_phi_key_fully_redacted(self) -> None:
        data = {"patient_id": "abc123", "model": "xgboost"}
        result = redact_dict(data)
        assert result["patient_id"] == "[REDACTED]"
        assert result["model"] == "xgboost"

    def test_nested_dict_redacted(self) -> None:
        data = {"record": {"patient_name": "John Doe", "score": 0.87}}
        result = redact_dict(data)
        assert result["record"]["patient_name"] == "[REDACTED]"
        assert result["record"]["score"] == 0.87

    def test_list_of_strings_run_through_redact_phi(self) -> None:
        data = {"tags": ["555-123-4567", "normal-tag"]}
        result = redact_dict(data)
        assert "555-123-4567" not in result["tags"]

    def test_list_of_dicts_recursed(self) -> None:
        data = {"patients": [{"patient_id": "p1"}, {"patient_id": "p2"}]}
        result = redact_dict(data)
        for entry in result["patients"]:
            assert entry["patient_id"] == "[REDACTED]"

    def test_non_phi_keys_with_phi_values_still_redacted_by_pattern(self) -> None:
        data = {"notes": "call 555-987-6543 for follow-up"}
        result = redact_dict(data)
        assert "555-987-6543" not in result["notes"]

    def test_non_string_values_passed_through(self) -> None:
        data = {"risk_score": 0.73, "age": 45, "active": True}
        result = redact_dict(data)
        assert result["risk_score"] == 0.73
        assert result["age"] == 45
        assert result["active"] is True

    def test_original_dict_not_mutated(self) -> None:
        original = {"patient_id": "secret"}
        _ = redact_dict(original)
        assert original["patient_id"] == "secret"


class TestPhiRedactingFilter:
    """Tests for the PhiRedactingFilter logging filter."""

    def _make_record(self, msg: str) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        return record

    def test_filter_returns_true(self) -> None:
        f = PhiRedactingFilter()
        record = self._make_record("hello")
        assert f.filter(record) is True

    def test_filter_redacts_ssn_in_message(self) -> None:
        f = PhiRedactingFilter()
        record = self._make_record("Patient SSN 123-45-6789 processed")
        f.filter(record)
        assert "123-45-6789" not in record.msg

    def test_filter_redacts_string_args_tuple(self) -> None:
        f = PhiRedactingFilter()
        record = self._make_record("info: %s")
        record.args = ("email: test@example.com",)
        f.filter(record)
        assert "test@example.com" not in str(record.args)

    def test_filter_redacts_dict_args(self) -> None:
        f = PhiRedactingFilter()
        record = self._make_record("info: %(patient_id)s")
        record.args = {"patient_id": "secret-123"}
        f.filter(record)
        assert record.args["patient_id"] == "[REDACTED]"  # type: ignore[index]

    def test_non_phi_message_unchanged(self) -> None:
        f = PhiRedactingFilter()
        msg = "Model AUC-ROC: 0.92 on validation set"
        record = self._make_record(msg)
        f.filter(record)
        assert record.msg == msg

    def test_filter_plugs_into_handler(self, caplog: pytest.LogCaptureFixture) -> None:
        """Integration-style: verify PHI does not reach captured log output."""
        logger = logging.getLogger("phi_test")
        logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.addFilter(PhiRedactingFilter())
        logger.addHandler(handler)
        logger.propagate = False

        with caplog.at_level(logging.INFO, logger="phi_test"):
            logger.info("Patient born 1990-01-01 has risk score 0.3")

        for record in caplog.records:
            assert "1990-01-01" not in record.getMessage()
