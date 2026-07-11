"""Unit tests for libs/common/logging.py."""

import json
import logging
from io import StringIO

import pytest

from libs.common.logging import configure_logging, get_logger


@pytest.fixture(autouse=True)
def reset_root_logger() -> None:
    """Restore root logger state after each test."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)


class TestConfigureLogging:
    def test_sets_log_level(self) -> None:
        configure_logging(level="WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_single_handler_after_configure(self) -> None:
        configure_logging()
        assert len(logging.getLogger().handlers) == 1

    def test_reconfigure_does_not_duplicate_handlers(self) -> None:
        configure_logging()
        configure_logging()
        assert len(logging.getLogger().handlers) == 1

    def test_output_is_valid_json(self) -> None:
        stream = StringIO()
        configure_logging(level="DEBUG", enable_phi_redaction=False)
        root = logging.getLogger()
        root.handlers.clear()
        handler = logging.StreamHandler(stream)
        from libs.common.logging import _JsonFormatter

        handler.setFormatter(_JsonFormatter())
        root.addHandler(handler)

        logging.getLogger("test.json").info("hello world")

        output = stream.getvalue().strip()
        assert output, "No log output produced"
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_phi_redacted_in_output(self) -> None:
        stream = StringIO()
        configure_logging(level="DEBUG", enable_phi_redaction=True)
        root = logging.getLogger()
        root.handlers.clear()
        handler = logging.StreamHandler(stream)
        from libs.common.logging import _JsonFormatter
        from libs.common.phi import PhiRedactingFilter

        handler.setFormatter(_JsonFormatter())
        handler.addFilter(PhiRedactingFilter())
        root.addHandler(handler)

        logging.getLogger("phi.output").warning("SSN: 999-88-7777")
        assert "999-88-7777" not in stream.getvalue()


class TestGetLogger:
    def test_returns_logger_with_correct_name(self) -> None:
        logger = get_logger("my.module")
        assert logger.name == "my.module"

    def test_same_name_returns_same_instance(self) -> None:
        assert get_logger("same.name") is get_logger("same.name")
