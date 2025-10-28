"""Tests for custom exceptions."""

from typing import cast

import pytest
from pydantic_core import ErrorDetails

from uipath_lowcode.lowcode_agent_graph_builder.exceptions import (
    ConfigurationError,
    InputValidationError,
)


class TestConfigurationError:
    """Test ConfigurationError exception."""

    def test_basic_instantiation(self):
        """Test creating ConfigurationError with message."""
        error = ConfigurationError("Config not found")
        assert str(error) == "Config not found"
        assert isinstance(error, Exception)

    def test_raise_and_catch(self):
        """Test raising and catching ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Invalid config"):
            raise ConfigurationError("Invalid config")


class TestInputValidationError:
    """Test InputValidationError exception."""

    def test_basic_instantiation(self):
        """Test creating InputValidationError with message."""
        error = InputValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert error.validation_errors is None

    def test_with_validation_errors(self):
        """Test creating InputValidationError with validation details."""
        validation_errors = cast(
            list[ErrorDetails],
            [
                {
                    "loc": ("field",),
                    "msg": "Field required",
                    "type": "missing",
                    "input": {},
                },
                {
                    "loc": ("other",),
                    "msg": "Invalid type",
                    "type": "type_error",
                    "input": {},
                },
            ],
        )
        error = InputValidationError("Schema validation failed", validation_errors)

        assert str(error) == "Schema validation failed"
        assert error.validation_errors == validation_errors
        assert len(error.validation_errors) == 2

    def test_raise_with_errors(self):
        """Test raising InputValidationError with errors."""
        errors = cast(
            list[ErrorDetails],
            [{"loc": ("name",), "msg": "Required", "type": "missing", "input": {}}],
        )
        with pytest.raises(InputValidationError) as exc_info:
            raise InputValidationError("Invalid input", errors)

        assert exc_info.value.validation_errors == errors
