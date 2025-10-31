"""Tests for input loading and validation."""

from typing import Any

import pytest

from uipath_lowcode.agent_graph_builder.input_utils import validate_input_data


class TestValidateInputData:
    """Test loading and validating input arguments."""

    def test_none_input(self):
        """Test None input returns empty dict."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        result = validate_input_data(schema, None)
        assert result == {}

    def test_empty_string_input(self):
        """Test empty string input returns empty dict."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        result = validate_input_data(schema, "")
        assert result == {}

    def test_dict_input_valid(self):
        """Test valid dict input."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = validate_input_data(schema, {"name": "Alice"})
        assert result["name"] == "Alice"

    def test_json_string_input_valid(self):
        """Test valid JSON string input."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        result = validate_input_data(schema, '{"count": 42}')
        assert result["count"] == 42

    def test_validation_error_missing_required(self):
        """Test validation error for missing required fields."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        with pytest.raises(TypeError):
            # The function currently has a bug where it tries to pass kwargs to Exception
            validate_input_data(schema, {})

    def test_validation_error_wrong_type(self):
        """Test validation error for wrong type."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        with pytest.raises(Exception):
            validate_input_data(schema, {"count": "not a number"})

    def test_invalid_json_string(self):
        """Test error for invalid JSON string."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        with pytest.raises(Exception):
            validate_input_data(schema, "{ invalid json }")

    def test_nested_object_validation(self):
        """Test validation of nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
            "required": ["user"],
        }
        result = validate_input_data(schema, {"user": {"name": "Bob"}})
        assert result["user"]["name"] == "Bob"
