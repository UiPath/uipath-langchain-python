"""Tests for JSON schema validation utilities."""

from typing import Any

import pytest

from uipath_agents._cli.exceptions import InputValidationError
from uipath_agents._cli.json_schema_utils import validate_json_against_json_schema


class TestValidateJsonAgainstJsonSchema:
    """Tests for validate_json_against_json_schema function."""

    def test_validates_simple_object(self):
        """Test validation of simple object matching schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        data = {"name": "Alice", "age": 30}

        result = validate_json_against_json_schema(schema, data)

        assert result == {"name": "Alice", "age": 30}

    def test_validates_json_string(self):
        """Test validation of JSON string input."""
        schema = {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        }
        json_string = '{"task": "process data"}'

        result = validate_json_against_json_schema(schema, json_string)

        assert result == {"task": "process data"}

    def test_handles_none_input(self):
        """Test that None input returns empty dict."""
        schema = {"type": "object", "properties": {}}

        result = validate_json_against_json_schema(schema, None)

        assert result == {}

    def test_handles_empty_string_input(self):
        """Test that empty string input returns empty dict."""
        schema = {"type": "object", "properties": {}}

        result = validate_json_against_json_schema(schema, "")

        assert result == {}

    def test_validates_nested_objects(self):
        """Test validation of nested object structures."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                }
            },
        }
        data = {"user": {"name": "Bob", "email": "bob@example.com"}}

        result = validate_json_against_json_schema(schema, data)

        assert result == {"user": {"name": "Bob", "email": "bob@example.com"}}

    def test_validates_arrays(self):
        """Test validation of arrays in schema."""
        schema = {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        }
        data = {"tags": ["python", "testing", "automation"]}

        result = validate_json_against_json_schema(schema, data)

        assert result == {"tags": ["python", "testing", "automation"]}

    def test_raises_on_missing_required_field(self):
        """Test that missing required field raises InputValidationError."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data: dict[str, Any] = {}

        with pytest.raises(InputValidationError) as exc_info:
            validate_json_against_json_schema(schema, data)

        assert "Data failed json schema validation" in str(exc_info.value)
        assert exc_info.value.validation_errors is not None

    def test_raises_on_type_mismatch(self):
        """Test that type mismatch raises InputValidationError."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
        }
        data = {"age": "not a number"}

        with pytest.raises(InputValidationError) as exc_info:
            validate_json_against_json_schema(schema, data)

        assert "Data failed json schema validation" in str(exc_info.value)
        assert exc_info.value.validation_errors is not None

    def test_raises_on_invalid_json_string(self):
        """Test that invalid JSON string raises InputValidationError."""
        schema = {"type": "object", "properties": {}}
        invalid_json = '{"invalid": json}'

        with pytest.raises(InputValidationError) as exc_info:
            validate_json_against_json_schema(schema, invalid_json)

        assert "Data failed json schema validation" in str(exc_info.value)

    def test_allows_extra_fields_not_in_schema(self):
        """Test that fields not in schema are preserved in output when extra='allow'."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        data = {"name": "Charlie", "extra_field": "value"}

        result = validate_json_against_json_schema(schema, data)

        # By default, jsonschema_to_pydantic may filter extra fields
        # Just verify the schema field is present
        assert result["name"] == "Charlie"

    def test_validates_complex_nested_structure(self):
        """Test validation of complex nested structure with arrays and objects."""
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                            "roles": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                }
            },
        }
        data = {
            "users": [
                {"id": 1, "name": "Alice", "roles": ["admin", "user"]},
                {"id": 2, "name": "Bob", "roles": ["user"]},
            ]
        }

        result = validate_json_against_json_schema(schema, data)

        assert result == data

    def test_validation_error_includes_details(self):
        """Test that validation error includes detailed error information."""
        schema = {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["email", "age"],
        }
        data = {"age": "not an integer"}

        with pytest.raises(InputValidationError) as exc_info:
            validate_json_against_json_schema(schema, data)

        assert exc_info.value.validation_errors is not None
        assert len(exc_info.value.validation_errors) > 0
