"""Tests for JSON schema validation in jsonschema_pydantic_converter."""

from typing import Any

import pytest
from pydantic import BaseModel

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    _is_non_object_schema,
    _wrap_non_object_schema,
    create_model,
    has_underscore_fields,
)


class TestHasUnderscoreFieldsReturnsTrue:
    """Scenarios where underscore fields are present."""

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "_hidden": {"type": "string"},
                    },
                },
                id="top-level underscore field",
            ),
            pytest.param(
                {
                    "type": "object",
                    "properties": {
                        "outer": {
                            "type": "object",
                            "properties": {
                                "_secret": {"type": "integer"},
                            },
                        },
                    },
                },
                id="nested underscore field in object properties",
            ),
            pytest.param(
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "$defs": {
                        "Inner": {
                            "type": "object",
                            "properties": {
                                "_internal": {"type": "boolean"},
                            },
                        },
                    },
                },
                id="underscore field in $defs",
            ),
            pytest.param(
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "definitions": {
                        "Inner": {
                            "type": "object",
                            "properties": {
                                "_private": {"type": "number"},
                            },
                        },
                    },
                },
                id="underscore field in definitions",
            ),
            pytest.param(
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "_element_id": {"type": "string"},
                        },
                    },
                },
                id="underscore field in array items",
            ),
            pytest.param(
                {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {
                                "_merged": {"type": "string"},
                            },
                        },
                    ],
                },
                id="underscore field in allOf sub-schema",
            ),
            pytest.param(
                {
                    "anyOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "_variant": {"type": "integer"},
                            },
                        },
                    ],
                },
                id="underscore field in anyOf sub-schema",
            ),
            pytest.param(
                {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "_choice": {"type": "boolean"},
                            },
                        },
                    ],
                },
                id="underscore field in oneOf sub-schema",
            ),
            pytest.param(
                {
                    "not": {
                        "type": "object",
                        "properties": {
                            "_excluded": {"type": "string"},
                        },
                    },
                },
                id="underscore field in not sub-schema",
            ),
            pytest.param(
                {
                    "if": {
                        "type": "object",
                        "properties": {
                            "_condition": {"type": "boolean"},
                        },
                    },
                    "then": {"type": "object", "properties": {}},
                },
                id="underscore field in if",
            ),
            pytest.param(
                {
                    "if": {
                        "type": "object",
                        "properties": {"flag": {"type": "boolean"}},
                    },
                    "then": {
                        "type": "object",
                        "properties": {
                            "_result": {"type": "string"},
                        },
                    },
                },
                id="underscore field in then",
            ),
            pytest.param(
                {
                    "if": {
                        "type": "object",
                        "properties": {"flag": {"type": "boolean"}},
                    },
                    "then": {"type": "object", "properties": {}},
                    "else": {
                        "type": "object",
                        "properties": {
                            "_fallback": {"type": "string"},
                        },
                    },
                },
                id="underscore field in else",
            ),
            pytest.param(
                {
                    "type": "object",
                    "properties": {
                        "level1": {
                            "type": "object",
                            "properties": {
                                "level2": {
                                    "type": "object",
                                    "properties": {
                                        "level3": {
                                            "type": "object",
                                            "properties": {
                                                "_deep": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                id="deeply nested underscore field",
            ),
        ],
    )
    def test_returns_true(self, schema: dict[str, Any]) -> None:
        assert has_underscore_fields(schema) is True


class TestHasUnderscoreFieldsReturnsFalse:
    """Scenarios where no underscore fields exist."""

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param(
                {},
                id="empty schema",
            ),
            pytest.param(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
                id="flat schema with normal property names",
            ),
            pytest.param(
                {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "properties": {
                                "firstName": {"type": "string"},
                                "lastName": {"type": "string"},
                            },
                        },
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "value": {"type": "integer"},
                                },
                            },
                        },
                    },
                    "$defs": {
                        "Address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                            },
                        },
                    },
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {
                                "extra": {"type": "string"},
                            },
                        },
                    ],
                },
                id="complex schema with no underscores anywhere",
            ),
        ],
    )
    def test_returns_false(self, schema: dict[str, Any]) -> None:
        assert has_underscore_fields(schema) is False


class TestCreateModelRejectsUnderscoreFields:
    def test_top_level_underscore_field(self) -> None:
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "_hidden": {"type": "string"},
            },
        }
        with pytest.raises(AgentStartupError) as exc_info:
            create_model(schema)
        assert exc_info.value.error_info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.UNDERSCORE_SCHEMA
        )

    def test_nested_underscore_field(self) -> None:
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "_secret": {"type": "integer"},
                    },
                },
            },
        }
        with pytest.raises(AgentStartupError) as exc_info:
            create_model(schema)
        assert exc_info.value.error_info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.UNDERSCORE_SCHEMA
        )

    def test_underscore_error_includes_tool_name(self) -> None:
        schema = {
            "type": "object",
            "properties": {"_hidden": {"type": "string"}},
        }
        with pytest.raises(AgentStartupError) as exc_info:
            create_model(schema, tool_name="MyTool")
        assert "MyTool" in exc_info.value.error_info.detail


class TestIsNonObjectSchema:
    """Tests for the _is_non_object_schema helper."""

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param({"type": "array", "items": {"type": "string"}}, id="array"),
            pytest.param({"type": "string"}, id="string"),
            pytest.param({"type": "integer"}, id="integer"),
            pytest.param({"type": "number"}, id="number"),
            pytest.param({"type": "boolean"}, id="boolean"),
            pytest.param({"type": "null"}, id="null"),
        ],
    )
    def test_non_object_types_detected(self, schema: dict[str, Any]) -> None:
        assert _is_non_object_schema(schema) is True

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param(
                {"type": "object", "properties": {"name": {"type": "string"}}},
                id="object",
            ),
            pytest.param({}, id="empty schema"),
            pytest.param({"properties": {"x": {"type": "string"}}}, id="no type"),
        ],
    )
    def test_object_or_typeless_not_detected(self, schema: dict[str, Any]) -> None:
        assert _is_non_object_schema(schema) is False


class TestWrapNonObjectSchema:
    """Tests for the _wrap_non_object_schema helper."""

    def test_wraps_array_in_object_envelope(self) -> None:
        schema: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
        wrapped = _wrap_non_object_schema(schema)
        assert wrapped["type"] == "object"
        assert "result" in wrapped["properties"]
        assert wrapped["properties"]["result"] is schema
        assert wrapped["required"] == ["result"]

    def test_hoists_defs_to_wrapper(self) -> None:
        schema: dict[str, Any] = {
            "type": "array",
            "items": {"$ref": "#/$defs/Comment"},
            "$defs": {
                "Comment": {
                    "type": "object",
                    "properties": {"body": {"type": "string"}},
                }
            },
        }
        wrapped = _wrap_non_object_schema(schema)
        assert "$defs" in wrapped
        assert "Comment" in wrapped["$defs"]

    def test_hoists_definitions_to_wrapper(self) -> None:
        schema: dict[str, Any] = {
            "type": "array",
            "items": {"$ref": "#/definitions/Item"},
            "definitions": {
                "Item": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                }
            },
        }
        wrapped = _wrap_non_object_schema(schema)
        assert "definitions" in wrapped
        assert "Item" in wrapped["definitions"]


class TestCreateModelNonObjectSchemas:
    """Tests for create_model handling non-object root schemas (PC-4127)."""

    def test_array_of_strings(self) -> None:
        schema: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
        model = create_model(schema)
        assert issubclass(model, BaseModel)
        assert "result" in model.model_fields

    def test_array_of_objects(self) -> None:
        """Simulates the 'Get Comments' Integration Service response schema."""
        schema: dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "body": {"type": "string"},
                    "created": {"type": "string"},
                },
            },
        }
        model = create_model(schema)
        assert issubclass(model, BaseModel)
        assert "result" in model.model_fields
        # model_json_schema must work (used by @mockable)
        json_schema = model.model_json_schema()
        assert json_schema["type"] == "object"

    def test_string_schema(self) -> None:
        schema: dict[str, Any] = {"type": "string"}
        model = create_model(schema)
        assert issubclass(model, BaseModel)
        assert "result" in model.model_fields

    def test_integer_schema(self) -> None:
        schema: dict[str, Any] = {"type": "integer"}
        model = create_model(schema)
        assert issubclass(model, BaseModel)
        assert "result" in model.model_fields

    def test_boolean_schema(self) -> None:
        schema: dict[str, Any] = {"type": "boolean"}
        model = create_model(schema)
        assert issubclass(model, BaseModel)
        assert "result" in model.model_fields

    def test_number_schema(self) -> None:
        schema: dict[str, Any] = {"type": "number"}
        model = create_model(schema)
        assert issubclass(model, BaseModel)
        assert "result" in model.model_fields

    def test_object_schema_unchanged(self) -> None:
        """Object schemas should continue to work without wrapping."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
        }
        model = create_model(schema)
        assert issubclass(model, BaseModel)
        assert "name" in model.model_fields
        assert "count" in model.model_fields
        # Should NOT have a 'result' wrapper
        assert "result" not in model.model_fields

    def test_tool_name_in_error_on_schema_failure(self) -> None:
        """Error messages should include tool_name when provided."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"_bad": {"type": "string"}},
        }
        with pytest.raises(AgentStartupError) as exc_info:
            create_model(schema, tool_name="Get Comments")
        assert "Get Comments" in exc_info.value.error_info.detail
