"""Tests for JSON schema validation in jsonschema_pydantic_converter."""

from typing import Any

import pytest

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
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
