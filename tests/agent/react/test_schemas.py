"""Tests for JSON schema validation in jsonschema_pydantic_converter."""

from typing import Any

import pytest
from pydantic import BaseModel

from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    _RESERVED_FIELD_NAMES,
    _rename_reserved_properties,
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


class TestCreateModelWithUnderscoreFields:
    """Tests for create_model aliasing of underscore-prefixed fields."""

    def test_underscore_field_creates_valid_model(self) -> None:
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "_hidden": {"type": "string"},
                "name": {"type": "string"},
            },
        }
        model = create_model(schema)

        assert issubclass(model, BaseModel)
        assert callable(model.model_json_schema)

    def test_underscore_field_validate_and_dump(self) -> None:
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "_hidden": {"type": "string"},
                "name": {"type": "string"},
            },
        }
        model = create_model(schema)

        instance = model.model_validate({"_hidden": "secret", "name": "alice"})
        dumped = instance.model_dump()

        assert dumped == {"_hidden": "secret", "name": "alice"}

    def test_underscore_field_json_schema_shows_original(self) -> None:
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "_hidden": {"type": "string"},
            },
        }
        model = create_model(schema)

        json_schema = model.model_json_schema()
        assert "_hidden" in json_schema["properties"]

    def test_underscore_field_that_would_also_shadow_basemodel(self) -> None:
        """'_schema' strips to 'schema' which is reserved — should still work."""
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "_schema": {"type": "string"},
                "name": {"type": "string"},
            },
        }
        model = create_model(schema)

        instance = model.model_validate({"_schema": "val", "name": "n"})
        assert instance.model_dump() == {"_schema": "val", "name": "n"}

    def test_underscore_field_collision_with_stripped_name(self) -> None:
        """Schema has both '_hidden' and 'hidden' — no collision after rename."""
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "_hidden": {"type": "string"},
                "hidden": {"type": "string"},
            },
        }
        model = create_model(schema)

        instance = model.model_validate({"_hidden": "a", "hidden": "b"})
        dumped = instance.model_dump()

        assert dumped["_hidden"] == "a"
        assert dumped["hidden"] == "b"


class TestRenameReservedProperties:
    """Tests for _rename_reserved_properties schema pre-processing."""

    def test_renames_schema_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "schema": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["schema", "name"],
        }
        modified, renames = _rename_reserved_properties(schema)

        assert "schema_" in modified["properties"]
        assert "schema" not in modified["properties"]
        assert "name" in modified["properties"]
        assert renames == {"schema_": "schema"}
        assert modified["required"] == ["schema_", "name"]

    def test_renames_multiple_reserved_fields(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "schema": {"type": "string"},
                "copy": {"type": "string"},
                "validate": {"type": "string"},
                "name": {"type": "string"},
            },
        }
        modified, renames = _rename_reserved_properties(schema)

        assert "schema_" in modified["properties"]
        assert "copy_" in modified["properties"]
        assert "validate_" in modified["properties"]
        assert "name" in modified["properties"]
        assert len(renames) == 3

    def test_handles_collision_with_existing_field(self) -> None:
        """When 'schema_' already exists, 'schema' should become 'schema__'."""
        schema = {
            "type": "object",
            "properties": {
                "schema": {"type": "string"},
                "schema_": {"type": "string"},
            },
        }
        modified, renames = _rename_reserved_properties(schema)

        assert "schema__" in modified["properties"]
        assert "schema_" in modified["properties"]
        assert renames["schema__"] == "schema"

    def test_renames_in_defs(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {
                        "schema": {"type": "string"},
                    },
                    "required": ["schema"],
                },
            },
        }
        modified, renames = _rename_reserved_properties(schema)

        inner = modified["$defs"]["Inner"]
        assert "schema_" in inner["properties"]
        assert inner["required"] == ["schema_"]

    def test_does_not_modify_original_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {"schema": {"type": "string"}},
        }
        _rename_reserved_properties(schema)

        assert "schema" in schema["properties"]

    def test_no_renames_for_normal_fields(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        modified, renames = _rename_reserved_properties(schema)

        assert renames == {}
        assert modified["properties"] == schema["properties"]

    def test_renames_underscore_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "_hidden": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["_hidden", "name"],
        }
        modified, renames = _rename_reserved_properties(schema)

        assert "hidden" in modified["properties"]
        assert "_hidden" not in modified["properties"]
        assert renames == {"hidden": "_hidden"}
        assert modified["required"] == ["hidden", "name"]

    def test_underscore_field_stripped_to_reserved_name(self) -> None:
        """'_schema' strips to 'schema' which is reserved — gets extra '_'."""
        schema = {
            "type": "object",
            "properties": {
                "_schema": {"type": "string"},
            },
        }
        modified, renames = _rename_reserved_properties(schema)

        assert "schema_" in modified["properties"]
        assert renames == {"schema_": "_schema"}

    def test_underscore_field_collision_with_existing(self) -> None:
        """'_hidden' strips to 'hidden', but 'hidden' already exists."""
        schema = {
            "type": "object",
            "properties": {
                "_hidden": {"type": "string"},
                "hidden": {"type": "string"},
            },
        }
        modified, renames = _rename_reserved_properties(schema)

        assert "hidden" in modified["properties"]
        assert "hidden_" in modified["properties"]
        assert renames["hidden_"] == "_hidden"

    def test_mixed_underscore_and_reserved(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "_secret": {"type": "string"},
                "schema": {"type": "string"},
                "name": {"type": "string"},
            },
        }
        modified, renames = _rename_reserved_properties(schema)

        assert "secret" in modified["properties"]
        assert "schema_" in modified["properties"]
        assert "name" in modified["properties"]
        assert renames == {"secret": "_secret", "schema_": "schema"}


class TestCreateModelWithReservedFields:
    """Tests for create_model handling of reserved field names."""

    def test_schema_field_creates_valid_model(self) -> None:
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "schema": {"type": "string"},
                "name": {"type": "string"},
            },
        }
        model = create_model(schema)

        assert issubclass(model, BaseModel)
        # BaseModel methods should still work
        assert callable(model.model_json_schema)
        assert callable(model.model_validate)

    def test_model_validate_accepts_original_names(self) -> None:
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "schema": {"type": "string"},
                "name": {"type": "string"},
            },
        }
        model = create_model(schema)

        instance = model.model_validate({"schema": "test_val", "name": "alice"})
        assert instance is not None

    def test_model_dump_outputs_original_names(self) -> None:
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "schema": {"type": "string"},
                "name": {"type": "string"},
            },
        }
        model = create_model(schema)

        instance = model.model_validate({"schema": "test_val", "name": "alice"})
        dumped = instance.model_dump()

        assert dumped == {"schema": "test_val", "name": "alice"}

    def test_model_dump_json_mode_outputs_original_names(self) -> None:
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "schema": {"type": "string"},
            },
        }
        model = create_model(schema)

        instance = model.model_validate({"schema": "val"})
        dumped = instance.model_dump(mode="json")

        assert dumped == {"schema": "val"}

    def test_model_json_schema_shows_original_names(self) -> None:
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "schema": {"type": "string"},
                "copy": {"type": "integer"},
            },
        }
        model = create_model(schema)

        json_schema = model.model_json_schema()
        assert "schema" in json_schema["properties"]
        assert "copy" in json_schema["properties"]
        assert "schema_" not in json_schema["properties"]
        assert "copy_" not in json_schema["properties"]

    def test_multiple_reserved_fields(self) -> None:
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "schema": {"type": "string"},
                "copy": {"type": "string"},
                "validate": {"type": "string"},
                "name": {"type": "string"},
            },
        }
        model = create_model(schema)

        instance = model.model_validate(
            {"schema": "s", "copy": "c", "validate": "v", "name": "n"}
        )
        dumped = instance.model_dump()

        assert dumped == {"schema": "s", "copy": "c", "validate": "v", "name": "n"}

    def test_model_fields_field_does_not_shadow(self) -> None:
        """'model_fields' is in Pydantic's protected namespace — must not crash."""
        schema = {
            "title": "Input",
            "type": "object",
            "properties": {
                "model_fields": {"type": "string"},
                "name": {"type": "string"},
            },
        }
        model = create_model(schema)

        # model.model_fields should still be the Pydantic descriptor, not a field value
        assert isinstance(model.model_fields, dict)
        instance = model.model_validate({"model_fields": "test", "name": "n"})
        assert instance.model_dump() == {"model_fields": "test", "name": "n"}

    def test_reserved_field_names_constant_contains_known_problematic_names(
        self,
    ) -> None:
        known_problematic = {"schema", "copy", "validate", "dict", "json", "construct"}
        assert known_problematic.issubset(_RESERVED_FIELD_NAMES)
