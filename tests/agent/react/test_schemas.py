"""Tests for JSON schema validation in jsonschema_pydantic_converter."""

from typing import Any

from pydantic import BaseModel

from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model,
)


class TestCreateModelJsonSchemaRoundtrip:
    """Verify that model_json_schema() works on models produced by create_model().

    Pydantic re-resolves forward references via sys.modules[cls.__module__],
    so all types must be registered in the pseudo-module.
    """

    def test_single_ref(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "address": {"$ref": "#/$defs/Address"},
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
        }
        model = create_model(schema)
        result = model.model_json_schema()
        defs = result.get("$defs") or result.get("definitions") or {}
        assert len(defs) == 1

    def test_deeply_nested_defs_chain(self) -> None:
        """Type B is only referenced by type A (not by root). Old code missed B."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "order": {"$ref": "#/$defs/Order"},
            },
            "$defs": {
                "Order": {
                    "type": "object",
                    "properties": {
                        "item": {"$ref": "#/$defs/Item"},
                        "quantity": {"type": "integer"},
                    },
                },
                "Item": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                    },
                },
            },
        }
        model = create_model(schema)
        result = model.model_json_schema()
        defs = result.get("$defs") or result.get("definitions") or {}
        assert len(defs) == 2

    def test_three_level_defs_chain(self) -> None:
        """Root -> A -> B -> C. C is two levels removed from root fields."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "company": {"$ref": "#/$defs/Company"},
            },
            "$defs": {
                "Company": {
                    "type": "object",
                    "properties": {
                        "department": {"$ref": "#/$defs/Department"},
                    },
                },
                "Department": {
                    "type": "object",
                    "properties": {
                        "manager": {"$ref": "#/$defs/Employee"},
                    },
                },
                "Employee": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                    },
                },
            },
        }
        model = create_model(schema)
        result = model.model_json_schema()
        defs = result.get("$defs") or result.get("definitions") or {}
        assert len(defs) == 3

    def test_enum_in_defs(self) -> None:
        """Non-BaseModel types (enums) in $defs must also be registered."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "status": {"$ref": "#/$defs/Status"},
            },
            "$defs": {
                "Status": {
                    "type": "string",
                    "enum": ["active", "inactive", "pending"],
                },
            },
        }
        model = create_model(schema)
        result = model.model_json_schema()
        assert result is not None

    def test_array_of_nested_refs(self) -> None:
        """Array field referencing a $def type that itself has a $ref."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Task"},
                },
            },
            "$defs": {
                "Task": {
                    "type": "object",
                    "properties": {
                        "assignee": {"$ref": "#/$defs/Person"},
                        "title": {"type": "string"},
                    },
                },
                "Person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                },
            },
        }
        model = create_model(schema)
        result = model.model_json_schema()
        defs = result.get("$defs") or result.get("definitions") or {}
        assert len(defs) == 2


class TestCreateModelWithUnderscoreFields:
    """Tests for create_model handling of underscore-prefixed fields."""

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
        model = create_model(schema)
        instance = model.model_validate({"outer": {"_secret": 42}})
        dumped = instance.model_dump()
        assert dumped["outer"]["_secret"] == 42


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
        assert callable(model.model_json_schema)

    def test_reserved_field_validate_and_dump(self) -> None:
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
