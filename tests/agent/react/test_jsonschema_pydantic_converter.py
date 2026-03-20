"""Tests for jsonschema_pydantic_converter wrapper — create_model().

Covers $ref/$defs handling across all code paths that produce Pydantic models
from JSON schemas: tool schemas, agent input/output, static_args round-trips,
schema cleaning, and conversational schema merging.
"""

import copy
from typing import Any

import pytest
from pydantic import BaseModel

from uipath_langchain.agent.exceptions import AgentStartupError
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model

# --- Fixtures: reusable schema fragments ---


@pytest.fixture()
def contact_def() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "fullname": {"type": "string"},
            "email": {"type": "string"},
        },
    }


@pytest.fixture()
def schema_with_defs(contact_def: dict[str, Any]) -> dict[str, Any]:
    """Schema with properly matched $ref and $defs."""
    return {
        "type": "object",
        "properties": {
            "owner": {"$ref": "#/$defs/Contact"},
        },
        "$defs": {
            "Contact": contact_def,
        },
    }


# --- 1. Dangling $ref (unresolvable type references) ---


class TestDanglingRef:
    """Schemas where $ref points to a type not in $defs."""

    def test_ref_to_missing_defs_raises(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "owner": {"$ref": "#/$defs/Contact"},
            },
        }
        with pytest.raises(AgentStartupError, match=r"Contact.*could not be resolved"):
            create_model(schema)

    def test_bare_ref_without_defs_raises(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "owner": {"$ref": "Contact"},
            },
        }
        with pytest.raises(AgentStartupError, match=r"Contact.*could not be resolved"):
            create_model(schema)

    def test_nested_defs_with_root_relative_ref_raises(self) -> None:
        """$defs inside 'items' with root-relative $ref paths.

        When $defs are placed inside a nested object (e.g. array items)
        but $ref uses root-relative paths (#/$defs/...), the converter
        cannot reach the definitions.
        """
        schema = {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "author": {"$ref": "#/$defs/Person"},
                            "title": {"type": "string"},
                        },
                        "$defs": {
                            "Person": {
                                "type": "object",
                                "properties": {
                                    "Name": {"type": "string"},
                                    "Email": {"type": "string"},
                                },
                            },
                            "Timestamp": {
                                "type": "string",
                                "format": "date-time",
                            },
                        },
                    },
                }
            },
        }
        with pytest.raises(AgentStartupError, match=r"Person.*could not be resolved"):
            create_model(schema)

    def test_ref_to_partial_defs_raises_for_missing_type(self) -> None:
        """$defs has some types but not the one referenced."""
        schema = {
            "type": "object",
            "properties": {
                "report": {"$ref": "#/$defs/Report"},
            },
            "$defs": {
                "Report": {
                    "type": "object",
                    "properties": {
                        "reviewer": {"$ref": "#/$defs/Contact"},
                    },
                },
                # Contact is missing
            },
        }
        with pytest.raises(AgentStartupError, match=r"Contact.*could not be resolved"):
            create_model(schema)


# --- 2. Valid $ref/$defs (happy paths) ---


class TestValidDefs:
    """Schemas where $ref and $defs are properly matched."""

    def test_ref_with_matching_defs(self, schema_with_defs: dict[str, Any]) -> None:
        model = create_model(schema_with_defs)
        assert model.__pydantic_complete__
        assert "owner" in model.model_fields

    def test_cross_referencing_defs(self, contact_def: dict[str, Any]) -> None:
        schema = {
            "type": "object",
            "properties": {
                "report": {"$ref": "#/$defs/Report"},
            },
            "$defs": {
                "Report": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "reviewer": {"$ref": "#/$defs/Contact"},
                    },
                },
                "Contact": contact_def,
            },
        }
        model = create_model(schema)
        assert model.__pydantic_complete__

    def test_definitions_keyword(self, contact_def: dict[str, Any]) -> None:
        """Old-style 'definitions' keyword (not $defs)."""
        schema = {
            "type": "object",
            "properties": {
                "owner": {"$ref": "#/definitions/Contact"},
            },
            "definitions": {
                "Contact": contact_def,
            },
        }
        model = create_model(schema)
        assert model.__pydantic_complete__


# --- 3. Static args round-trip (model → JSON schema → model) ---


class TestStaticArgsRoundTrip:
    """Simulates static_args.py: model_json_schema() → modify → create_model().

    The round-trip regenerates $defs with Pydantic-chosen keys.
    All $ref entries must still resolve after the round-trip.
    """

    def test_round_trip_preserves_defs(self, schema_with_defs: dict[str, Any]) -> None:
        model = create_model(schema_with_defs)
        round_tripped = model.model_json_schema()
        model2 = create_model(round_tripped)
        assert model2.__pydantic_complete__

    def test_round_trip_with_cross_refs(self, contact_def: dict[str, Any]) -> None:
        schema = {
            "type": "object",
            "properties": {
                "report": {"$ref": "#/$defs/Report"},
                "reviewer": {"$ref": "#/$defs/Contact"},
            },
            "$defs": {
                "Report": {
                    "type": "object",
                    "properties": {
                        "owner": {"$ref": "#/$defs/Contact"},
                        "items": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Contact"},
                        },
                    },
                },
                "Contact": contact_def,
            },
        }
        model = create_model(schema)
        round_tripped = model.model_json_schema()
        assert "$defs" in round_tripped or "definitions" in round_tripped
        model2 = create_model(round_tripped)
        assert model2.__pydantic_complete__

    def test_round_trip_after_property_removal(
        self, schema_with_defs: dict[str, Any]
    ) -> None:
        """Simulates static_args removing a property from the schema."""
        model = create_model(schema_with_defs)
        round_tripped = model.model_json_schema()

        # Add a simple field, then remove it (like static_args does)
        round_tripped["properties"]["extra"] = {"type": "string"}
        round_tripped["properties"].pop("extra")

        model2 = create_model(round_tripped)
        assert model2.__pydantic_complete__


# --- 4. Schema cleaning (integration tools) ---


class TestSchemaCleaning:
    """Tests that schema cleaning functions don't break $ref/$defs pairing."""

    def test_remove_asterisk_cleans_defs_keys(self) -> None:
        """remove_asterisk_from_properties cleans $defs keys alongside $ref values."""
        from uipath_langchain.agent.tools.integration_tool import (
            remove_asterisk_from_properties,
        )

        schema = {
            "type": "object",
            "properties": {
                "items[*]": {"$ref": "#/$defs/Record[*]"},
            },
            "$defs": {
                "Record[*]": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        cleaned = remove_asterisk_from_properties(schema)

        # $ref is cleaned
        assert "[*]" not in cleaned["properties"]["items"]["$ref"]
        # $defs key is also cleaned
        assert "Record" in cleaned["$defs"]
        assert "Record[*]" not in cleaned["$defs"]

        # The cleaned schema can be converted to a Pydantic model
        model = create_model(cleaned)
        assert model.__pydantic_complete__

    def test_remove_asterisk_with_definitions_keyword_works(self) -> None:
        """remove_asterisk correctly cleans keys when using 'definitions'."""
        from uipath_langchain.agent.tools.integration_tool import (
            remove_asterisk_from_properties,
        )

        schema = {
            "type": "object",
            "properties": {
                "items[*]": {"$ref": "#/definitions/Record[*]"},
            },
            "definitions": {
                "Record[*]": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        cleaned = remove_asterisk_from_properties(schema)
        assert "Record" in cleaned["definitions"]
        model = create_model(cleaned)
        assert model.__pydantic_complete__

    def test_remove_asterisk_no_asterisks_passthrough(self) -> None:
        from uipath_langchain.agent.tools.integration_tool import (
            remove_asterisk_from_properties,
        )

        schema = {
            "type": "object",
            "properties": {
                "owner": {"$ref": "#/$defs/Contact"},
            },
            "$defs": {
                "Contact": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        cleaned = remove_asterisk_from_properties(schema)
        model = create_model(cleaned)
        assert model.__pydantic_complete__


# --- 5. Conversational schema merging ---


class TestSchemaMerging:
    """Simulates _merge_system_model_into_existing_schema from factory.py."""

    @staticmethod
    def _merge(
        existing: dict[str, Any] | None, system_model: type[BaseModel]
    ) -> dict[str, Any]:
        """Replicates factory.py merge logic."""
        system_schema = system_model.model_json_schema()
        if not existing:
            return system_schema

        schema = copy.deepcopy(existing)
        if "properties" not in schema:
            schema["properties"] = {}
        schema["properties"].update(system_schema["properties"])

        if "$defs" in system_schema:
            if "$defs" not in schema:
                schema["$defs"] = {}
            schema["$defs"].update(system_schema["$defs"])

        if "required" in system_schema:
            system_required = set(system_schema["required"])
            current_required = set(schema.get("required", []))
            schema["required"] = list(current_required | system_required)

        return schema

    def test_merge_adds_system_defs(self) -> None:
        class SystemInput(BaseModel):
            messages: list[str] = []

        existing = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }
        merged = self._merge(existing, SystemInput)
        model = create_model(merged)
        assert model.__pydantic_complete__
        assert "messages" in model.model_fields

    def test_merge_preserves_user_defs(self, contact_def: dict[str, Any]) -> None:
        class SystemInput(BaseModel):
            messages: list[str] = []

        existing = {
            "type": "object",
            "properties": {
                "owner": {"$ref": "#/$defs/Contact"},
            },
            "$defs": {
                "Contact": contact_def,
            },
        }
        merged = self._merge(existing, SystemInput)
        assert "Contact" in merged["$defs"]

        model = create_model(merged)
        assert model.__pydantic_complete__
        assert "owner" in model.model_fields

    def test_merge_with_conflicting_defs_names(self) -> None:
        """System and user schemas both define a type with the same name."""

        class Metadata(BaseModel):
            source: str = "system"

        class SystemInput(BaseModel):
            meta: Metadata = Metadata()

        existing = {
            "type": "object",
            "properties": {
                "meta": {"$ref": "#/$defs/Metadata"},
            },
            "$defs": {
                "Metadata": {
                    "type": "object",
                    "properties": {"source": {"type": "string", "default": "user"}},
                },
            },
        }
        merged = self._merge(existing, SystemInput)
        model = create_model(merged)
        assert model.__pydantic_complete__


# --- 6. Pseudo-module isolation across multiple create_model calls ---


class TestPseudoModuleIsolation:
    """Multiple create_model calls share one pseudo-module.

    Each call must produce a working model regardless of prior calls.
    """

    def test_sequential_models_with_same_def_name(
        self, contact_def: dict[str, Any]
    ) -> None:
        schema_a = {
            "type": "object",
            "properties": {"owner": {"$ref": "#/$defs/Contact"}},
            "$defs": {"Contact": contact_def},
        }
        schema_b = {
            "type": "object",
            "properties": {"author": {"$ref": "#/$defs/Contact"}},
            "$defs": {
                "Contact": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                }
            },
        }

        model_a = create_model(schema_a)
        model_b = create_model(schema_b)

        assert model_a.__pydantic_complete__
        assert model_b.__pydantic_complete__
        assert "owner" in model_a.model_fields
        assert "author" in model_b.model_fields

    def test_model_from_simple_schema_after_complex(
        self, contact_def: dict[str, Any]
    ) -> None:
        complex_schema = {
            "type": "object",
            "properties": {
                "report": {"$ref": "#/$defs/Report"},
            },
            "$defs": {
                "Report": {
                    "type": "object",
                    "properties": {
                        "owner": {"$ref": "#/$defs/Contact"},
                    },
                },
                "Contact": contact_def,
            },
        }
        simple_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        create_model(complex_schema)
        model = create_model(simple_schema)

        assert model.__pydantic_complete__
        assert "name" in model.model_fields
