"""Tests for jsonschema_pydantic_converter — create_model() and create_output_model()."""

import copy
import logging
from typing import Any

import pytest

from uipath_langchain.agent.exceptions import AgentStartupError
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    _UNRESOLVED_TYPE_TITLE,
    _neutralize_dangling_refs,
    _ref_resolves,
    create_model,
    create_output_model,
)

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

    def test_malformed_ref_path_raises(self) -> None:
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
        assert "report" in model.model_fields

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
        assert "owner" in model.model_fields


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
        assert "owner" in model2.model_fields

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
        assert "report" in model2.model_fields
        assert "reviewer" in model2.model_fields

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
        assert "owner" in model2.model_fields


# --- 4. Pseudo-module isolation across multiple create_model calls ---


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

        complex_model = create_model(complex_schema)
        model = create_model(simple_schema)

        assert complex_model.__pydantic_complete__
        assert "report" in complex_model.model_fields
        assert model.__pydantic_complete__
        assert "name" in model.model_fields


# --- 5. Output-schema pre-pass: dangling $refs are neutralized, never fatal ---


class TestRefResolves:
    """_ref_resolves: only a local pointer with a present target resolves."""

    def test_present_local_ref_resolves(self, schema_with_defs: dict[str, Any]) -> None:
        assert _ref_resolves("#/$defs/Contact", schema_with_defs) is True

    def test_missing_local_ref_does_not_resolve(self) -> None:
        assert _ref_resolves("#/$defs/Missing", {"$defs": {}}) is False

    def test_definitions_keyword_resolves(self) -> None:
        root = {"definitions": {"Foo": {"type": "object"}}}
        assert _ref_resolves("#/definitions/Foo", root) is True

    def test_nested_pointer(self) -> None:
        root = {"$defs": {"A": {"$defs": {"B": {"type": "string"}}}}}
        assert _ref_resolves("#/$defs/A/$defs/B", root) is True
        assert _ref_resolves("#/$defs/A/$defs/Missing", root) is False

    def test_external_and_bare_refs_do_not_resolve(self) -> None:
        assert _ref_resolves("https://example.com/Foo", {}) is False
        assert _ref_resolves("#", {}) is False
        assert _ref_resolves("Contact", {}) is False


class TestNeutralizeDanglingRefs:
    """_neutralize_dangling_refs: surgical, in-place, preserves valid nodes."""

    def test_valid_schema_returned_unchanged(
        self, schema_with_defs: dict[str, Any]
    ) -> None:
        sanitized, dropped = _neutralize_dangling_refs(schema_with_defs)
        assert dropped == []
        assert sanitized == schema_with_defs

    def test_dangling_top_level_ref_neutralized(self) -> None:
        schema = {
            "type": "object",
            "properties": {"amount": {"$ref": "#/$defs/Missing"}},
        }
        sanitized, dropped = _neutralize_dangling_refs(schema)
        assert dropped == ["#/$defs/Missing"]
        node = sanitized["properties"]["amount"]
        assert "$ref" not in node
        assert node["title"] == _UNRESOLVED_TYPE_TITLE
        assert "#/$defs/Missing" in node["description"]

    def test_valid_sibling_and_valid_ref_preserved(
        self, contact_def: dict[str, Any]
    ) -> None:
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "owner": {"$ref": "#/$defs/Contact"},
                "amount": {"$ref": "#/$defs/Missing"},
            },
            "$defs": {"Contact": contact_def},
        }
        sanitized, dropped = _neutralize_dangling_refs(schema)
        assert dropped == ["#/$defs/Missing"]
        assert sanitized["properties"]["status"] == {"type": "string"}
        assert sanitized["properties"]["owner"] == {"$ref": "#/$defs/Contact"}
        assert sanitized["properties"]["amount"]["title"] == _UNRESOLVED_TYPE_TITLE

    def test_nested_in_array_items_neutralized(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "rows": {"type": "array", "items": {"$ref": "#/$defs/Missing"}},
            },
        }
        sanitized, dropped = _neutralize_dangling_refs(schema)
        assert dropped == ["#/$defs/Missing"]
        items = sanitized["properties"]["rows"]["items"]
        assert items["title"] == _UNRESOLVED_TYPE_TITLE

    def test_dangling_ref_inside_valid_def_neutralized(self) -> None:
        schema = {
            "type": "object",
            "properties": {"w": {"$ref": "#/$defs/Wrapper"}},
            "$defs": {
                "Wrapper": {
                    "type": "object",
                    "properties": {"x": {"$ref": "#/$defs/Missing"}},
                },
            },
        }
        sanitized, dropped = _neutralize_dangling_refs(schema)
        assert dropped == ["#/$defs/Missing"]
        # outer, resolvable ref kept; inner dangling ref neutralized
        assert sanitized["properties"]["w"] == {"$ref": "#/$defs/Wrapper"}
        inner = sanitized["$defs"]["Wrapper"]["properties"]["x"]
        assert inner["title"] == _UNRESOLVED_TYPE_TITLE

    def test_does_not_mutate_input(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/Missing"}},
        }
        original = copy.deepcopy(schema)
        _neutralize_dangling_refs(schema)
        assert schema == original


class TestCreateOutputModel:
    """create_output_model: never crashes on dangling refs; keeps valid fields."""

    def test_neutralized_field_accepts_any_and_keeps_original_ref(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "amount": {"$ref": "#/$defs/Nullableofdecimal", "type": "object"},
            },
        }
        model = create_output_model(schema, "my_tool")
        props = model.model_json_schema()["properties"]
        assert "status" in props  # valid sibling preserved
        assert props["amount"]["title"] == _UNRESOLVED_TYPE_TITLE
        assert "Nullableofdecimal" in props["amount"]["description"]
        # Permissive: every value kind validates (a typed field would raise here).
        for value in [1.5, "text", {"k": 1}, [1, 2], True, None]:
            model.model_validate({"status": "ok", "amount": value})

    def test_valid_schema_matches_create_model_fields(
        self, schema_with_defs: dict[str, Any]
    ) -> None:
        """A valid schema is passed through untouched: same fields as create_model."""
        via_output = create_output_model(schema_with_defs, "t")
        via_direct = create_model(schema_with_defs)
        assert via_output.model_fields.keys() == via_direct.model_fields.keys()
        assert "owner" in via_output.model_fields

    def test_many_fields_distinct_refs_create_no_named_types(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "f1": {"$ref": "#/$defs/A"},
                "f2": {"$ref": "#/$defs/A"},
                "f3": {"$ref": "#/$defs/B"},
                "f4": {"$ref": "#/$defs/B"},
                "f5": {"$ref": "#/$defs/C"},
            },
        }
        model = create_output_model(schema, "t")
        js = model.model_json_schema()
        # no named types are generated -> nothing to collide / deduplicate
        assert js.get("$defs", {}) == {}
        for field in ["f1", "f2", "f3", "f4", "f5"]:
            assert js["properties"][field]["title"] == _UNRESOLVED_TYPE_TITLE
        model.model_validate(
            {"f1": 1, "f2": "x", "f3": None, "f4": [1], "f5": {"a": 1}}
        )

    def test_non_ref_conversion_failure_is_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A conversion failure that is NOT a dangling ref must propagate (fail
        loud) -- there is deliberately no last-resort fallback that would swallow
        an unexpected malformation into a degraded model."""
        import uipath_langchain.agent.react.jsonschema_pydantic_converter as mod

        # Capture a real AgentStartupError, then make create_model raise it.
        try:
            mod.create_model(
                {"type": "object", "properties": {"x": {"$ref": "#/$defs/Missing"}}}
            )
            raise AssertionError("expected AgentStartupError")
        except AgentStartupError as exc:
            captured = exc

        def always_raise(_schema: dict[str, Any]) -> Any:
            raise captured

        monkeypatch.setattr(mod, "create_model", always_raise)

        with pytest.raises(AgentStartupError):
            mod.create_output_model(
                {"type": "object", "properties": {"y": {"type": "string"}}}, "t"
            )

    def test_neutralized_refs_are_logged_with_tool_and_ref(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dropped refs must be logged (named) so the degradation is debuggable."""
        schema = {
            "type": "object",
            "properties": {"amount": {"$ref": "#/$defs/Nullableofdecimal"}},
        }
        with caplog.at_level(logging.WARNING):
            create_output_model(schema, "my_tool")
        assert "my_tool" in caplog.text
        assert "#/$defs/Nullableofdecimal" in caplog.text
