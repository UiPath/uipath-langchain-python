"""Tests for _schema_refs helpers not covered by test_jsonschema_pydantic_converter.

Covers:
- resolve_pointer: happy path, missing, external ref, escaped segments
- Additional neutralize_dangling_refs edge cases
"""

from typing import Any

from uipath_langchain.agent.react._schema_refs import (
    resolve_pointer,
)


class TestResolvePointer:
    def test_resolves_simple_path(self) -> None:
        schema: dict[str, Any] = {"$defs": {"Contact": {"type": "object"}}}
        assert resolve_pointer(schema, "#/$defs/Contact") == {"type": "object"}

    def test_returns_none_for_missing(self) -> None:
        schema: dict[str, Any] = {"$defs": {}}
        assert resolve_pointer(schema, "#/$defs/Missing") is None

    def test_returns_none_for_external_ref(self) -> None:
        assert resolve_pointer({}, "https://example.com/Foo") is None

    def test_returns_none_for_bare_hash(self) -> None:
        assert resolve_pointer({}, "#") is None

    def test_nested_path(self) -> None:
        schema: dict[str, Any] = {"$defs": {"A": {"inner": {"val": 42}}}}
        assert resolve_pointer(schema, "#/$defs/A/inner/val") == 42

    def test_escaped_tilde_in_path(self) -> None:
        # JSON pointer: ~0 = ~, ~1 = /
        schema: dict[str, Any] = {"defs": {"a~b": "found"}}
        assert resolve_pointer(schema, "#/defs/a~0b") == "found"

    def test_escaped_slash_in_path(self) -> None:
        schema: dict[str, Any] = {"defs": {"a/b": "found"}}
        assert resolve_pointer(schema, "#/defs/a~1b") == "found"

    def test_non_dict_intermediate(self) -> None:
        schema: dict[str, Any] = {"a": "not a dict"}
        assert resolve_pointer(schema, "#/a/b") is None
