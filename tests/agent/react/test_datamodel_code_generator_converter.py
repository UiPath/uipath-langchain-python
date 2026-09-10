"""Tests for internal helpers in _datamodel_code_generator_converter.

These helpers are thoroughly exercised indirectly through scenario tests, but
these unit tests pin specific edge cases and make regressions easier to locate.
"""

from typing import Any

from pydantic import BaseModel, Field

from uipath_langchain.agent.react._datamodel_code_generator_converter import (
    _child_paths,
    _definition_type_name,
    _fields_by_json_name,
    _is_unenforceable,
    _iter_refs,
    _models_in,
    _nested_class_name,
    _root_class_name,
    _unresolved_type_name,
    _valid_identifier,
    _values_at_path,
)

# ---------------------------------------------------------------------------
# _definition_type_name
# ---------------------------------------------------------------------------


class TestDefinitionTypeName:
    def test_standard_defs_ref(self) -> None:
        assert _definition_type_name("#/$defs/Contact") == "__Contact"

    def test_definitions_keyword_ref(self) -> None:
        assert _definition_type_name("#/definitions/Contact") == "__Contact"

    def test_nested_defs_ref(self) -> None:
        # "$defs" segments are stripped from the name
        name = _definition_type_name("#/$defs/Outer/$defs/Inner")
        assert "Outer" in name
        assert "Inner" in name.lower() or "inner" in name

    def test_special_characters_sanitized(self) -> None:
        name = _definition_type_name("#/$defs/My-Type.V2")
        assert "-" not in name
        assert "." not in name

    def test_external_ref(self) -> None:
        name = _definition_type_name("https://example.com/schemas/Foo")
        assert "Foo" in name


# ---------------------------------------------------------------------------
# _unresolved_type_name
# ---------------------------------------------------------------------------


class TestUnresolvedTypeName:
    def test_simple_ref(self) -> None:
        assert _unresolved_type_name("#/$defs/Contact") == "Contact"

    def test_trailing_slash(self) -> None:
        assert _unresolved_type_name("#/$defs/Contact/") == "Contact"

    def test_bare_ref(self) -> None:
        assert _unresolved_type_name("Contact") == "Contact"


# ---------------------------------------------------------------------------
# _iter_refs
# ---------------------------------------------------------------------------


class TestIterRefs:
    def test_no_refs(self) -> None:
        assert list(_iter_refs({"type": "object"})) == []

    def test_single_ref(self) -> None:
        assert list(_iter_refs({"$ref": "#/$defs/A"})) == ["#/$defs/A"]

    def test_nested_refs(self) -> None:
        schema = {
            "properties": {
                "a": {"$ref": "#/$defs/A"},
                "b": {"type": "array", "items": {"$ref": "#/$defs/B"}},
            }
        }
        refs = list(_iter_refs(schema))
        assert "#/$defs/A" in refs
        assert "#/$defs/B" in refs

    def test_refs_in_list(self) -> None:
        schema = {"anyOf": [{"$ref": "#/$defs/X"}, {"$ref": "#/$defs/Y"}]}
        assert set(_iter_refs(schema)) == {"#/$defs/X", "#/$defs/Y"}

    def test_non_string_ref_ignored(self) -> None:
        assert list(_iter_refs({"$ref": 42})) == []


# ---------------------------------------------------------------------------
# _valid_identifier
# ---------------------------------------------------------------------------


class TestValidIdentifier:
    def test_normal_name(self) -> None:
        assert _valid_identifier("Contact", "Model") == "Contact"

    def test_leading_underscore_stripped(self) -> None:
        assert _valid_identifier("___Foo", "Model") == "Foo"

    def test_leading_digit_gets_fallback(self) -> None:
        result = _valid_identifier("123Type", "Model")
        assert result.startswith("Model")

    def test_keyword_gets_suffix(self) -> None:
        assert _valid_identifier("class", "Model") == "class_"

    def test_empty_string_uses_fallback(self) -> None:
        assert _valid_identifier("", "Fallback") == "Fallback"

    def test_all_underscores_uses_fallback(self) -> None:
        result = _valid_identifier("___", "Fallback")
        assert result == "Fallback"


# ---------------------------------------------------------------------------
# _root_class_name / _nested_class_name
# ---------------------------------------------------------------------------


class TestClassNames:
    def test_root_preserves_title(self) -> None:
        assert _root_class_name("MyModel") == "MyModel"

    def test_root_sanitizes_special_chars(self) -> None:
        name = _root_class_name("My-Model.V2")
        assert "-" not in name
        assert "." not in name

    def test_nested_pascal_case(self) -> None:
        assert _nested_class_name("order_item") == "OrderItem"

    def test_nested_strips_special(self) -> None:
        name = _nested_class_name("my-type.v2")
        # Should be PascalCase
        assert name[0].isupper()


# ---------------------------------------------------------------------------
# _values_at_path
# ---------------------------------------------------------------------------


class TestValuesAtPath:
    def test_empty_path(self) -> None:
        assert _values_at_path(42, ()) == [42]

    def test_dict_path(self) -> None:
        assert _values_at_path({"a": {"b": 1}}, ("a", "b")) == [1]

    def test_array_expansion(self) -> None:
        data = [{"x": 1}, {"x": 2}]
        assert _values_at_path(data, ("[]", "x")) == [1, 2]

    def test_missing_key(self) -> None:
        assert _values_at_path({"a": 1}, ("b",)) == []

    def test_array_marker_on_non_list(self) -> None:
        assert _values_at_path("not a list", ("[]",)) == []

    def test_nested_array(self) -> None:
        data = {"items": [{"vals": [1, 2]}, {"vals": [3]}]}
        assert _values_at_path(data, ("items", "[]", "vals")) == [[1, 2], [3]]


# ---------------------------------------------------------------------------
# _is_unenforceable
# ---------------------------------------------------------------------------


class TestIsUnenforceable:
    def test_not_keyword(self) -> None:
        assert _is_unenforceable({"not": {"type": "null"}}) is True

    def test_prefix_items(self) -> None:
        assert _is_unenforceable({"prefixItems": [{"type": "string"}]}) is True

    def test_empty_enum(self) -> None:
        assert _is_unenforceable({"enum": []}) is True

    def test_normal_schema(self) -> None:
        assert _is_unenforceable({"type": "string"}) is False

    def test_non_empty_enum(self) -> None:
        assert _is_unenforceable({"enum": ["a", "b"]}) is False


# ---------------------------------------------------------------------------
# _fields_by_json_name
# ---------------------------------------------------------------------------


class TestFieldsByJsonName:
    def test_no_alias(self) -> None:
        class M(BaseModel):
            name: str

        result = _fields_by_json_name(M)
        assert "name" in result

    def test_with_alias(self) -> None:
        class M(BaseModel):
            content_type: str = Field(alias="Content-Type")

        result = _fields_by_json_name(M)
        assert "Content-Type" in result
        assert "content_type" not in result


# ---------------------------------------------------------------------------
# _models_in
# ---------------------------------------------------------------------------


class TestModelsIn:
    def test_plain_type(self) -> None:
        assert _models_in(str) == []

    def test_basemodel_subclass(self) -> None:
        class M(BaseModel):
            pass

        assert _models_in(M) == [M]

    def test_optional_model(self) -> None:
        from typing import Optional

        class M(BaseModel):
            pass

        result = _models_in(Optional[M])
        assert M in result

    def test_list_of_models(self) -> None:
        class M(BaseModel):
            pass

        result = _models_in(list[M])
        assert M in result


# ---------------------------------------------------------------------------
# _child_paths
# ---------------------------------------------------------------------------


class TestChildPaths:
    def test_properties(self) -> None:
        node: dict[str, Any] = {
            "properties": {"a": {"type": "string"}, "b": {"type": "int"}}
        }
        paths = list(_child_paths(node, ()))
        subs = {p[1] for p in paths}
        assert ("a",) in subs
        assert ("b",) in subs

    def test_items(self) -> None:
        node: dict[str, Any] = {"items": {"type": "string"}}
        paths = list(_child_paths(node, ("arr",)))
        assert any(p[1] == ("arr", "[]") for p in paths)

    def test_combiners(self) -> None:
        node: dict[str, Any] = {"anyOf": [{"type": "string"}, {"type": "int"}]}
        paths = list(_child_paths(node, ("x",)))
        # combiners keep the parent path
        assert all(p[1] == ("x",) for p in paths)
        assert len(paths) == 2
