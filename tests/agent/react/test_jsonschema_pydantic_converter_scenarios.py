"""Scenario coverage for schema -> model conversion.

The contract tests in ``test_jsonschema_pydantic_converter.py`` cover dangling
refs, the static-args round trip and pseudo-module isolation. This module covers
the behaviour the runtime depends on but that no single caller asserts: the
inline object holding a ``$ref`` that once produced an incomplete model,
constraints the generated types cannot carry, property names that are not valid
Python identifiers, and the marker/module lookups job-attachment discovery
relies on.
"""

import json
import sys
from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ValidationError
from pydantic import create_model as pydantic_create_model
from uipath.core.feature_flags import FeatureFlags

from uipath_langchain.agent.attachments.pydantic_json import get_json_paths_by_type
from uipath_langchain.agent.react import (
    _datamodel_code_generator_converter,
    _legacy_converter,
)
from uipath_langchain.agent.react import (
    jsonschema_pydantic_converter as converter,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    DATAMODEL_CODE_GENERATOR_CONVERTER_FF,
    create_model,
    create_output_model,
)
from uipath_langchain.agent.tools.base_uipath_structured_tool import (
    BaseUiPathStructuredTool,
)

# Every test in this module runs against both backends. Requested through
# ``pytestmark`` rather than autouse so the parametrization is explicit.
pytestmark = pytest.mark.usefixtures("schema_backend")


@pytest.fixture(params=[False, True], ids=["legacy", "datamodel_code_generator"])
def schema_backend(request: pytest.FixtureRequest) -> Any:
    """Run every test in this module against both conversion backends.

    The two are meant to be interchangeable behind
    ``DATAMODEL_CODE_GENERATOR_CONVERTER_FF``, so the contract is only proven if
    it holds either way.
    """
    FeatureFlags.reset_flags()
    FeatureFlags.configure_flags({DATAMODEL_CODE_GENERATOR_CONVERTER_FF: request.param})
    yield request.param
    FeatureFlags.reset_flags()


def _clone_like_langchain(model: type[BaseModel]) -> type[BaseModel]:
    """Copy field annotations into a new model in a different module.

    This is what ``BaseTool.tool_call_schema`` does via ``_create_subset_model``,
    and it is where an incompletely-defined nested model surfaces.
    """
    clone = pydantic_create_model(  # type: ignore[call-overload]
        model.__name__,
        **{
            name: (field.annotation, field)
            for name, field in model.model_fields.items()
        },
    )
    clone.model_json_schema()  # forces the core schema to be built
    return clone


def _nested_models(model: type[BaseModel], field_name: str) -> list[type[BaseModel]]:
    annotation = model.model_fields[field_name].annotation
    return [
        arg
        for arg in getattr(annotation, "__args__", ())
        if isinstance(arg, type) and issubclass(arg, BaseModel)
    ]


# --- an object written inline that contains a $ref ----------------------------


CREATE_ISSUE = {
    "title": "Create_Issue",
    "type": "object",
    "properties": {
        # written inline rather than under $defs, and holding a $ref
        "fields": {
            "type": "object",
            "properties": {
                "project": {"$ref": "#/$defs/Project"},
                "summary": {"type": "string"},
            },
        },
    },
    "$defs": {
        "Project": {"type": "object", "properties": {"key": {"type": "string"}}},
    },
}


class TestInlineObjectWithRef:
    """The nested model must be fully defined, not just the root.

    Asserted against both backends: this is a shared guarantee, not a reason to
    pick one over the other.
    """

    def test_nested_model_is_complete(self) -> None:
        model = create_model(CREATE_ISSUE)
        assert model.__pydantic_complete__
        nested = _nested_models(model, "fields")
        assert nested, "the inline object should have produced a model"
        for inner in nested:
            assert inner.__pydantic_complete__

    def test_clone_into_another_module_succeeds(self) -> None:
        _clone_like_langchain(create_model(CREATE_ISSUE))

    def test_survives_the_static_args_round_trip(self) -> None:
        """model -> JSON Schema -> inline a $ref -> model, as static args does."""
        first = create_model(CREATE_ISSUE)
        round_tripped = first.model_json_schema()

        # schema_editing inlines a copy of the $ref it navigates through, leaving
        # sibling properties pointing at $defs.
        holder = round_tripped["properties"]["fields"]
        target = holder["anyOf"][0] if "anyOf" in holder else holder
        ref = target.get("$ref")
        if ref:
            name = ref.rsplit("/", 1)[1]
            inlined = json.loads(json.dumps(round_tripped["$defs"][name]))
            if "anyOf" in holder:
                holder["anyOf"][0] = inlined
            else:
                round_tripped["properties"]["fields"] = inlined

        second = create_model(round_tripped)
        assert second.__pydantic_complete__
        for inner in _nested_models(second, "fields"):
            assert inner.__pydantic_complete__
        _clone_like_langchain(second)

    def test_inline_object_nested_inside_a_def(self) -> None:
        schema = {
            "title": "Root",
            "type": "object",
            "properties": {"outer": {"$ref": "#/$defs/Outer"}},
            "$defs": {
                "Outer": {
                    "type": "object",
                    "properties": {
                        "inline": {
                            "type": "object",
                            "properties": {"leaf": {"$ref": "#/$defs/Leaf"}},
                        }
                    },
                },
                "Leaf": {"type": "object", "properties": {"k": {"type": "string"}}},
            },
        }
        model = create_model(schema)
        (outer,) = _nested_models(model, "outer")
        assert outer.__pydantic_complete__
        for inline in _nested_models(outer, "inline"):
            assert inline.__pydantic_complete__

    def test_tool_definition_reaches_the_model(self) -> None:
        """The failure mode was at tool-binding time, not tool execution."""
        model = create_model(CREATE_ISSUE)
        tool = StructuredTool(
            name="Create_Issue",
            description="Create an issue",
            args_schema=model,
            func=lambda **kwargs: kwargs,
        )
        spec = convert_to_openai_tool(tool)
        assert spec["function"]["name"] == "Create_Issue"

    def test_root_keeps_the_title_from_the_schema(self) -> None:
        """The root's name is the title the model sees; it must not be rewritten."""
        assert create_model(CREATE_ISSUE).model_json_schema()["title"] == "Create_Issue"


# --- constraints the generated types cannot carry ----------------------------


class TestUnenforceableConstraints:
    """``not``, ``prefixItems`` and an empty ``enum`` have no Pydantic annotation.

    They must still reject invalid input rather than silently accept anything.
    """

    @pytest.mark.parametrize(
        ("name", "schema", "invalid", "valid"),
        [
            (
                "not",
                {"type": "object", "properties": {"n": {"not": {"type": "string"}}}},
                {"n": "a string"},
                {"n": 42},
            ),
            (
                "prefixItems",
                {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "array",
                            "prefixItems": [{"type": "string"}, {"type": "integer"}],
                        }
                    },
                },
                {"pair": [1, "wrong order"]},
                {"pair": ["ok", 5]},
            ),
            (
                "empty enum",
                {"type": "object", "properties": {"e": {"type": "string", "enum": []}}},
                {"e": "anything"},
                None,
            ),
            (
                "inside an array",
                {
                    "type": "object",
                    "properties": {
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"n": {"not": {"type": "string"}}},
                            },
                        }
                    },
                },
                {"rows": [{"n": "a string"}]},
                {"rows": [{"n": 1}]},
            ),
            (
                "behind a $ref",
                {
                    "type": "object",
                    "properties": {"wrapped": {"$ref": "#/$defs/Wrapper"}},
                    "$defs": {
                        "Wrapper": {
                            "type": "object",
                            "properties": {"n": {"not": {"type": "string"}}},
                        }
                    },
                },
                {"wrapped": {"n": "a string"}},
                {"wrapped": {"n": 1}},
            ),
        ],
    )
    def test_constraint_is_enforced(
        self,
        name: str,
        schema: dict[str, Any],
        invalid: dict[str, Any],
        valid: dict[str, Any] | None,
    ) -> None:
        model = create_model(schema)
        with pytest.raises(ValidationError):
            model.model_validate(invalid)
        if valid is not None:
            model.model_validate(valid)

    def test_schema_without_such_keywords_is_not_wrapped(self) -> None:
        """The guard is only added where it is needed."""
        model = create_model(
            {"type": "object", "properties": {"a": {"type": "string"}}}
        )
        model.model_validate({"a": "x"})
        assert model.__pydantic_complete__


# --- property names that are not valid Python identifiers --------------------


class TestPropertyNaming:
    """Sanitized field names must not change what goes on the wire."""

    @pytest.mark.parametrize(
        "json_name",
        [
            "project key",  # space
            "project-key",  # hyphen
            "schema",  # shadows a Pydantic member
            "model_fields",  # shadows a Pydantic member
            "copy",  # shadows a Pydantic member
            "class",  # Python keyword
            "_leading",  # leading underscore
        ],
    )
    def test_original_name_round_trips(self, json_name: str) -> None:
        model = create_model(
            {
                "type": "object",
                "properties": {json_name: {"type": "string"}},
                "required": [json_name],
            }
        )
        instance = model.model_validate({json_name: "value"})
        # serialize_by_alias is what puts the original JSON name back on the wire
        assert instance.model_dump()[json_name] == "value"

    def test_llm_facing_schema_uses_the_original_name(self) -> None:
        model = create_model(
            {"type": "object", "properties": {"project key": {"type": "string"}}}
        )
        assert "project key" in model.model_json_schema()["properties"]

    @pytest.mark.parametrize(
        "json_name",
        [
            "plain",
            "Content-Type",  # routine in REST connector schemas
            "@odata.type",
            "first-name",
            "a b",
            "x.y",
            "1st",
            "class",  # Python keyword
            "schema",  # shadows a Pydantic member
            "copy",  # shadows a Pydantic member
        ],
    )
    def test_tool_invocation_delivers_the_original_name(self, json_name: str) -> None:
        """A tool call must reach the handler keyed by the declared JSON name.

        Asserting on ``model_dump()`` is not enough: LangChain builds the handler
        kwargs by dumping the validated model and reading each dumped key back off
        the instance with ``getattr``. Because dumps here are alias-keyed, that is
        a lookup by JSON name, which a sanitized field only answers through
        ``UiPathDatamodelCodeGeneratorBaseModel.__getattr__``. Names that shadow a
        Pydantic member need the alias fix-up in
        ``BaseUiPathStructuredTool._parse_input`` on top, since normal lookup
        succeeds there and returns the inherited method.
        """
        received: dict[str, Any] = {}

        def handler(**kwargs: Any) -> str:
            received.update(kwargs)
            return "called"

        model = create_model(
            {
                "type": "object",
                "properties": {json_name: {"type": "string"}},
                "required": [json_name],
            }
        )
        tool = BaseUiPathStructuredTool(
            name="a_tool", description="a tool", args_schema=model, func=handler
        )

        tool_input: dict[str, Any] = {json_name: "value"}
        assert tool.invoke(tool_input) == "called"
        assert received == tool_input


class TestFormatKeyword:
    """A ``format`` annotates a value; it must not change what the value is."""

    @pytest.mark.parametrize(
        ("json_type", "fmt", "value"),
        [
            ("string", "password", "hunter2"),  # must not serialize as a mask
            ("string", "email", "a@b.com"),  # must not need email-validator
            ("string", "ulid", "01ARZ3NDEKTSV4RRFFQ69G5FAV"),
            ("string", "date-time", "2024-01-02T03:04:05"),  # naive, as models emit
            ("string", "uuid", "550e8400-e29b-41d4-a716-446655440500"),
            ("string", "decimal", "1.5"),
            ("string", "binary", "abc"),
            ("string", "uri", "https://example.com/x"),
            ("string", "path", "/tmp/x"),
            ("string", "ipv4", "1.2.3.4"),
            ("integer", "int64", 9007199254740993),
            ("integer", "date-time", 17),
            ("number", "decimal", 1.5),
            ("number", "time-delta", 2.5),
        ],
    )
    def test_format_does_not_change_the_value(
        self, json_type: str, fmt: str, value: Any
    ) -> None:
        """The value survives the round trip unchanged, and stays serializable.

        Both assertions matter. Retyping on ``format`` costs the value itself in
        the worst case -- ``password`` comes back as a mask rather than the
        credential -- and costs serializability in the rest, since a ``UUID`` or
        ``Decimal`` reaches a connector through ``json.dumps``.
        """
        model = create_model(
            {
                "type": "object",
                "properties": {"v": {"type": json_type, "format": fmt}},
                "required": ["v"],
            }
        )
        dumped = model.model_validate({"v": value}).model_dump()
        assert dumped["v"] == value
        assert json.loads(json.dumps(dumped)) == {"v": value}


# --- marker and module contracts --------------------------------------------


class TestLookupContracts:
    """Job-attachment discovery finds types by name through the pseudo-module."""

    @pytest.mark.parametrize("def_name", ["job-attachment", "Job_attachment"])
    def test_definition_is_reachable_by_its_marker_name(self, def_name: str) -> None:
        model = create_model(
            {
                "type": "object",
                "properties": {"attachment": {"$ref": f"#/definitions/{def_name}"}},
                "definitions": {
                    def_name: {
                        "type": "object",
                        "properties": {"ID": {"type": "string"}},
                        "required": ["ID"],
                    }
                },
            }
        )
        # the name job_attachments.py asks for
        assert get_json_paths_by_type(model, "__Job_attachment") == ["$.attachment"]

    def test_root_lives_in_a_registered_module(self) -> None:
        model = create_model(CREATE_ISSUE)
        module = sys.modules.get(model.__module__)
        assert module is not None, "qualified-name lookups need a real module"
        assert getattr(module, model.__name__, None) is model

    def test_datamodel_code_generator_also_rehomes_inline_models(self) -> None:
        """Every class is published, including objects written inline.

        The legacy backend only re-homes the types it collected from ``$defs``,
        so an inline model keeps the converter's own module -- see
        ``TestLegacyBackendDeltas``.
        """
        FeatureFlags.configure_flags({DATAMODEL_CODE_GENERATOR_CONVERTER_FF: True})
        model = create_model(CREATE_ISSUE)
        module = sys.modules[model.__module__]
        inner_models = _nested_models(model, "fields")
        assert inner_models
        for inner in inner_models:
            assert inner.__module__ == model.__module__
            assert getattr(module, inner.__name__, None) is inner

    def test_same_definition_name_in_two_schemas_stays_separate(self) -> None:
        def schema(field: str) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {"item": {"$ref": "#/$defs/Shared"}},
                "$defs": {
                    "Shared": {
                        "type": "object",
                        "properties": {field: {"type": "string"}},
                    }
                },
            }

        first = create_model(schema("alpha"))
        second = create_model(schema("beta"))
        assert first.__module__ != second.__module__
        (first_inner,) = _nested_models(first, "item")
        (second_inner,) = _nested_models(second, "item")
        assert first_inner is not second_inner
        assert "alpha" in first_inner.model_fields
        assert "beta" in second_inner.model_fields


# --- additionalProperties / extra --------------------------------------------


class TestAdditionalProperties:
    def test_unset_allows_extra(self) -> None:
        model = create_model(
            {"type": "object", "properties": {"a": {"type": "string"}}}
        )
        instance = model.model_validate({"a": "x", "unexpected": 1})
        assert instance.model_dump()["unexpected"] == 1

    def test_false_forbids_extra(self) -> None:
        model = create_model(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"a": {"type": "string"}},
            }
        )
        with pytest.raises(ValidationError):
            model.model_validate({"a": "x", "unexpected": 1})

    def test_typed_additional_properties_are_validated(self) -> None:
        model = create_model(
            {
                "type": "object",
                "properties": {
                    "meta": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    }
                },
            }
        )
        model.model_validate({"meta": {"count": 3}})
        with pytest.raises(ValidationError):
            model.model_validate({"meta": {"count": "not a number"}})


# --- root shapes -------------------------------------------------------------


class TestRootNameCollision:
    """The model handed back is the document's own, whatever the generator named it.

    A schema whose ``title`` matches a ``$defs`` entry puts the two in
    competition for one class name, and the generator resolves the root last,
    so the definition can win the name the root asked for. Recovering the root
    by that name then returns the definition -- and nothing raises: the
    LLM-facing schema, the validation and the payload all describe the wrong
    type, and the argument the caller did send is swallowed as an extra.
    """

    _DEFINITION = {"type": "object", "properties": {"x": {"type": "string"}}}

    @pytest.mark.parametrize(
        ("schema", "properties"),
        [
            (
                {
                    "type": "object",
                    "title": "Root",
                    "properties": {"r": {"$ref": "#/$defs/Root"}},
                    "required": ["r"],
                    "$defs": {"Root": _DEFINITION},
                },
                {"r"},
            ),
            (
                {
                    "type": "object",
                    "properties": {"r": {"$ref": "#/$defs/Model"}},
                    "required": ["r"],
                    "$defs": {"Model": _DEFINITION},
                },
                {"r"},
            ),
            (
                {
                    "type": "object",
                    "title": "Root",
                    "properties": {"r": {"$ref": "#/$defs/Root"}},
                    "required": ["r"],
                    "$defs": {
                        "Root": _DEFINITION,
                        "Spare": {
                            "type": "object",
                            "properties": {"s": {"type": "string"}},
                        },
                    },
                },
                {"r"},
            ),
            (
                {
                    "type": "object",
                    "title": "Root",
                    "properties": {"a": {"$ref": "#/$defs/Root"}},
                    "required": ["a"],
                    "$defs": {
                        "Root": {
                            "type": "object",
                            "properties": {"a": {"type": "string"}},
                        }
                    },
                },
                {"a"},
            ),
            (
                {
                    "type": "object",
                    "title": "Root",
                    "properties": {"r": {"$ref": "#/$defs/Inner"}},
                    "required": ["r"],
                    "$defs": {"Inner": _DEFINITION},
                },
                {"r"},
            ),
        ],
        ids=[
            "title_collides_with_a_definition",
            "untitled_root_against_a_definition_named_Model",
            "collision_alongside_an_unreferenced_definition",
            "collision_where_the_definition_declares_the_same_name",
            "no_collision",
        ],
    )
    def test_root_carries_the_documents_own_properties(
        self, schema: dict[str, Any], properties: set[str]
    ) -> None:
        """The root's properties and its ``required`` both survive the collision.

        ``required`` is asserted separately because the properties alone do not
        always give the swap away -- where the definition happens to declare the
        same name, the returned model looks right and only the lost
        ``required`` shows that it is the wrong class.
        """
        model = create_model(schema)
        fields = {
            (field.alias or name): field for name, field in model.model_fields.items()
        }
        assert set(fields) == properties
        assert {
            name for name, field in fields.items() if field.is_required()
        } == properties

    def test_the_declared_argument_is_not_swallowed_as_an_extra(self) -> None:
        """The symptom the swap produces: the real argument silently disappears."""
        model = create_model(
            {
                "type": "object",
                "title": "Root",
                "properties": {"r": {"$ref": "#/$defs/Root"}},
                "required": ["r"],
                "$defs": {"Root": self._DEFINITION},
            }
        )
        assert model.model_validate({"r": {"x": "v"}}).model_dump() == {"r": {"x": "v"}}
        with pytest.raises(ValidationError):
            model.model_validate({})


class TestRootShapes:
    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "object", "properties": {}},
            {"type": "object"},
            {"type": "array", "items": {"type": "string"}},
            {"type": "string"},
        ],
    )
    def test_root_converts(self, schema: dict[str, Any]) -> None:
        model = create_model(schema)
        assert isinstance(model, type) and issubclass(model, BaseModel)

    def test_deeply_nested_inline_objects(self) -> None:
        node: dict[str, Any] = {"type": "string"}
        for _ in range(6):
            node = {"type": "object", "properties": {"child": node}}
        model = create_model({"title": "Deep", **node})
        _clone_like_langchain(model)

    def test_recursive_definition(self) -> None:
        model = create_model(
            {
                "title": "Node",
                "type": "object",
                "properties": {"next": {"$ref": "#/$defs/Node"}},
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "next": {"$ref": "#/$defs/Node"},
                        },
                    }
                },
            }
        )
        model.model_validate({"next": {"value": "a", "next": {"value": "b"}}})


# --- output schemas ----------------------------------------------------------


class TestOutputModel:
    def test_dangling_ref_is_neutralized_not_fatal(self) -> None:
        model = create_output_model(
            {
                "type": "object",
                "properties": {
                    "good": {"type": "string"},
                    "bad": {"$ref": "#/$defs/Missing"},
                },
            },
            "some_tool",
        )
        # the unresolvable field accepts anything, and the valid sibling survives
        model.model_validate({"good": "x", "bad": {"anything": True}})
        assert "good" in model.model_json_schema()["properties"]


# --- how the two backends still differ --------------------------------------


class TestBackendDifferences:
    """Behaviour that is not identical across backends, pinned deliberately.

    An incomplete nested model is not among them -- both backends produce a
    complete one. What is left is why the datamodel-code-generator backend is
    still worth having.
    """

    @pytest.fixture(autouse=True)
    def schema_backend(self) -> Any:
        """Override the module fixture: these tests switch backends themselves."""
        FeatureFlags.reset_flags()
        yield
        FeatureFlags.reset_flags()

    def _inline_model(self, model: type[BaseModel]) -> type[BaseModel]:
        (inner,) = _nested_models(model, "fields")
        return inner

    def test_legacy_leaves_inline_models_on_a_shared_module(self) -> None:
        """The legacy wrapper only re-homes the types it collected from ``$defs``.

        An inline object's model keeps the converter's own module, which every
        conversion shares -- and those class names repeat across schemas, so a
        qualified-name lookup there can land on another schema's class. The code
        generator gives each conversion its own module for every class it makes.
        """
        FeatureFlags.reset_flags()
        FeatureFlags.configure_flags({DATAMODEL_CODE_GENERATOR_CONVERTER_FF: False})
        legacy_model = create_model(CREATE_ISSUE)
        legacy_inline = self._inline_model(legacy_model)
        assert legacy_inline.__module__ != legacy_model.__module__

        FeatureFlags.configure_flags({DATAMODEL_CODE_GENERATOR_CONVERTER_FF: True})
        generated_model = create_model(CREATE_ISSUE)
        inline = self._inline_model(generated_model)
        assert inline.__module__ == generated_model.__module__

    def test_datamodel_code_generator_names_types_after_the_schema(self) -> None:
        """The ``$defs`` names reach the language model, so they carry meaning.

        Legacy names every generated type ``DynamicType_N``; the code generator
        derives the name from the schema, which is what the model then sees.
        """
        FeatureFlags.reset_flags()
        FeatureFlags.configure_flags({DATAMODEL_CODE_GENERATOR_CONVERTER_FF: False})
        legacy_defs = set(create_model(CREATE_ISSUE).model_json_schema()["$defs"])
        assert all(name.startswith("DynamicType") for name in legacy_defs)

        FeatureFlags.configure_flags({DATAMODEL_CODE_GENERATOR_CONVERTER_FF: True})
        generated_defs = set(create_model(CREATE_ISSUE).model_json_schema()["$defs"])
        assert "Project" in generated_defs

    def test_root_title_is_the_same_either_way(self) -> None:
        """What the model is told the tool is called must not depend on the flag."""
        titles = set()
        for enabled in (False, True):
            FeatureFlags.reset_flags()
            FeatureFlags.configure_flags(
                {DATAMODEL_CODE_GENERATOR_CONVERTER_FF: enabled}
            )
            titles.add(create_model(CREATE_ISSUE).model_json_schema()["title"])
        assert titles == {"Create_Issue"}

    def test_additional_properties_only_object(self) -> None:
        """An object declared solely by ``additionalProperties`` is a string map.

        The code generator types it as a dict; legacy wraps it in a model.
        Validation and serialization agree, which is what callers depend on.
        """
        schema = {
            "type": "object",
            "properties": {
                "meta": {"type": "object", "additionalProperties": {"type": "string"}}
            },
        }
        for enabled in (False, True):
            FeatureFlags.reset_flags()
            FeatureFlags.configure_flags(
                {DATAMODEL_CODE_GENERATOR_CONVERTER_FF: enabled}
            )
            model = create_model(schema)
            assert model.model_validate({"meta": {"a": "1"}}).model_dump() == {
                "meta": {"a": "1"}
            }
            with pytest.raises(ValidationError):
                model.model_validate({"meta": {"a": 1}})


class TestBackendSelection:
    """The feature flag picks the backend; off is the default."""

    @pytest.fixture(autouse=True)
    def schema_backend(self) -> Any:
        """Override the module fixture: these tests set the flag themselves.

        Without this they would also be parametrized over both backends, and the
        default-value test would run with the flag forced on.
        """
        FeatureFlags.reset_flags()
        yield
        FeatureFlags.reset_flags()

    def test_defaults_to_the_legacy_backend(self) -> None:
        assert not converter._datamodel_code_generator_enabled()

    @pytest.mark.parametrize(
        ("enabled", "expected"),
        [(False, _legacy_converter), (True, _datamodel_code_generator_converter)],
        ids=["legacy", "datamodel_code_generator"],
    )
    def test_flag_selects_the_backend(self, enabled: bool, expected: Any) -> None:
        FeatureFlags.configure_flags({DATAMODEL_CODE_GENERATOR_CONVERTER_FF: enabled})
        assert converter._datamodel_code_generator_enabled() is enabled

        calls: list[dict[str, Any]] = []
        original = expected.create_model

        def spy(schema: dict[str, Any]) -> Any:
            calls.append(schema)
            return original(schema)

        expected.create_model = spy
        try:
            create_model({"type": "object", "properties": {"a": {"type": "string"}}})
        finally:
            expected.create_model = original
        assert len(calls) == 1, "the selected backend should have been used"

    def test_output_model_honours_the_flag_too(self) -> None:
        FeatureFlags.configure_flags({DATAMODEL_CODE_GENERATOR_CONVERTER_FF: True})
        model = create_output_model(
            {"type": "object", "properties": {"a": {"type": "string"}}}, "t"
        )
        assert model.__pydantic_complete__
