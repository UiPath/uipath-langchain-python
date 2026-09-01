"""Schema-to-model conversion via ``datamodel-code-generator``.

The newer backend. Compared with :mod:`._legacy_converter` it names generated
types after the schema rather than ``DynamicType_N`` -- those names reach the
language model through ``$defs`` -- homes every class it produces in this
conversion's pseudo-module, and repairs property names that are not valid Python
identifiers while keeping the declared names on the wire.

Both backends resolve ``$ref``s completely.

Everything below the two public entry points exists to keep three runtime
contracts intact:

* generated classes live in a per-conversion pseudo-module registered in
  ``sys.modules``, so qualified-name lookups resolve (LangGraph checkpoint
  deserialization, and :mod:`uipath_langchain.agent.attachments.pydantic_json`);
* classes reached through a ``$ref`` carry ``__uipath_marker_name__`` holding the
  name that ``$defs`` entry maps to, which is how job-attachment discovery finds
  them;
* schemas whose ``$ref`` targets are missing fail at startup with a message
  naming the type, rather than producing a model that breaks later.
"""

import itertools
import keyword
import re
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any, Type, cast, get_args

import jsonschema
from datamodel_code_generator import (
    Formatter,
    GenerateConfig,
    InputFileType,
    generate_dynamic_models,
)
from datamodel_code_generator.parser.jsonschema import json_schema_data_formats
from pydantic import BaseModel, model_validator

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode

from ._datamodel_code_generator_base import UiPathDatamodelCodeGeneratorBaseModel
from ._schema_refs import ref_resolves, resolve_pointer

# Prefix for the per-conversion pseudo-modules that let qualified-name lookups
# resolve each schema's generated classes.
_DYNAMIC_MODULE_PREFIX = "jsonschema_pydantic_converter._dynamic"

_dynamic_module_counter = itertools.count()

# Import path of the base class every generated model derives from. The generator
# takes this as a string and emits the import itself.
_BASE_CLASS = (
    f"{UiPathDatamodelCodeGeneratorBaseModel.__module__}"
    f".{UiPathDatamodelCodeGeneratorBaseModel.__name__}"
)

# Pin every ``format`` back to its type's own default, so a format annotates the
# value without changing it. Left alone, the generator retypes on format --
# ``password`` to ``SecretStr``, which serializes as a mask rather than the
# credential; ``email`` and ``ulid`` to types whose validators are not installed,
# which fails conversion outright; ``date-time`` to a timezone-aware datetime,
# rejecting the naive stamps models emit; and a dozen more to objects that are no
# longer JSON-serializable once dumped. None of that is what this backend is for,
# and the legacy backend does none of it, so the two must agree here.
#
# Derived from the generator's own table rather than listed, so a format added
# upstream is covered without a change here.
_FORMAT_PINS = [
    f"{base}+{name}={base}"
    for base, formats in json_schema_data_formats.items()
    for name, type_ in formats.items()
    if type_ is not formats.get("default")
]

# JSON Schema keywords that have no equivalent in Pydantic's type system, so the
# generator drops them and the field ends up accepting anything. Enforcement for
# these is restored by _build_constraint_guard.
_UNENFORCEABLE_KEYWORDS = ("not", "prefixItems")

# Keywords that nest a subschema describing the same value as their parent.
_COMBINERS = ("anyOf", "oneOf", "allOf")

_INVALID_SCHEMA_TITLE = "Invalid schema"


def _invalid_schema(detail: str) -> AgentStartupError:
    """The startup error raised for every schema this backend cannot convert."""
    return AgentStartupError(
        code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
        title=_INVALID_SCHEMA_TITLE,
        detail=detail,
    )


def _create_dynamic_module() -> ModuleType:
    """Create a pseudo-module unique to one conversion, since class names repeat."""
    module_name = f"{_DYNAMIC_MODULE_PREFIX}_{next(_dynamic_module_counter)}"
    pseudo_module = ModuleType(module_name)
    sys.modules[module_name] = pseudo_module
    return pseudo_module


def _definition_type_name(ref: str) -> str:
    """The name a ``$ref`` maps to, byte-compatible with the legacy backend."""

    def sanitize(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    if ref.startswith("#/"):
        parts = [
            sanitize(part)
            for part in ref[2:].split("/")
            if part not in ("$defs", "definitions")
        ]
        return "__" + "_".join(parts).capitalize()
    return "__" + sanitize(ref.split("/")[-1]).capitalize()


def _unresolved_type_name(ref: str) -> str:
    """The bare type name to show a user for an unresolvable ``$ref``."""
    return ref.rstrip("/").split("/")[-1] or ref


def _iter_refs(node: Any) -> Any:
    """Yield every ``$ref`` string in a schema, at any depth."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            yield ref
        for value in node.values():
            yield from _iter_refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_refs(item)


def _assert_refs_resolve(schema: dict[str, Any]) -> None:
    """Fail with a user-facing error if any ``$ref`` target is missing."""
    for ref in _iter_refs(schema):
        if ref_resolves(ref, schema):
            continue
        raise _invalid_schema(
            f"Type '{_unresolved_type_name(ref)}' could not be resolved. "
            f"Check that all $ref targets have matching entries in $defs."
        )


def _valid_identifier(name: str, fallback: str) -> str:
    """Coerce `name` into something usable as a Python class name."""
    # the generator treats a leading underscore as private and withholds the class
    cleaned = name.lstrip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{fallback}{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def _root_class_name(title: str) -> str:
    """The root's class name, preserving the schema's ``title`` where possible."""
    return _valid_identifier(re.sub(r"[^0-9a-zA-Z_]", "_", title), "Model")


def _nested_class_name(name: str) -> str:
    """A Pascal-case class name for a ``$defs`` entry or an inline object."""
    parts = [part for part in re.split(r"[^0-9a-zA-Z]+", name) if part]
    pascal = "".join(part[:1].upper() + part[1:] for part in parts)
    return _valid_identifier(pascal, "Model")


def _generate(schema: dict[str, Any]) -> tuple[dict[str, type], str]:
    """Generate the model classes for `schema`, plus the name of its root class."""
    order: list[str] = []

    def record_name(name: str) -> str:
        # The root is named first, so the first call is the only one whose name
        # is user-visible; everything after it is an internal class.
        cleaned = _root_class_name(name) if not order else _nested_class_name(name)
        order.append(cleaned)
        return cleaned

    config = GenerateConfig(
        input_file_type=InputFileType.JsonSchema,
        base_class=_BASE_CLASS,
        strict_refs=True,
        # The generator renders Python source and execs it. Nothing ever reads
        # that source, so skip black/isort: ~3x faster, and the library is
        # moving to external formatters being opt-in anyway.
        formatters=[Formatter.BUILTIN],
        allow_population_by_field_name=True,
        custom_class_name_generator=record_name,
        type_mappings=_FORMAT_PINS,
    )
    module_name = f"{_DYNAMIC_MODULE_PREFIX}_gen_{next(_dynamic_module_counter)}"
    try:
        models = generate_dynamic_models(
            schema,
            config=config,
            module_name=module_name,
            # Each conversion gets its own classes, so one caller mutating a
            # model cannot affect another agent built from an identical schema.
            cache_size=0,
        )
    except AgentStartupError:
        raise
    except Exception as exc:
        raise _invalid_schema(
            f"The schema could not be converted to a model: {exc}"
        ) from exc

    if not models:
        raise _invalid_schema("The schema produced no model.")

    return models, _root_name(schema, models, order)


def _referenced_model_ids(models: dict[str, type]) -> set[int]:
    """The ids of the models that another model reaches through a field."""
    referenced: set[int] = set()
    for model in models.values():
        for field in cast("type[BaseModel]", model).model_fields.values():
            for nested in _models_in(field.annotation):
                # Recursion makes a model reach itself; that does not make it
                # somebody else's child.
                if nested is not model:
                    referenced.add(id(nested))
    return referenced


def _models_with_root_properties(
    schema: dict[str, Any], models: dict[str, type]
) -> list[str]:
    """The models carrying exactly the properties the document itself declares."""
    properties = schema.get("properties")
    declared = set(properties) if isinstance(properties, dict) else set()
    if not declared:
        return []
    return [
        name
        for name, model in models.items()
        if set(_fields_by_json_name(cast("type[BaseModel]", model))) == declared
    ]


def _root_name(
    schema: dict[str, Any], models: dict[str, type], requested: list[str]
) -> str:
    """The name of the model generated for the document as a whole.

    Not simply the name the root asked for. The generator renames on collision
    and resolves the root last, so a ``$defs`` entry can win the requested name
    while the root is renamed around it -- a schema titled ``Root`` alongside
    ``$defs/Root`` hands back the definition, with the document's own
    properties and ``required`` silently gone. The requested name is not
    dependable either way: names are singularized after this callback returns,
    and it is called more than once for some subschemas.

    So the root is identified by structure -- it carries the document's own
    properties, and nothing else refers to it -- with the requested names left
    to break ties between shapes structure cannot separate (an ``allOf`` root,
    whose properties come from its branches).
    """
    candidates = _models_with_root_properties(schema, models) or list(models)
    referenced = _referenced_model_ids(models)
    unreferenced = [name for name in candidates if id(models[name]) not in referenced]
    candidates = unreferenced or candidates
    for name in requested:
        if name in candidates:
            return name
    return candidates[0]


def _models_in(annotation: Any) -> list[type[BaseModel]]:
    """Every model class reachable inside a type annotation, containers unwrapped."""
    found: list[type[BaseModel]] = []
    stack: list[Any] = [annotation]
    while stack:
        current = stack.pop()
        if isinstance(current, type) and issubclass(current, BaseModel):
            found.append(current)
        else:
            stack.extend(get_args(current))
    return found


def _fields_by_json_name(model: type[BaseModel]) -> dict[str, Any]:
    """Map each JSON property name to its field, honouring generated aliases."""
    return {(field.alias or name): field for name, field in model.model_fields.items()}


def _models_for_property(
    models: list[type[BaseModel]], json_name: str
) -> list[type[BaseModel]]:
    """The models describing property `json_name` of any of `models`."""
    found: list[type[BaseModel]] = []
    for model in models:
        field = _fields_by_json_name(model).get(json_name)
        if field is not None:
            found.extend(_models_in(field.annotation))
    return found


def _child_nodes(
    node: dict[str, Any], models: list[type[BaseModel]]
) -> Iterator[tuple[Any, list[type[BaseModel]]]]:
    """Yield each subschema of `node` with the models it describes."""
    for combiner in _COMBINERS:
        for sub in node.get(combiner) or []:
            yield sub, models

    properties = node.get("properties")
    if isinstance(properties, dict):
        for json_name, sub in properties.items():
            yield sub, _models_for_property(models, json_name)

    for keyword_name in ("items", "additionalProperties"):
        sub = node.get(keyword_name)
        if isinstance(sub, dict):
            yield sub, models


def _tag_referenced_models(
    root: type[BaseModel], schema: dict[str, Any], pseudo_module: ModuleType
) -> None:
    """Name every model a ``$ref`` points at, and publish it on the module."""
    # Walks schema and model tree together rather than matching on class names:
    # the generator renames on collision, so only the structural walk is reliable.
    visited: set[tuple[int, int]] = set()

    def name_models(models: list[type[BaseModel]], type_name: str) -> None:
        for model in models:
            if not hasattr(model, "__uipath_marker_name__"):
                cast(Any, model).__uipath_marker_name__ = type_name
            if not hasattr(pseudo_module, type_name):
                setattr(pseudo_module, type_name, model)

    def visit(node: Any, models: list[type[BaseModel]]) -> None:
        if not isinstance(node, dict) or not models:
            return
        key = (id(node), id(models[0]))
        if key in visited:
            return
        visited.add(key)

        ref = node.get("$ref")
        if isinstance(ref, str):
            name_models(models, _definition_type_name(ref))
            visit(resolve_pointer(schema, ref), models)
            return

        for sub, sub_models in _child_nodes(node, models):
            visit(sub, sub_models)

    visit(schema, [root])


def _register_models(models: dict[str, type], pseudo_module: ModuleType) -> None:
    """Publish every generated class on the pseudo-module and re-home it there."""
    for name, model in models.items():
        setattr(pseudo_module, name, model)
        if isinstance(model, type) and issubclass(model, BaseModel):
            model.__module__ = pseudo_module.__name__
            setattr(pseudo_module, model.__name__, model)


def _unenforceable_constraints(
    schema: dict[str, Any],
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    """Locate subschemas using a keyword the generator cannot express as a type."""
    found: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    seen: set[int] = set()

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if not isinstance(node, dict) or id(node) in seen:
            return
        seen.add(id(node))

        ref = node.get("$ref")
        if isinstance(ref, str):
            visit(resolve_pointer(schema, ref), path)
            return

        if _is_unenforceable(node):
            found.append((path, node))
            return

        for sub, sub_path in _child_paths(node, path):
            visit(sub, sub_path)

    visit(schema, ())
    return found


def _is_unenforceable(node: dict[str, Any]) -> bool:
    """Whether `node` states a rule no Pydantic annotation can carry."""
    if any(name in node for name in _UNENFORCEABLE_KEYWORDS):
        return True
    # An empty enum permits nothing, which no annotation expresses either.
    return node.get("enum") == []


def _child_paths(
    node: dict[str, Any], path: tuple[str, ...]
) -> Iterator[tuple[Any, tuple[str, ...]]]:
    """Yield each subschema of `node` with the path leading to its values."""
    properties = node.get("properties")
    if isinstance(properties, dict):
        for json_name, sub in properties.items():
            yield sub, (*path, json_name)

    items = node.get("items")
    if isinstance(items, dict):
        yield items, (*path, "[]")

    for combiner in _COMBINERS:
        for sub in node.get(combiner) or []:
            yield sub, path


def _build_constraint_guard(
    constraints: list[tuple[tuple[str, ...], dict[str, Any]]],
) -> Any:
    """Return a callable enforcing constraints the generated types cannot carry."""
    validators = [
        (path, jsonschema.Draft202012Validator(subschema))
        for path, subschema in constraints
    ]

    def enforce(data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for path, validator in validators:
            for value in _values_at_path(data, path):
                error = jsonschema.exceptions.best_match(validator.iter_errors(value))
                if error is not None:
                    location = ".".join(path) or "value"
                    raise ValueError(f"{location}: {error.message}")
        return data

    return enforce


def _values_at_path(value: Any, path: tuple[str, ...]) -> list[Any]:
    """Every value `path` reaches inside `value`; ``"[]"`` means each array item."""
    if not path:
        return [value]
    head, rest = path[0], path[1:]
    if head == "[]":
        if not isinstance(value, list):
            return []
        return [found for item in value for found in _values_at_path(item, rest)]
    if isinstance(value, dict) and head in value:
        return _values_at_path(value[head], rest)
    return []


def _guard_unenforceable(
    root: type[BaseModel], schema: dict[str, Any]
) -> type[BaseModel]:
    """Wrap `root` so constraints the generated types dropped are still applied."""
    constraints = _unenforceable_constraints(schema)
    if not constraints:
        return root

    # A "before" model validator may be a plain callable taking the raw input,
    # which is exactly the shape _build_constraint_guard returns.
    enforce = model_validator(mode="before")(_build_constraint_guard(constraints))

    guarded = type(
        root.__name__,
        (root,),
        {
            "__module__": root.__module__,
            "__doc__": root.__doc__,
            "_uipath_enforce_unrepresentable": enforce,
        },
    )
    return cast(Type[BaseModel], guarded)


def create_model(
    schema: dict[str, Any],
) -> Type[BaseModel]:
    """Convert a JSON schema dict to a Pydantic model."""
    _assert_refs_resolve(schema)

    models, root_name = _generate(schema)
    root = cast(Type[BaseModel], models[root_name])

    pseudo_module = _create_dynamic_module()
    _register_models(models, pseudo_module)
    _tag_referenced_models(root, schema, pseudo_module)

    root = _guard_unenforceable(root, schema)
    root.__module__ = pseudo_module.__name__
    setattr(pseudo_module, root.__name__, root)

    return root
