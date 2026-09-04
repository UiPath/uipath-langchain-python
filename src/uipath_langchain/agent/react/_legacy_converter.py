"""Schema-to-model conversion via ``jsonschema-pydantic-converter``.

The original backend, kept selectable so the newer code-generator path can be
rolled out behind a flag. See :mod:`jsonschema_pydantic_converter` (the façade in
this package) for how the two are chosen.

An object written inline (not under ``$defs``) that contains a ``$ref`` produces a
fully defined nested model, so a consumer copying the field annotations into a
model in another module gets a complete type.

Differences that remain against the code-generator backend, none of them
failures: generated types are named ``DynamicType_N`` rather than after the
schema, and those names reach the language model through ``$defs``; models built
for inline objects keep the converter's own module rather than this conversion's
pseudo-module; and an object declared solely by ``additionalProperties`` becomes
a model rather than a dict.
"""

import inspect
import itertools
import sys
from types import ModuleType
from typing import Any, Type, cast

from jsonschema_pydantic_converter import transform_with_modules
from pydantic import BaseModel, PydanticUndefinedAnnotation

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode

# Prefix for the per-conversion pseudo-modules that let get_type_hints()
# resolve each schema's forward references.
_DYNAMIC_MODULE_PREFIX = "jsonschema_pydantic_converter._dynamic"

_dynamic_module_counter = itertools.count()


def _create_dynamic_module() -> ModuleType:
    """Create a pseudo-module unique to one schema conversion.

    The converter reuses generic class names (``DynamicType_0``, ...) across
    schemas, so a shared module would let qualified-name lookups (e.g.
    LangGraph checkpoint deserialization) resolve to a class generated from a
    different schema.
    """
    module_name = f"{_DYNAMIC_MODULE_PREFIX}_{next(_dynamic_module_counter)}"
    pseudo_module = ModuleType(module_name)
    sys.modules[module_name] = pseudo_module
    return pseudo_module


def create_model(
    schema: dict[str, Any],
) -> Type[BaseModel]:
    """Convert a JSON schema dict to a Pydantic model.

    Raises:
        AgentStartupError: If the schema contains a type that cannot be resolved.
    """
    try:
        model, namespace = transform_with_modules(schema)
    except PydanticUndefinedAnnotation as e:
        # Strip the __ prefix the converter adds to forward references
        # so the user sees the original type name from their JSON schema.
        type_name = e.name.lstrip("_") if e.name else None
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Invalid schema",
            detail=(
                f"Type '{type_name}' could not be resolved. "
                f"Check that all $ref targets have matching entries in $defs."
            ),
        ) from e

    pseudo_module = _create_dynamic_module()

    for type_name, type_def in namespace.items():
        setattr(pseudo_module, type_name, type_def)
        if inspect.isclass(type_def) and issubclass(type_def, BaseModel):
            type_def.__module__ = pseudo_module.__name__
            # the namespace key is a forward-ref alias, not the class's
            # __name__; register under __name__ too so qualified-name lookups
            # (e.g. checkpoint deserialization) resolve.
            setattr(pseudo_module, type_def.__name__, type_def)
            # per-class marker for lookups by the schema's original type name.
            cast(Any, type_def).__uipath_marker_name__ = type_name

    setattr(pseudo_module, model.__name__, model)
    model.__module__ = pseudo_module.__name__

    return model
