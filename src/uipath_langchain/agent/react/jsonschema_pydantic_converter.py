import inspect
import logging
import sys
from types import ModuleType
from typing import Any, Type, get_args, get_origin

from jsonschema_pydantic_converter import transform_with_modules
from pydantic import BaseModel
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode

logger = logging.getLogger("uipath")

# Shared pseudo-module for all dynamically created types
# This allows get_type_hints() to resolve forward references
_DYNAMIC_MODULE_NAME = "jsonschema_pydantic_converter._dynamic"

# Non-object JSON Schema types that require wrapping in an object envelope
_NON_OBJECT_TYPES = {"array", "string", "integer", "number", "boolean", "null"}


def _get_or_create_dynamic_module() -> ModuleType:
    """Get or create the shared pseudo-module for dynamic types."""
    if _DYNAMIC_MODULE_NAME not in sys.modules:
        pseudo_module = ModuleType(_DYNAMIC_MODULE_NAME)
        pseudo_module.__doc__ = (
            "Shared module for dynamically generated Pydantic models from JSON schemas"
        )
        sys.modules[_DYNAMIC_MODULE_NAME] = pseudo_module
    return sys.modules[_DYNAMIC_MODULE_NAME]


def _is_non_object_schema(schema: dict[str, Any]) -> bool:
    """Check whether the root schema type is a non-object JSON Schema type.

    Returns True for schemas like ``{"type": "array", ...}`` or
    ``{"type": "string"}`` where ``transform_with_modules`` would fail
    because it only supports object-typed root schemas.
    """
    schema_type = schema.get("type")
    return isinstance(schema_type, str) and schema_type in _NON_OBJECT_TYPES


def _wrap_non_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a non-object schema inside an object envelope.

    The original schema becomes the ``result`` property of a new object
    schema so that ``transform_with_modules`` can process it.  Any
    top-level ``$defs`` / ``definitions`` are hoisted to the wrapper so
    that ``$ref`` pointers keep working.
    """
    wrapper: dict[str, Any] = {
        "type": "object",
        "properties": {"result": schema},
        "required": ["result"],
    }
    # Hoist shared definition blocks so $ref resolution still works
    for defs_key in ("$defs", "definitions"):
        if defs_key in schema:
            wrapper[defs_key] = schema[defs_key]
    return wrapper


def create_model(
    schema: dict[str, Any],
    *,
    tool_name: str | None = None,
) -> Type[BaseModel]:
    """Convert a JSON Schema dict to a Pydantic BaseModel class.

    Non-object root schemas (e.g. ``type: array``) are automatically
    wrapped in an object envelope with a single ``result`` property so
    that the return value is always a ``BaseModel`` subclass.

    Args:
        schema: JSON Schema dictionary.
        tool_name: Optional tool name included in error messages to make
            failures easier to diagnose.
    """
    if has_underscore_fields(schema):
        detail = (
            "Schema properties starting with '_' are currently not supported. "
            "If they are unavoidable, please contact UiPath Support"
        )
        if tool_name:
            detail = f"Tool '{tool_name}': {detail}"
        raise AgentStartupError(
            code=AgentStartupErrorCode.UNDERSCORE_SCHEMA,
            title="Schema contains properties starting with '_'",
            detail=detail,
            category=UiPathErrorCategory.USER,
        )

    effective_schema = schema
    if _is_non_object_schema(schema):
        logger.debug(
            "Wrapping non-object schema (type=%s) in object envelope%s",
            schema.get("type"),
            f" for tool '{tool_name}'" if tool_name else "",
        )
        effective_schema = _wrap_non_object_schema(schema)

    try:
        model, namespace = transform_with_modules(effective_schema)
    except Exception as exc:
        detail = f"Failed to convert JSON schema to model: {exc}"
        if tool_name:
            detail = f"Tool '{tool_name}': {detail}"
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Invalid tool schema",
            detail=detail,
            category=UiPathErrorCategory.SYSTEM,
        ) from exc
    corrected_namespace: dict[str, Any] = {}

    def collect_types(annotation: Any) -> None:
        """Recursively collect all BaseModel types from an annotation."""
        # Unwrap generic types like List, Optional, etc.
        origin = get_origin(annotation)
        if origin is not None:
            for arg in get_args(annotation):
                collect_types(arg)

        elif inspect.isclass(annotation) and issubclass(annotation, BaseModel):
            # Find the original name for this type from the namespace
            for type_name, type_def in namespace.items():
                # Match by class name since rebuild may create new instances
                if (
                    hasattr(annotation, "__name__")
                    and hasattr(type_def, "__name__")
                    and annotation.__name__ == type_def.__name__
                ):
                    # Store the actual annotation type, not the old namespace one
                    annotation.__name__ = type_name
                    corrected_namespace[type_name] = annotation
                    break

    # Collect all types from field annotations
    for field_info in model.model_fields.values():
        collect_types(field_info.annotation)

    # Get the shared pseudo-module and populate it with this schema's types
    # This ensures that forward references can be resolved by get_type_hints()
    # when the model is used with external libraries (e.g., LangGraph)
    pseudo_module = _get_or_create_dynamic_module()

    # Populate the pseudo-module with all types from the namespace
    # Use the original names so forward references resolve correctly
    for type_name, type_def in corrected_namespace.items():
        setattr(pseudo_module, type_name, type_def)

    setattr(pseudo_module, model.__name__, model)

    # Update the model's __module__ to point to the shared pseudo-module
    model.__module__ = _DYNAMIC_MODULE_NAME

    # Update the __module__ of all generated types in the namespace
    for type_def in corrected_namespace.values():
        if inspect.isclass(type_def) and issubclass(type_def, BaseModel):
            type_def.__module__ = _DYNAMIC_MODULE_NAME
    return model


def has_underscore_fields(schema: dict[str, Any]) -> bool:
    properties: dict[str, Any] = schema.get("properties", {})
    for key, value in properties.items():
        if key.startswith("_"):
            return True
        if isinstance(value, dict) and has_underscore_fields(value):
            return True

    defs: dict[str, Any] = schema.get("$defs") or schema.get("definitions") or {}
    for definition in defs.values():
        if isinstance(definition, dict) and has_underscore_fields(definition):
            return True

    items = schema.get("items")
    if isinstance(items, dict) and has_underscore_fields(items):
        return True

    for keyword in ("allOf", "anyOf", "oneOf"):
        for sub_schema in schema.get(keyword, []):
            if isinstance(sub_schema, dict) and has_underscore_fields(sub_schema):
                return True

    not_schema = schema.get("not")
    if isinstance(not_schema, dict) and has_underscore_fields(not_schema):
        return True

    for keyword in ("if", "then", "else"):
        sub_schema = schema.get(keyword)
        if isinstance(sub_schema, dict) and has_underscore_fields(sub_schema):
            return True

    return False
