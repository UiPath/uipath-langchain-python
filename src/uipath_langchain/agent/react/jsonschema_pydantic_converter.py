import inspect
import sys
from types import ModuleType
from typing import Any, Type

from jsonschema_pydantic_converter import transform_with_modules
from pydantic import BaseModel, PydanticUndefinedAnnotation

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode

# Shared pseudo-module for all dynamically created types
# This allows get_type_hints() to resolve forward references
_DYNAMIC_MODULE_NAME = "jsonschema_pydantic_converter._dynamic"


def _get_or_create_dynamic_module() -> ModuleType:
    """Get or create the shared pseudo-module for dynamic types."""
    if _DYNAMIC_MODULE_NAME not in sys.modules:
        pseudo_module = ModuleType(_DYNAMIC_MODULE_NAME)
        pseudo_module.__doc__ = (
            "Shared module for dynamically generated Pydantic models from JSON schemas"
        )
        sys.modules[_DYNAMIC_MODULE_NAME] = pseudo_module
    return sys.modules[_DYNAMIC_MODULE_NAME]


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

    pseudo_module = _get_or_create_dynamic_module()

    for type_name, type_def in namespace.items():
        setattr(pseudo_module, type_name, type_def)
        if inspect.isclass(type_def) and issubclass(type_def, BaseModel):
            type_def.__module__ = _DYNAMIC_MODULE_NAME

    setattr(pseudo_module, model.__name__, model)
    model.__module__ = _DYNAMIC_MODULE_NAME

    return model
