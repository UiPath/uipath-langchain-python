import inspect
import sys
from types import ModuleType
from typing import Any, Type

from jsonschema_pydantic_converter import transform_with_modules
from pydantic import BaseModel

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
    model, namespace = transform_with_modules(schema)

    pseudo_module = _get_or_create_dynamic_module()

    for type_name, type_def in namespace.items():
        setattr(pseudo_module, type_name, type_def)
        if inspect.isclass(type_def) and issubclass(type_def, BaseModel):
            type_def.__module__ = _DYNAMIC_MODULE_NAME

    setattr(pseudo_module, model.__name__, model)
    model.__module__ = _DYNAMIC_MODULE_NAME

    return model
