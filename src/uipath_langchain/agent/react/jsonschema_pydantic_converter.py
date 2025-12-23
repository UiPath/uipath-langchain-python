import inspect
import sys
from types import ModuleType
from typing import Any, Type, get_args, get_origin

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
