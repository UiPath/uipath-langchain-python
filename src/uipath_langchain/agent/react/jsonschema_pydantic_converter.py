import inspect
import sys
from types import ModuleType
from typing import Any, Type

from jsonschema_pydantic_converter import transform_with_modules
from pydantic import BaseModel
from uipath.runtime.errors import UiPathErrorCategory

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
    if has_underscore_fields(schema):
        raise AgentStartupError(
            code=AgentStartupErrorCode.UNDERSCORE_SCHEMA,
            title="Schema contains properties starting with '_'",
            detail="Schema properties starting with '_' are currently not supported. If they are unavoidable, please contact UiPath Support",
            category=UiPathErrorCategory.USER,
        )

    model, namespace = transform_with_modules(schema)

    pseudo_module = _get_or_create_dynamic_module()

    for type_name, type_def in namespace.items():
        setattr(pseudo_module, type_name, type_def)
        if inspect.isclass(type_def) and issubclass(type_def, BaseModel):
            type_def.__module__ = _DYNAMIC_MODULE_NAME

    setattr(pseudo_module, model.__name__, model)
    model.__module__ = _DYNAMIC_MODULE_NAME

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
