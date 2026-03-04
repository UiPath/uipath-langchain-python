import inspect
import sys
from types import ModuleType
from typing import Any, Type, get_args, get_origin

from jsonschema_pydantic_converter import transform_with_modules
from pydantic import BaseModel

# Shared pseudo-module for all dynamically created types
# This allows get_type_hints() to resolve forward references
_DYNAMIC_MODULE_NAME = "jsonschema_pydantic_converter._dynamic"

# Field names that shadow BaseModel attributes and must be renamed.
# Computed from BaseModel's public interface to stay future-proof across Pydantic versions.
_RESERVED_FIELD_NAMES: frozenset[str] = frozenset(
    name for name in dir(BaseModel) if not name.startswith("_")
)


def _get_or_create_dynamic_module() -> ModuleType:
    """Get or create the shared pseudo-module for dynamic types."""
    if _DYNAMIC_MODULE_NAME not in sys.modules:
        pseudo_module = ModuleType(_DYNAMIC_MODULE_NAME)
        pseudo_module.__doc__ = (
            "Shared module for dynamically generated Pydantic models from JSON schemas"
        )
        sys.modules[_DYNAMIC_MODULE_NAME] = pseudo_module
    return sys.modules[_DYNAMIC_MODULE_NAME]


def _needs_rename(name: str) -> bool:
    """Check if a JSON Schema property name needs renaming for Pydantic compatibility."""
    return name.startswith("_") or name in _RESERVED_FIELD_NAMES


def _safe_field_name(
    original: str, existing_keys: set[str], used_keys: set[str]
) -> str:
    """Generate a Pydantic-safe field name from a JSON Schema property name.

    Strips leading underscores and avoids collisions with BaseModel attributes
    and other property names (both original and already-renamed).
    """
    name = original.lstrip("_") or "field"
    if name in _RESERVED_FIELD_NAMES:
        name += "_"
    while name in existing_keys or name in used_keys:
        name += "_"
    return name


def _rename_reserved_properties(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Rename JSON Schema properties that are invalid as Pydantic field names.

    Handles two cases:
    - Properties starting with ``_`` (Pydantic treats these as private attributes)
    - Properties that shadow ``BaseModel`` attributes (e.g. ``schema``, ``copy``)

    Returns:
        Tuple of (modified schema copy, {new_field_name: original_name}).
    """
    renames: dict[str, str] = {}

    def _process(s: dict[str, Any]) -> dict[str, Any]:
        result = s.copy()

        if "properties" in result:
            existing_keys = set(result["properties"].keys())
            used_keys: set[str] = set()
            new_props: dict[str, Any] = {}

            for key, value in result["properties"].items():
                if _needs_rename(key):
                    new_key = _safe_field_name(key, existing_keys, used_keys)
                    renames[new_key] = key
                else:
                    new_key = key

                used_keys.add(new_key)
                new_props[new_key] = (
                    _process(value) if isinstance(value, dict) else value
                )
            result["properties"] = new_props

            if "required" in result:
                # Build a lookup from original→renamed for this level only
                local_renames = {v: k for k, v in renames.items() if v in existing_keys}
                result["required"] = [
                    local_renames.get(name, name) for name in result["required"]
                ]

        for defs_key in ("$defs", "definitions"):
            if defs_key in result:
                result[defs_key] = {
                    k: (_process(v) if isinstance(v, dict) else v)
                    for k, v in result[defs_key].items()
                }

        if "items" in result and isinstance(result["items"], dict):
            result["items"] = _process(result["items"])

        for keyword in ("allOf", "anyOf", "oneOf"):
            if keyword in result:
                result[keyword] = [
                    _process(sub) if isinstance(sub, dict) else sub
                    for sub in result[keyword]
                ]

        if "not" in result and isinstance(result["not"], dict):
            result["not"] = _process(result["not"])

        for keyword in ("if", "then", "else"):
            if keyword in result and isinstance(result[keyword], dict):
                result[keyword] = _process(result[keyword])

        return result

    modified = _process(schema)
    return modified, renames


def _apply_field_aliases(
    model: Type[BaseModel],
    namespace: dict[str, Any],
    renames: dict[str, str],
) -> None:
    """Add aliases to renamed fields so serialization/validation uses original names.

    Iterates the root model and all nested models from the namespace. For any
    field whose name appears in ``renames``, sets alias/validation_alias/
    serialization_alias to the original property name and enables
    ``populate_by_name`` + ``serialize_by_alias`` in the model config.
    """
    if not renames:
        return

    all_models = [model]
    for v in namespace.values():
        if inspect.isclass(v) and issubclass(v, BaseModel):
            all_models.append(v)

    for m in all_models:
        needs_rebuild = False
        for field_name, field_info in m.model_fields.items():
            if field_name in renames:
                original_name = renames[field_name]
                field_info.alias = original_name
                field_info.validation_alias = original_name
                field_info.serialization_alias = original_name
                needs_rebuild = True

        if needs_rebuild:
            m.model_config = {
                **m.model_config,
                "populate_by_name": True,
                "serialize_by_alias": True,
            }
            m.model_rebuild(force=True)


def create_model(
    schema: dict[str, Any],
) -> Type[BaseModel]:
    processed_schema, renames = _rename_reserved_properties(schema)
    model, namespace = transform_with_modules(processed_schema)
    _apply_field_aliases(model, namespace, renames)
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
