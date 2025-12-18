"""Schema modifier for dynamic tool schemas with enum values."""

import copy
import json
from typing import Any

from uipath_langchain._utils._jsonpath import parse_jsonpath_segments


class SchemaModificationError(ValueError):
    """Raised when a schema modification fails."""

    pass


def apply_static_value_to_schema(
    schema: dict[str, Any],
    json_path: str,
    value: Any,
    is_sensitive: bool,
) -> dict[str, Any]:
    """Apply the static argument value to a specific field in the schema.

    Args:
        schema: The schema dict to modify in place
        json_path: JSON path like "$['config']['host']"
        value: The static value
        is_sensitive: Whether the value is sensitive

    Raises:
        InvalidSchemaPathError: If the JSON path does not exist in the schema
    """
    path_parts = parse_jsonpath_segments(json_path)
    if not path_parts:
        raise SchemaModificationError("Empty JSON path")

    try:
        parent_path = path_parts[:-1]
        parent_object_schema = _navigate_schema_inlining_refs(schema, parent_path)
        field_name = path_parts[-1]

        if is_sensitive:
            _apply_sensitive_schema_modification(parent_object_schema, field_name)
        else:
            properties = parent_object_schema["properties"]
            _apply_const_schema_modification(properties, field_name, value)

        return schema
    except KeyError as e:
        raise SchemaModificationError(
            f"Invalid schema path {json_path} for schema {schema}"
        ) from e


def _resolve_definition(schema: dict[str, Any], ref_path: str) -> dict[str, Any]:
    """Resolve a $ref pointer to the actual schema definition.

    Args:
        schema: The root schema containing $defs
        ref_path: The $ref string like "#/$defs/Config"

    Returns:
        A deep copy of the resolved schema definition

    Raises:
        ValueError: If the $ref is malformed or not found
    """
    if not ref_path.startswith("#/"):
        raise ValueError(f"Invalid $ref format: {ref_path}")

    ref_path_parts = ref_path[2:].split("/")

    current = schema
    for key in ref_path_parts:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"$ref not found: {ref_path}")
        current = current[key]

    if not isinstance(current, dict):
        raise ValueError(f"$ref does not point to a schema object: {ref_path}")

    return copy.deepcopy(current)


def _navigate_schema_inlining_refs(
    schema: dict[str, Any],
    json_path: list[str],
) -> dict[str, Any]:
    """Navigate to a location in schema using JSON path, inlining $refs as needed.

    Args:
        schema: The root schema
        json_path: JSON path like ['config', 'host']

    Returns:
        The schema dict at that location

    Raises:
        KeyError: If any key in the path does not exist
    """
    if not json_path:
        return schema

    def _inline_ref_if_present(container: dict[str, Any], field_name: str) -> None:
        """Inlines a $ref if present in container[field_name]."""
        if "$ref" in container[field_name]:
            resolved = _resolve_definition(schema, container[field_name]["$ref"])
            container[field_name] = resolved

    current = schema
    for key in json_path:
        schema_type = current.get("type")

        if schema_type == "object":
            _inline_ref_if_present(current["properties"], key)
            current = current["properties"][key]

        elif schema_type == "array" and key == "items":
            _inline_ref_if_present(current, "items")
            current = current["items"]

        else:
            raise SchemaModificationError(
                f"Invalid schema type {schema_type} for key {key} in schema {schema}"
            )

    return current


def _apply_sensitive_schema_modification(
    parent_object_schema: dict[str, Any],
    field_name: str,
) -> None:
    """Apply modifications for sensitive static parameters.

    - Remove from required list of the parent object
    - Set description indicating pre-configured value

    Args:
        parent_object_schema: The object schema containing this field (with 'required' array and 'properties')
        field_name: The name of the field in the parent object's properties

    Raises:
        KeyError: If the field does not exist in properties
    """
    field_schema = parent_object_schema["properties"][field_name]

    field_schema["description"] = (
        "This argument is pre-configured with a static value "
        "and will be overwritten on tool call"
    )

    required_fields: list[str] = parent_object_schema.get("required", [])
    if field_name in required_fields:
        required_fields.remove(field_name)


def _apply_const_schema_modification(
    properties_object: dict[str, Any],
    field_name: str,
    value: Any,
) -> None:
    """Apply enum modifications for non-sensitive static parameters.

    - Primitives: Apply enum: [value]
    - Objects: Recursively apply enum to primitive leaves
    - Arrays: Convert to string with JSON-serialized enum
    """
    field_schema = properties_object[field_name]
    schema_type = field_schema.get("type")

    if schema_type in ("string", "number", "integer", "boolean"):
        field_schema["enum"] = [value]
    elif schema_type == "array":
        properties_object[field_name] = {"type": "string", "enum": [json.dumps(value)]}
    elif schema_type == "object":
        assert isinstance(value, dict), "Object static value should be a dictionary"
        for prop_name, prop_value in value.items():
            _apply_const_schema_modification(
                field_schema["properties"], prop_name, prop_value
            )
