"""JSON schema modifier for static argument values."""

import copy
import json
from typing import Any

from jsonpath_ng import (  # type: ignore[import-untyped]
    Child,
    Fields,
    JSONPath,
    Root,
    Slice,
    parse,
)

STATIC_ARGUMENT_DESCRIPTION = (
    "This argument is pre-configured and will be overwritten. Leave it empty"
)


class SchemaModificationError(ValueError):
    """Raised when a schema modification fails."""

    pass


def parse_jsonpath_segments(json_path: str) -> list[str]:
    """Parse JSON path $['a']['b'] into list of segments ['a', 'b']."""
    jsonpath_expr = parse(json_path)
    return _extract_segments(jsonpath_expr)


def _extract_segments(expr: JSONPath) -> list[str]:
    """Recursively extract segments from a JSONPath expression tree.

    Args:
        expr: The JSONPath expression node (Root, Child, Fields, etc.)
    """
    parts: list[str] = []

    match expr:
        case Root():
            pass
        case Fields():
            # Leaf node, add it to the path
            parts.extend(expr.fields)
        case Child():
            # Child node, walk left then right to maintain order
            parts.extend(_extract_segments(expr.left))
            parts.extend(_extract_segments(expr.right))
        case Slice(start=None, end=None, step=None):
            parts.append("*")

    return parts


def apply_static_value_to_schema(
    schema: dict[str, Any],
    json_path: str,
    value: Any,
    is_sensitive: bool,
) -> dict[str, Any]:
    """Apply the static argument value to a specific field in the schema."""
    path_parts = parse_jsonpath_segments(json_path)
    if not path_parts:
        raise SchemaModificationError("Empty JSON path")

    try:
        if is_sensitive:
            _apply_sensitive_schema_modification(schema, path_parts)
        else:
            _apply_const_schema_modification(schema, path_parts, value)

        return schema
    except KeyError as e:
        raise SchemaModificationError(
            f"Invalid schema path {json_path} for schema {schema}"
        ) from e


def _navigate_schema_inlining_refs(
    schema: dict[str, Any],
    json_path: list[str],
) -> dict[str, Any]:
    """Navigate to a location in schema using JSON path, inlining $refs as needed."""
    if not json_path:
        return schema

    current = schema
    for key in json_path:
        current = _skip_anyof_inlining_ref(schema, current)

        schema_type = current.get("type")

        if schema_type == "object":
            _inline_ref_if_present(schema, current["properties"], key)
            current = current["properties"][key]
        elif schema_type == "array" and key == "*":
            _inline_ref_if_present(schema, current, "items")
            current = current["items"]
        else:
            raise SchemaModificationError(
                f"Invalid schema type {schema_type} for key {key} in schema {schema}"
            )

        current = _skip_anyof_inlining_ref(schema, current)

    return current


def _apply_sensitive_schema_modification(
    schema: dict[str, Any],
    path_parts: list[str],
) -> None:
    """Apply modifications for sensitive static parameters."""
    field_name = path_parts[-1]
    if field_name == "*":
        # We want to mark the array property as sensitive, not its items field.
        # So we need to navigate to the parent of the array property.
        assert len(path_parts) >= 2, "$[*] is not a valid JSON schema path"
        parent_path = path_parts[:-2]
        field_name = path_parts[-2]
    else:
        parent_path = path_parts[:-1]

    parent_object_schema = _navigate_schema_inlining_refs(schema, parent_path)
    field_schema = parent_object_schema["properties"][field_name]
    field_schema = _skip_anyof_inlining_ref(schema, field_schema)

    field_schema["description"] = STATIC_ARGUMENT_DESCRIPTION

    required_fields: list[str] = parent_object_schema.get("required", [])
    if field_name in required_fields:
        required_fields.remove(field_name)


def _apply_const_schema_modification(
    schema: dict[str, Any],
    path_parts: list[str],
    static_value: Any,
) -> None:
    """Apply constant enum modifications for non-sensitive static arguments.

    - Primitives: enum: [value]
    - Objects: Recursively apply enum to primitive leaves
    - Arrays: enum: [json.dumps(value)]
    """

    def _apply_recursive(
        properties_object: dict[str, Any],
        field_name: str,
        value: Any,
    ) -> None:
        field_schema = properties_object[field_name]
        field_schema = _skip_anyof_inlining_ref(schema, field_schema)
        schema_type = field_schema.get("type")

        match schema_type:
            case "string" | "number" | "integer" | "boolean":
                field_schema["enum"] = [value]
            case "array":
                properties_object[field_name] = {
                    "type": "string",
                    "enum": [json.dumps(value)],
                }
            case "object":
                assert isinstance(value, dict), (
                    "Object static value should be a dictionary"
                )
                for prop_name, prop_value in value.items():
                    _apply_recursive(field_schema["properties"], prop_name, prop_value)

    parent_path = path_parts[:-1]
    parent_object_schema = _navigate_schema_inlining_refs(schema, parent_path)
    field_name = path_parts[-1]

    parent_type = parent_object_schema.get("type")
    if parent_type == "array" and field_name == "*":
        _inline_ref_if_present(schema, parent_object_schema, "items")
        _apply_recursive(parent_object_schema, "items", static_value)
    elif parent_type == "object":
        _apply_recursive(parent_object_schema["properties"], field_name, static_value)
    else:
        raise SchemaModificationError(
            f"{parent_object_schema} at path {path_parts} must be a container"
        )


def _resolve_definition(schema: dict[str, Any], ref_path: str) -> dict[str, Any]:
    """Resolve a $ref pointer to the actual schema definition.

    Args:
        schema: The root schema containing $defs
        ref_path: The $ref string like "#/$defs/Config"

    Returns:
        A deep copy of the resolved schema definition
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


def _inline_ref_if_present(
    schema: dict[str, Any],
    container: dict[Any, Any] | list[Any],
    key: Any,
) -> None:
    """Inlines a $ref if present in container[key]."""
    try:
        candidate = container[key]
        if isinstance(candidate, dict) and "$ref" in candidate:
            resolved = _resolve_definition(schema, candidate["$ref"])
            container[key] = resolved
    except (KeyError, IndexError, TypeError) as e:
        raise SchemaModificationError(
            f"Unable to inline ref at '{key}' in container type {type(container).__name__}"
        ) from e


def _skip_anyof_inlining_ref(
    schema: dict[str, Any], container: dict[str, Any]
) -> dict[str, Any]:
    """Enters an anyOf union and inlines the ref if present.

    Args:
        schema: The root schema containing $defs
        container: The container of the anyOf, modified in place

    Returns:
        The non-null schema of the anyOf, inlined if needed. If there is no anyOf, returns the container.
    """

    def _index_of_non_null_schema(
        field_schema: dict[str, Any],
    ) -> int | None:
        """Pydantic represents optional fields as an anyOf union of the field schema and null.
        This function extracts the index of the anyOf's non-null item.

        Returns:
            The index of the non-null schema, or None if it's not a nullable field.
        """
        union_list = field_schema.get("anyOf")
        if union_list is None or len(union_list) != 2:
            return None

        [schema_0, schema_1] = union_list
        if schema_0.get("type") == "null":
            return 1
        elif schema_1.get("type") == "null":
            return 0
        return None

    target_index = _index_of_non_null_schema(container)
    if target_index is None:
        return container

    _inline_ref_if_present(schema, container["anyOf"], target_index)
    return container["anyOf"][target_index]
