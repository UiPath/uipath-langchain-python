import sys
from typing import Any, ForwardRef, Union, get_args, get_origin

from jsonpath_ng import parse  # type: ignore[import-untyped]
from pydantic import BaseModel


def get_json_paths_by_type(model: type[BaseModel], type_name: str) -> list[str]:
    """Get JSONPath expressions for all fields that reference a specific type.

    This function recursively traverses nested Pydantic models to find all paths
    that lead to fields of the specified type.

    Args:
        model: A Pydantic model class
        type_name: The name of the type to search for (e.g., "Job_attachment")

    Returns:
        List of JSONPath expressions using standard JSONPath syntax.
        For array fields, uses [*] to indicate all array elements.

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "attachment": {"$ref": "#/definitions/job-attachment"},
        ...         "attachments": {
        ...             "type": "array",
        ...             "items": {"$ref": "#/definitions/job-attachment"}
        ...         }
        ...     },
        ...     "definitions": {
        ...         "job-attachment": {"type": "object", "properties": {"id": {"type": "string"}}}
        ...     }
        ... }
        >>> model = transform(schema)
        >>> _get_json_paths_by_type(model, "Job_attachment")
        ['$.attachment', '$.attachments[*]']
    """

    def _recursive_search(
        current_model: type[BaseModel], current_path: str
    ) -> list[str]:
        """Recursively search for fields of the target type."""
        json_paths = []

        target_type = _get_target_type(current_model, type_name)
        matches_type = _create_type_matcher(type_name, target_type)

        for field_name, field_info in current_model.model_fields.items():
            annotation = field_info.annotation

            if current_path:
                field_path = f"{current_path}.{field_name}"
            else:
                field_path = f"$.{field_name}"

            annotation = _unwrap_optional(annotation)
            origin = get_origin(annotation)

            if matches_type(annotation):
                json_paths.append(field_path)
                continue

            if origin is list:
                args = get_args(annotation)
                if args:
                    list_item_type = args[0]
                    if matches_type(list_item_type):
                        json_paths.append(f"{field_path}[*]")
                        continue

                    if _is_pydantic_model(list_item_type):
                        nested_paths = _recursive_search(
                            list_item_type, f"{field_path}[*]"
                        )
                        json_paths.extend(nested_paths)
                        continue

            if _is_pydantic_model(annotation):
                nested_paths = _recursive_search(annotation, field_path)
                json_paths.extend(nested_paths)

        return json_paths

    return _recursive_search(model, "")


def extract_values_by_paths(
    obj: dict[str, Any] | BaseModel, json_paths: list[str]
) -> list[Any]:
    """Extract values from an object using JSONPath expressions.

    Args:
        obj: The object (dict or Pydantic model) to extract values from
        json_paths: List of JSONPath expressions. **Paths are assumed to be disjoint**
                    (non-overlapping). If paths overlap, duplicate values will be returned.

    Returns:
        List of all extracted values (flattened)

    Example:
        >>> obj = {
        ...     "attachment": {"id": "123"},
        ...     "attachments": [{"id": "456"}, {"id": "789"}]
        ... }
        >>> paths = ['$.attachment', '$.attachments[*]']
        >>> _extract_values_by_paths(obj, paths)
        [{'id': '123'}, {'id': '456'}, {'id': '789'}]
    """
    data = obj.model_dump() if isinstance(obj, BaseModel) else obj

    results = []
    for json_path in json_paths:
        expr = parse(json_path)
        matches = expr.find(data)
        results.extend([match.value for match in matches])

    return results


def _get_target_type(model: type[BaseModel], type_name: str) -> Any:
    """Get the target type from the model's module.

    Args:
        model: A Pydantic model class
        type_name: The name of the type to search for

    Returns:
        The target type if found, None otherwise
    """
    model_module = sys.modules.get(model.__module__)
    if model_module and hasattr(model_module, type_name):
        return getattr(model_module, type_name)
    return None


def _create_type_matcher(type_name: str, target_type: Any) -> Any:
    """Create a function that checks if an annotation matches the target type.

    Args:
        type_name: The name of the type to match
        target_type: The actual type object (can be None)

    Returns:
        A function that takes an annotation and returns True if it matches
    """

    def matches_type(annotation: Any) -> bool:
        """Check if an annotation matches the target type name."""
        if isinstance(annotation, ForwardRef):
            return annotation.__forward_arg__ == type_name
        if isinstance(annotation, str):
            return annotation == type_name
        if hasattr(annotation, "__name__") and annotation.__name__ == type_name:
            return True
        if target_type is not None and annotation is target_type:
            return True
        return False

    return matches_type


def _unwrap_optional(annotation: Any) -> Any:
    """Unwrap Optional/Union types to get the underlying type.

    Args:
        annotation: The type annotation to unwrap

    Returns:
        The unwrapped type, or the original if not Optional/Union
    """
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            return non_none_args[0]
    return annotation


def _is_pydantic_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)
