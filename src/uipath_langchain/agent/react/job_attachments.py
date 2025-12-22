"""Job attachment utilities for ReAct Agent."""

import copy
import sys
import uuid
from typing import Any, ForwardRef, Union, get_args, get_origin

from jsonpath_ng import parse  # type: ignore[import-untyped]
from pydantic import BaseModel
from uipath.platform.attachments import Attachment


def get_job_attachments(
    schema: type[BaseModel],
    data: dict[str, Any] | BaseModel,
) -> list[Attachment]:
    """Extract job attachments from data based on schema and convert to Attachment objects.

    Args:
        schema: The Pydantic model class defining the data structure
        data: The data object (dict or Pydantic model) to extract attachments from

    Returns:
        List of Attachment objects
    """
    job_attachment_paths = get_job_attachment_paths(schema)
    job_attachments = _extract_values_by_paths(data, job_attachment_paths)

    result = []
    for attachment in job_attachments:
        if isinstance(attachment, BaseModel):
            attachment_dict = attachment.model_dump(by_alias=True)
            result.append(Attachment.model_validate(attachment_dict))
        elif isinstance(attachment, dict):
            result.append(Attachment.model_validate(attachment))
        else:
            result.append(Attachment.model_validate(attachment))

    return result


def get_job_attachment_paths(model: type[BaseModel]) -> list[str]:
    """Get JSONPath expressions for all job attachment fields in a Pydantic model.

    Args:
        model: The Pydantic model class to analyze

    Returns:
        List of JSONPath expressions pointing to job attachment fields
    """
    return _get_json_paths_by_type(model, "Job_attachment")


def _get_json_paths_by_type(model: type[BaseModel], type_name: str) -> list[str]:
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


def replace_job_attachment_ids(
    json_paths: list[str],
    tool_args: dict[str, Any],
    state: dict[str, Attachment],
    errors: list[str],
) -> dict[str, Any]:
    """Replace job attachment IDs in tool_args with full attachment objects from state.

    For each JSON path, this function finds matching objects in tool_args and
    replaces them with corresponding attachment objects from state. The matching
    is done by looking up the object's 'ID' field in the state dictionary.

    If an ID is not a valid UUID or is not present in state, an error message
    is added to the errors list.

    Args:
        json_paths: List of JSONPath expressions (e.g., ["$.attachment", "$.attachments[*]"])
        tool_args: The dictionary containing tool arguments to modify
        state: Dictionary mapping attachment UUID strings to Attachment objects
        errors: List to collect error messages for invalid or missing IDs

    Returns:
        Modified copy of tool_args with attachment IDs replaced by full objects

    Example:
        >>> state = {
        ...     "123e4567-e89b-12d3-a456-426614174000": Attachment(id="123e4567-e89b-12d3-a456-426614174000", name="file1.pdf"),
        ...     "223e4567-e89b-12d3-a456-426614174001": Attachment(id="223e4567-e89b-12d3-a456-426614174001", name="file2.pdf")
        ... }
        >>> tool_args = {
        ...     "attachment": {"ID": "123"},
        ...     "other_field": "value"
        ... }
        >>> paths = ['$.attachment']
        >>> errors = []
        >>> replace_job_attachment_ids(paths, tool_args, state, errors)
        {'attachment': {'ID': '123', 'name': 'file1.pdf', ...}, 'other_field': 'value'}
    """
    result = copy.deepcopy(tool_args)

    for json_path in json_paths:
        expr = parse(json_path)
        matches = expr.find(result)

        for match in matches:
            current_value = match.value

            if isinstance(current_value, dict) and "ID" in current_value:
                attachment_id_str = str(current_value["ID"])

                try:
                    uuid.UUID(attachment_id_str)
                except (ValueError, AttributeError):
                    errors.append(
                        _create_job_attachment_error_message(attachment_id_str)
                    )
                    continue

                if attachment_id_str in state:
                    replacement_value = state[attachment_id_str]
                    match.full_path.update(
                        result, replacement_value.model_dump(by_alias=True, mode="json")
                    )
                else:
                    errors.append(
                        _create_job_attachment_error_message(attachment_id_str)
                    )

    return result


def _create_job_attachment_error_message(attachment_id_str: str) -> str:
    return (
        f"Could not find JobAttachment with ID='{attachment_id_str}'. "
        f"Try again invoking the tool and please make sure that you pass "
        f"valid JobAttachment IDs associated with existing JobAttachments in the current context."
    )


def _extract_values_by_paths(
    obj: dict[str, Any] | BaseModel, json_paths: list[str]
) -> list[Any]:
    """Extract values from an object using JSONPath expressions.

    Args:
        obj: The object (dict or Pydantic model) to extract values from
        json_paths: List of JSONPath expressions (e.g., ["$.attachment", "$.attachments[*]"])

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
