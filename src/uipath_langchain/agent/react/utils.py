"""ReAct Agent loop utilities."""

import sys
import uuid
from typing import Any, ForwardRef, Sequence, Union, get_args, get_origin

from jsonpath_ng import parse  # type: ignore[import-untyped]
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel
from uipath.agent.react import END_EXECUTION_TOOL
from uipath.platform.attachments import Attachment

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model


def resolve_input_model(
    input_schema: dict[str, Any] | None,
) -> type[BaseModel]:
    """Resolve the input model from the input schema."""
    if input_schema:
        return create_model(input_schema)

    return BaseModel


def resolve_output_model(
    output_schema: dict[str, Any] | None,
) -> type[BaseModel]:
    """Fallback to default end_execution tool schema when no agent output schema is provided."""
    if output_schema:
        return create_model(output_schema)

    return END_EXECUTION_TOOL.args_schema


def count_consecutive_thinking_messages(messages: Sequence[BaseMessage]) -> int:
    """Count consecutive AIMessages without tool calls at end of message history."""
    if not messages:
        return 0

    count = 0
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            break

        if message.tool_calls:
            break

        if not message.content:
            break

        count += 1

    return count


def add_job_attachments(
    left: dict[uuid.UUID, Attachment], right: dict[uuid.UUID, Attachment]
) -> dict[uuid.UUID, Attachment]:
    """Merge attachment dictionaries, with right values taking precedence.

    This reducer function merges two dictionaries of attachments by UUID.
    If the same UUID exists in both dictionaries, the value from 'right' takes precedence.

    Args:
        left: Existing dictionary of attachments keyed by UUID
        right: New dictionary of attachments to merge

    Returns:
        Merged dictionary with right values overriding left values for duplicate keys
    """
    if not right:
        return left

    if not left:
        return right

    return {**left, **right}


def get_job_attachments(
    schema: type[BaseModel],
    data: dict[str, Any],
) -> list[Attachment]:
    """Extract job attachments from data based on schema and convert to Attachment objects.

    Args:
        schema: The Pydantic model class defining the data structure
        data: The data object (dict or Pydantic model) to extract attachments from

    Returns:
        List of Attachment objects
    """
    job_attachment_paths = _get_job_attachment_paths(schema)
    job_attachments = _extract_values_by_paths(data, job_attachment_paths)

    result = []
    for attachment in job_attachments:
        if isinstance(attachment, BaseModel):
            # Convert Pydantic model to dict and create Attachment
            attachment_dict = attachment.model_dump(by_alias=True)
            result.append(Attachment.model_validate(attachment_dict))
        elif isinstance(attachment, dict):
            # Already a dict, create Attachment directly
            result.append(Attachment.model_validate(attachment))
        else:
            # Try to convert to Attachment as-is
            result.append(Attachment.model_validate(attachment))

    return result


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
    """Check if annotation is a Pydantic model.

    Args:
        annotation: The type annotation to check

    Returns:
        True if the annotation is a Pydantic model class
    """
    try:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)
    except TypeError:
        return False


def _get_job_attachment_paths(model: type[BaseModel]) -> list[str]:
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
        >>> get_json_paths_by_type(model, "Job_attachment")
        ['$.attachment', '$.attachments[*]']
    """

    def _recursive_search(
        current_model: type[BaseModel], current_path: str
    ) -> list[str]:
        """Recursively search for fields of the target type."""
        json_paths = []

        # Get the target type and create a matcher function
        target_type = _get_target_type(current_model, type_name)
        matches_type = _create_type_matcher(type_name, target_type)

        for field_name, field_info in current_model.model_fields.items():
            annotation = field_info.annotation

            # Build the path for this field
            if current_path:
                field_path = f"{current_path}.{field_name}"
            else:
                field_path = f"$.{field_name}"

            # Unwrap Optional/Union types
            annotation = _unwrap_optional(annotation)
            origin = get_origin(annotation)

            # Check if this field matches the target type
            if matches_type(annotation):
                json_paths.append(field_path)
                continue

            # Check if this is a list of the target type or nested models
            if origin is list:
                args = get_args(annotation)
                if args:
                    list_item_type = args[0]
                    if matches_type(list_item_type):
                        json_paths.append(f"{field_path}[*]")
                        continue
                    # Check if it's a list of nested models
                    if _is_pydantic_model(list_item_type):
                        nested_paths = _recursive_search(
                            list_item_type, f"{field_path}[*]"
                        )
                        json_paths.extend(nested_paths)
                        continue

            # Check if this field is a nested Pydantic model that we should traverse
            if _is_pydantic_model(annotation):
                nested_paths = _recursive_search(annotation, field_path)
                json_paths.extend(nested_paths)

        return json_paths

    return _recursive_search(model, "")


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
        >>> extract_values_by_paths(obj, paths)
        [{'id': '123'}, {'id': '456'}, {'id': '789'}]
    """
    # Convert Pydantic model to dict if needed
    data = obj.model_dump() if isinstance(obj, BaseModel) else obj

    results = []
    for json_path in json_paths:
        expr = parse(json_path)
        matches = expr.find(data)
        results.extend([match.value for match in matches])

    return results
