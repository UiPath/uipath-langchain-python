"""Job attachment utilities for ReAct Agent."""

import copy
import uuid
from typing import Any

from jsonpath_ng import parse  # type: ignore[import-untyped]
from pydantic import BaseModel
from uipath.platform.attachments import Attachment

from .json_utils import extract_values_by_paths, get_json_paths_by_type


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
    job_attachments = extract_values_by_paths(data, job_attachment_paths)

    result = []
    for attachment in job_attachments:
        result.append(Attachment.model_validate(attachment, from_attributes=True))

    return result


def get_job_attachment_paths(model: type[BaseModel]) -> list[str]:
    """Get JSONPath expressions for all job attachment fields in a Pydantic model.

    Args:
        model: The Pydantic model class to analyze

    Returns:
        List of JSONPath expressions pointing to job attachment fields
    """
    return get_json_paths_by_type(model, "Job_attachment")


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
        f"Try invoking the tool again and please make sure that you pass "
        f"valid JobAttachment IDs associated with existing JobAttachments in the current context."
    )
