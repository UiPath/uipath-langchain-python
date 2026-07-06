"""Job attachment utilities for ReAct Agent."""

import copy
import uuid
from typing import Any, Sequence

from jsonpath_ng import parse  # type: ignore[import-untyped]
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError
from uipath.platform.attachments import Attachment
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
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
        List of Attachment objects.

    Raises:
        AgentRuntimeError: If a tool-output attachment fails validation (e.g. its
            ID is not a valid UUID). This is unrecoverable invalid data and is
            surfaced as a SYSTEM failure rather than silently skipped.
    """
    job_attachment_paths = get_job_attachment_paths(schema)
    job_attachments = extract_values_by_paths(data, job_attachment_paths)

    result = []
    for att in job_attachments:
        if not att:
            continue
        try:
            attachment = Attachment.model_validate(att, from_attributes=True)
        except ValidationError as e:
            id_error = _attachment_id_uuid_error(e)
            if id_error:
                raise AgentRuntimeError(
                    code=AgentRuntimeErrorCode.INVALID_ATTACHMENT_ID,
                    title="Invalid attachment id",
                    detail=(
                        f"A tool returned a job attachment with id {id_error.get('input')!r}, "
                        f"which is not a valid UUID. The agent cannot proceed with an "
                        f"invalid attachment."
                    ),
                    category=UiPathErrorCategory.SYSTEM,
                ) from e
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.OUTPUT_VALIDATION_ERROR,
                title="Invalid job attachment",
                detail=(
                    f"A tool returned a job attachment that does not match the "
                    f"expected shape — {_describe_validation_errors(e)}. "
                    f"Verify the tool's output provides valid attachment fields; the "
                    f"agent cannot proceed with an invalid attachment."
                ),
                category=UiPathErrorCategory.SYSTEM,
            ) from e
        result.append(attachment)

    return result


def _attachment_id_uuid_error(exc: ValidationError) -> Any | None:
    id_field = Attachment.model_fields["id"]
    id_field_names = ("id", id_field.validation_alias, id_field.alias)
    for err in exc.errors():
        if err.get("type") not in ("uuid_parsing", "uuid_type"):
            continue
        if any(
            err.get("loc") == (name,)
            for name in id_field_names
            if isinstance(name, str)
        ):
            return err
    return None


def _describe_validation_errors(exc: ValidationError) -> str:
    """Render a pydantic ValidationError as a short, human-readable field list.

    Reports each failing field path and reason (e.g. ``'MimeType': Field required``)
    without echoing the offending input values, so the message is actionable and
    safe to surface.
    """
    issues = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err.get("loc", ())) or "attachment"
        issues.append(f"'{field}': {err.get('msg', 'invalid value')}")
    return "; ".join(issues)


def get_job_attachment_paths(model: type[BaseModel]) -> list[str]:
    """Get JSONPath expressions for all job attachment fields in a Pydantic model.

    Args:
        model: The Pydantic model class to analyze

    Returns:
        List of JSONPath expressions pointing to job attachment fields
    """
    return get_json_paths_by_type(model, "__Job_attachment")


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


def parse_attachments_from_conversation_messages(
    messages: Sequence[BaseMessage],
) -> dict[str, Attachment]:
    """Parse attachments from HumanMessage additional_kwargs.

    Extracts attachment information from HumanMessages where additional_kwargs
    contains an 'attachments' list with attachment details.

    Args:
        messages: Sequence of messages to parse

    Returns:
        Dictionary mapping attachment ID to Attachment objects
    """
    attachments: dict[str, Attachment] = {}

    for message in messages:
        if not isinstance(message, HumanMessage):
            continue

        kwargs = getattr(message, "additional_kwargs", None)
        if not kwargs:
            continue

        # Handle attachments list in additional_kwargs
        attachment_list = kwargs.get("attachments", [])
        for att in attachment_list:
            id = att.get("id")
            full_name = att.get("full_name")
            mime_type = att.get("mime_type")

            if id and full_name:
                attachments[str(id)] = Attachment(
                    id=id,
                    full_name=full_name,
                    mime_type=mime_type,
                )

    return attachments
