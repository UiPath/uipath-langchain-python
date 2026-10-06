"""Discovery and verification of job-attachment fields in an agent's output schema.

An output schema may declare fields that hold a file (a job attachment). The
agent fills one with a reference returned by the create-file tool, or by any
other tool that produced a file.

Verification closes the loop. Nothing stops a model from inventing an attachment
id or editing a reference, so at termination every attachment reference in the
output is looked up in Orchestrator and rebuilt from the attachment it names. An
attachment a child job produced is linked to that job only, so each one is also
linked to this job, which then lists every file it outputs.
"""

import copy
import uuid
from typing import Any, NamedTuple, Sequence

from jsonpath_ng import parse  # type: ignore[import-untyped]
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError
from uipath.platform import UiPath
from uipath.platform.attachments import Attachment
from uipath.platform.common import UiPathConfig
from uipath.platform.errors import EnrichedException

from .job_attachments import get_job_attachment_paths
from .mime_types import guess_mime_type
from .pydantic_json import extract_values_by_paths


class OutputFileField(NamedTuple):
    """One declared output field that holds a file."""

    path: str
    """JSONPath to the field, e.g. ``$.report`` or ``$.exports[*]``."""

    name: str
    """The field's name as the agent sees it."""

    description: str
    """The field's description from the schema; empty when none was authored."""

    required: bool
    """Whether the schema requires the field to be filled."""


def get_output_file_fields(model: type[BaseModel]) -> list[OutputFileField]:
    """Describe every job-attachment field declared by an output model.

    Only top-level fields carry a name, description, and required flag that are
    meaningful to state in a prompt; a nested attachment still gets a path so it
    is verified, described by its path alone.
    """
    by_json_key = {
        field_info.alias or field_name: field_info
        for field_name, field_info in model.model_fields.items()
    }
    fields = []
    for path in get_job_attachment_paths(model):
        json_key = _json_key_from_path(path)
        field_info = by_json_key.get(json_key)
        fields.append(
            OutputFileField(
                path=path,
                name=json_key,
                description=(field_info.description or "") if field_info else "",
                required=field_info.is_required() if field_info else False,
            )
        )
    return fields


def _json_key_from_path(path: str) -> str:
    """The first segment of a JSONPath, e.g. ``$.exports[*]`` -> ``exports``.

    The segment is the field's JSON key. The converter aliases any property
    whose name collides with a BaseModel attribute: ``schema`` becomes
    ``schema_`` with alias ``schema``.
    """
    return path.removeprefix("$.").split(".")[0].split("[")[0]


def missing_output_files(
    fields: list[OutputFileField], output: dict[str, Any]
) -> list[OutputFileField]:
    """Required file fields the agent left empty.

    A path that resolves to ``None`` counts as empty: an optional-shaped field
    the model declined to fill still matches its JSONPath.
    """
    return [
        field
        for field in fields
        if field.required and not _filled_values(output, field.path)
    ]


def _filled_values(output: dict[str, Any], path: str) -> list[dict[str, Any]]:
    """Attachment-shaped values at ``path``, skipping empty ones."""
    return [
        value
        for value in extract_values_by_paths(output, [path])
        if isinstance(value, dict) and value
    ]


def malformed_output_files(
    fields: list[OutputFileField], output: dict[str, Any]
) -> list[OutputFileField]:
    """File fields holding something ``Attachment`` will not accept."""
    malformed = []
    for field in fields:
        for value in _filled_values(output, field.path):
            try:
                Attachment.model_validate(value, from_attributes=True)
            except ValidationError:
                malformed.append(field)
                break
    return malformed


def output_attachment_ids(
    fields: list[OutputFileField], output: dict[str, Any]
) -> list[str]:
    """Every attachment id referenced by the output's file fields."""
    ids = []
    for field in fields:
        for value in _filled_values(output, field.path):
            if value.get("ID"):
                ids.append(str(value["ID"]))
    return ids


async def _lookup_attachment(uipath: UiPath, attachment_id: str) -> str | None:
    """The attachment's file name, or None when no such attachment exists."""
    try:
        key = uuid.UUID(attachment_id)
    except ValueError:
        return None
    try:
        info = await uipath.attachments.get_blob_file_access_uri_async(
            key=key, folder_key=UiPathConfig.folder_key
        )
    except EnrichedException as e:
        if e.status_code in (400, 403, 404):
            return None
        raise
    return info.name


async def resolve_output_attachments(
    fields: list[OutputFileField], output: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Rebuild every file reference from its attachment and link it to this job.

    Returns the output with each reference replaced by the attachment's real
    ``ID``, ``FullName`` and ``MimeType``, and the ids that name no attachment.
    Without a job there is nothing to look up or link to, so the output passes
    through unchanged.
    """
    referenced = list(dict.fromkeys(output_attachment_ids(fields, output)))
    if not referenced or not UiPathConfig.job_key:
        return output, []

    uipath = UiPath()
    lookups = {id: await _lookup_attachment(uipath, id) for id in referenced}
    unknown = [id for id, name in lookups.items() if name is None]
    if unknown:
        return output, unknown
    names = {id: name for id, name in lookups.items() if name is not None}

    job_key = uuid.UUID(str(UiPathConfig.job_key))
    linked = {
        str(key).lower()
        for key in await uipath.jobs.list_attachments_async(
            job_key=job_key, folder_key=UiPathConfig.folder_key
        )
    }
    for id in referenced:
        if id.lower() not in linked:
            await uipath.jobs.link_attachment_async(
                attachment_key=uuid.UUID(id),
                job_key=job_key,
                folder_key=UiPathConfig.folder_key,
            )

    resolved = copy.deepcopy(output)
    for field in fields:
        for match in parse(field.path).find(resolved):
            value = match.value
            if isinstance(value, dict) and value.get("ID"):
                name = names[str(value["ID"])]
                match.full_path.update(
                    resolved,
                    {
                        "ID": str(value["ID"]),
                        "FullName": name,
                        "MimeType": guess_mime_type(name),
                    },
                )
    return resolved, []


class OutputFilesCheck(NamedTuple):
    """The verdict on an output's file fields."""

    problem: str | None
    """Why the output cannot be accepted yet, or None when it can."""

    output: dict[str, Any]
    """The output with every file reference rebuilt from its attachment."""


DEFAULT_MAX_OUTPUT_FILE_RETRIES = 2


def _file_creator(file_tool_name: str | None) -> str:
    return f"`{file_tool_name}`" if file_tool_name else "a tool that creates files"


def _missing_files_message(
    fields: list[OutputFileField], file_tool_name: str | None
) -> str:
    names = ", ".join(f"'{field.name}'" for field in fields)
    creator = _file_creator(file_tool_name)
    return (
        f"Execution cannot end: the output field(s) {names} must hold a file and "
        f"are empty. Put in each field a file reference returned by a tool. If no "
        f"tool has produced the file yet, create it with {creator}. Then end "
        f"execution again."
    )


def _malformed_files_message(
    fields: list[OutputFileField], file_tool_name: str | None
) -> str:
    names = ", ".join(f"'{field.name}'" for field in fields)
    return (
        f"Execution cannot end: the output field(s) {names} do not hold a usable "
        f"file reference. Use a reference a tool returned, unchanged and "
        f"complete, rather than assembling one by hand. If you don't have one, "
        f"create it with {_file_creator(file_tool_name)}."
    )


def _unknown_ids_message(ids: list[str], file_tool_name: str | None) -> str:
    listed = ", ".join(f"'{id}'" for id in ids)
    return (
        f"Execution cannot end: the attachment reference(s) {listed} in the "
        f"output do not name an existing file. Use a reference a tool returned, "
        f"unchanged and complete. If you don't have one, create the file with "
        f"{_file_creator(file_tool_name)} and use the reference it returns."
    )


async def check_output_files(
    fields: list[OutputFileField],
    output: dict[str, Any],
    file_tool_name: str | None = None,
) -> OutputFilesCheck:
    """Whether this output can be accepted, and the output to accept.

    Checked in order: a required file field left empty, a field holding
    something that is not a usable reference, then a reference naming no
    attachment. Each message is written for the agent to act on, so it names
    the field or reference to fix.
    """
    missing = missing_output_files(fields, output)
    if missing:
        return OutputFilesCheck(_missing_files_message(missing, file_tool_name), output)

    malformed = malformed_output_files(fields, output)
    if malformed:
        return OutputFilesCheck(
            _malformed_files_message(malformed, file_tool_name), output
        )

    resolved, unknown = await resolve_output_attachments(fields, output)
    if unknown:
        return OutputFilesCheck(_unknown_ids_message(unknown, file_tool_name), output)

    return OutputFilesCheck(None, resolved)


def has_attachment_fields(
    tools: Sequence[BaseTool], output_model: type[BaseModel] | None
) -> bool:
    """Whether the output or any tool argument declares a job-attachment field."""
    if output_model is not None and get_job_attachment_paths(output_model):
        return True
    return any(
        isinstance(tool.args_schema, type)
        and issubclass(tool.args_schema, BaseModel)
        and bool(get_job_attachment_paths(tool.args_schema))
        for tool in tools
    )


def build_files_prompt(*, file_tool_name: str | None, with_workspace: bool) -> str:
    """Explain to the agent what fills a file field."""
    sentences = [
        "**Files.** Every file field, in your output or in a tool's arguments, "
        "holds a UiPath attachment reference (`ID`, `FullName`, `MimeType`) "
        "returned by a tool or given in your input."
    ]
    if with_workspace:
        sentences.append(
            "Files in your workspace are not attachments until a tool turns them "
            "into one."
        )
    sentences.append("Use a reference you already have.")
    if file_tool_name:
        sentences.append(f"If you don't have one, create it with `{file_tool_name}`.")
    sentences.append("Never write a reference yourself.")
    return " ".join(sentences)
