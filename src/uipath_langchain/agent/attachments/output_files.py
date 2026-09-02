"""Discovery and verification of job-attachment fields in an agent's output schema.

An output schema may declare fields that hold a file (a job attachment). The
agent has no way to fill such a field on its own, so the runtime injects the
``create_output_file`` tool and tells the agent, in the system prompt, which
fields expect a file and what to write into them.

Verification closes the loop. Nothing stops a model from inventing an attachment
id, so at termination every attachment reference in the output is checked against
the attachments actually linked to this job. A reference that is not there did
not come from the tool.
"""

import uuid
from typing import Any, NamedTuple

from pydantic import BaseModel
from uipath.platform import UiPath
from uipath.platform.common import UiPathConfig

from .constants import OUTPUT_FILE_TOOL_NAME
from .job_attachments import get_job_attachment_paths
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
    fields = []
    for path in get_job_attachment_paths(model):
        name = _field_name_from_path(path)
        field_info = model.model_fields.get(name)
        fields.append(
            OutputFileField(
                path=path,
                name=name,
                description=(field_info.description or "") if field_info else "",
                required=field_info.is_required() if field_info else False,
            )
        )
    return fields


def _field_name_from_path(path: str) -> str:
    """The first segment of a JSONPath, e.g. ``$.exports[*]`` -> ``exports``."""
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


async def unlinked_output_attachment_ids(
    fields: list[OutputFileField], output: dict[str, Any]
) -> list[str]:
    """Referenced attachment ids that are not linked to the current job.

    Returns an empty list when there is no job to check against — a local run
    stores attachments outside Orchestrator, so there is nothing to verify.
    """
    referenced = output_attachment_ids(fields, output)
    if not referenced or not UiPathConfig.job_key:
        return []

    uipath = UiPath()
    linked = {
        str(key).lower()
        for key in await uipath.jobs.list_attachments_async(
            job_key=uuid.UUID(str(UiPathConfig.job_key)),
            folder_key=UiPathConfig.folder_key,
        )
    }
    return [id for id in referenced if id.lower() not in linked]


_PROMPT_HEADER = """\
**Output files**
These output fields hold a file, and the only way to fill one is with the \
reference returned by the `{tool}` tool. Put each returned reference in its \
matching field.
"""

_PROMPT_REQUIRED_RULE = """\
Create every required file before you end execution."""

_PROMPT_OPTIONAL_RULE = """\
Create an optional file only when it serves the request; leaving one empty is a \
valid answer."""

_PROMPT_FORMAT_RULE = """\
If a field's description names a file format, use that format. Otherwise choose \
the format that best fits the content, and give the file an extension that \
matches it."""

_PROMPT_WORKSPACE_RULE = """\
For anything you have already written to a file, or any non-text file, pass its \
workspace path as `file_path` rather than re-emitting the body as `content`."""


def build_output_files_prompt(
    fields: list[OutputFileField],
    *,
    tool_name: str,
    with_workspace: bool = False,
) -> str:
    """Describe the declared output file fields and how to fill them.

    Returns an empty string when the output schema declares no file field, so
    the caller can append the result unconditionally.
    """
    if not fields:
        return ""

    lines = [_PROMPT_HEADER.format(tool=tool_name)]
    for field in fields:
        suffix = " (required)" if field.required else " (optional)"
        description = f" — {field.description}" if field.description else ""
        lines.append(f"- `{field.name}`{suffix}{description}")
    lines.append("")
    # Stated per kind that is actually declared, so an all-optional schema is
    # never told to produce a file and an all-required one is never told it may
    # skip one.
    if any(field.required for field in fields):
        lines.append(_PROMPT_REQUIRED_RULE)
    if any(not field.required for field in fields):
        lines.append(_PROMPT_OPTIONAL_RULE)
    lines.append(_PROMPT_FORMAT_RULE)
    if with_workspace:
        lines.append(_PROMPT_WORKSPACE_RULE)
    return "\n".join(lines)


DEFAULT_MAX_OUTPUT_FILE_RETRIES = 2


def _missing_files_message(fields: list[OutputFileField]) -> str:
    names = ", ".join(f"'{field.name}'" for field in fields)
    return (
        f"Execution cannot end: the output field(s) {names} must hold a file and "
        f"are empty. Call `{OUTPUT_FILE_TOOL_NAME}` once per field, put each returned "
        f"reference in its field, then end execution again."
    )


def _unlinked_ids_message(ids: list[str]) -> str:
    listed = ", ".join(f"'{id}'" for id in ids)
    return (
        f"Execution cannot end: the attachment reference(s) {listed} in the "
        f"output do not belong to this job. Only a reference returned by "
        f"`{OUTPUT_FILE_TOOL_NAME}` (or by a tool that produced a file) is valid. Create "
        f"the file with `{OUTPUT_FILE_TOOL_NAME}` and use the reference it returns."
    )


async def diagnose_output_files(
    fields: list[OutputFileField], output: dict[str, Any]
) -> str | None:
    """Why this output cannot be accepted yet, or None when it can.

    Checked in order: a required file field left empty, then a reference to an
    attachment that is not linked to this job. The message is written for the
    agent to act on, so it names the field and the tool to call.
    """
    missing = missing_output_files(fields, output)
    if missing:
        return _missing_files_message(missing)

    unlinked = await unlinked_output_attachment_ids(fields, output)
    if unlinked:
        return _unlinked_ids_message(unlinked)

    return None
