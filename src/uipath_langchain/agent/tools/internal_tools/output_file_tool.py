"""Internal tool that publishes agent-authored content as a job attachment.

Injected automatically — never configured by the user — whenever the agent's
output schema declares a job-attachment field. The tool creates the attachment,
links it to the current job, and returns the attachment ticket; the agent then
places that ticket in the declared output field.

Two content sources, and which one is offered depends on the agent flavour:

- ``content`` — the file body inline. The only source a standard agent has,
  since it owns no filesystem. Text formats only.
- ``file_path`` — a path in the agent's own workspace, offered only when a
  filesystem backend is present (advanced agents). Preferred there: the body
  never round-trips through the model, so large and binary files work.
"""

import mimetypes
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import UiPathConfig
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.structured_tool_with_output_type import (
    StructuredToolWithOutputType,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperMixin

from ...attachments.constants import OUTPUT_FILE_TOOL_NAME
from .schema_utils import JOB_ATTACHMENT_DEFINITION

__all__ = ["OUTPUT_FILE_TOOL_NAME", "create_output_file_tool", "guess_mime_type"]


_DEFAULT_MIME_TYPE = "application/octet-stream"

# mimetypes has no entry for these on every supported Python, and they are among
# the formats an agent is most likely to pick for a generated document.
_EXTRA_MIME_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".jsonl": "application/jsonl",
}

_TOOL_DESCRIPTION = (
    "Create a file and attach it to this job, then return the attachment "
    "reference to put in the agent output field that expects a file. Call this "
    "before ending execution: an output file field can only be filled with a "
    "reference this tool returned."
)

_FILE_NAME_DESCRIPTION = (
    "File name including the extension, e.g. 'summary.md' or 'accounts.csv'. "
    "The extension determines the file's MIME type, so it must match the "
    "format of the content."
)

_CONTENT_DESCRIPTION = "The full text content of the file."

_FILE_PATH_DESCRIPTION = (
    "Path of an existing file in your workspace to publish, e.g. '/report.md'. "
    "Prefer this over 'content' for anything you have already written to a "
    "file, and use it for any non-text file."
)


@runtime_checkable
class _BoundedPathBackend(Protocol):
    """The part of a filesystem backend this tool needs: bounded path resolution."""

    def _resolve_path(self, file_path: str) -> Path: ...


def output_file_tool_output_schema() -> dict[str, Any]:
    """The tool's output schema: a single job-attachment ticket under ``file``."""
    return {
        "type": "object",
        "properties": {
            "file": {
                "$ref": "#/definitions/job-attachment",
                "description": "Reference to the created file. Use this value for the output file field.",
            }
        },
        "required": ["file"],
        "definitions": {"job-attachment": JOB_ATTACHMENT_DEFINITION},
    }


def _input_schema(*, with_file_path: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "file_name": {"type": "string", "description": _FILE_NAME_DESCRIPTION},
        "content": {"type": "string", "description": _CONTENT_DESCRIPTION},
    }
    if with_file_path:
        properties["file_path"] = {
            "type": "string",
            "description": _FILE_PATH_DESCRIPTION,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": ["file_name"],
    }


def guess_mime_type(file_name: str) -> str:
    """Resolve a file's MIME type from its extension."""
    suffix = Path(file_name).suffix.lower()
    if suffix in _EXTRA_MIME_TYPES:
        return _EXTRA_MIME_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(file_name)
    return guessed or _DEFAULT_MIME_TYPE


def _resolve_source_path(backend: Any, file_path: str) -> Path:
    """Resolve a workspace-relative path, rejecting anything outside the workspace."""
    if not isinstance(backend, _BoundedPathBackend):
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.FILE_ERROR,
            title="Workspace file paths are not available",
            detail=(
                f"'{OUTPUT_FILE_TOOL_NAME}' received a 'file_path' but this agent "
                "has no workspace to read it from. Pass the file body in 'content' instead."
            ),
            category=UiPathErrorCategory.SYSTEM,
        )
    # The backend's own resolver keeps the path inside the workspace root; it
    # raises on traversal, so no separate containment check is needed here.
    return backend._resolve_path(file_path)


class _OutputFileTool(StructuredToolWithOutputType, ToolWrapperMixin):
    """Output type plus a state-updating wrapper, as the other attachment-producing tools have."""


def create_output_file_tool(backend: Any | None = None) -> _OutputFileTool:
    """Create the ``create_output_file`` tool.

    Args:
        backend: The agent's filesystem backend, when it has one. ``file_path``
            is offered only for a backend that can resolve a workspace path;
            for anything else the tool accepts inline ``content`` only, rather
            than advertising an argument every use of which would fail.
    """
    with_file_path = isinstance(backend, _BoundedPathBackend)
    input_model = create_model(_input_schema(with_file_path=with_file_path))
    output_model = create_model(output_file_tool_output_schema())

    async def create_output_file_fn(**kwargs: Any) -> dict[str, Any]:
        file_name = kwargs.get("file_name")
        content = kwargs.get("content")
        file_path = kwargs.get("file_path")

        if not file_name:
            raise ValueError("'file_name' is required.")
        if not content and not file_path:
            raise ValueError(
                "Provide the file body in 'content'"
                + (
                    ", or an existing workspace path in 'file_path'."
                    if with_file_path
                    else "."
                )
            )
        if content and file_path:
            raise ValueError("'content' and 'file_path' are mutually exclusive.")

        # basename only: file_name reaches us from the model, and it names the
        # attachment rather than a path on disk.
        attachment_name = Path(file_name).name

        @mockable(
            name=OUTPUT_FILE_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=[],
        )
        async def publish_output_file(**_tool_kwargs: Any) -> dict[str, Any]:
            source_path = (
                _resolve_source_path(backend, file_path) if file_path else None
            )
            if source_path is not None and not source_path.is_file():
                raise ValueError(
                    f"'{file_path}' does not exist in your workspace. Write the "
                    "file first, or pass its body in 'content'."
                )

            uipath = UiPath()
            attachment_id = await uipath.jobs.create_attachment_async(
                name=attachment_name,
                content=content if source_path is None else None,
                source_path=str(source_path) if source_path is not None else None,
                job_key=UiPathConfig.job_key,
                folder_key=UiPathConfig.folder_key,
            )
            return {
                "ID": str(attachment_id),
                "FullName": attachment_name,
                "MimeType": guess_mime_type(attachment_name),
            }

        return {"file": await publish_output_file(**kwargs)}

    # Imported here to avoid a circular import at module load.
    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    tool = _OutputFileTool(
        name=OUTPUT_FILE_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        args_schema=input_model,
        coroutine=create_output_file_fn,
        output_type=output_model,
        metadata={
            "tool_type": "internal",
            "display_name": OUTPUT_FILE_TOOL_NAME,
            "args_schema": input_model,
            "output_schema": output_model,
        },
    )
    tool.set_tool_wrappers(
        awrapper=get_job_attachment_wrapper(output_type=output_model)
    )
    return tool
