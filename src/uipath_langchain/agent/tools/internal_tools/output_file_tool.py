"""Internal tool that publishes agent-authored content as a job attachment.

Injected automatically — never configured by the user — whenever the agent's
output schema declares a job-attachment field. The tool creates the attachment,
links it to the current job, and returns the attachment ticket; the agent then
places that ticket in the declared output field.

Two content sources, and which one is offered depends on the agent flavour:

- ``content`` — the body inline. The only source a standard agent has, since it
  owns no filesystem. Text formats only.
- ``file_path`` — a path in the agent's own workspace, offered only when the
  backend exposes a workspace root (advanced agents). Preferred there: the body
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
from .schema_utils import single_attachment_schema

__all__ = ["OUTPUT_FILE_TOOL_NAME", "create_output_file_tool", "guess_mime_type"]


_DEFAULT_MIME_TYPE = "application/octet-stream"

# mimetypes has no entry for these on every supported Python.
_EXTRA_MIME_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".jsonl": "application/jsonl",
}

_TOOL_DESCRIPTION = (
    "Create an Orchestrator attachment on this job and return its reference, "
    "for an agent output field that expects a file. This creates an attachment, "
    "not a file on disk."
)

_FILE_NAME_DESCRIPTION = (
    "Name the attachment carries, including the extension, e.g. 'summary.md' "
    "or 'accounts.csv'. The extension determines its MIME type, so it must "
    "match the format of the content."
)

_CONTENT_DESCRIPTION = "The full text content of the attachment."

_FILE_PATH_DESCRIPTION = (
    "Path of a file you wrote to your workspace with the filesystem tools, "
    "e.g. '/report.md'. Prefer this over 'content' for a file that already "
    "exists there, and use it for any non-text file."
)


class _OutputFileRejected(Exception):
    """A model-correctable rejection, reported to the agent rather than raised."""


@runtime_checkable
class _WorkspaceBackend(Protocol):
    """The part of a filesystem backend this tool needs: the workspace root.

    ``cwd`` is deepagents' public root attribute, and the same one the
    input-attachment path writes through.
    """

    cwd: Path


def output_file_tool_output_schema() -> dict[str, Any]:
    """The tool's output schema: a single job-attachment ticket under ``file``."""
    return single_attachment_schema(
        "file",
        "Reference to the created file. Use this value for the output file field.",
    )


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
    """Resolve a model-supplied virtual path, rejecting anything outside the root.

    Containment is re-checked after ``resolve()``, which is what catches a
    symlink pointing out of the workspace.
    """
    if not isinstance(backend, _WorkspaceBackend):
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.FILE_ERROR,
            title="Workspace file paths are not available",
            detail=(
                f"'{OUTPUT_FILE_TOOL_NAME}' received a 'file_path' but this agent "
                "has no workspace to read it from. Pass the file body in 'content' instead."
            ),
            category=UiPathErrorCategory.SYSTEM,
        )

    virtual_path = file_path if file_path.startswith("/") else f"/{file_path}"
    if ".." in virtual_path or virtual_path.startswith("~"):
        raise _OutputFileRejected(f"Path traversal is not allowed: {file_path!r}")

    root = Path(backend.cwd).resolve()
    resolved = (root / virtual_path.lstrip("/")).resolve()
    if resolved != root and root not in resolved.parents:
        raise _OutputFileRejected(f"{file_path!r} is outside your workspace")
    return resolved


class _OutputFileTool(StructuredToolWithOutputType, ToolWrapperMixin):
    """Output type plus a state-updating wrapper, as the other attachment-producing tools have."""


def create_output_file_tool(backend: Any | None = None) -> _OutputFileTool:
    """Create the ``create_output_file`` tool.

    Args:
        backend: The agent's filesystem backend, when it has one. ``file_path``
            is offered only for a backend that exposes a workspace root;
            otherwise the tool accepts inline ``content`` only.
    """
    with_file_path = isinstance(backend, _WorkspaceBackend)
    input_model = create_model(_input_schema(with_file_path=with_file_path))
    output_model = create_model(output_file_tool_output_schema())

    async def create_output_file_fn(**kwargs: Any) -> dict[str, Any]:
        file_name = kwargs.get("file_name")
        content = kwargs.get("content")
        file_path = kwargs.get("file_path")

        # Returned rather than raised: an exception here faults the whole run.
        if not file_name:
            return {"error": "'file_name' is required."}
        if not content and not file_path:
            return {
                "error": "Provide the file body in 'content'"
                + (
                    ", or the path of a file you already wrote to your "
                    "workspace in 'file_path'."
                    if with_file_path
                    else "."
                )
            }
        if content and file_path:
            return {"error": "'content' and 'file_path' are mutually exclusive."}

        # file_name comes from the model; it names the attachment, not a path.
        attachment_name = Path(file_name).name

        @mockable(
            name=OUTPUT_FILE_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=[],
        )
        async def publish_output_file(**_tool_kwargs: Any) -> dict[str, Any]:
            try:
                source_path = (
                    _resolve_source_path(backend, file_path) if file_path else None
                )
            except _OutputFileRejected as rejection:
                return {"error": str(rejection)}
            if source_path is not None and not source_path.is_file():
                return {
                    "error": (
                        f"'{file_path}' is not a file in your workspace. Note "
                        "that passing 'content' uploads directly and leaves no "
                        "file behind, so a file you created that way cannot be "
                        "referenced by path. Write it with the filesystem tools "
                        "first, or pass its body in 'content'."
                    )
                }

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

        published = await publish_output_file(**kwargs)
        if "error" in published:
            return published
        return {"file": published}

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
