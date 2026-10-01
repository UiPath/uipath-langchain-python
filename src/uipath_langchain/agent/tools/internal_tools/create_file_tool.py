"""Internal tool that publishes agent-authored content as a job attachment.

Opt-in: the agent author adds it as the ``create-file`` built-in tool. It
creates the attachment, links it to the current job, and returns the attachment
reference, which the agent passes to a tool that accepts a file or places in an
output field that expects one.

Two content sources:

- ``content`` — the file body inline. Text formats only.
- ``filePath`` — a path in the agent's own workspace, usable only when the
  backend exposes a workspace root (advanced agents). The body never
  round-trips through the model, so large and binary files work.

The model sees the input schema saved with the resource. Agent Builder saves all
three arguments and marks ``filePath`` read-only for standard agents.
"""

from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from langchain_core.tools import BaseTool, ToolException
from uipath.agent.models.agent import AgentInternalToolResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import UiPathConfig

from uipath_langchain.agent.attachments.mime_types import guess_mime_type
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name

from .schema_utils import single_attachment_schema

__all__ = [
    "CreateFileTool",
    "create_file_tool",
    "create_file_tool_name",
]


@runtime_checkable
class _WorkspaceBackend(Protocol):
    """The part of a filesystem backend this tool needs: the workspace root.

    ``cwd`` is deepagents' public root attribute, and the same one the
    input-attachment path writes through.
    """

    cwd: Path


def create_file_tool_output_schema() -> dict[str, Any]:
    """The tool's output schema: a single job-attachment ticket under ``file``."""
    return single_attachment_schema(
        "file",
        "Reference to the created file. Use this value, unchanged, wherever a "
        "file is expected.",
    )


def _resolve_source_path(backend: _WorkspaceBackend, file_path: str) -> Path:
    """Resolve a model-supplied virtual path, rejecting anything outside the root.

    Containment is re-checked after ``resolve()``, which is what catches a
    symlink pointing out of the workspace.
    """
    virtual_path = file_path if file_path.startswith("/") else f"/{file_path}"
    if ".." in virtual_path or virtual_path.startswith("~"):
        raise ToolException(f"Path traversal is not allowed: {file_path!r}")

    root = Path(backend.cwd).resolve()
    resolved = (root / virtual_path.lstrip("/")).resolve()
    if resolved != root and root not in resolved.parents:
        raise ToolException(f"{file_path!r} is outside your workspace")
    return resolved


class CreateFileTool(StructuredToolWithArgumentProperties):
    """The create-file tool, recognisable by type whatever the author named it."""

    resource: AgentInternalToolResourceConfig
    workspace: Any | None = None

    def with_workspace(self, backend: Any) -> "CreateFileTool":
        """This tool rebuilt against ``backend``, keeping metadata set on it since."""
        tool = create_file_tool(self.resource, backend)
        tool.metadata = {**(self.metadata or {}), **(tool.metadata or {})}
        return tool


def create_file_tool_name(tools: Sequence[BaseTool]) -> str | None:
    """The name the model knows the create-file tool by, if the agent has one."""
    return next((tool.name for tool in tools if isinstance(tool, CreateFileTool)), None)


def create_file_tool(
    resource: AgentInternalToolResourceConfig, backend: Any | None = None
) -> CreateFileTool:
    """Create the create-file tool from its resource.

    Args:
        resource: The saved ``create-file`` resource. Its name, description,
            input schema and argument properties are what the model sees.
        backend: The agent's filesystem backend, when it has one. ``filePath``
            resolves only against a backend that exposes a workspace root.
    """
    workspace = backend if isinstance(backend, _WorkspaceBackend) else None
    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model(resource.input_schema)
    output_model = create_model(create_file_tool_output_schema())

    async def create_file_fn(**kwargs: Any) -> dict[str, Any]:
        file_name = kwargs.get("fileName")
        content = kwargs.get("content")
        file_path = kwargs.get("filePath")

        if not file_name:
            raise ToolException("'fileName' is required.")
        if file_path and workspace is None:
            raise ToolException(
                "This agent has no workspace to read 'filePath' from. Pass the "
                "file body in 'content' instead."
            )
        if not content and not file_path:
            raise ToolException(
                "Provide the file body in 'content'"
                + (
                    ", or an existing workspace path in 'filePath'."
                    if workspace is not None
                    else "."
                )
            )
        if content and file_path:
            raise ToolException("'content' and 'filePath' are mutually exclusive.")

        # fileName comes from the model; it names the attachment, not a path.
        attachment_name = Path(file_name).name

        @mockable(
            name=tool_name,
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=[],
        )
        async def publish_file(**_tool_kwargs: Any) -> dict[str, Any]:
            source_path = (
                _resolve_source_path(workspace, file_path)
                if file_path and workspace is not None
                else None
            )
            if source_path is not None and not source_path.is_file():
                raise ToolException(
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

        return {"file": await publish_file(**kwargs)}

    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    tool = CreateFileTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=create_file_fn,
        handle_tool_error=True,
        output_type=output_model,
        resource=resource,
        workspace=workspace,
        argument_properties=resource.argument_properties,
        metadata={
            "tool_type": "internal",
            "display_name": tool_name,
            "args_schema": input_model,
            "output_schema": output_model,
        },
    )
    tool.set_tool_wrappers(
        awrapper=get_job_attachment_wrapper(output_type=output_model)
    )
    return tool
