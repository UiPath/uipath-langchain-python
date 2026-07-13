"""Advanced agent utilities."""

import asyncio
import copy
import logging
import uuid
from pathlib import Path
from typing import Any, NamedTuple, cast

from deepagents.backends import BackendProtocol, FilesystemBackend
from deepagents.backends.protocol import BackendFactory
from jsonpath_ng import parse as jsonpath_parse  # type: ignore[import-untyped]
from langchain.tools import ToolRuntime
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel
from uipath.platform import UiPath
from uipath.platform.attachments import Attachment

from .types import AdvancedAgentGraphState

logger = logging.getLogger(__name__)

# --- Workspace memory layout ---
# Durable memory lives under <workspace>/memory/: MEMORY.md is the always-loaded
# index, entries live in <workspace>/memory/<name>.md. deepagents' MemoryMiddleware
# handles loading/injection, but backed by the agent's FilesystemBackend (run-scoped,
# persisted via WorkspaceHydrator) rather than the cross-run StoreBackend.
MEMORY_DIR_NAME = "memory"
MEMORY_INDEX_FILENAME = "MEMORY.md"
WORKSPACE_FILESYSTEM_BACKEND_ATTR = "is_uipath_workspace_filesystem_backend"

# Virtual path handed to MemoryMiddleware as a source; the agent's virtual-mode
# FilesystemBackend resolves it under the workspace root.
MEMORY_INDEX_VIRTUAL_PATH = f"/{MEMORY_DIR_NAME}/{MEMORY_INDEX_FILENAME}"


def is_workspace_filesystem_backend(
    backend: BackendProtocol | BackendFactory | None,
) -> bool:
    """Return whether a backend is backed by a runtime workspace filesystem."""
    return isinstance(backend, FilesystemBackend) or (
        callable(backend)
        and getattr(backend, WORKSPACE_FILESYSTEM_BACKEND_ATTR, False) is True
    )


def _resolve_filesystem_backend(
    backend: BackendProtocol | BackendFactory | None,
    *,
    state: BaseModel | None = None,
    config: RunnableConfig | None = None,
) -> FilesystemBackend:
    if isinstance(backend, FilesystemBackend):
        return backend
    if callable(backend) and is_workspace_filesystem_backend(backend):
        resolved = backend(
            ToolRuntime(
                state=state,
                context=None,
                config=config or {},
                stream_writer=lambda _: None,
                tool_call_id=None,
                store=None,
            )
        )
        if isinstance(resolved, FilesystemBackend):
            return resolved
        raise TypeError(
            "UiPath workspace backend factory must resolve to FilesystemBackend, "
            f"got {type(resolved).__name__}"
        )
    raise NotImplementedError(
        "Advanced agent with input attachments requires a FilesystemBackend, "
        f"got {type(backend).__name__}"
    )


def create_state_with_input(
    input_schema: type[BaseModel] | None,
) -> type[AdvancedAgentGraphState]:
    """Create combined state by merging AdvancedAgentGraphState with the input schema."""
    if input_schema is None:
        return AdvancedAgentGraphState
    CompleteState = type(
        "CompleteAdvancedAgentGraphState",
        (AdvancedAgentGraphState, input_schema),
        {},
    )
    cast(type[BaseModel], CompleteState).model_rebuild()
    return CompleteState


class _AttachmentDownload(NamedTuple):
    """One input attachment to download and patch back into the args."""

    location: Any
    attachment_id: uuid.UUID
    file_name: str
    ticket: dict[str, Any]


async def resolve_input_attachments(
    backend: BackendProtocol | BackendFactory | None,
    attachment_paths: list[str],
    input_args: dict[str, Any],
    *,
    state: BaseModel | None = None,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Download attachment-shaped inputs into the backend and add a ``FilePath``.

    Each ticket is streamed to ``<backend.cwd>/<ID>_<name>`` and augmented with a
    ``FilePath`` so the agent's file tools can open it.
    """
    backend = _resolve_filesystem_backend(backend, state=state, config=config)

    result = copy.deepcopy(input_args)
    client = UiPath()

    worklist: list[_AttachmentDownload] = []
    for path_expr in attachment_paths:
        for match in jsonpath_parse(path_expr).find(result):
            ticket = match.value
            if not isinstance(ticket, dict) or "ID" not in ticket:
                continue
            att = Attachment.model_validate(ticket, from_attributes=True)
            worklist.append(
                _AttachmentDownload(
                    location=match.full_path,
                    attachment_id=att.id,
                    # basename only: full_name is caller-controlled, keep the
                    # download inside the workspace (no path traversal)
                    file_name=f"{att.id}_{Path(att.full_name).name}",
                    ticket=ticket,
                )
            )

    logger.info(
        "Downloading %d input attachment(s) into %s", len(worklist), backend.cwd
    )

    await asyncio.gather(
        *(
            client.attachments.download_async(
                key=item.attachment_id,
                destination_path=str(backend.cwd / item.file_name),
            )
            for item in worklist
        )
    )
    for item in worklist:
        item.location.update(result, {**item.ticket, "FilePath": f"/{item.file_name}"})
    return result
