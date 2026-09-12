"""Advanced agent utilities."""

import asyncio
import copy
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

from deepagents.backends import BackendProtocol, FilesystemBackend
from jsonpath_ng import parse as jsonpath_parse  # type: ignore[import-untyped]
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict
from uipath.platform import UiPath
from uipath.platform.attachments import Attachment

from ..._utils._attachments import (
    ATTACHMENTS_BLOCK_PREFIX,
    render_attachments_block,
)
from .types import AdvancedAgentGraphState

logger = logging.getLogger(__name__)

# --- Workspace memory layout ---
# Durable memory lives under <workspace>/memory/: MEMORY.md is the always-loaded
# index, entries live in <workspace>/memory/<name>.md. deepagents' MemoryMiddleware
# handles loading/injection, but backed by the agent's FilesystemBackend (run-scoped,
# persisted via WorkspaceHydrator) rather than the cross-run StoreBackend.
MEMORY_DIR_NAME = "memory"
MEMORY_INDEX_FILENAME = "MEMORY.md"

# Virtual path handed to MemoryMiddleware as a source; the agent's virtual-mode
# FilesystemBackend resolves it under the workspace root.
MEMORY_INDEX_VIRTUAL_PATH = f"/{MEMORY_DIR_NAME}/{MEMORY_INDEX_FILENAME}"


def create_state_with_input(
    input_schema: type[BaseModel] | None,
    *,
    base: type[BaseModel] = AdvancedAgentGraphState,
    name: str = "CompleteAdvancedAgentGraphState",
    model_config: ConfigDict | None = None,
) -> Any:
    """Create combined state by merging ``base`` with the input schema."""
    if input_schema is None:
        return base
    namespace: dict[str, Any] = {}
    if model_config is not None:
        namespace["model_config"] = model_config
    CompleteState = type(name, (base, input_schema), namespace)
    cast(type[BaseModel], CompleteState).model_rebuild()
    return CompleteState


def _workspace_file_name(attachment_id: uuid.UUID, full_name: str) -> str:
    # basename only: full_name is caller-controlled, keep the download inside
    # the workspace (no path traversal)
    return f"{attachment_id}_{Path(full_name).name}"


class _AttachmentDownload(NamedTuple):
    """One input attachment to download and patch back into the args."""

    location: Any
    attachment_id: uuid.UUID
    file_name: str
    ticket: dict[str, Any]


async def resolve_input_attachments(
    backend: BackendProtocol | None,
    attachment_paths: list[str],
    input_args: dict[str, Any],
) -> dict[str, Any]:
    """Download attachment-shaped inputs into the backend and add a ``FilePath``.

    Each ticket is streamed to ``<backend.cwd>/<ID>_<name>`` and augmented with a
    ``FilePath`` so the agent's file tools can open it. FilesystemBackend only.
    """
    if not isinstance(backend, FilesystemBackend):
        raise NotImplementedError(
            "Advanced agent with input attachments requires a FilesystemBackend, "
            f"got {type(backend).__name__}"
        )

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
                    file_name=_workspace_file_name(att.id, att.full_name),
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


def _with_attachments_block(
    message: AnyMessage, attachments: list[dict[str, Any]]
) -> AnyMessage:
    rendered = render_attachments_block(attachments)
    content = [
        {**block, "text": rendered}
        if isinstance(block, dict)
        and isinstance(block.get("text"), str)
        and block["text"].startswith(ATTACHMENTS_BLOCK_PREFIX)
        else block
        for block in message.content
    ]
    return message.model_copy(
        update={
            "content": content,
            "additional_kwargs": {
                **message.additional_kwargs,
                "attachments": attachments,
            },
        }
    )


def _with_file_paths(
    attachments: list[dict[str, Any]], paths: dict[uuid.UUID, Path]
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for attachment in attachments:
        path = paths.get(uuid.UUID(str(attachment["id"])))
        if path is None:
            resolved.append(
                {key: value for key, value in attachment.items() if key != "file_path"}
            )
        else:
            resolved.append({**attachment, "file_path": f"/{path.name}"})
    return resolved


async def _download_missing(
    paths: dict[uuid.UUID, Path], workspace: Path
) -> dict[uuid.UUID, Path]:
    """Fetch the attachments not already in the workspace, dropping those that fail."""
    missing = {key: path for key, path in paths.items() if not path.exists()}
    if not missing:
        return paths

    logger.info("Downloading %d message attachment(s) into %s", len(missing), workspace)
    client = UiPath()
    outcomes = await asyncio.gather(
        *(
            client.attachments.download_async(key=key, destination_path=str(path))
            for key, path in missing.items()
        ),
        return_exceptions=True,
    )
    downloaded = dict(paths)
    for key, outcome in zip(missing, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            logger.warning("Attachment %s could not be downloaded: %s", key, outcome)
            # a failed download leaves a truncated file behind, which would then
            # pass for a complete one on the next exchange
            missing[key].unlink(missing_ok=True)
            del downloaded[key]
    return downloaded


async def resolve_message_attachments(
    backend: BackendProtocol | None,
    messages: Sequence[AnyMessage],
) -> list[AnyMessage]:
    """Download attachments referenced by messages and add their ``file_path``.

    Each attachment is streamed to ``<backend.cwd>/<id>_<name>``, the layout
    input attachments already use, and its entry in the message's attachment
    block gains the path the agent's file tools can open. Files already in the
    workspace are left alone, so replaying a conversation history downloads
    nothing. An attachment that cannot be downloaded is left without a path
    rather than failing the exchange. Returns only the messages that changed.
    """
    candidates = [
        message
        for message in messages
        if message.additional_kwargs.get("attachments")
        and isinstance(message.content, list)
    ]
    if not candidates:
        return []
    if not isinstance(backend, FilesystemBackend):
        logger.warning(
            "Message attachments stay unopenable: %s has no workspace to download into",
            type(backend).__name__,
        )
        return []

    paths: dict[uuid.UUID, Path] = {}
    for message in candidates:
        for attachment in message.additional_kwargs["attachments"]:
            attachment_id = uuid.UUID(str(attachment["id"]))
            paths[attachment_id] = backend.cwd / _workspace_file_name(
                attachment_id, attachment["full_name"]
            )

    paths = await _download_missing(paths, backend.cwd)
    return [
        _with_attachments_block(
            message, _with_file_paths(message.additional_kwargs["attachments"], paths)
        )
        for message in candidates
    ]
