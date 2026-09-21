"""Download the attachments a tool returns into the advanced agent's workspace.

An advanced agent reads files with its filesystem tools, so a job attachment is
useful to it only once it is on disk. Input attachments and chat attachments are
downloaded before the loop starts. A ticket that a tool returns mid-run, such as
a process tool's output file, a batch transform result or a child agent's output,
was not: on this path tools run through deepagents' tool node, so the
job-attachment wrapper of the standard agent never sees the result either. This
middleware closes that gap at the tool boundary. Every ticket found in a tool
result is streamed into the workspace under the layout input attachments use and
gains a ``FilePath`` the agent can open, in the main agent and in every subagent
alike.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple, cast

from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import AgentMiddleware, AgentState, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ..attachments.constants import OUTPUT_FILE_TOOL_NAME
from .utils import _download_missing, _workspace_file_name

logger = logging.getLogger(__name__)

# The agent wrote and named this file itself; the ticket points back at a copy of
# what is already in the workspace.
_TOOLS_WITHOUT_DOWNLOAD: frozenset[str] = frozenset({OUTPUT_FILE_TOOL_NAME})

ToolCallHandler = Callable[[ToolCallRequest], ToolMessage | Command[Any]]
AsyncToolCallHandler = Callable[
    [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
]


class _TicketRef(NamedTuple):
    attachment_id: uuid.UUID
    full_name: str


def _as_ticket(value: Any) -> _TicketRef | None:
    """The attachment a JobAttachment-shaped object refers to, else None."""
    if not isinstance(value, dict):
        return None
    full_name = value.get("FullName")
    if not isinstance(full_name, str) or not full_name:
        return None
    try:
        attachment_id = uuid.UUID(str(value["ID"]))
    except (KeyError, ValueError, AttributeError, TypeError):
        return None
    return _TicketRef(attachment_id, full_name)


def _iter_tickets(payload: Any) -> Iterator[tuple[dict[str, Any], _TicketRef]]:
    if isinstance(payload, dict):
        ref = _as_ticket(payload)
        if ref is not None:
            yield payload, ref
            return
        for value in payload.values():
            yield from _iter_tickets(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_tickets(value)


def find_tickets(payload: Any) -> list[dict[str, Any]]:
    """Every JobAttachment-shaped object in a parsed tool result, in document order.

    Detection is structural rather than schema-driven: an MCP tool, the code
    interpreter or a child agent's free-form output can all carry a ticket that
    no output schema declares. A ticket's own fields, such as ``Metadata``, are
    not searched.
    """
    return [ticket for ticket, _ in _iter_tickets(payload)]


class _Slot(NamedTuple):
    """One JSON document inside a tool message's content."""

    block_index: int | None
    """``None`` for string content; the block index for a text content block."""

    payload: Any


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _slots(content: str | list[Any]) -> list[_Slot]:
    if isinstance(content, str):
        payload = _parse_json(content)
        return [_Slot(None, payload)] if isinstance(payload, (dict, list)) else []
    slots: list[_Slot] = []
    for block_index, block in enumerate(content):
        if not isinstance(block, dict) or not isinstance(block.get("text"), str):
            continue
        payload = _parse_json(block["text"])
        if isinstance(payload, (dict, list)):
            slots.append(_Slot(block_index, payload))
    return slots


def _dump(payload: Any) -> str:
    # the serialization the tool node applied to the original output
    return json.dumps(payload, ensure_ascii=False)


def _content_with(content: str | list[Any], slots: list[_Slot]) -> str | list[Any]:
    if isinstance(content, str):
        return _dump(slots[0].payload)
    rewritten = list(content)
    for slot in slots:
        if slot.block_index is not None:
            rewritten[slot.block_index] = {
                **rewritten[slot.block_index],
                "text": _dump(slot.payload),
            }
    return rewritten


def _tool_messages(result: ToolMessage | Command[Any]) -> list[ToolMessage]:
    if isinstance(result, ToolMessage):
        return [result]
    if isinstance(result, Command) and isinstance(result.update, dict):
        messages = result.update.get("messages")
        if isinstance(messages, list):
            return [m for m in messages if isinstance(m, ToolMessage)]
    return []


def _with_messages(
    result: ToolMessage | Command[Any], rewritten: dict[int, ToolMessage]
) -> ToolMessage | Command[Any]:
    if isinstance(result, ToolMessage):
        return rewritten.get(id(result), result)
    update = cast(dict[str, Any], result.update)
    messages = [rewritten.get(id(m), m) for m in update["messages"]]
    return dataclasses.replace(result, update={**update, "messages": messages})


class ToolAttachmentsMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Give every attachment a tool returns a ``FilePath`` in the workspace.

    Each ticket in a tool result is downloaded to ``<backend.cwd>/<ID>_<name>``,
    the layout input attachments already use, and the ticket in the tool message
    gains ``FilePath``. A file already in the workspace is not fetched again. A
    ticket whose download fails is left without a path rather than failing the
    tool call; the agent still holds a reference it can hand to other tools.
    Error results and results of :data:`_TOOLS_WITHOUT_DOWNLOAD` pass through
    untouched.
    """

    def __init__(self, backend: FilesystemBackend) -> None:
        self.backend = backend

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: ToolCallHandler
    ) -> ToolMessage | Command[Any]:
        result = handler(request)
        if self._skips(request):
            return result
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.resolve(result))
        logger.warning(
            "Tool attachments stay unopenable: the tool ran synchronously inside a "
            "running event loop, where they cannot be downloaded"
        )
        return result

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: AsyncToolCallHandler
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        if self._skips(request):
            return result
        return await self.resolve(result)

    @staticmethod
    def _skips(request: ToolCallRequest) -> bool:
        return request.tool_call["name"] in _TOOLS_WITHOUT_DOWNLOAD

    async def resolve(
        self, result: ToolMessage | Command[Any]
    ) -> ToolMessage | Command[Any]:
        """Return ``result`` with a path on every ticket whose file is in the workspace.

        The result comes back unchanged, same object, when it carries no ticket.
        """
        parsed = [
            (message, slots)
            for message in _tool_messages(result)
            if message.status != "error"
            for slots in (_slots(message.content),)
            if slots
        ]
        tickets = [
            found
            for _, slots in parsed
            for slot in slots
            for found in _iter_tickets(slot.payload)
        ]
        if not tickets:
            return result

        paths: dict[uuid.UUID, Path] = {
            ref.attachment_id: self.backend.cwd
            / _workspace_file_name(ref.attachment_id, ref.full_name)
            for _, ref in tickets
        }
        downloaded = await _download_missing(paths, self.backend.cwd)
        for ticket, ref in tickets:
            path = downloaded.get(ref.attachment_id)
            if path is None:
                ticket.pop("FilePath", None)
            else:
                ticket["FilePath"] = f"/{path.name}"

        rewritten = {
            id(message): message.model_copy(
                update={"content": _content_with(message.content, slots)}
            )
            for message, slots in parsed
        }
        return _with_messages(result, rewritten)
