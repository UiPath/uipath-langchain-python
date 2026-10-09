"""A2A singleton tool — one tool per remote agent.

Each tool maintains conversation context (task_id/context_id) across calls
using deterministic persistence via LangGraph graph state (tools_storage).

Authentication uses the UiPath SDK Bearer token, resolved lazily on first call.
Client lifecycle is managed by the caller via ``A2aClient.dispose()`` or the
``open_a2a_tools`` async context manager.
"""

import asyncio
import json
import weakref
from contextlib import AsyncExitStack, asynccontextmanager, nullcontext
from logging import getLogger
from typing import Any, AsyncGenerator, Awaitable, Callable, Sequence
from urllib.parse import urlparse

import httpx
from a2a.client import Client
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import (
    AgentCard,
    AgentInterface,
    Message,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
)
from a2a.utils.errors import InvalidRequestError
from google.protobuf.json_format import ParseDict
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, Field
from uipath._utils import resource_override
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.agent.models.agent import (
    AgentA2aResourceConfig,
    AgentConversationalAgentToolResourceConfig,
)
from uipath.core.tracing.span_utils import UiPathSpanUtils
from uipath.platform.common import header_folder, resolve_service_url
from uipath.platform.errors import FolderNotFoundException

from uipath_langchain._utils import get_execution_folder_path
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.base_uipath_structured_tool import (
    BaseUiPathStructuredTool,
)
from uipath_langchain.agent.tools.tool_node import (
    ToolWrapperMixin,
    ToolWrapperReturnType,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name
from uipath_langchain.chat.hitl import REQUIRE_CONVERSATIONAL_CONFIRMATION

logger = getLogger(__name__)

A2aResource = AgentA2aResourceConfig | AgentConversationalAgentToolResourceConfig

_MAX_CONVERSATIONS = 20

_CONVERSATION_HINT = (
    "Each reply includes a conversation_id. Pass it to continue that conversation. "
    "Set new_conversation=true to start a separate one for an unrelated topic or "
    "for parallel work. If you pass neither, the most recent open conversation may "
    "continue or a new one starts, so pass conversation_id whenever you mean to "
    "continue a specific conversation."
)

# The A2A terminal task states
_TERMINAL_TASK_STATES = frozenset(
    {
        "completed",
        "canceled",
        "failed",
        "rejected",
    }
)


class A2aToolInput(BaseModel):
    """Input schema for A2A agent tool."""

    message: str = Field(description="The message to send to the remote agent.")


class A2aConversationInput(BaseModel):
    """Input schema for a conversational agent tool."""

    message: str = Field(description="The message to send to the agent.")
    conversation_id: str | None = Field(
        default=None,
        description="Continue the conversation with this id, taken from an earlier reply.",
    )
    new_conversation: bool | None = Field(
        default=False,
        description="Start a separate conversation. Do not combine with conversation_id.",
    )


class A2aStructuredToolWithWrapper(BaseUiPathStructuredTool, ToolWrapperMixin):
    pass


class A2aClient:
    """Wraps an A2A client and its underlying httpx.AsyncClient for lifecycle management.

    The A2A client is initialized lazily on first ``get()`` call to avoid blocking
    tool creation. The caller must call ``dispose()`` to close the HTTP connection
    pool when done.
    """

    def __init__(
        self,
        agent_card: AgentCard,
        resource_name: str,
        protocol_version: str | None = "1.0",
        conversational: bool = False,
        process_name: str | None = None,
        folder_path: str | None = None,
    ) -> None:
        self._agent_card = agent_card
        self._resource_name = resource_name
        self._protocol_version = protocol_version
        # Set for conversational agents, which are resolved by Orchestrator
        # release instead of the Remote A2A registry.
        self._conversational = conversational
        self._process_name = process_name
        self._folder_path = folder_path
        self._lock = asyncio.Lock()
        self._client: Client | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def get(self) -> Client:
        """Get (or lazily create) the A2A client."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    from a2a.client import ClientConfig, ClientFactory
                    from uipath.platform import UiPath

                    if self._protocol_version is None:
                        raise ValueError(
                            f"Remote A2A agent '{self._resource_name}' has no compatible "
                            "JSON-RPC endpoint for A2A 1.0 or 0.3"
                        )

                    sdk = UiPath()
                    if self._conversational:
                        if not self._process_name:
                            raise ValueError(
                                f"conversational agent tool '{self._resource_name}' "
                                "has no processName"
                            )
                        a2a_url = await _resolve_conversational_a2a_url(
                            sdk, self._process_name, self._folder_path
                        )
                    else:
                        agent = await sdk.remote_a2a.retrieve_async(
                            name=self._resource_name,
                            folder_path=get_execution_folder_path(),
                        )
                        if not agent.a2a_url:
                            raise ValueError(
                                f"Remote A2A agent '{self._resource_name}' has no a2a_url configured"
                            )
                        a2a_url = agent.a2a_url
                    runtime_card = AgentCard()
                    runtime_card.CopyFrom(self._agent_card)
                    runtime_card.ClearField("supported_interfaces")
                    runtime_card.supported_interfaces.append(
                        AgentInterface(
                            url=a2a_url,
                            protocol_binding="JSONRPC",
                            protocol_version=self._protocol_version,
                        )
                    )

                    client_kwargs = get_httpx_client_kwargs(
                        headers={"Authorization": f"Bearer {sdk._config.secret}"},
                    )
                    client_kwargs["timeout"] = httpx.Timeout(300.0, connect=10.0)
                    self._http_client = httpx.AsyncClient(**client_kwargs)
                    self._client = ClientFactory(
                        ClientConfig(
                            httpx_client=self._http_client,
                            streaming=False,
                            accepted_output_modes=list(
                                runtime_card.default_output_modes
                            ),
                        )
                    ).create(runtime_card)
        return self._client

    async def dispose(self) -> None:
        """Close the underlying HTTP client and release the A2A client."""
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                logger.warning("Failed to close A2A httpx client", exc_info=True)
            finally:
                self._http_client = None
                self._client = None


@resource_override(resource_type="process", resource_identifier="process_name")
async def _resolve_conversational_a2a_url(
    sdk: Any, process_name: str, folder_path: str | None
) -> str:
    """Resolve the AgentHub A2A URL of a conversational agent's Orchestrator release.

    The release is looked up by process name in the folder (a solution-deployed
    process binding overrides both). The configured folder falls back to the
    execution folder when it does not resolve. The URL is
    ``{agenthub}/a2a/{folderKey}/{releaseId}``.
    """
    configured_folder = folder_path
    execution_folder = get_execution_folder_path()
    candidates = [
        path for path in dict.fromkeys([folder_path, execution_folder]) if path
    ]
    if not candidates:
        raise ValueError(
            f"Conversational agent '{process_name}' has no folder: none is "
            "configured and no execution folder is set"
        )

    folder_key = None
    for candidate in candidates:
        try:
            folder_key = await sdk.folders.retrieve_folder_key_async(candidate)
        except FolderNotFoundException:
            continue
        if configured_folder and candidate != configured_folder:
            logger.warning(
                "Folder '%s' of conversational agent '%s' not found, "
                "falling back to execution folder '%s'",
                configured_folder,
                process_name,
                candidate,
            )
        folder_path = candidate
        break
    if not folder_key:
        raise ValueError(
            f"Conversational agent '{process_name}': folder "
            f"{' / '.join(repr(c) for c in candidates)} not found"
        )

    escaped_name = process_name.replace("'", "''")
    response = await sdk.api_client.request_async(
        "GET",
        "/orchestrator_/odata/Releases",
        params={
            "$filter": f"Name eq '{escaped_name}'",
            "$select": "Id,Name,ProcessType,IsConversational",
        },
        headers=header_folder(folder_key, None),
    )
    release = next(
        (
            r
            for r in response.json().get("value", [])
            if r.get("ProcessType") == "Agent" and r.get("IsConversational") is True
        ),
        None,
    )
    if release is None:
        raise ValueError(
            f"'{process_name}' in folder '{folder_path}' is not a published "
            "conversational agent"
        )

    path = f"agenthub_/a2a/{folder_key}/{release['Id']}"
    return resolve_service_url(path) or f"{sdk._config.base_url.rstrip('/')}/{path}"


def _extract_text(obj: Task | Message) -> str:
    """Extract text content from a Task or Message response."""
    if isinstance(obj, Message):
        return get_message_text(obj)
    if (
        obj.HasField("status")
        and obj.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        and obj.status.HasField("message")
    ):
        return get_message_text(obj.status.message)
    if obj.artifacts:
        return "\n".join(filter(None, (get_artifact_text(a) for a in obj.artifacts)))
    if obj.HasField("status") and obj.status.HasField("message"):
        return get_message_text(obj.status.message)
    for message in reversed(obj.history):
        if message.role == Role.ROLE_AGENT:
            return get_message_text(message)
    return ""


def _task_state_name(state: int) -> str:
    """Return the stable lowercase task-state value used by tool state."""
    name = TaskState.Name(state).removeprefix("TASK_STATE_").lower()
    return "unknown" if name == "unspecified" else name


def _format_response(text: str, state: str) -> str:
    """Build a structured tool response the LLM can act on."""
    return json.dumps({"agent_response": text, "task_state": state})


def _build_description(card: AgentCard) -> str:
    """Build a tool description from an agent card."""
    parts = []
    if card.description:
        parts.append(card.description)
    if card.skills:
        for skill in card.skills:
            skill_desc = skill.name or ""
            if skill.description:
                skill_desc += f": {skill.description}"
            if skill_desc:
                parts.append(f"Skill: {skill_desc}")
    if parts:
        return " | ".join(parts)
    # The card URL is resolved lazily at runtime, so it is empty or stale here;
    # fall back to the agent name rather than exposing an internal/blank URL.
    return f"Remote A2A agent: {card.name}" if card.name else "Remote A2A agent"


def _cached_agent_card(config: A2aResource) -> dict[str, Any] | None:
    if isinstance(config, AgentConversationalAgentToolResourceConfig):
        return config.properties.cached_agent_card
    return config.cached_agent_card


def _build_metadata_card(config: A2aResource) -> AgentCard:
    """Build v1 card metadata without retaining cached transport endpoints."""
    card = AgentCard()
    cached_agent_card = _cached_agent_card(config)
    if cached_agent_card:
        ParseDict(cached_agent_card, card, ignore_unknown_fields=True)

    if not card.name:
        card.name = config.name
    if not card.description and config.description:
        card.description = config.description
    if not card.default_input_modes:
        card.default_input_modes.append("text/plain")
    if not card.default_output_modes:
        card.default_output_modes.append("text/plain")

    # Cached cards may point directly at third-party agents. Invocation must
    # always go through the binding-aware AgentHub proxy resolved at runtime.
    card.ClearField("supported_interfaces")
    return card


def _select_protocol_version(cached_card: dict[str, Any] | None) -> str | None:
    """Select the best JSON-RPC protocol advertised by a cached agent card."""
    if not isinstance(cached_card, dict):
        return None

    interfaces = cached_card.get("supportedInterfaces")
    if isinstance(interfaces, list):
        for interface in interfaces:
            if (
                isinstance(interface, dict)
                and _is_http_endpoint(interface.get("url"))
                and _is_jsonrpc(interface.get("protocolBinding"))
                and _is_protocol_version(interface.get("protocolVersion"), 1, 0)
            ):
                return "1.0"

    if (
        _is_http_endpoint(cached_card.get("url"))
        and _is_jsonrpc(cached_card.get("preferredTransport", "JSONRPC"))
        and _is_protocol_version(cached_card.get("protocolVersion", "0.3.0"), 0, 3)
    ):
        return "0.3"

    return None


def _is_http_endpoint(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _is_jsonrpc(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() == "JSONRPC"


def _is_protocol_version(value: Any, major: int, minor: int) -> bool:
    if not isinstance(value, str):
        return False
    parts = value.strip().split(".")
    if not 2 <= len(parts) <= 4 or not all(
        part.isascii() and part.isdigit() for part in parts
    ):
        return False
    return int(parts[0]) == major and int(parts[1]) == minor


async def _send_a2a_message_or_raise(
    client: Client,
    agent_label: str,
    *,
    message: str,
    task_id: str | None,
    context_id: str | None,
    message_state: str = "completed",
) -> tuple[str, str, str | None, str | None]:
    """Send a message to an A2A agent and return the response; request errors raise.

    ``message_state`` is the task state reported when the agent replies with a
    plain message instead of a task.

    Returns:
        Tuple of (response_text, task_state, new_task_id, new_context_id).
    """
    if task_id or context_id:
        logger.info(
            "A2A continue task=%s context=%s to %s", task_id, context_id, agent_label
        )
    else:
        logger.info("A2A new message to %s", agent_label)

    a2a_message = new_text_message(
        message,
        role=Role.ROLE_USER,
        task_id=task_id,
        context_id=context_id,
    )

    text = ""
    state = "unknown"
    new_task_id = task_id
    new_context_id = context_id

    async for response in client.send_message(SendMessageRequest(message=a2a_message)):
        if response.HasField("message"):
            text = _extract_text(response.message)
            new_context_id = response.message.context_id or new_context_id
            state = message_state
            break
        if response.HasField("task"):
            task = response.task
            text = _extract_text(task)
            new_task_id = task.id or new_task_id
            new_context_id = task.context_id or new_context_id
            if task.HasField("status"):
                state = _task_state_name(task.status.state)
            break

    return (text or "No response received.", state, new_task_id, new_context_id)


async def _send_a2a_message(
    client: Client,
    agent_label: str,
    *,
    message: str,
    task_id: str | None,
    context_id: str | None,
) -> tuple[str, str, str | None, str | None]:
    """Send a message to a remote A2A agent and return the response.

    Returns:
        Tuple of (response_text, task_state, new_task_id, new_context_id).
    """
    try:
        return await _send_a2a_message_or_raise(
            client,
            agent_label,
            message=message,
            task_id=task_id,
            context_id=context_id,
        )
    except Exception as e:
        logger.exception("A2A request to %s failed", agent_label)
        return (f"Error: {e}", "error", task_id, context_id)


class _A2aRequestError(Exception):
    """A request to the agent failed after the client was resolved."""


_ConversationSend = Callable[
    [str, str | None, str | None], Awaitable[tuple[str, str, str | None, str | None]]
]


def _conversation_error(error: str, conversation_id: str | None, message: str) -> str:
    return json.dumps(
        {"error": error, "conversation_id": conversation_id, "message": message}
    )


def _record_conversation(
    entry: dict[str, Any],
    conversation_id: str,
    task_id: str | None,
    *,
    closed: bool,
    make_current: bool,
) -> dict[str, Any]:
    """Return a copy of ``entry`` with the conversation recorded as just used.

    ``last_used`` is a counter derived from the stored entries, so the state is
    deterministic. Beyond ``_MAX_CONVERSATIONS`` the least recently used
    conversation is evicted, closed ones first.
    """
    conversations = {k: dict(v) for k, v in entry["conversations"].items()}
    last_used = max((c["last_used"] for c in conversations.values()), default=0) + 1
    conversations[conversation_id] = {
        "task_id": task_id,
        "state": "closed" if closed else "open",
        "last_used": last_used,
    }
    current = conversation_id if make_current else entry["current"]
    if closed and current == conversation_id:
        current = None
    while len(conversations) > _MAX_CONVERSATIONS:
        evicted = min(
            (cid for cid in conversations if cid != conversation_id),
            key=lambda cid: (
                conversations[cid]["state"] != "closed",
                conversations[cid]["last_used"],
            ),
        )
        del conversations[evicted]
        if current == evicted:
            current = None
    return {"current": current, "conversations": conversations}


def _is_terminal_state_error(error: _A2aRequestError) -> bool:
    return isinstance(error.__cause__, InvalidRequestError) and (
        "terminal state" in str(error).lower()
    )


def _load_entry(stored: dict[str, Any]) -> dict[str, Any]:
    """Read a ``tools_storage`` entry, migrating the earlier single-conversation shape."""
    if "conversations" not in stored and stored.get("context_id"):
        context_id = stored["context_id"]
        return {
            "current": context_id,
            "conversations": {
                context_id: {
                    "task_id": stored.get("task_id"),
                    "state": "open",
                    "last_used": 1,
                }
            },
        }
    return {
        "current": stored.get("current"),
        "conversations": dict(stored.get("conversations") or {}),
    }


def _lock_for(
    locks: weakref.WeakValueDictionary[str, asyncio.Lock], cid: str
) -> asyncio.Lock:
    lock = locks.get(cid)
    if lock is None:
        lock = locks[cid] = asyncio.Lock()
    return lock


async def _converse(
    args: A2aConversationInput,
    load: Callable[[], dict[str, Any]],
    save: Callable[[dict[str, Any]], None],
    send: _ConversationSend,
    locks: weakref.WeakValueDictionary[str, asyncio.Lock],
    *,
    use_current: bool = True,
) -> str:
    """Send one message to a conversational agent and return the JSON result.

    The conversation is ``new_conversation`` (no IDs), an explicit
    ``conversation_id`` (sent as the context even when unknown, e.g. from
    replayed history), or the entry's ``current`` one when ``use_current``.
    Sends to the same conversation are serialized through ``locks``, which
    holds a lock only while a call uses it.
    """
    if args.conversation_id and args.new_conversation:
        return _conversation_error(
            "invalid_arguments",
            None,
            "Pass either conversation_id or new_conversation=true, not both.",
        )

    cid = (
        None
        if args.new_conversation
        else args.conversation_id or (load()["current"] if use_current else None)
    )
    lock = _lock_for(locks, cid) if cid else nullcontext()
    async with lock:
        entry = load()
        known = entry["conversations"].get(cid) if cid else None
        if known and known["state"] == "closed":
            return _conversation_error(
                "conversation_closed",
                cid,
                "This conversation has ended. Start a new one with "
                "new_conversation=true.",
            )
        task_id = known["task_id"] if known else None

        try:
            text, state, new_task_id, new_context_id = await send(
                args.message, task_id, cid
            )
        except _A2aRequestError as e:
            if cid and _is_terminal_state_error(e):
                logger.info("A2A conversation %s has ended: %s", cid, e)
                save(
                    _record_conversation(
                        load(), cid, task_id, closed=True, make_current=False
                    )
                )
                return _conversation_error(
                    "conversation_closed",
                    cid,
                    "This conversation has ended. Start a new one with "
                    "new_conversation=true.",
                )
            logger.exception("A2A conversation request failed")
            return _conversation_error("request_failed", cid, f"Error: {e}")

        new_cid = new_context_id or new_task_id or cid
        if new_cid:
            latest = load()
            save(
                _record_conversation(
                    latest,
                    new_cid,
                    new_task_id or task_id,
                    closed=state in _TERMINAL_TASK_STATES,
                    make_current=new_cid != cid or latest["current"] is None,
                )
            )
        result = {
            "agent_response": text,
            "conversation_id": new_cid,
            "task_state": state,
        }
        if cid and new_cid != cid:
            result["note"] = (
                "The previous conversation was not found; a new conversation was "
                "started."
            )
        return json.dumps(result)


def _create_conversation_tool(
    name: str,
    description: str,
    metadata: dict[str, str],
    send: _ConversationSend,
) -> BaseTool:
    """Create the LLM-facing tool for a conversational agent.

    The model addresses conversations by the ``conversation_id`` returned in each
    result. In the ReAct graph the wrapper keeps them in ``tools_storage``. Where
    tools run without a wrapper (advanced mode) nothing is stored: tool instances
    outlive a run, so a call that passes no ``conversation_id`` starts a new
    conversation instead of continuing one from an earlier run.
    """
    locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
        weakref.WeakValueDictionary()
    )

    async def _send(
        *,
        message: str,
        conversation_id: str | None = None,
        new_conversation: bool | None = False,
    ) -> str:
        return await _converse(
            A2aConversationInput(
                message=message,
                conversation_id=conversation_id,
                new_conversation=new_conversation,
            ),
            lambda: {"current": None, "conversations": {}},
            lambda entry: None,
            send,
            locks,
            use_current=False,
        )

    async def _conversation_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        entry = [_load_entry(state.inner_state.tools_storage.get(tool.name) or {})]

        result = await _converse(
            A2aConversationInput.model_validate(call["args"]),
            lambda: entry[0],
            lambda updated: entry.__setitem__(0, updated),
            send,
            locks,
        )

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=result,
                        name=call["name"],
                        tool_call_id=call["id"],
                    )
                ],
                "inner_state": {"tools_storage": {tool.name: entry[0]}},
            }
        )

    tool = A2aStructuredToolWithWrapper(
        name=name,
        description=f"{description} {_CONVERSATION_HINT}",
        coroutine=_send,
        args_schema=A2aConversationInput,
        metadata=metadata,
    )
    tool.set_tool_wrappers(awrapper=_conversation_wrapper)
    return tool


def _create_a2a_tool(
    config: A2aResource, a2a_client: A2aClient, agent_card: AgentCard
) -> BaseTool:
    """Create a single LangChain tool for A2A communication.

    Conversation context (task_id/context_id) is persisted deterministically
    in LangGraph's graph state via tools_storage, ensuring reliable
    multi-turn conversations with the remote agent.
    """
    display_name = agent_card.name or config.name
    tool_name = sanitize_tool_name(config.name)
    tool_description = _build_description(agent_card)
    metadata = {
        "tool_type": "a2a",
        "display_name": display_name,
        "resource_name": config.name,
    }
    if isinstance(config, AgentConversationalAgentToolResourceConfig):
        agent_label = config.properties.process_name or config.name
        metadata["process_name"] = config.properties.process_name or ""
    else:
        agent_label = config.slug
        metadata["slug"] = config.slug

    async def _invoke(
        *,
        message: str,
        task_id: str | None,
        context_id: str | None,
        raise_errors: bool = False,
    ) -> tuple[str, str, str | None, str | None]:
        """Send one message to the remote agent inside an A2A trace span.

        The span is parented under the active tool-call span (via
        ``UiPathSpanUtils.get_parent_context``) so the remote call nests under
        the tool in the Execution Trace, and is marked
        ``uipath.custom_instrumentation`` so the LLMOps exporter keeps it. The
        a2a-sdk's own transport spans are disabled in this package's __init__,
        so this is the single node representing the call.
        """
        parent_ctx = UiPathSpanUtils.get_parent_context()
        tracer = otel_trace.get_tracer(__name__)
        with tracer.start_as_current_span(display_name, context=parent_ctx) as span:
            # "openinference.span.kind" drives the SpanType shown in the UI;
            # "toolCall" is the recognized type for a tool invocation.
            span.set_attribute("openinference.span.kind", "toolCall")
            span.set_attribute("type", "toolCall")
            span.set_attribute("span_type", "toolCall")
            span.set_attribute("uipath.custom_instrumentation", True)
            span.set_attribute("tool_type", "a2a")
            span.set_attribute("input", message)
            span.set_attribute("input.value", message)

            if raise_errors:
                try:
                    client = await a2a_client.get()
                    # A reply without a task is an open conversation, not a finished one.
                    (
                        text,
                        response_state,
                        new_task_id,
                        new_context_id,
                    ) = await _send_a2a_message_or_raise(
                        client,
                        agent_label,
                        message=message,
                        task_id=task_id,
                        context_id=context_id,
                        message_state="input_required",
                    )
                except Exception as e:
                    raise _A2aRequestError(str(e)) from e
            else:
                client = await a2a_client.get()
                (
                    text,
                    response_state,
                    new_task_id,
                    new_context_id,
                ) = await _send_a2a_message(
                    client,
                    agent_label,
                    message=message,
                    task_id=task_id,
                    context_id=context_id,
                )

            span.set_attribute("output", text)
            span.set_attribute("output.value", text)
            span.set_attribute("task_state", response_state)
            if raise_errors and new_context_id:
                span.set_attribute("conversation_id", new_context_id)
            if response_state == "error":
                span.set_status(otel_trace.StatusCode.ERROR, text)
            return text, response_state, new_task_id, new_context_id

    if isinstance(config, AgentConversationalAgentToolResourceConfig):

        async def _send_to_conversation(
            message: str, task_id: str | None, context_id: str | None
        ) -> tuple[str, str, str | None, str | None]:
            return await _invoke(
                message=message,
                task_id=task_id,
                context_id=context_id,
                raise_errors=True,
            )

        return _create_conversation_tool(
            tool_name, tool_description, metadata, _send_to_conversation
        )

    async def _send(*, message: str) -> str:
        text, state, _, _ = await _invoke(
            message=message, task_id=None, context_id=None
        )
        return _format_response(text, state)

    async def _a2a_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        prior = state.inner_state.tools_storage.get(tool.name) or {}
        task_id = prior.get("task_id")
        context_id = prior.get("context_id")

        text, task_state, new_task_id, new_context_id = await _invoke(
            message=call["args"]["message"],
            task_id=task_id,
            context_id=context_id,
        )

        # The server rejects messages to a terminal task, so start a new task
        # next turn, keeping context_id to stay in the same conversation.
        if task_state in _TERMINAL_TASK_STATES:
            new_task_id = None

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=_format_response(text, task_state),
                        name=call["name"],
                        tool_call_id=call["id"],
                    )
                ],
                "inner_state": {
                    "tools_storage": {
                        tool.name: {
                            "task_id": new_task_id,
                            "context_id": new_context_id,
                        }
                    }
                },
            }
        )

    tool = A2aStructuredToolWithWrapper(
        name=tool_name,
        description=tool_description,
        coroutine=_send,
        args_schema=A2aToolInput,
        metadata=metadata,
    )
    tool.set_tool_wrappers(awrapper=_a2a_wrapper)
    return tool


def create_a2a_tools_and_clients(
    resources: Sequence[A2aResource],
    *,
    is_conversational: bool = False,
) -> tuple[list[BaseTool], list[A2aClient]]:
    """Create A2A tools and their associated clients from resource configurations.

    Each enabled A2A resource gets a dedicated ``A2aClient`` (with its own
    httpx.AsyncClient). The caller is responsible for calling ``dispose()``
    on each returned client when done.

    For automatic client lifecycle management, prefer ``open_a2a_tools``.

    Args:
        resources: List of A2A resource configurations from agent.json.
        is_conversational: Whether the agent itself is conversational; gates
            ``requireConversationalConfirmation`` on conversational agent tools.

    Returns:
        Tuple of (tools, clients) where:
        - tools: BaseTool instances, one per enabled A2A resource
        - clients: A2aClient instances that need to be disposed when done
    """
    tools: list[BaseTool] = []
    clients: list[A2aClient] = []

    for resource in resources:
        if resource.is_enabled is False:
            logger.info("Skipping disabled A2A resource '%s'", resource.name)
            continue

        logger.info("Creating A2A tool for resource '%s'", resource.name)

        agent_card = _build_metadata_card(resource)

        protocol_version = _select_protocol_version(_cached_agent_card(resource))
        if isinstance(resource, AgentConversationalAgentToolResourceConfig):
            # AgentHub serves conversational agents over both A2A 0.3 and 1.0.
            a2a_client = A2aClient(
                agent_card,
                resource.name,
                protocol_version=protocol_version or "1.0",
                conversational=True,
                process_name=resource.properties.process_name,
                folder_path=resource.properties.folder_path,
            )
        else:
            a2a_client = A2aClient(
                agent_card, resource.name, protocol_version=protocol_version
            )
        tool = _create_a2a_tool(resource, a2a_client, agent_card)
        if (
            is_conversational
            and isinstance(resource, AgentConversationalAgentToolResourceConfig)
            and resource.properties.require_conversational_confirmation
        ):
            tool.metadata = {
                **(tool.metadata or {}),
                REQUIRE_CONVERSATIONAL_CONFIRMATION: True,
            }
        tools.append(tool)
        clients.append(a2a_client)

    return tools, clients


@asynccontextmanager
async def open_a2a_tools(
    resources: Sequence[A2aResource],
    *,
    is_conversational: bool = False,
) -> AsyncGenerator[list[BaseTool], None]:
    """Open A2A tools with automatic client lifecycle management.

    Wraps ``create_a2a_tools_and_clients`` in an ``AsyncExitStack`` so each
    ``A2aClient`` is disposed when the context exits.

    Args:
        resources: List of A2A resource configurations.
        is_conversational: Whether the agent itself is conversational.

    Yields:
        List of BaseTool instances for all enabled A2A resources.
    """
    async with AsyncExitStack() as stack:
        tools, clients = create_a2a_tools_and_clients(
            resources, is_conversational=is_conversational
        )
        for client in clients:
            stack.push_async_callback(client.dispose)
        yield tools
