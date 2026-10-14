"""Conversational agent tool — call a UiPath conversational agent over A2A.

A conversational agent is a published Orchestrator release. It is resolved by
process name in the execution folder, like a process tool, and reached through
AgentHub's ``a2a/{folderKey}/{releaseId}`` endpoint. Unlike a Remote A2A agent
(``a2a_tool``) it is not looked up in the Remote A2A registry, and the model
addresses its conversations by a ``conversation_id`` it passes back.

In the ReAct graph the conversations are kept in ``tools_storage``. Where tools
run without a wrapper (advanced mode) nothing is stored. Client lifecycle is
managed by the caller via ``ConversationalAgentA2aClient.dispose()`` or the
``open_conversational_agent_tools`` async context manager.
"""

import asyncio
import json
import os
from contextlib import AsyncExitStack, asynccontextmanager
from logging import getLogger
from typing import Any, AsyncGenerator, Awaitable, Callable, Sequence

import httpx
from a2a.client import Client
from a2a.helpers import new_text_message
from a2a.types import (
    AgentCard,
    AgentInterface,
    Role,
    SendMessageRequest,
)
from a2a.utils.errors import InvalidRequestError, TaskNotFoundError
from google.protobuf.json_format import ParseDict
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, Field
from uipath._utils import resource_override
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.agent.models.agent import AgentConversationalAgentToolResourceConfig
from uipath.core.tracing.span_utils import UiPathSpanUtils
from uipath.platform.common import (
    header_folder,
    inject_routing_headers,
    resolve_service_url,
)
from uipath.platform.errors import FolderNotFoundException

from uipath_langchain._utils import get_execution_folder_path
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.tool_node import ToolWrapperReturnType
from uipath_langchain.agent.tools.utils import sanitize_tool_name
from uipath_langchain.chat.hitl import REQUIRE_CONVERSATIONAL_CONFIRMATION

from .a2a_tool import (
    _TERMINAL_TASK_STATES,
    A2aStructuredToolWithWrapper,
    _build_description,
    _extract_text,
    _select_protocol_version,
    _task_state_name,
)

logger = getLogger(__name__)

_MAX_CONVERSATIONS = 20

_CONVERSATION_HINT = (
    "Each reply includes a conversation_id. Pass it to continue that conversation. "
    "Set new_conversation=true to start a separate one for an unrelated topic or "
    "for parallel work. If you pass neither, the most recent open conversation may "
    "continue or a new one starts, so pass conversation_id whenever you mean to "
    "continue a specific conversation."
)


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


@resource_override(resource_type="process", resource_identifier="process_name")
async def _resolve_conversational_a2a_url(
    sdk: Any, process_name: str, folder_path: str | None
) -> tuple[str, bool]:
    """Resolve the AgentHub A2A URL of a conversational agent's Orchestrator release.

    As for process tools, the release is looked up by process name in the
    execution folder (``folder_path``); ``properties.folderPath`` is not used. A
    solution ``process.<name>`` binding takes precedence: ``resource_override``
    replaces both the name and the folder. Without a path, ``UIPATH_FOLDER_KEY``
    supplies the folder key, as for Remote A2A. The URL is
    ``{agenthub}/a2a/{folderKey}/{releaseId}``. Returns the URL and whether it
    uses a service override, which requires explicit tenant/account headers.
    """
    if folder_path is not None:
        try:
            folder_key = await sdk.folders.retrieve_folder_key_async(folder_path)
        except FolderNotFoundException as e:
            raise ValueError(
                f"Conversational agent '{process_name}': folder '{folder_path}' not found"
            ) from e
    else:
        folder_key = os.getenv("UIPATH_FOLDER_KEY")

    if not folder_key:
        raise ValueError(
            f"Conversational agent '{process_name}' has no folder: "
            "no execution folder path or key is set"
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
            f"'{process_name}' in folder '{folder_path or folder_key}' is not a published "
            "conversational agent"
        )

    path = f"agenthub_/a2a/{folder_key}/{release['Id']}"
    override_url = resolve_service_url(path)
    if override_url:
        return override_url, True
    return f"{sdk._config.base_url.rstrip('/')}/{path}", False


class ConversationalAgentA2aClient:
    """Wraps an A2A client and its httpx.AsyncClient for a conversational agent.

    Same lifecycle as ``A2aClient``: the A2A client is created lazily on the
    first ``get()`` call, and the caller must call ``dispose()`` to close the
    HTTP connection pool when done. The URL comes from the agent's Orchestrator
    release instead of the Remote A2A registry.
    """

    def __init__(
        self,
        agent_card: AgentCard,
        resource_name: str,
        process_name: str | None,
        protocol_version: str,
    ) -> None:
        self._agent_card = agent_card
        self._resource_name = resource_name
        self._process_name = process_name
        self._protocol_version = protocol_version
        self._lock = asyncio.Lock()
        self._client: Client | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def get(self) -> Client:
        """Get (or lazily create) the A2A client."""
        if self._client is not None:
            return self._client

        async with self._lock:
            if self._client is not None:
                return self._client

            from a2a.client import ClientConfig, ClientFactory
            from uipath.platform import UiPath

            sdk = UiPath()
            if not self._process_name:
                raise ValueError(
                    f"conversational agent tool '{self._resource_name}' "
                    "has no processName"
                )
            a2a_url, is_service_override = await _resolve_conversational_a2a_url(
                sdk, self._process_name, get_execution_folder_path()
            )

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
            headers = {"Authorization": f"Bearer {sdk._config.secret}"}
            if is_service_override:
                inject_routing_headers(headers)
            client_kwargs = get_httpx_client_kwargs(headers=headers)
            client_kwargs["timeout"] = httpx.Timeout(300.0, connect=10.0)
            self._http_client = httpx.AsyncClient(**client_kwargs)
            self._client = ClientFactory(
                ClientConfig(
                    httpx_client=self._http_client,
                    streaming=False,
                    accepted_output_modes=list(runtime_card.default_output_modes),
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


class _A2aRequestError(Exception):
    """A request to the agent failed, including resolving the client."""


_ConversationSend = Callable[
    [str, str | None, str | None], Awaitable[tuple[str, str, str | None, str | None]]
]


async def _send_message(
    client: Client,
    agent_label: str,
    *,
    message: str,
    task_id: str | None,
    context_id: str | None,
) -> tuple[str, str, str | None, str | None]:
    """Send a message to a conversational agent and return the response.

    A reply that is a plain message instead of a task is AgentHub's
    conversational reply shape: the conversation is still open, so it is
    reported as ``input_required``. Request errors propagate.

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
            state = "input_required"
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


def _conversation_error(error: str, conversation_id: str | None, message: str) -> str:
    return json.dumps(
        {"error": error, "conversation_id": conversation_id, "message": message}
    )


def _record_conversation(
    conversation_state: dict[str, Any],
    conversation_id: str,
    task_id: str | None,
    *,
    closed: bool,
    make_current: bool,
) -> dict[str, Any]:
    """Return a copy of the state with the conversation recorded as just used.

    ``last_used`` is a counter derived from the stored entries, so the state is
    deterministic. Beyond ``_MAX_CONVERSATIONS`` the least recently used
    conversation is evicted, closed ones first.
    """
    conversations = {
        key: dict(value) for key, value in conversation_state["conversations"].items()
    }
    last_used = max((c["last_used"] for c in conversations.values()), default=0) + 1
    conversations[conversation_id] = {
        "task_id": task_id,
        "state": "closed" if closed else "open",
        "last_used": last_used,
    }
    current = conversation_id if make_current else conversation_state["current"]
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


def _load_conversation_state(stored: dict[str, Any]) -> dict[str, Any]:
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


async def _converse(
    args: A2aConversationInput,
    conversation_state: dict[str, Any],
    send: _ConversationSend,
) -> tuple[str, dict[str, Any]]:
    """Send one message and return the JSON result and updated conversation state.

    The conversation is ``new_conversation`` (no IDs), an explicit
    ``conversation_id`` (sent as the context even when unknown, e.g. from
    replayed history), or the state's ``current`` one.
    Callers are responsible for ordering messages to the same conversation.
    """
    if args.conversation_id and args.new_conversation:
        return (
            _conversation_error(
                "invalid_arguments",
                None,
                "Pass either conversation_id or new_conversation=true, not both.",
            ),
            conversation_state,
        )

    conversation_id = args.conversation_id or conversation_state["current"]
    if args.new_conversation:
        conversation_id = None

    known_conversation = (
        conversation_state["conversations"].get(conversation_id)
        if conversation_id
        else None
    )
    if known_conversation and known_conversation["state"] == "closed":
        return (
            _conversation_error(
                "conversation_closed",
                conversation_id,
                "This conversation has ended. Start a new one with new_conversation=true.",
            ),
            conversation_state,
        )
    task_id = known_conversation["task_id"] if known_conversation else None

    try:
        text, task_state, new_task_id, new_context_id = await send(
            args.message, task_id, conversation_id
        )
    except _A2aRequestError as e:
        task_not_found = isinstance(e.__cause__, TaskNotFoundError)
        if conversation_id and (task_not_found or _is_terminal_state_error(e)):
            logger.info("A2A conversation %s is unavailable: %s", conversation_id, e)
            conversation_state = _record_conversation(
                conversation_state,
                conversation_id,
                task_id,
                closed=True,
                make_current=False,
            )
            message = (
                "This conversation is no longer available. "
                if task_not_found
                else "This conversation has ended. "
            )
            return (
                _conversation_error(
                    "conversation_closed",
                    conversation_id,
                    message + "Start a new one with new_conversation=true.",
                ),
                conversation_state,
            )
        logger.exception("A2A conversation request failed")
        return (
            _conversation_error("request_failed", conversation_id, f"Error: {e}"),
            conversation_state,
        )

    response_conversation_id = new_context_id or new_task_id or conversation_id
    if response_conversation_id:
        conversation_state = _record_conversation(
            conversation_state,
            response_conversation_id,
            new_task_id or task_id,
            closed=task_state in _TERMINAL_TASK_STATES,
            make_current=response_conversation_id != conversation_id
            or conversation_state["current"] is None,
        )
    result = {
        "agent_response": text,
        "conversation_id": response_conversation_id,
        "task_state": task_state,
    }
    if conversation_id and response_conversation_id != conversation_id:
        result["note"] = (
            "The previous conversation was not found; a new conversation was started."
        )
    return json.dumps(result), conversation_state


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

    async def _send_stateless(
        *,
        message: str,
        conversation_id: str | None = None,
        new_conversation: bool | None = False,
    ) -> str:
        result, _ = await _converse(
            A2aConversationInput(
                message=message,
                conversation_id=conversation_id,
                new_conversation=new_conversation,
            ),
            {"current": None, "conversations": {}},
            send,
        )
        return result

    async def _conversation_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        conversation_state = _load_conversation_state(
            state.inner_state.tools_storage.get(tool.name) or {}
        )

        result, updated_state = await _converse(
            A2aConversationInput.model_validate(call["args"]),
            conversation_state,
            send,
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
                "inner_state": {"tools_storage": {tool.name: updated_state}},
            }
        )

    tool = A2aStructuredToolWithWrapper(
        name=name,
        description=f"{description} {_CONVERSATION_HINT}",
        coroutine=_send_stateless,
        args_schema=A2aConversationInput,
        metadata=metadata,
    )
    tool.set_tool_wrappers(awrapper=_conversation_wrapper)
    return tool


def _build_agent_card(config: AgentConversationalAgentToolResourceConfig) -> AgentCard:
    """Build v1 card metadata without retaining cached transport endpoints.

    Mirrors ``a2a_tool._build_metadata_card``, which reads the cached card from
    the Remote A2A resource instead of ``properties``.
    """
    card = AgentCard()
    if config.properties.cached_agent_card:
        ParseDict(config.properties.cached_agent_card, card, ignore_unknown_fields=True)

    if not card.name:
        card.name = config.name
    if not card.description and config.description:
        card.description = config.description
    if not card.default_input_modes:
        card.default_input_modes.append("text/plain")
    if not card.default_output_modes:
        card.default_output_modes.append("text/plain")

    # Invocation always goes through the AgentHub endpoint resolved at runtime.
    card.ClearField("supported_interfaces")
    return card


def _create_conversational_agent_tool(
    config: AgentConversationalAgentToolResourceConfig,
    a2a_client: ConversationalAgentA2aClient,
    agent_card: AgentCard,
) -> BaseTool:
    """Create the LangChain tool for one conversational agent."""
    display_name = agent_card.name or config.name
    agent_label = config.properties.process_name or config.name
    metadata = {
        "tool_type": "a2a",
        "display_name": display_name,
        "resource_name": config.name,
        "process_name": config.properties.process_name or "",
    }

    async def _invoke(
        message: str, task_id: str | None, context_id: str | None
    ) -> tuple[str, str, str | None, str | None]:
        """Send one message to the agent inside an A2A trace span.

        The span is parented under the active tool-call span (via
        ``UiPathSpanUtils.get_parent_context``) so the call nests under the tool
        in the Execution Trace, and is marked ``uipath.custom_instrumentation``
        so the LLMOps exporter keeps it. The a2a-sdk's own transport spans are
        disabled in this package's __init__, so this is the single node
        representing the call. Failures, including resolving the client, are
        raised as ``_A2aRequestError``.
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

            try:
                client = await a2a_client.get()
                text, state, new_task_id, new_context_id = await _send_message(
                    client,
                    agent_label,
                    message=message,
                    task_id=task_id,
                    context_id=context_id,
                )
            except Exception as e:
                raise _A2aRequestError(str(e)) from e

            span.set_attribute("output", text)
            span.set_attribute("output.value", text)
            span.set_attribute("task_state", state)
            if new_context_id:
                span.set_attribute("conversation_id", new_context_id)
            return text, state, new_task_id, new_context_id

    return _create_conversation_tool(
        sanitize_tool_name(config.name),
        _build_description(agent_card),
        metadata,
        _invoke,
    )


def create_conversational_agent_tools_and_clients(
    resources: Sequence[AgentConversationalAgentToolResourceConfig],
    *,
    is_conversational: bool,
) -> tuple[list[BaseTool], list[ConversationalAgentA2aClient]]:
    """Create conversational agent tools and their clients from resource configurations.

    Each enabled resource gets a dedicated ``ConversationalAgentA2aClient`` (with
    its own httpx.AsyncClient). The caller is responsible for calling
    ``dispose()`` on each returned client when done.

    For automatic client lifecycle management, prefer
    ``open_conversational_agent_tools``.

    Args:
        resources: List of conversational agent tool resource configurations
            from agent.json.
        is_conversational: Whether the agent itself is conversational; gates
            ``requireConversationalConfirmation`` on the tools.

    Returns:
        Tuple of (tools, clients) where:
        - tools: BaseTool instances, one per enabled resource
        - clients: ConversationalAgentA2aClient instances that need to be
          disposed when done
    """
    tools: list[BaseTool] = []
    clients: list[ConversationalAgentA2aClient] = []

    for resource in resources:
        if resource.is_enabled is False:
            logger.info(
                "Skipping disabled conversational agent resource '%s'", resource.name
            )
            continue

        logger.info(
            "Creating conversational agent tool for resource '%s'", resource.name
        )

        agent_card = _build_agent_card(resource)
        # AgentHub serves conversational agents over both A2A 0.3 and 1.0.
        protocol_version = (
            _select_protocol_version(resource.properties.cached_agent_card) or "1.0"
        )
        a2a_client = ConversationalAgentA2aClient(
            agent_card,
            resource.name,
            resource.properties.process_name,
            protocol_version,
        )
        tool = _create_conversational_agent_tool(resource, a2a_client, agent_card)
        if (
            is_conversational
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
async def open_conversational_agent_tools(
    resources: Sequence[AgentConversationalAgentToolResourceConfig],
    *,
    is_conversational: bool,
) -> AsyncGenerator[list[BaseTool], None]:
    """Open conversational agent tools with automatic client lifecycle management.

    Wraps ``create_conversational_agent_tools_and_clients`` in an
    ``AsyncExitStack`` so each ``ConversationalAgentA2aClient`` is disposed when
    the context exits.

    Args:
        resources: List of conversational agent tool resource configurations.
        is_conversational: Whether the agent itself is conversational.

    Yields:
        List of BaseTool instances for all enabled resources.
    """
    async with AsyncExitStack() as stack:
        tools, clients = create_conversational_agent_tools_and_clients(
            resources, is_conversational=is_conversational
        )
        for client in clients:
            stack.push_async_callback(client.dispose)
        yield tools
