"""A2A singleton tool — one tool per remote agent.

Each tool maintains conversation context (task_id/context_id) across calls
using deterministic persistence via LangGraph graph state (tools_storage).

Authentication uses the UiPath SDK Bearer token, resolved lazily on first call.
Client lifecycle is managed by the caller via ``A2aClient.dispose()`` or the
``open_a2a_tools`` async context manager.
"""

import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager
from logging import getLogger
from typing import Any, AsyncGenerator
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
from google.protobuf.json_format import ParseDict
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, Field
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.agent.models.agent import AgentA2aResourceConfig
from uipath.core.tracing.span_utils import UiPathSpanUtils

from uipath_langchain._utils import get_execution_folder_path
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.base_uipath_structured_tool import (
    BaseUiPathStructuredTool,
)
from uipath_langchain.agent.tools.static_args import wrap_tools_with_static_args
from uipath_langchain.agent.tools.tool_node import (
    ToolWrapperMixin,
    ToolWrapperReturnType,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name

logger = getLogger(__name__)

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
    ) -> None:
        self._agent_card = agent_card
        self._resource_name = resource_name
        self._protocol_version = protocol_version
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
                    folder_path = get_execution_folder_path()
                    agent = await sdk.remote_a2a.retrieve_async(
                        name=self._resource_name,
                        folder_path=folder_path,
                    )
                    if not agent.a2a_url:
                        raise ValueError(
                            f"Remote A2A agent '{self._resource_name}' has no a2a_url configured"
                        )
                    runtime_card = AgentCard()
                    runtime_card.CopyFrom(self._agent_card)
                    runtime_card.ClearField("supported_interfaces")
                    runtime_card.supported_interfaces.append(
                        AgentInterface(
                            url=agent.a2a_url,
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


def _build_metadata_card(config: AgentA2aResourceConfig) -> AgentCard:
    """Build v1 card metadata without retaining cached transport endpoints."""
    card = AgentCard()
    if config.cached_agent_card:
        ParseDict(config.cached_agent_card, card, ignore_unknown_fields=True)

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

    try:
        text = ""
        state = "unknown"
        new_task_id = task_id
        new_context_id = context_id

        async for response in client.send_message(
            SendMessageRequest(message=a2a_message)
        ):
            if response.HasField("message"):
                text = _extract_text(response.message)
                new_context_id = response.message.context_id or new_context_id
                state = "completed"
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

    except Exception as e:
        logger.exception("A2A request to %s failed", agent_label)
        return (f"Error: {e}", "error", task_id, context_id)


def _create_a2a_tool(
    config: AgentA2aResourceConfig, a2a_client: A2aClient, agent_card: AgentCard
) -> BaseTool:
    """Create a single LangChain tool for A2A communication.

    Conversation context (task_id/context_id) is persisted deterministically
    in LangGraph's graph state via tools_storage, ensuring reliable
    multi-turn conversations with the remote agent.
    """
    raw_name = agent_card.name or config.name
    tool_name = sanitize_tool_name(raw_name)
    tool_description = _build_description(agent_card)
    agent_label = config.slug

    metadata = {
        "tool_type": "a2a",
        "display_name": raw_name,
        "slug": config.slug,
    }

    async def _invoke(
        *, message: str, task_id: str | None, context_id: str | None
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
        with tracer.start_as_current_span(raw_name, context=parent_ctx) as span:
            # "openinference.span.kind" drives the SpanType shown in the UI;
            # "toolCall" is the recognized type for a tool invocation.
            span.set_attribute("openinference.span.kind", "toolCall")
            span.set_attribute("type", "toolCall")
            span.set_attribute("span_type", "toolCall")
            span.set_attribute("uipath.custom_instrumentation", True)
            span.set_attribute("tool_type", "a2a")
            span.set_attribute("input", message)
            span.set_attribute("input.value", message)

            client = await a2a_client.get()
            text, response_state, new_task_id, new_context_id = await _send_a2a_message(
                client,
                agent_label,
                message=message,
                task_id=task_id,
                context_id=context_id,
            )

            span.set_attribute("output", text)
            span.set_attribute("output.value", text)
            span.set_attribute("task_state", response_state)
            if response_state == "error":
                span.set_status(otel_trace.StatusCode.ERROR, text)
            return text, response_state, new_task_id, new_context_id

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
    resources: list[AgentA2aResourceConfig],
) -> tuple[list[BaseTool], list[A2aClient]]:
    """Create A2A tools and their associated clients from resource configurations.

    Each enabled A2A resource gets a dedicated ``A2aClient`` (with its own
    httpx.AsyncClient). The caller is responsible for calling ``dispose()``
    on each returned client when done.

    For automatic client lifecycle management, prefer ``open_a2a_tools``.

    Args:
        resources: List of A2A resource configurations from agent.json.

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

        a2a_client = A2aClient(
            agent_card,
            resource.name,
            protocol_version=_select_protocol_version(resource.cached_agent_card),
        )
        tool = _create_a2a_tool(resource, a2a_client, agent_card)
        tools.append(tool)
        clients.append(a2a_client)

    return wrap_tools_with_static_args(tools), clients


@asynccontextmanager
async def open_a2a_tools(
    resources: list[AgentA2aResourceConfig],
) -> AsyncGenerator[list[BaseTool], None]:
    """Open A2A tools with automatic client lifecycle management.

    Wraps ``create_a2a_tools_and_clients`` in an ``AsyncExitStack`` so each
    ``A2aClient`` is disposed when the context exits.

    Args:
        resources: List of A2A resource configurations.

    Yields:
        List of BaseTool instances for all enabled A2A resources.
    """
    async with AsyncExitStack() as stack:
        tools, clients = create_a2a_tools_and_clients(resources)
        for client in clients:
            stack.push_async_callback(client.dispose)
        yield tools
