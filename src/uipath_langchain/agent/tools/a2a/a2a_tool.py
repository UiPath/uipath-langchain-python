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
from typing import AsyncGenerator
from uuid import uuid4

import httpx
from a2a.client import Client
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TextPart,
)
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel, Field
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.agent.models.agent import AgentA2aResourceConfig

from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.base_uipath_structured_tool import (
    BaseUiPathStructuredTool,
)
from uipath_langchain.agent.tools.tool_node import (
    ToolWrapperMixin,
    ToolWrapperReturnType,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name

logger = getLogger(__name__)


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

    def __init__(self, agent_card: AgentCard) -> None:
        self._agent_card = agent_card
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

                    sdk = UiPath()
                    client_kwargs = get_httpx_client_kwargs(
                        headers={"Authorization": f"Bearer {sdk._config.secret}"},
                    )
                    client_kwargs["timeout"] = httpx.Timeout(300.0, connect=10.0)
                    self._http_client = httpx.AsyncClient(**client_kwargs)
                    self._client = await ClientFactory.connect(
                        self._agent_card,
                        client_config=ClientConfig(
                            httpx_client=self._http_client,
                            streaming=False,
                        ),
                    )
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
    parts: list[Part] = []

    if isinstance(obj, Message):
        parts = obj.parts or []
    elif isinstance(obj, Task):
        if obj.status and obj.status.state == TaskState.input_required:
            if obj.status.message:
                parts = obj.status.message.parts or []
        else:
            if obj.artifacts:
                for artifact in obj.artifacts:
                    parts.extend(artifact.parts or [])
            if not parts and obj.status and obj.status.message:
                parts = obj.status.message.parts or []
            if not parts and obj.history:
                for msg in reversed(obj.history):
                    if msg.role == Role.agent:
                        parts = msg.parts or []
                        break

    texts = []
    for part in parts:
        if isinstance(part.root, TextPart):
            texts.append(part.root.text)
    return "\n".join(texts) if texts else ""


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
    return " | ".join(parts) if parts else f"Remote A2A agent at {card.url}"


def _resolve_a2a_url(config: AgentA2aResourceConfig) -> str:
    """Resolve the A2A endpoint URL from the cached agent card."""
    if config.cached_agent_card and "url" in config.cached_agent_card:
        return config.cached_agent_card["url"]
    raise ValueError(f"A2A resource '{config.name}' has no URL in cachedAgentCard")


async def _send_a2a_message(
    client: Client,
    a2a_url: str,
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
            "A2A continue task=%s context=%s to %s", task_id, context_id, a2a_url
        )
    else:
        logger.info("A2A new message to %s", a2a_url)

    a2a_message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=message))],
        message_id=str(uuid4()),
        task_id=task_id,
        context_id=context_id,
    )

    try:
        text = ""
        state = "unknown"
        new_task_id = task_id
        new_context_id = context_id

        async for event in client.send_message(a2a_message):
            if isinstance(event, Message):
                text = _extract_text(event)
                new_context_id = event.context_id
                state = "completed"
                break
            else:
                task, update = event
                new_task_id = task.id
                new_context_id = task.context_id
                state = task.status.state.value if task.status else "unknown"
                if update is None:
                    text = _extract_text(task)
                    break
                elif isinstance(update, TaskArtifactUpdateEvent):
                    for part in update.artifact.parts or []:
                        if isinstance(part.root, TextPart):
                            text += part.root.text

        return (text or "No response received.", state, new_task_id, new_context_id)

    except Exception as e:
        logger.exception("A2A request to %s failed", a2a_url)
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
    a2a_url = _resolve_a2a_url(config)

    metadata = {
        "tool_type": "a2a",
        "display_name": raw_name,
        "slug": config.slug,
    }

    async def _send(*, message: str) -> str:
        client = await a2a_client.get()
        text, state, _, _ = await _send_a2a_message(
            client, a2a_url, message=message, task_id=None, context_id=None
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

        client = await a2a_client.get()
        text, task_state, new_task_id, new_context_id = await _send_a2a_message(
            client,
            a2a_url,
            message=call["args"]["message"],
            task_id=task_id,
            context_id=context_id,
        )

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

        if resource.cached_agent_card:
            agent_card = AgentCard(**resource.cached_agent_card)
        else:
            agent_card = AgentCard(
                url="",
                name=resource.name,
                description=resource.description or "",
                version="1.0.0",
                skills=[],
                capabilities={},
                default_input_modes=["text/plain"],
                default_output_modes=["text/plain"],
            )

        a2a_client = A2aClient(agent_card)
        tool = _create_a2a_tool(resource, a2a_client, agent_card)
        tools.append(tool)
        clients.append(a2a_client)

    return tools, clients


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
