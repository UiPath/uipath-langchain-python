"""A2A singleton tool — one tool per remote agent.

Each tool maintains conversation context (task_id/context_id) across calls
using deterministic persistence via LangGraph graph state (tools_storage).

Authentication uses the UiPath SDK Bearer token, resolved lazily on first call.
"""

from __future__ import annotations

import asyncio
import json
from logging import getLogger
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
    return ""


async def create_a2a_agent_tools(
    resources: list[AgentA2aResourceConfig],
) -> list[BaseTool]:
    """Create A2A tools from a list of A2A resource configurations.

    Each enabled A2A resource becomes a single tool representing the remote agent.
    Conversation context (task_id/context_id) is persisted in LangGraph graph state.

    Args:
        resources: List of A2A resource configurations from agent.json.

    Returns:
        List of BaseTool instances, one per enabled A2A resource.
    """
    tools: list[BaseTool] = []

    for resource in resources:
        if resource.is_enabled is False:
            logger.info("Skipping disabled A2A resource '%s'", resource.name)
            continue

        logger.info("Creating A2A tool for resource '%s'", resource.name)
        tool = _create_a2a_tool(resource)
        tools.append(tool)

    return tools


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


def _create_a2a_tool(config: AgentA2aResourceConfig) -> BaseTool:
    """Create a single LangChain tool for A2A communication.

    Conversation context (task_id/context_id) is persisted deterministically
    in LangGraph's graph state via tools_storage, ensuring reliable
    multi-turn conversations with the remote agent.
    """
    if config.cached_agent_card:
        agent_card = AgentCard(**config.cached_agent_card)
    else:
        agent_card = AgentCard(
            url="",
            name=config.name,
            description=config.description or "",
            version="1.0.0",
            skills=[],
            capabilities={},
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
        )

    raw_name = agent_card.name or config.name
    tool_name = sanitize_tool_name(raw_name)
    tool_description = _build_description(agent_card)
    a2a_url = _resolve_a2a_url(config)

    _lock = asyncio.Lock()
    _client: Client | None = None
    _http_client: httpx.AsyncClient | None = None

    async def _ensure_client() -> Client:
        nonlocal _client, _http_client
        if _client is None:
            async with _lock:
                if _client is None:
                    from a2a.client import ClientConfig, ClientFactory
                    from uipath.platform import UiPath

                    sdk = UiPath()
                    client_kwargs = get_httpx_client_kwargs(
                        headers={"Authorization": f"Bearer {sdk._config.secret}"},
                    )
                    _http_client = httpx.AsyncClient(**client_kwargs)
                    _client = await ClientFactory.connect(
                        a2a_url,
                        client_config=ClientConfig(
                            httpx_client=_http_client,
                            streaming=False,
                        ),
                    )
        return _client

    metadata = {
        "tool_type": "a2a",
        "display_name": raw_name,
        "slug": config.slug,
    }

    async def _send(*, message: str) -> str:
        client = await _ensure_client()
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

        client = await _ensure_client()
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
