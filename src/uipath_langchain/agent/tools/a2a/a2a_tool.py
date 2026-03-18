"""A2A singleton tool — one tool per remote agent with closure-managed context.

Each tool maintains its own task_id/context_id in a closure for conversation
continuity. Authentication uses the UiPath SDK Bearer token, resolved lazily
on first call.
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
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from uipath.agent.models.agent import AgentA2aResourceConfig

from uipath_langchain.agent.tools.base_uipath_structured_tool import (
    BaseUiPathStructuredTool,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name

logger = getLogger(__name__)


class A2aToolInput(BaseModel):
    """Input schema for A2A agent tool."""

    message: str = Field(description="The message to send to the remote agent.")


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


def fetch_agent_card(
    agent_card_url: str, headers: dict[str, str] | None = None
) -> AgentCard:
    """Fetch the agent card from a remote A2A endpoint (synchronous)."""
    response = httpx.get(agent_card_url, headers=headers or {}, timeout=30)
    response.raise_for_status()
    return AgentCard(**response.json())


def _resolve_a2a_url(config: AgentA2aResourceConfig) -> str:
    """Resolve the A2A endpoint URL from config.

    Uses a2aUrl if available, otherwise derives from agentCardUrl
    by stripping the .well-known path.
    """
    a2a_url = getattr(config, "a2a_url", None)
    if a2a_url:
        return a2a_url
    return config.agent_card_url.replace("/.well-known/agent-card.json", "")


async def create_a2a_agent_tools(
    resources: list[AgentA2aResourceConfig],
) -> list[BaseTool]:
    """Create A2A tools from a list of A2A resource configurations.

    Each enabled A2A resource becomes a single tool representing the remote agent.

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
        if resource.is_active is False:
            logger.info("Skipping inactive A2A resource '%s'", resource.name)
            continue

        logger.info("Creating A2A tool for resource '%s'", resource.name)
        tool = _create_a2a_singleton_tool(resource)
        tools.append(tool)

    return tools


def _create_a2a_singleton_tool(config: AgentA2aResourceConfig) -> BaseTool:
    """Create a single LangChain tool for A2A communication.

    All connection and conversation state is managed in a closure.
    The httpx client and A2A protocol client are created lazily on first call.

    Args:
        config: A2A resource configuration from agent.json.

    Returns:
        A BaseTool that sends messages to the remote A2A agent.
    """
    if config.cached_agent_card:
        agent_card = AgentCard(**config.cached_agent_card)
    else:
        agent_card = fetch_agent_card(config.agent_card_url)

    raw_name = agent_card.name or config.name
    tool_name = sanitize_tool_name(raw_name)
    tool_description = _build_description(agent_card)
    a2a_url = _resolve_a2a_url(config)

    _lock = asyncio.Lock()
    _client: Client | None = None
    _http_client: httpx.AsyncClient | None = None
    _task_id: str | None = None
    _context_id: str | None = None

    async def _ensure_client() -> Client:
        nonlocal _client, _http_client
        if _client is None:
            async with _lock:
                if _client is None:
                    from a2a.client import ClientConfig, ClientFactory
                    from uipath.platform import UiPath

                    sdk = UiPath()
                    _http_client = httpx.AsyncClient(
                        timeout=120,
                        headers={"Authorization": f"Bearer {sdk._config.secret}"},
                    )
                    _client = await ClientFactory.connect(
                        a2a_url,
                        client_config=ClientConfig(
                            httpx_client=_http_client,
                            streaming=False,
                        ),
                    )
        return _client  # type: ignore[return-value]

    async def _send(*, message: str) -> str:
        nonlocal _task_id, _context_id
        client = await _ensure_client()

        if _task_id or _context_id:
            logger.info(
                "A2A continue task=%s context=%s to %s",
                _task_id,
                _context_id,
                a2a_url,
            )
        else:
            logger.info("A2A new message to %s", a2a_url)

        a2a_message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=message))],
            message_id=str(uuid4()),
            task_id=_task_id,
            context_id=_context_id,
        )

        try:
            text = ""
            state = "unknown"

            async for event in client.send_message(a2a_message):
                if isinstance(event, Message):
                    text = _extract_text(event)
                    _context_id = event.context_id
                    state = "completed"
                    break
                else:
                    task, update = event
                    _task_id = task.id
                    _context_id = task.context_id
                    state = task.status.state.value if task.status else "unknown"
                    if update is None:
                        text = _extract_text(task)
                        break
                    elif isinstance(update, TaskArtifactUpdateEvent):
                        for part in update.artifact.parts or []:
                            if isinstance(part.root, TextPart):
                                text += part.root.text

            return _format_response(text or "No response received.", state)

        except Exception as e:
            logger.exception("A2A request to %s failed", a2a_url)
            return _format_response(f"Error: {e}", "error")

    return BaseUiPathStructuredTool(
        name=tool_name,
        description=tool_description,
        coroutine=_send,
        args_schema=A2aToolInput,
        metadata={
            "tool_type": "a2a",
            "display_name": raw_name,
            "slug": config.slug,
        },
    )
