"""Tests for A2A tool creation and URL resolution.

The proxy URL is resolved lazily at runtime by retrieving the remote agent
through the binding-aware SDK (``remote_a2a.retrieve_async``), mirroring how
MCP servers are resolved — not from a field baked into the resource config.
"""

import asyncio
import gc
import json
import os
import weakref
from types import SimpleNamespace
from typing import Any, cast

import pytest
from a2a.client import Client
from a2a.types import (
    AgentCard,
    Artifact,
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.errors import InvalidRequestError
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from uipath.agent.models.agent import (
    AgentA2aResourceConfig,
    AgentConversationalAgentToolProperties,
    AgentConversationalAgentToolResourceConfig,
    AgentToolType,
)
from uipath.platform.common import GenericResourceOverwrite, ResourceOverwritesContext
from uipath.platform.errors import FolderNotFoundException

import uipath_langchain.agent.tools.a2a.a2a_tool as a2a_tool
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.a2a.a2a_tool import (
    A2aClient,
    A2aConversationInput,
    A2aStructuredToolWithWrapper,
    A2aToolInput,
    _build_description,
    _send_a2a_message,
    create_a2a_tools_and_clients,
)
from uipath_langchain.agent.tools.tool_node import (
    create_tool_node,
    wrap_tools_with_error_handling,
)
from uipath_langchain.chat.hitl import REQUIRE_CONVERSATIONAL_CONFIRMATION

PROXY_URL = (
    "https://cloud.uipath.com/org/tenant/agenthub_/a2a/remote/folder/remote-agent-slug"
)
CACHED_URL = "https://internal.example.com/agents/remote-agent"


def _make_resource(
    *,
    name: str = "remote-agent",
    cached_agent_card: dict[str, Any] | None = None,
    is_enabled: bool = True,
) -> AgentA2aResourceConfig:
    """Build an A2A resource config for tests."""
    return AgentA2aResourceConfig(
        id="resource-id",
        name=name,
        description="A remote A2A agent",
        is_enabled=is_enabled,
        slug="remote-agent-slug",
        folder_path="Shared",
        cached_agent_card=cached_agent_card,
    )


def _cached_card() -> dict[str, Any]:
    return {
        "url": CACHED_URL,
        "name": "Remote Agent",
        "description": "cached",
        "version": "1.0.0",
        "skills": [],
        "capabilities": {},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
    }


def test_create_tools_builds_one_client_per_enabled_resource() -> None:
    resource = _make_resource(cached_agent_card=_cached_card())

    tools, clients = create_a2a_tools_and_clients([resource])

    assert len(tools) == 1
    assert len(clients) == 1
    assert clients[0]._resource_name == "remote-agent"
    # The card is built from the cached card; the URL is not yet resolved.
    assert clients[0]._agent_card.name == "Remote Agent"


def test_create_tools_builds_default_card_without_cached_card() -> None:
    resource = _make_resource(cached_agent_card=None)

    tools, clients = create_a2a_tools_and_clients([resource])

    assert len(tools) == 1
    assert clients[0]._resource_name == "remote-agent"
    assert clients[0]._agent_card.name == "remote-agent"


def test_tool_is_named_after_the_resource_and_titled_by_the_card() -> None:
    resource = _make_resource(
        name="Research Briefing", cached_agent_card=_cached_card()
    )

    tools, _ = create_a2a_tools_and_clients([resource])

    assert tools[0].name == "Research_Briefing"
    assert tools[0].metadata == {
        "tool_type": "a2a",
        "display_name": "Remote Agent",
        "slug": "remote-agent-slug",
        "resource_name": "Research Briefing",
    }


def test_resources_sharing_a_card_get_distinct_tools() -> None:
    resources = [
        _make_resource(name="Briefing EU", cached_agent_card=_cached_card()),
        _make_resource(name="Briefing US", cached_agent_card=_cached_card()),
    ]

    tools, _ = create_a2a_tools_and_clients(resources)

    assert [t.name for t in tools] == ["Briefing_EU", "Briefing_US"]
    assert set(create_tool_node(tools)) == {"Briefing_EU", "Briefing_US"}


def test_create_tools_skips_disabled_resource() -> None:
    resource = _make_resource(is_enabled=False)

    tools, clients = create_a2a_tools_and_clients([resource])

    assert tools == []
    assert clients == []


class _FakeRemoteA2aService:
    def __init__(self, a2a_url: str | None) -> None:
        self._a2a_url = a2a_url
        self.calls: list[tuple[str, str | None]] = []

    async def retrieve_async(
        self,
        *,
        name: str,
        folder_path: str | None,
    ):
        self.calls.append((name, folder_path))
        return SimpleNamespace(a2a_url=self._a2a_url)


class _FakeSdk:
    def __init__(self, a2a_url: str | None) -> None:
        self.remote_a2a = _FakeRemoteA2aService(a2a_url)
        self._config = SimpleNamespace(secret="token")


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, sdk: Any) -> dict[str, Any]:
    """Patch the SDK, folder-path resolver, and A2A client factory."""
    import uipath.platform as uipath_platform
    from a2a.client import ClientFactory

    monkeypatch.setattr(uipath_platform, "UiPath", lambda: sdk)
    monkeypatch.setattr(a2a_tool, "get_execution_folder_path", lambda: "Shared")

    connected = SimpleNamespace(value="connected")

    captured: dict[str, AgentCard] = {}

    def _fake_create(self: ClientFactory, agent_card: AgentCard):
        captured["card"] = agent_card
        return connected

    monkeypatch.setattr(ClientFactory, "create", _fake_create)
    return {"connected": connected, "captured": captured}


async def test_client_resolves_proxy_url_via_retrieve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk(a2a_url=PROXY_URL)
    handles = _patch_runtime(monkeypatch, sdk)

    card = AgentCard(
        name="Remote Agent",
        description="cached",
        version="1.0.0",
        skills=[],
        capabilities={},
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )
    client = A2aClient(card, resource_name="Remote Agent")

    result = await client.get()

    assert result is handles["connected"]
    # URL resolved from the retrieved agent's a2a_url, not the cached card URL.
    runtime_card = handles["captured"]["card"]
    assert runtime_card.supported_interfaces[0].url == PROXY_URL
    assert runtime_card.supported_interfaces[0].protocol_binding == "JSONRPC"
    assert runtime_card.supported_interfaces[0].protocol_version == "1.0"
    assert sdk.remote_a2a.calls == [("Remote Agent", "Shared")]

    await client.dispose()


async def test_client_raises_when_agent_has_no_proxy_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk(a2a_url=None)
    _patch_runtime(monkeypatch, sdk)

    card = AgentCard(
        name="Remote Agent",
        description="",
        version="1.0.0",
        skills=[],
        capabilities={},
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )
    client = A2aClient(card, resource_name="Remote Agent")

    with pytest.raises(ValueError, match="has no a2a_url"):
        await client.get()


def _agent_message(text: str, context_id: str | None = None) -> StreamResponse:
    return StreamResponse(
        message=Message(
            role=Role.ROLE_AGENT,
            parts=[Part(text=text)],
            message_id="msg-1",
            context_id=context_id or "",
        )
    )


class _FakeA2aClient:
    """Minimal stand-in for the a2a Client whose send_message yields events."""

    def __init__(self, events: list[StreamResponse]) -> None:
        self._events = events
        self.sent: list[SendMessageRequest] = []

    async def send_message(self, request: SendMessageRequest):
        self.sent.append(request)
        for event in self._events:
            yield event


class _RaisingA2aClient:
    async def send_message(self, request: SendMessageRequest):
        raise RuntimeError("boom")
        yield  # pragma: no cover - marks this as an async generator


def test_build_description_falls_back_to_name() -> None:
    card = AgentCard(
        name="Finance Agent",
        description="",
        version="1.0.0",
        skills=[],
        capabilities={},
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )
    assert _build_description(card) == "Remote A2A agent: Finance Agent"


def test_build_description_generic_when_no_name() -> None:
    card = AgentCard(
        name="",
        description="",
        version="1.0.0",
        skills=[],
        capabilities={},
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )
    assert _build_description(card) == "Remote A2A agent"


async def test_send_a2a_message_returns_text() -> None:
    client = _FakeA2aClient([_agent_message("pong", context_id="ctx-1")])
    text, state, task_id, context_id = await _send_a2a_message(
        cast(Client, client),
        "finance-agent",
        message="ping",
        task_id=None,
        context_id=None,
    )
    assert text == "pong"
    assert state == "completed"
    assert context_id == "ctx-1"


async def test_send_a2a_message_continues_existing_task() -> None:
    client = _FakeA2aClient([_agent_message("again", context_id="ctx-2")])
    text, _, _, context_id = await _send_a2a_message(
        cast(Client, client),
        "finance-agent",
        message="more",
        task_id="task-1",
        context_id="ctx-2",
    )
    assert text == "again"
    assert context_id == "ctx-2"
    # The continued message carries the prior task/context ids.
    assert client.sent[0].message.task_id == "task-1"


async def test_send_a2a_message_handles_error() -> None:
    text, state, _, _ = await _send_a2a_message(
        cast(Client, _RaisingA2aClient()),
        "finance-agent",
        message="ping",
        task_id=None,
        context_id=None,
    )
    assert state == "error"
    assert "boom" in text


async def test_tool_send_coroutine_sends_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = _make_resource(cached_agent_card=_cached_card())
    tools, clients = create_a2a_tools_and_clients([resource])
    fake = _FakeA2aClient([_agent_message("pong", context_id="ctx-1")])

    async def _get():
        return fake

    monkeypatch.setattr(clients[0], "get", _get)

    result = await tools[0].ainvoke({"message": "ping"})
    assert "pong" in result


async def test_tool_wrapper_persists_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = _make_resource(cached_agent_card=_cached_card())
    tools, clients = create_a2a_tools_and_clients([resource])
    tool = cast(A2aStructuredToolWithWrapper, tools[0])
    fake = _FakeA2aClient([_agent_message("pong", context_id="ctx-9")])

    async def _get():
        return fake

    monkeypatch.setattr(clients[0], "get", _get)
    wrapper: Any = tool.awrapper
    assert wrapper is not None

    call = {"name": tool.name, "args": {"message": "ping"}, "id": "call-1"}
    command = await wrapper(tool, call, AgentGraphState())

    stored = command.update["inner_state"]["tools_storage"][tool.name]
    assert stored["context_id"] == "ctx-9"
    assert "pong" in command.update["messages"][0].content


def _completed_task(
    *, task_id: str = "task-1", context_id: str = "ctx-1", text: str = "done"
) -> StreamResponse:
    return StreamResponse(
        task=Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            artifacts=[
                Artifact(
                    artifact_id="artifact-1",
                    parts=[Part(text=text)],
                )
            ],
        )
    )


def _input_required_task(
    *, task_id: str = "task-1", context_id: str = "ctx-1", text: str = "need more"
) -> StreamResponse:
    return StreamResponse(
        task=Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(
                state=TaskState.TASK_STATE_INPUT_REQUIRED,
                message=Message(
                    role=Role.ROLE_AGENT,
                    parts=[Part(text=text)],
                    message_id="status-msg",
                ),
            ),
        )
    )


async def test_tool_wrapper_drops_task_id_after_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed (terminal) task is not reused: the next turn starts a new
    task while keeping the conversation context."""
    resource = _make_resource(cached_agent_card=_cached_card())
    tools, clients = create_a2a_tools_and_clients([resource])
    tool = cast(A2aStructuredToolWithWrapper, tools[0])
    fake = _FakeA2aClient([_completed_task(task_id="task-1", context_id="ctx-1")])

    async def _get():
        return fake

    monkeypatch.setattr(clients[0], "get", _get)
    wrapper: Any = tool.awrapper
    assert wrapper is not None

    call = {"name": tool.name, "args": {"message": "ping"}, "id": "call-1"}
    command = await wrapper(tool, call, AgentGraphState())

    stored = command.update["inner_state"]["tools_storage"][tool.name]
    assert stored["task_id"] is None
    assert stored["context_id"] == "ctx-1"


async def test_tool_wrapper_keeps_task_id_when_not_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-terminal task (input-required) keeps its task_id so the next turn
    continues the same task."""
    resource = _make_resource(cached_agent_card=_cached_card())
    tools, clients = create_a2a_tools_and_clients([resource])
    tool = cast(A2aStructuredToolWithWrapper, tools[0])
    fake = _FakeA2aClient([_input_required_task(task_id="task-1", context_id="ctx-1")])

    async def _get():
        return fake

    monkeypatch.setattr(clients[0], "get", _get)
    wrapper: Any = tool.awrapper
    assert wrapper is not None

    call = {"name": tool.name, "args": {"message": "ping"}, "id": "call-1"}
    command = await wrapper(tool, call, AgentGraphState())

    stored = command.update["inner_state"]["tools_storage"][tool.name]
    assert stored["task_id"] == "task-1"
    assert stored["context_id"] == "ctx-1"


def test_a2a_sdk_telemetry_suppressed_by_default() -> None:
    """Importing the a2a package disables the a2a-sdk's own OTel transport spans.

    The package __init__ runs (via the module imports above) before the a2a-sdk
    is imported and sets the suppression default.
    """
    assert os.environ.get("OTEL_INSTRUMENTATION_A2A_SDK_ENABLED") == "false"


async def test_tool_invocation_emits_custom_a2a_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoking the tool emits one custom-instrumentation A2A span carrying the
    toolCall kind and the message/response."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, "get_tracer", provider.get_tracer)

    resource = _make_resource(cached_agent_card=_cached_card())
    tools, clients = create_a2a_tools_and_clients([resource])
    fake = _FakeA2aClient([_agent_message("pong", context_id="ctx-1")])

    async def _get():
        return fake

    monkeypatch.setattr(clients[0], "get", _get)

    result = await tools[0].ainvoke({"message": "ping"})
    assert "pong" in result

    a2a_spans = [s for s in exporter.get_finished_spans() if s.name == "Remote Agent"]
    assert len(a2a_spans) == 1
    attrs = a2a_spans[0].attributes or {}
    assert attrs["uipath.custom_instrumentation"] is True
    assert attrs["openinference.span.kind"] == "toolCall"
    assert attrs["tool_type"] == "a2a"
    assert attrs["input.value"] == "ping"
    assert attrs["output.value"] == "pong"


async def test_tool_invocation_marks_span_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed remote call sets the A2A span status to ERROR."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, "get_tracer", provider.get_tracer)

    resource = _make_resource(cached_agent_card=_cached_card())
    tools, clients = create_a2a_tools_and_clients([resource])

    async def _get():
        return _RaisingA2aClient()

    monkeypatch.setattr(clients[0], "get", _get)

    await tools[0].ainvoke({"message": "ping"})

    a2a_spans = [s for s in exporter.get_finished_spans() if s.name == "Remote Agent"]
    assert len(a2a_spans) == 1
    assert a2a_spans[0].status.status_code == StatusCode.ERROR


# --- Conversational agents -------------------------------------------------

CONVERSATIONAL_BASE_URL = "https://cloud.uipath.com/org/tenant"
FOLDER_KEY = "11111111-2222-3333-4444-555555555555"


def _make_conversational_resource(
    *,
    folder_path: str | None = "Shared/Agents",
    cached_agent_card: dict[str, Any] | None = None,
    is_enabled: bool = True,
    process_name: str | None = "My Conversational Agent",
    require_confirmation: bool = False,
) -> AgentConversationalAgentToolResourceConfig:
    return AgentConversationalAgentToolResourceConfig(
        id="conv-id",
        name="Support Agent",
        description="Answers support questions",
        is_enabled=is_enabled,
        type=AgentToolType.CONVERSATIONAL_AGENT,
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        properties=AgentConversationalAgentToolProperties(
            process_name=process_name,
            folder_path=folder_path,
            cached_agent_card=cached_agent_card,
            require_conversational_confirmation=require_confirmation,
        ),
    )


def _release(
    release_id: int,
    *,
    name: str = "My Conversational Agent",
    process_type: str = "Agent",
    is_conversational: bool | None = True,
) -> dict[str, Any]:
    return {
        "Id": release_id,
        "Name": name,
        "ProcessType": process_type,
        "IsConversational": is_conversational,
    }


class _FakeFolders:
    """Mirrors FolderService: raises FolderNotFoundException for unknown paths."""

    def __init__(self, folder_keys: dict[str, str]) -> None:
        self._folder_keys = folder_keys
        self.calls: list[str | None] = []

    async def retrieve_folder_key_async(self, folder_path: str | None) -> str | None:
        self.calls.append(folder_path)
        if folder_path not in self._folder_keys:
            raise FolderNotFoundException(folder_path)
        return self._folder_keys[folder_path]


class _FakeApiClient:
    def __init__(self, releases: list[dict[str, Any]]) -> None:
        self._releases = releases
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def request_async(self, method: str, url: str, **kwargs: Any):
        self.calls.append((method, url, kwargs))
        return SimpleNamespace(json=lambda: {"value": self._releases})


class _FakeConversationalSdk:
    def __init__(
        self,
        releases: list[dict[str, Any]],
        folder_keys: dict[str, str] | None = None,
    ) -> None:
        self.folders = _FakeFolders(
            {"Shared/Agents": FOLDER_KEY, "Shared": FOLDER_KEY}
            if folder_keys is None
            else folder_keys
        )
        self.api_client = _FakeApiClient(releases)
        self.remote_a2a = _FakeRemoteA2aService(a2a_url=None)
        self._config = SimpleNamespace(
            secret="token", base_url=CONVERSATIONAL_BASE_URL + "/"
        )


def test_conversational_config_creates_tool_from_resource_labels() -> None:
    resource = _make_conversational_resource()

    tools, clients = create_a2a_tools_and_clients([resource])

    assert len(tools) == 1
    assert len(clients) == 1
    assert tools[0].name == "Support_Agent"
    assert tools[0].description.startswith("Answers support questions")
    assert tools[0].metadata == {
        "tool_type": "a2a",
        "display_name": "Support Agent",
        "resource_name": "Support Agent",
        "process_name": "My Conversational Agent",
    }
    assert not hasattr(resource, "slug")


def test_conversational_tool_labels_prefer_cached_card() -> None:
    resource = _make_conversational_resource(
        cached_agent_card={**_cached_card(), "description": "Cached description"}
    )

    tools, _ = create_a2a_tools_and_clients([resource])

    assert tools[0].name == "Support_Agent"
    assert tools[0].description.startswith("Cached description")
    assert tools[0].metadata is not None
    assert tools[0].metadata["display_name"] == "Remote Agent"


def test_conversational_disabled_resource_is_skipped() -> None:
    tools, clients = create_a2a_tools_and_clients(
        [_make_conversational_resource(is_enabled=False)]
    )

    assert tools == []
    assert clients == []


async def test_conversational_client_resolves_release_on_first_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk([_release(4242)])
    handles = _patch_runtime(monkeypatch, sdk)
    _, clients = create_a2a_tools_and_clients([_make_conversational_resource()])

    # Nothing is resolved while the graph is being built.
    assert sdk.api_client.calls == []

    result = await clients[0].get()

    assert result is handles["connected"]
    interface = handles["captured"]["card"].supported_interfaces[0]
    assert interface.url == f"{CONVERSATIONAL_BASE_URL}/agenthub_/a2a/{FOLDER_KEY}/4242"
    assert interface.protocol_binding == "JSONRPC"
    assert interface.protocol_version == "1.0"
    assert sdk.folders.calls == ["Shared/Agents"]
    ((method, url, kwargs),) = sdk.api_client.calls
    assert (method, url) == ("GET", "/orchestrator_/odata/Releases")
    assert kwargs["params"]["$filter"] == "Name eq 'My Conversational Agent'"
    assert kwargs["params"]["$select"] == "Id,Name,ProcessType,IsConversational"
    assert kwargs["headers"] == {"x-uipath-folderkey": FOLDER_KEY}
    # The remote agent lookup is never used for conversational agents.
    assert sdk.remote_a2a.calls == []

    await clients[0].get()
    assert len(sdk.api_client.calls) == 1

    await clients[0].dispose()


async def test_conversational_client_falls_back_to_execution_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk([_release(7)])
    _patch_runtime(monkeypatch, sdk)
    _, clients = create_a2a_tools_and_clients(
        [_make_conversational_resource(folder_path=None)]
    )

    await clients[0].get()

    assert sdk.folders.calls == ["Shared"]
    await clients[0].dispose()


async def test_conversational_client_honors_solution_resource_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk(
        [_release(9, name="Deployed Agent")],
        {"Deployed/Folder": FOLDER_KEY},
    )
    _patch_runtime(monkeypatch, sdk)
    _, clients = create_a2a_tools_and_clients([_make_conversational_resource()])

    async def _overwrites():
        return {
            "process.My Conversational Agent": GenericResourceOverwrite(
                resource_type="process",
                name="Deployed Agent",
                folder_path="Deployed/Folder",
            )
        }

    async with ResourceOverwritesContext(_overwrites):
        await clients[0].get()

    assert sdk.folders.calls == ["Deployed/Folder"]
    assert sdk.api_client.calls[0][2]["params"]["$filter"] == "Name eq 'Deployed Agent'"
    await clients[0].dispose()


async def test_conversational_missing_release_is_a_request_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk([])
    _patch_runtime(monkeypatch, sdk)
    tools, _ = create_a2a_tools_and_clients([_make_conversational_resource()])
    tool = cast(A2aStructuredToolWithWrapper, tools[0])
    wrapper: Any = tool.awrapper

    result, _ = await _converse(tool, wrapper, None)

    assert result["error"] == "request_failed"
    assert result["conversation_id"] is None
    assert "My Conversational Agent" in result["message"]
    assert "Shared/Agents" in result["message"]
    assert "is not a published conversational agent" in result["message"]


async def test_conversational_configured_folder_not_found_falls_back_to_execution_folder(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sdk = _FakeConversationalSdk([_release(5)], {"Shared": FOLDER_KEY})
    _patch_runtime(monkeypatch, sdk)
    _, clients = create_a2a_tools_and_clients([_make_conversational_resource()])

    with caplog.at_level("WARNING"):
        await clients[0].get()

    assert sdk.folders.calls == ["Shared/Agents", "Shared"]
    assert "Shared/Agents" in caplog.text
    assert "Shared" in caplog.text
    assert "My Conversational Agent" in caplog.text
    await clients[0].dispose()


async def test_conversational_no_folder_resolves_is_a_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk([_release(5)], {})
    _patch_runtime(monkeypatch, sdk)
    _, clients = create_a2a_tools_and_clients([_make_conversational_resource()])

    with pytest.raises(ValueError, match="My Conversational Agent.*Shared/Agents"):
        await clients[0].get()
    assert sdk.api_client.calls == []


async def test_conversational_no_folder_configured_or_executing_is_a_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk([_release(5)])
    _patch_runtime(monkeypatch, sdk)
    monkeypatch.setattr(a2a_tool, "get_execution_folder_path", lambda: None)
    _, clients = create_a2a_tools_and_clients(
        [_make_conversational_resource(folder_path=None)]
    )

    with pytest.raises(ValueError, match="My Conversational Agent.*no folder"):
        await clients[0].get()
    assert sdk.folders.calls == []


async def test_conversational_non_conversational_release_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk(
        [
            _release(1, process_type="Process", is_conversational=None),
            _release(2, is_conversational=False),
        ]
    )
    _patch_runtime(monkeypatch, sdk)
    _, clients = create_a2a_tools_and_clients([_make_conversational_resource()])

    with pytest.raises(ValueError, match="not a published conversational agent"):
        await clients[0].get()


async def test_conversational_release_is_picked_among_same_named_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk(
        [
            _release(1, process_type="Process", is_conversational=None),
            _release(2, is_conversational=False),
            _release(3),
        ]
    )
    handles = _patch_runtime(monkeypatch, sdk)
    _, clients = create_a2a_tools_and_clients([_make_conversational_resource()])

    await clients[0].get()

    interface = handles["captured"]["card"].supported_interfaces[0]
    assert interface.url.endswith(f"/agenthub_/a2a/{FOLDER_KEY}/3")
    await clients[0].dispose()


async def test_conversational_process_name_quote_is_escaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk([_release(6, name="Bob's Agent")])
    _patch_runtime(monkeypatch, sdk)
    resource = _make_conversational_resource()
    resource.properties.process_name = "Bob's Agent"
    _, clients = create_a2a_tools_and_clients([resource])

    await clients[0].get()

    assert sdk.api_client.calls[0][2]["params"]["$filter"] == "Name eq 'Bob''s Agent'"
    await clients[0].dispose()


async def test_conversational_client_honors_agenthub_service_url_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk([_release(8)])
    handles = _patch_runtime(monkeypatch, sdk)
    monkeypatch.setenv("UIPATH_SERVICE_URL_AGENTHUB", "http://localhost:5100/")
    _, clients = create_a2a_tools_and_clients([_make_conversational_resource()])

    await clients[0].get()

    interface = handles["captured"]["card"].supported_interfaces[0]
    assert interface.url == f"http://localhost:5100/a2a/{FOLDER_KEY}/8"
    await clients[0].dispose()


async def test_conversational_cached_v03_card_selects_protocol_0_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk([_release(4)])
    handles = _patch_runtime(monkeypatch, sdk)
    _, clients = create_a2a_tools_and_clients(
        [_make_conversational_resource(cached_agent_card=_cached_card())]
    )

    await clients[0].get()

    interface = handles["captured"]["card"].supported_interfaces[0]
    assert interface.protocol_version == "0.3"
    await clients[0].dispose()


async def _run_through_tool_node(
    tools: list[Any], args: dict[str, Any]
) -> dict[str, Any]:
    tool_node = wrap_tools_with_error_handling(create_tool_node(tools))[tools[0].name]
    state = AgentGraphState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[{"name": tools[0].name, "args": args, "id": "call-1"}],
            )
        ]
    )
    result = await tool_node.ainvoke(state)
    update = cast(
        dict[str, Any], result.update if isinstance(result, Command) else result
    )
    message = update["messages"][0]
    assert isinstance(message, ToolMessage)
    return cast(dict[str, Any], json.loads(str(message.content)))


async def test_conversational_missing_release_becomes_request_failed_tool_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk([])
    _patch_runtime(monkeypatch, sdk)
    tools, _ = create_a2a_tools_and_clients([_make_conversational_resource()])

    content = await _run_through_tool_node(tools, {"message": "hi"})

    assert content["error"] == "request_failed"
    assert "My Conversational Agent" in content["message"]
    assert "is not a published conversational agent" in content["message"]


async def test_conversational_missing_process_name_is_a_request_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeConversationalSdk([_release(1)])
    _patch_runtime(monkeypatch, sdk)
    tools, _ = create_a2a_tools_and_clients(
        [_make_conversational_resource(process_name=None)]
    )

    content = await _run_through_tool_node(tools, {"message": "hi"})

    assert content["error"] == "request_failed"
    assert "conversational agent tool 'Support Agent' has no processName" in str(
        content["message"]
    )
    assert sdk.api_client.calls == []


def test_conversational_confirmation_is_set_per_tool_for_conversational_agents() -> (
    None
):
    first = _make_conversational_resource(require_confirmation=True)
    second = _make_conversational_resource(require_confirmation=False)
    second.name = "Support Agent"  # same sanitized name as the first

    tools, _ = create_a2a_tools_and_clients([first, second], is_conversational=True)

    assert [
        (t.metadata or {}).get(REQUIRE_CONVERSATIONAL_CONFIRMATION) for t in tools
    ] == [True, None]


def test_conversational_confirmation_requires_a_conversational_agent() -> None:
    resource = _make_conversational_resource(require_confirmation=True)

    tools, _ = create_a2a_tools_and_clients([resource])

    assert REQUIRE_CONVERSATIONAL_CONFIRMATION not in (tools[0].metadata or {})


# --- Conversation handles (conversational agent tool) ----------------------


class _ScriptedA2aClient:
    """a2a Client stand-in that replays a script of responses or exceptions."""

    def __init__(
        self, script: list[StreamResponse | Exception], delay: float = 0.0
    ) -> None:
        self._script = list(script)
        self._delay = delay
        self.sent: list[SendMessageRequest] = []
        self.active = 0
        self.max_active = 0

    async def send_message(self, request: SendMessageRequest):
        self.sent.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            item = self._script.pop(0)
        finally:
            self.active -= 1
        if isinstance(item, Exception):
            raise item
        yield item


def _open_task(cid: str, text: str = "ok") -> StreamResponse:
    return _input_required_task(task_id=cid, context_id=cid, text=text)


def _terminal_task(
    cid: str, state: TaskState | str = TaskState.TASK_STATE_FAILED
) -> StreamResponse:
    return StreamResponse(
        task=Task(
            id=cid,
            context_id=cid,
            status=TaskStatus(
                state=state,
                message=Message(
                    role=Role.ROLE_AGENT,
                    parts=[Part(text="it failed")],
                    message_id="status-msg",
                ),
            ),
        )
    )


def _conversation_tool(
    monkeypatch: pytest.MonkeyPatch, client: _ScriptedA2aClient
) -> tuple[A2aStructuredToolWithWrapper, Any]:
    tools, clients = create_a2a_tools_and_clients([_make_conversational_resource()])

    async def _get():
        return client

    monkeypatch.setattr(clients[0], "get", _get)
    tool = cast(A2aStructuredToolWithWrapper, tools[0])
    assert tool.awrapper is not None
    return tool, tool.awrapper


async def _converse(
    tool: A2aStructuredToolWithWrapper,
    wrapper: Any,
    storage: dict[str, Any] | None,
    **args: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = AgentGraphState()
    if storage is not None:
        state.inner_state.tools_storage[tool.name] = storage
    call = {"name": tool.name, "args": {"message": "hi", **args}, "id": "call-1"}
    command = await wrapper(tool, call, state)
    result = json.loads(command.update["messages"][0].content)
    entry = command.update["inner_state"]["tools_storage"][tool.name]
    return result, entry


def _entry(
    conversations: dict[str, tuple[str, int]], current: str | None
) -> dict[str, Any]:
    return {
        "current": current,
        "conversations": {
            cid: {"task_id": cid, "state": state, "last_used": used}
            for cid, (state, used) in conversations.items()
        },
    }


def test_conversational_tool_exposes_handle_arguments() -> None:
    tools, _ = create_a2a_tools_and_clients([_make_conversational_resource()])

    assert tools[0].args_schema is A2aConversationInput
    assert set(A2aConversationInput.model_fields) == {
        "message",
        "conversation_id",
        "new_conversation",
    }
    assert "conversation_id" in tools[0].description
    assert "new_conversation=true" in tools[0].description


def test_remote_tool_keeps_its_input_and_state_shape() -> None:
    tools, _ = create_a2a_tools_and_clients([_make_resource()])

    assert tools[0].args_schema is A2aToolInput
    assert "conversation_id" not in tools[0].description


async def test_remote_tool_state_shape_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools, clients = create_a2a_tools_and_clients([_make_resource()])
    tool = cast(A2aStructuredToolWithWrapper, tools[0])
    fake = _FakeA2aClient([_agent_message("pong", context_id="ctx-9")])

    async def _get():
        return fake

    monkeypatch.setattr(clients[0], "get", _get)
    call = {"name": tool.name, "args": {"message": "ping"}, "id": "call-1"}
    command = await cast(Any, tool.awrapper)(tool, call, AgentGraphState())

    stored = command.update["inner_state"]["tools_storage"][tool.name]
    assert stored == {"task_id": None, "context_id": "ctx-9"}
    assert json.loads(command.update["messages"][0].content) == {
        "agent_response": "pong",
        "task_state": "completed",
    }


async def test_conversational_default_call_reuses_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("c1", "first"), _open_task("c1", "second")])
    tool, wrapper = _conversation_tool(monkeypatch, client)

    result, entry = await _converse(tool, wrapper, None)
    assert result == {
        "agent_response": "first",
        "conversation_id": "c1",
        "task_state": "input_required",
    }
    assert entry["current"] == "c1"
    assert entry["conversations"]["c1"]["state"] == "open"
    assert client.sent[0].message.context_id == ""

    result, entry = await _converse(tool, wrapper, entry)
    assert result["conversation_id"] == "c1"
    assert client.sent[1].message.context_id == "c1"
    assert client.sent[1].message.task_id == "c1"
    assert entry["conversations"]["c1"]["last_used"] == 2


async def test_conversational_new_conversation_changes_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("c2")])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    storage = _entry({"c1": ("open", 1)}, current="c1")

    result, entry = await _converse(tool, wrapper, storage, new_conversation=True)

    assert result["conversation_id"] == "c2"
    assert client.sent[0].message.context_id == ""
    assert client.sent[0].message.task_id == ""
    assert entry["current"] == "c2"
    assert set(entry["conversations"]) == {"c1", "c2"}


async def test_conversational_explicit_conversation_id_is_sent_as_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("c1")])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    storage = _entry({"c1": ("open", 1), "c2": ("open", 2)}, current="c2")

    result, entry = await _converse(tool, wrapper, storage, conversation_id="c1")

    assert result["conversation_id"] == "c1"
    assert client.sent[0].message.context_id == "c1"
    assert client.sent[0].message.task_id == "c1"
    assert entry["current"] == "c2"
    assert entry["conversations"]["c1"]["last_used"] == 3


async def test_conversational_unknown_conversation_id_is_still_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("from-history")])
    tool, wrapper = _conversation_tool(monkeypatch, client)

    result, entry = await _converse(tool, wrapper, None, conversation_id="from-history")

    assert result["conversation_id"] == "from-history"
    assert client.sent[0].message.context_id == "from-history"
    assert client.sent[0].message.task_id == ""
    assert entry["conversations"]["from-history"]["state"] == "open"


async def test_conversational_closed_conversation_is_rejected_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    storage = _entry({"c1": ("closed", 1)}, current=None)

    result, entry = await _converse(tool, wrapper, storage, conversation_id="c1")

    assert result["error"] == "conversation_closed"
    assert result["conversation_id"] == "c1"
    assert "new_conversation=true" in result["message"]
    assert client.sent == []
    assert entry == storage


async def test_conversational_terminal_task_closes_conversation_and_next_call_starts_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_terminal_task("c1"), _open_task("c2")])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    storage = _entry({"c1": ("open", 1)}, current="c1")

    result, entry = await _converse(tool, wrapper, storage)

    assert result["conversation_id"] == "c1"
    assert result["task_state"] == "failed"
    assert entry["current"] is None
    assert entry["conversations"]["c1"]["state"] == "closed"

    result, entry = await _converse(tool, wrapper, entry)

    assert client.sent[1].message.context_id == ""
    assert client.sent[1].message.task_id == ""
    assert result["conversation_id"] == "c2"
    assert entry["current"] == "c2"


async def test_conversational_server_terminal_state_error_means_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient(
        [InvalidRequestError("Cannot send a message to a task in a terminal state.")]
    )
    tool, wrapper = _conversation_tool(monkeypatch, client)
    storage = _entry({"c1": ("open", 1)}, current="c1")

    result, entry = await _converse(tool, wrapper, storage)

    assert result["error"] == "conversation_closed"
    assert result["conversation_id"] == "c1"
    assert entry["current"] is None
    assert entry["conversations"]["c1"]["state"] == "closed"


async def test_conversational_other_errors_are_request_failed_and_keep_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([RuntimeError("boom")])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    storage = _entry({"c1": ("open", 1)}, current="c1")

    result, entry = await _converse(tool, wrapper, storage)

    assert result["error"] == "request_failed"
    assert result["conversation_id"] == "c1"
    assert "boom" in result["message"]
    assert entry == storage


async def test_conversational_request_failed_on_new_conversation_has_null_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([RuntimeError("boom")])
    tool, wrapper = _conversation_tool(monkeypatch, client)

    result, entry = await _converse(tool, wrapper, None)

    assert result["error"] == "request_failed"
    assert result["conversation_id"] is None
    assert entry == {"current": None, "conversations": {}}


async def test_conversational_both_handle_arguments_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    storage = _entry({"c1": ("open", 1)}, current="c1")

    result, entry = await _converse(
        tool, wrapper, storage, conversation_id="c1", new_conversation=True
    )

    assert result["error"] == "invalid_arguments"
    assert client.sent == []
    assert entry == storage


async def test_conversational_cap_evicts_closed_conversations_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("new")])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    # c0 is the oldest and open; c5 is closed and newer than c0.
    conversations = {
        f"c{i}": ("closed" if i == 5 else "open", i + 1) for i in range(20)
    }
    storage = _entry(conversations, current="c19")

    _, entry = await _converse(tool, wrapper, storage, new_conversation=True)

    assert len(entry["conversations"]) == 20
    assert "c5" not in entry["conversations"]
    assert "c0" in entry["conversations"]
    assert "new" in entry["conversations"]

    # With no closed conversation left, the least recently used open one goes.
    client2 = _ScriptedA2aClient([_open_task("newer")])
    tool, wrapper = _conversation_tool(monkeypatch, client2)
    _, entry = await _converse(tool, wrapper, entry, new_conversation=True)

    assert len(entry["conversations"]) == 20
    assert "c0" not in entry["conversations"]
    assert entry["current"] == "newer"


async def test_conversational_stateless_send_honors_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("c1"), _open_task("c1"), _open_task("c2")])
    tool, _ = _conversation_tool(monkeypatch, client)

    first = json.loads(await tool.ainvoke({"message": "a"}))
    assert first["conversation_id"] == "c1"
    assert client.sent[0].message.context_id == ""

    explicit = json.loads(await tool.ainvoke({"message": "b", "conversation_id": "c1"}))
    assert explicit["conversation_id"] == "c1"
    assert client.sent[1].message.context_id == "c1"

    fresh = json.loads(await tool.ainvoke({"message": "c", "new_conversation": True}))
    assert fresh["conversation_id"] == "c2"
    assert client.sent[2].message.context_id == ""


async def test_conversational_stateless_send_validates_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([])
    tool, _ = _conversation_tool(monkeypatch, client)

    result = json.loads(
        await tool.ainvoke(
            {"message": "a", "conversation_id": "c1", "new_conversation": True}
        )
    )

    assert result["error"] == "invalid_arguments"
    assert client.sent == []


async def test_conversational_sends_to_one_conversation_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("c1"), _open_task("c1")], delay=0.05)
    tool, _ = _conversation_tool(monkeypatch, client)

    await asyncio.gather(
        tool.ainvoke({"message": "a", "conversation_id": "c1"}),
        tool.ainvoke({"message": "b", "conversation_id": "c1"}),
    )

    assert client.max_active == 1


async def test_conversational_sends_to_different_conversations_run_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("c1"), _open_task("c2")], delay=0.05)
    tool, _ = _conversation_tool(monkeypatch, client)

    await asyncio.gather(
        tool.ainvoke({"message": "a", "conversation_id": "c1"}),
        tool.ainvoke({"message": "b", "conversation_id": "c2"}),
    )

    assert client.max_active == 2


async def test_conversational_stateless_no_args_starts_a_new_conversation_each_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("run1"), _open_task("run2")])
    tool, _ = _conversation_tool(monkeypatch, client)

    first = json.loads(await tool.ainvoke({"message": "a"}))
    second = json.loads(await tool.ainvoke({"message": "b"}))

    assert (first["conversation_id"], second["conversation_id"]) == ("run1", "run2")
    assert client.sent[1].message.context_id == ""
    assert client.sent[1].message.task_id == ""


async def test_conversational_hint_says_what_omitting_both_arguments_does() -> None:
    tools, _ = create_a2a_tools_and_clients([_make_conversational_resource()])

    assert "If you pass neither" in tools[0].description
    assert "pass conversation_id whenever you mean to continue" in tools[0].description


async def test_conversational_null_new_conversation_means_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        A2aConversationInput(message="m", new_conversation=None).new_conversation
        is None
    )
    client = _ScriptedA2aClient([_open_task("c1")])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    storage = _entry({"c1": ("open", 1)}, current="c1")

    result, _ = await _converse(tool, wrapper, storage, new_conversation=None)

    assert result["conversation_id"] == "c1"
    assert client.sent[0].message.context_id == "c1"


async def test_conversational_replaced_conversation_id_adds_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("fresh"), _open_task("fresh")])
    tool, wrapper = _conversation_tool(monkeypatch, client)

    result, entry = await _converse(tool, wrapper, None, conversation_id="expired")

    assert result["conversation_id"] == "fresh"
    assert result["note"] == (
        "The previous conversation was not found; a new conversation was started."
    )
    assert entry["current"] == "fresh"

    result, _ = await _converse(tool, wrapper, entry, conversation_id="fresh")

    assert "note" not in result


async def test_conversational_explicit_id_becomes_current_when_there_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("c9")])
    tool, wrapper = _conversation_tool(monkeypatch, client)

    _, entry = await _converse(tool, wrapper, None, conversation_id="c9")

    assert entry["current"] == "c9"


async def test_conversational_locks_are_not_kept_for_invented_ids() -> None:
    locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
        weakref.WeakValueDictionary()
    )
    entry: dict[str, Any] = {"current": None, "conversations": {}}

    async def send(
        message: str, task_id: str | None, context_id: str | None
    ) -> tuple[str, str, str | None, str | None]:
        return "ok", "input_required", None, context_id

    for i in range(50):
        await _converse_args(
            A2aConversationInput(message="m", conversation_id=f"made-up-{i}"),
            entry,
            send,
            locks,
        )

    gc.collect()
    assert len(locks) == 0


async def _converse_args(
    args: A2aConversationInput,
    entry: dict[str, Any],
    send: Any,
    locks: weakref.WeakValueDictionary[str, asyncio.Lock],
) -> str:
    return await a2a_tool._converse(
        args, lambda: entry, lambda updated: entry.update(updated), send, locks
    )


async def test_conversational_span_carries_the_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, "get_tracer", provider.get_tracer)
    client = _ScriptedA2aClient([_open_task("c1")])
    tool, _ = _conversation_tool(monkeypatch, client)

    await tool.ainvoke({"message": "ping"})

    (span,) = [s for s in exporter.get_finished_spans() if s.name == "Support Agent"]
    assert (span.attributes or {})["conversation_id"] == "c1"


async def test_conversational_old_single_conversation_state_is_migrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("c1")])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    old = {"task_id": "t1", "context_id": "c1"}

    result, entry = await _converse(tool, wrapper, old)

    assert result["conversation_id"] == "c1"
    assert client.sent[0].message.context_id == "c1"
    assert client.sent[0].message.task_id == "t1"
    assert entry["current"] == "c1"
    assert entry["conversations"]["c1"]["state"] == "open"


async def test_conversational_non_terminal_invalid_request_is_request_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([InvalidRequestError("Unsupported content type")])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    storage = _entry({"c1": ("open", 1)}, current="c1")

    result, entry = await _converse(tool, wrapper, storage)

    assert result["error"] == "request_failed"
    assert entry == storage


async def test_conversational_other_error_mentioning_terminal_state_is_request_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([RuntimeError("task is in a terminal state")])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    storage = _entry({"c1": ("open", 1)}, current="c1")

    result, entry = await _converse(tool, wrapper, storage)

    assert result["error"] == "request_failed"
    assert entry == storage


async def test_conversational_plain_message_reply_keeps_the_conversation_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = [_agent_message("pong", context_id="c1")]
    client = _ScriptedA2aClient([*reply, *reply])
    tool, wrapper = _conversation_tool(monkeypatch, client)

    result, entry = await _converse(tool, wrapper, None)

    assert result["task_state"] == "input_required"
    assert result["conversation_id"] == "c1"
    assert entry["conversations"]["c1"]["state"] == "open"
    assert entry["current"] == "c1"

    result, _ = await _converse(tool, wrapper, entry)

    assert result["conversation_id"] == "c1"
    assert client.sent[1].message.context_id == "c1"


async def test_conversational_state_survives_the_tool_node_and_reducer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("c1"), _open_task("c2")])
    tools, clients = create_a2a_tools_and_clients([_make_conversational_resource()])

    async def _get():
        return client

    monkeypatch.setattr(clients[0], "get", _get)
    node = create_tool_node(tools)[tools[0].name]
    builder = StateGraph(AgentGraphState)
    builder.add_node("first", node)
    builder.add_node("second", node)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)

    final = await builder.compile().ainvoke(
        AgentGraphState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tools[0].name,
                            "args": {"message": "a"},
                            "id": "call-1",
                        },
                        {
                            "name": tools[0].name,
                            "args": {"message": "b", "new_conversation": True},
                            "id": "call-2",
                        },
                    ],
                )
            ]
        )
    )

    stored = final["inner_state"].tools_storage[tools[0].name]
    assert stored["current"] == "c2"
    assert set(stored["conversations"]) == {"c1", "c2"}
    assert stored["conversations"]["c2"]["last_used"] == 2
    assert [m.tool_call_id for m in final["messages"][1:]] == ["call-1", "call-2"]
    assert client.sent[1].message.context_id == ""


async def test_conversational_cap_eviction_clears_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("other")])
    tool, wrapper = _conversation_tool(monkeypatch, client)
    conversations = {f"c{i}": ("open", i + 1) for i in range(20)}
    storage = _entry(conversations, current="c0")

    _, entry = await _converse(tool, wrapper, storage, conversation_id="other")

    assert "c0" not in entry["conversations"]
    assert entry["current"] is None


async def test_conversational_confirmation_flow_runs_the_approved_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([_open_task("c7")])
    tools, clients = create_a2a_tools_and_clients(
        [_make_conversational_resource(require_confirmation=True)],
        is_conversational=True,
    )

    async def _get():
        return client

    monkeypatch.setattr(clients[0], "get", _get)
    approved = {"message": "edited", "conversation_id": "c7"}
    seen: list[dict[str, Any]] = []

    def _approve(tool_args: dict[str, Any], tool: Any) -> dict[str, Any]:
        seen.append(dict(tool_args))
        return approved

    monkeypatch.setattr("uipath_langchain.chat.hitl.request_approval", _approve)

    content = await _run_through_tool_node(
        tools, {"message": "original", "new_conversation": False}
    )

    assert seen[0]["message"] == "original"
    assert seen[0]["new_conversation"] is False
    # Edited arguments wrap the result in the args-modified envelope.
    assert content["meta"]["executed_args"] == approved
    assert content["result"]["conversation_id"] == "c7"
    assert client.sent[0].message.context_id == "c7"
    assert client.sent[0].message.parts[0].text == "edited"


async def test_conversational_confirmation_rejection_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedA2aClient([])
    tools, clients = create_a2a_tools_and_clients(
        [_make_conversational_resource(require_confirmation=True)],
        is_conversational=True,
    )

    async def _get():
        return client

    monkeypatch.setattr(clients[0], "get", _get)
    monkeypatch.setattr(
        "uipath_langchain.chat.hitl.request_approval", lambda tool_args, tool: None
    )

    await _run_through_tool_node(tools, {"message": "x", "conversation_id": "c1"})

    assert client.sent == []
