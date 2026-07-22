"""Tests for A2A tool creation and URL resolution.

The proxy URL is resolved lazily at runtime by retrieving the remote agent
through the binding-aware SDK (``remote_a2a.retrieve_async``), mirroring how
MCP servers are resolved — not from a field baked into the resource config.
"""

import os
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
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from uipath.agent.models.agent import AgentA2aResourceConfig

import uipath_langchain.agent.tools.a2a.a2a_tool as a2a_tool
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.a2a.a2a_tool import (
    A2aClient,
    A2aStructuredToolWithWrapper,
    _build_description,
    _send_a2a_message,
    create_a2a_tools_and_clients,
)

PROXY_URL = (
    "https://cloud.uipath.com/org/tenant/agenthub_/a2a/remote/folder/remote-agent-slug"
)
CACHED_URL = "https://internal.example.com/agents/remote-agent"


def _make_resource(
    *,
    cached_agent_card: dict[str, Any] | None = None,
    is_enabled: bool = True,
) -> AgentA2aResourceConfig:
    """Build an A2A resource config for tests."""
    return AgentA2aResourceConfig(
        id="resource-id",
        name="remote-agent",
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

    async def _fake_connect(agent_card, *, client_config):
        return connected

    monkeypatch.setattr(ClientFactory, "connect", staticmethod(_fake_connect))
    return {"connected": connected}


async def test_client_resolves_proxy_url_via_retrieve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk(a2a_url=PROXY_URL)
    handles = _patch_runtime(monkeypatch, sdk)

    card = AgentCard(
        url=CACHED_URL,
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
    assert card.url == PROXY_URL
    assert sdk.remote_a2a.calls == [("Remote Agent", "Shared")]

    await client.dispose()


async def test_client_raises_when_agent_has_no_proxy_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _FakeSdk(a2a_url=None)
    _patch_runtime(monkeypatch, sdk)

    card = AgentCard(
        url="",
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


def _agent_message(text: str, context_id: str | None = None) -> Message:
    return Message(
        role=Role.agent,
        parts=[Part(root=TextPart(text=text))],
        message_id="msg-1",
        context_id=context_id,
    )


class _FakeA2aClient:
    """Minimal stand-in for the a2a Client whose send_message yields events."""

    def __init__(self, events: list[object]) -> None:
        self._events = events
        self.sent: list[Message] = []

    async def send_message(self, message: Message):
        self.sent.append(message)
        for event in self._events:
            yield event


class _RaisingA2aClient:
    async def send_message(self, message: Message):
        raise RuntimeError("boom")
        yield  # pragma: no cover - marks this as an async generator


def test_build_description_falls_back_to_name() -> None:
    card = AgentCard(
        url="",
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
        url="",
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
    assert client.sent[0].task_id == "task-1"


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
) -> Task:
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=TaskState.completed),
        artifacts=[
            Artifact(
                artifact_id="artifact-1",
                parts=[Part(root=TextPart(text=text))],
            )
        ],
    )


def _input_required_task(
    *, task_id: str = "task-1", context_id: str = "ctx-1", text: str = "need more"
) -> Task:
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(
            state=TaskState.input_required,
            message=Message(
                role=Role.agent,
                parts=[Part(root=TextPart(text=text))],
                message_id="status-msg",
            ),
        ),
    )


async def test_tool_wrapper_drops_task_id_after_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed (terminal) task is not reused: the next turn starts a new
    task while keeping the conversation context."""
    resource = _make_resource(cached_agent_card=_cached_card())
    tools, clients = create_a2a_tools_and_clients([resource])
    tool = cast(A2aStructuredToolWithWrapper, tools[0])
    fake = _FakeA2aClient(
        [(_completed_task(task_id="task-1", context_id="ctx-1"), None)]
    )

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
    fake = _FakeA2aClient(
        [(_input_required_task(task_id="task-1", context_id="ctx-1"), None)]
    )

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
