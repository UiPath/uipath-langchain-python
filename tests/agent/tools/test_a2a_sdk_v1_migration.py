"""Migration contract tests for the A2A Python SDK v1."""

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import httpx
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
from uipath.agent.models.agent import AgentA2aResourceConfig

import uipath_langchain.agent.tools.a2a.a2a_tool as a2a_tool
from uipath_langchain.agent.tools.a2a.a2a_tool import (
    A2aClient,
    _send_a2a_message,
    create_a2a_tools_and_clients,
)

PROXY_URL = (
    "https://cloud.uipath.com/org/tenant/agenthub_/a2a/remote/folder/remote-agent"
)


def test_a2a_tool_imports_with_sdk_v1() -> None:
    """The A2A tool must not import APIs removed by a2a-sdk v1."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import uipath_langchain.agent.tools.a2a.a2a_tool",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


async def test_client_uses_strict_v1_jsonrpc_interface_at_proxy_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached endpoints must not replace the binding-aware AgentHub proxy."""
    import uipath.platform as uipath_platform
    from a2a.client import ClientFactory

    sdk = SimpleNamespace(
        remote_a2a=SimpleNamespace(
            retrieve_async=lambda **_: _async_value(SimpleNamespace(a2a_url=PROXY_URL))
        ),
        _config=SimpleNamespace(secret="token"),
    )
    monkeypatch.setattr(uipath_platform, "UiPath", lambda: sdk)
    monkeypatch.setattr(a2a_tool, "get_execution_folder_path", lambda: "Shared")

    captured: dict[str, Any] = {}
    connected = SimpleNamespace()

    def _fake_create(self: ClientFactory, card: AgentCard):
        captured["card"] = card
        return connected

    monkeypatch.setattr(ClientFactory, "create", _fake_create)

    metadata_card = AgentCard(name="Remote Agent", description="cached")
    client = A2aClient(metadata_card, resource_name="remote-agent")

    assert await client.get() is connected
    interfaces = list(captured["card"].supported_interfaces)
    assert len(interfaces) == 1
    assert interfaces[0].url == PROXY_URL
    assert interfaces[0].protocol_binding == "JSONRPC"
    assert interfaces[0].protocol_version == "1.0"
    assert client._http_client is not None
    assert client._http_client.headers["A2A-Version"] == "1.0"

    await client.dispose()


async def test_client_uses_v03_compat_for_legacy_only_cached_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy-only binding must use the SDK's v0.3 compatibility client."""
    import uipath.platform as uipath_platform

    sdk = SimpleNamespace(
        remote_a2a=SimpleNamespace(
            retrieve_async=lambda **_: _async_value(SimpleNamespace(a2a_url=PROXY_URL))
        ),
        _config=SimpleNamespace(secret="token"),
    )
    monkeypatch.setattr(uipath_platform, "UiPath", lambda: sdk)
    monkeypatch.setattr(a2a_tool, "get_execution_folder_path", lambda: "Shared")
    captured: dict[str, Any] = {}

    async def _handle_request(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured["headers"] = request.headers
        captured["payload"] = payload
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "kind": "message",
                    "messageId": "reply-1",
                    "role": "agent",
                    "parts": [{"kind": "text", "text": "pong"}],
                },
            },
        )

    monkeypatch.setattr(
        a2a_tool,
        "get_httpx_client_kwargs",
        lambda **kwargs: {
            "headers": kwargs["headers"],
            "transport": httpx.MockTransport(_handle_request),
        },
    )

    resource = AgentA2aResourceConfig(
        id="resource-id",
        name="remote-agent",
        description="resource description",
        is_enabled=True,
        slug="remote-agent",
        folder_path="Shared",
        cached_agent_card={
            "url": "https://legacy.example/a2a",
            "name": "Legacy Agent",
            "preferredTransport": "JSONRPC",
            "protocolVersion": "0.3.0",
            "capabilities": {},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
        },
    )
    _, clients = create_a2a_tools_and_clients([resource])

    connected = await clients[0].get()
    result = await _send_a2a_message(
        connected,
        "remote-agent",
        message="ping",
        task_id=None,
        context_id=None,
    )

    assert result == ("pong", "completed", None, None)
    assert captured["url"] == PROXY_URL
    assert captured["headers"]["A2A-Version"] == "0.3"
    assert captured["payload"]["method"] == "message/send"
    assert captured["payload"]["params"]["configuration"]["acceptedOutputModes"] == [
        "text/plain"
    ]

    await clients[0].dispose()


async def test_client_prefers_v1_when_cached_card_advertises_both_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blended card must negotiate v1 without probing the legacy endpoint."""
    import uipath.platform as uipath_platform
    from a2a.client import ClientFactory

    sdk = SimpleNamespace(
        remote_a2a=SimpleNamespace(
            retrieve_async=lambda **_: _async_value(SimpleNamespace(a2a_url=PROXY_URL))
        ),
        _config=SimpleNamespace(secret="token"),
    )
    monkeypatch.setattr(uipath_platform, "UiPath", lambda: sdk)
    monkeypatch.setattr(a2a_tool, "get_execution_folder_path", lambda: "Shared")

    captured: dict[str, Any] = {}
    connected = SimpleNamespace()

    def _fake_create(self: ClientFactory, card: AgentCard):
        captured["card"] = card
        return connected

    monkeypatch.setattr(ClientFactory, "create", _fake_create)

    resource = AgentA2aResourceConfig(
        id="resource-id",
        name="remote-agent",
        description="resource description",
        is_enabled=True,
        slug="remote-agent",
        folder_path="Shared",
        cached_agent_card={
            "url": "https://legacy.example/a2a",
            "name": "Blended Agent",
            "preferredTransport": "JSONRPC",
            "protocolVersion": "0.3.0",
            "supportedInterfaces": [
                {
                    "url": "https://v1.example/a2a",
                    "protocolBinding": "JSONRPC",
                    "protocolVersion": "1.0",
                }
            ],
            "capabilities": {},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
        },
    )
    _, clients = create_a2a_tools_and_clients([resource])

    assert await clients[0].get() is connected
    assert captured["card"].supported_interfaces[0].protocol_version == "1.0"
    assert clients[0]._http_client is not None
    assert clients[0]._http_client.headers["A2A-Version"] == "1.0"

    await clients[0].dispose()


async def test_client_rejects_cached_card_without_compatible_jsonrpc_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported card must fail before an A2A request can be dispatched."""
    import uipath.platform as uipath_platform
    from a2a.client import ClientFactory

    sdk = SimpleNamespace(
        remote_a2a=SimpleNamespace(
            retrieve_async=lambda **_: _async_value(SimpleNamespace(a2a_url=PROXY_URL))
        ),
        _config=SimpleNamespace(secret="token"),
    )
    monkeypatch.setattr(uipath_platform, "UiPath", lambda: sdk)
    monkeypatch.setattr(a2a_tool, "get_execution_folder_path", lambda: "Shared")
    create_calls = 0

    def _fake_create(self: ClientFactory, card: AgentCard):
        nonlocal create_calls
        create_calls += 1
        return SimpleNamespace()

    monkeypatch.setattr(ClientFactory, "create", _fake_create)

    resource = AgentA2aResourceConfig(
        id="resource-id",
        name="remote-agent",
        description="resource description",
        is_enabled=True,
        slug="remote-agent",
        folder_path="Shared",
        cached_agent_card={
            "name": "Unsupported Agent",
            "supportedInterfaces": [
                {
                    "url": "https://malformed.example/a2a",
                    "protocolBinding": "JSONRPC",
                    "protocolVersion": "1.0.invalid",
                }
            ],
            "capabilities": {},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
        },
    )
    _, clients = create_a2a_tools_and_clients([resource])

    with pytest.raises(ValueError, match="no compatible JSON-RPC endpoint"):
        await clients[0].get()
    assert create_calls == 0


async def _async_value(value: Any) -> Any:
    return value


def test_legacy_cached_card_remains_usable_as_tool_metadata() -> None:
    """A cached v0.3 card must not prevent creating a v1-backed tool."""
    resource = AgentA2aResourceConfig(
        id="resource-id",
        name="remote-agent",
        description="resource description",
        is_enabled=True,
        slug="remote-agent",
        folder_path="Shared",
        cached_agent_card={
            "url": "https://untrusted.example/ignored",
            "name": "Remote Agent",
            "description": "cached description",
            "version": "2.3.4",
            "skills": [
                {
                    "id": "answer",
                    "name": "Answer questions",
                    "description": "Answers a user question",
                    "tags": ["qa"],
                }
            ],
            "capabilities": {},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
        },
    )

    tools, clients = create_a2a_tools_and_clients([resource])

    assert tools[0].name == "Remote_Agent"
    assert "cached description" in tools[0].description
    assert "Answer questions" in tools[0].description
    assert clients[0]._agent_card.name == "Remote Agent"
    assert list(clients[0]._agent_card.supported_interfaces) == []


class _FakeClient:
    def __init__(self, responses: list[StreamResponse]) -> None:
        self.responses = responses
        self.sent: list[SendMessageRequest] = []

    async def send_message(self, request: SendMessageRequest):
        self.sent.append(request)
        for response in self.responses:
            yield response


async def test_send_message_uses_v1_request_and_reads_message_response() -> None:
    """A call must use v1 protobuf envelopes while retaining conversation IDs."""
    response = StreamResponse(
        message=Message(
            message_id="reply-1",
            context_id="context-1",
            role=Role.ROLE_AGENT,
            parts=[Part(text="pong")],
        )
    )
    client = _FakeClient([response])

    result = await _send_a2a_message(
        cast(Client, client),
        "remote-agent",
        message="ping",
        task_id="task-1",
        context_id="context-1",
    )

    assert result == ("pong", "completed", "task-1", "context-1")
    request = client.sent[0]
    assert request.message.role == Role.ROLE_USER
    assert request.message.parts[0].text == "ping"
    assert request.message.task_id == "task-1"
    assert request.message.context_id == "context-1"


async def test_send_message_reads_completed_task_artifacts() -> None:
    """A v1 task response must return artifacts and its continuation IDs."""
    response = StreamResponse(
        task=Task(
            id="task-2",
            context_id="context-2",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            artifacts=[Artifact(artifact_id="artifact-1", parts=[Part(text="done")])],
        )
    )
    client = _FakeClient([response])

    result = await _send_a2a_message(
        cast(Client, client),
        "remote-agent",
        message="work",
        task_id=None,
        context_id=None,
    )

    assert result == ("done", "completed", "task-2", "context-2")


async def test_send_message_reads_input_required_status_message() -> None:
    """An input-required task must expose the agent's follow-up question."""
    response = StreamResponse(
        task=Task(
            id="task-3",
            context_id="context-3",
            artifacts=[
                Artifact(
                    artifact_id="partial-1",
                    parts=[Part(text="stale partial result")],
                )
            ],
            status=TaskStatus(
                state=TaskState.TASK_STATE_INPUT_REQUIRED,
                message=Message(
                    message_id="question-1",
                    role=Role.ROLE_AGENT,
                    parts=[Part(text="Which account?")],
                ),
            ),
        )
    )
    client = _FakeClient([response])

    result = await _send_a2a_message(
        cast(Client, client),
        "remote-agent",
        message="continue",
        task_id="task-3",
        context_id="context-3",
    )

    assert result == (
        "Which account?",
        "input_required",
        "task-3",
        "context-3",
    )


async def test_send_message_falls_back_to_latest_agent_history() -> None:
    """A task without artifacts must surface its latest agent history entry."""
    response = StreamResponse(
        task=Task(
            id="task-4",
            context_id="context-4",
            status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            history=[
                Message(
                    message_id="user-1",
                    role=Role.ROLE_USER,
                    parts=[Part(text="ignored")],
                ),
                Message(
                    message_id="agent-1",
                    role=Role.ROLE_AGENT,
                    parts=[Part(text="Still working")],
                ),
            ],
        )
    )

    result = await _send_a2a_message(
        cast(Client, _FakeClient([response])),
        "remote-agent",
        message="status",
        task_id="task-4",
        context_id="context-4",
    )

    assert result == (
        "Still working",
        "working",
        "task-4",
        "context-4",
    )
