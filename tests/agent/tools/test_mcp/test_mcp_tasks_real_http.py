"""MCP Tasks over real HTTP: a task-shaped answer is polled, not surfaced.

Drives ``McpClient`` against a server that answers ``tools/call`` with
``resultType: "task"`` and finishes the task after N polls. These assert the
contract agent definitions depend on: the tool call returns the final
``CallToolResult``, the polling happens on the wire as ``tasks/get``, and the
era gate holds — a legacy-era connection never advertises the extension, so a
task can never arrive there.
"""

import json
from typing import Any

import pytest
from starlette.responses import Response

from uipath_langchain.agent.tools.mcp.mcp_tasks import (
    POLL_BUDGET_ENV_VAR,
    TASKS_POLLING_ENV_VAR,
)

from .real_server import (
    LEGACY_VERSION,
    MODERN_VERSION,
    PINNED_TOOL_NAME,
    PinnedVersionServer,
    RecordingGateway,
    connected_client,
    serve,
)


class TaskServingServer(PinnedVersionServer):
    """A pinned server whose tool call answers with a task handle.

    The task reports ``working`` for ``polls_before_done`` ``tasks/get`` calls
    and then reaches its terminal status, mirroring AgentHub projecting an
    Orchestrator job.

    Args:
        polls_before_done: ``tasks/get`` answers reporting ``working`` before
            the terminal one.
        terminal_status: The status the task finally reaches.
        poll_interval_ms: The interval the server asks the client to honor.
    """

    def __init__(
        self,
        *,
        polls_before_done: int = 1,
        terminal_status: str = "completed",
        poll_interval_ms: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(MODERN_VERSION, modern_only=True, **kwargs)
        self.polls_before_done = polls_before_done
        self.terminal_status = terminal_status
        self.poll_interval_ms = poll_interval_ms
        self.poll_count = 0
        self.task_calls = 0

    async def _handle(self, request: Any) -> Response:  # type: ignore[override]
        if request.method == "POST":
            body = json.loads(await request.body())
            if body.get("method") == "tasks/get":
                return self._tasks_get(body)
        return await super()._handle(request)

    def _call_tool(self, body: dict[str, Any]) -> Response:
        """Answer the tool call with a durable handle instead of a result."""
        params = body.get("params") or {}
        if params.get("name") != PINNED_TOOL_NAME:
            return super()._call_tool(body)

        self.task_calls += 1
        self.pending_total = int((params.get("arguments") or {}).get("a", 0)) + int(
            (params.get("arguments") or {}).get("b", 0)
        )
        return self._result(
            body["id"],
            {
                "resultType": "task",
                "taskId": "task-1",
                "status": "working",
                "statusMessage": "Orchestrator job created.",
                "ttlMs": None,
                "pollIntervalMs": self.poll_interval_ms,
            },
        )

    def _tasks_get(self, body: dict[str, Any]) -> Response:
        self.poll_count += 1
        if self.poll_count <= self.polls_before_done:
            return self._result(
                body["id"],
                {
                    "resultType": "complete",
                    "taskId": "task-1",
                    "status": "working",
                    "statusMessage": "Orchestrator job is Running.",
                    "ttlMs": None,
                    "pollIntervalMs": self.poll_interval_ms,
                },
            )

        terminal: dict[str, Any] = {
            "resultType": "complete",
            "taskId": "task-1",
            "status": self.terminal_status,
            "ttlMs": None,
        }
        if self.terminal_status == "completed":
            terminal["result"] = {
                "content": [{"type": "text", "text": str(self.pending_total)}],
                "structuredContent": {"result": self.pending_total},
                "isError": False,
            }
        elif self.terminal_status == "failed":
            terminal["error"] = {"code": -32003, "message": "the task failed"}
        return self._result(body["id"], terminal)


@pytest.mark.asyncio
async def test_a_task_answer_is_polled_to_completion() -> None:
    """The caller gets the final result; the polling is invisible to it."""
    endpoint = TaskServingServer(polls_before_done=2)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="modern") as client:
            result = await client.call_tool(PINNED_TOOL_NAME, {"a": 2, "b": 3})

    assert result.content[0].text == "5"  # type: ignore[union-attr]
    assert endpoint.task_calls == 1
    # Two working snapshots, then the terminal one.
    assert endpoint.poll_count == 3
    assert gateway.count("tasks/get") == 3
    assert gateway.count("tools/call") == 1


@pytest.mark.asyncio
async def test_a_synchronous_answer_is_returned_unchanged() -> None:
    """Declaring the capability must not disturb a server that answers inline."""
    endpoint = PinnedVersionServer(MODERN_VERSION, modern_only=True)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="modern") as client:
            result = await client.call_tool(PINNED_TOOL_NAME, {"a": 2, "b": 3})

    assert result.content[0].text == "5"  # type: ignore[union-attr]
    assert gateway.count("tasks/get") == 0


@pytest.mark.asyncio
async def test_the_client_advertises_the_tasks_extension_on_a_modern_wire() -> None:
    """The capability is declared per request, which is what lets a task arrive."""
    endpoint = TaskServingServer(polls_before_done=0)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="modern") as client:
            await client.call_tool(PINNED_TOOL_NAME, {"a": 1, "b": 1})

    call = gateway.for_rpc("tools/call")[0]
    capabilities = call.meta.get("io.modelcontextprotocol/clientCapabilities", {})
    assert "io.modelcontextprotocol/tasks" in capabilities.get("extensions", {})


@pytest.mark.asyncio
async def test_a_legacy_connection_never_advertises_the_extension() -> None:
    """The era gate: claimed shapes cannot be delivered on a handshake wire."""
    endpoint = PinnedVersionServer(LEGACY_VERSION)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="legacy") as client:
            result = await client.call_tool(PINNED_TOOL_NAME, {"a": 2, "b": 3})

    assert result.content[0].text == "5"  # type: ignore[union-attr]
    call = gateway.for_rpc("tools/call")[0]
    assert "io.modelcontextprotocol/clientCapabilities" not in call.meta


@pytest.mark.asyncio
async def test_the_kill_switch_stops_the_client_declaring_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the opt-out set, the extension is never constructed into the session."""
    monkeypatch.setenv(TASKS_POLLING_ENV_VAR, "false")
    endpoint = PinnedVersionServer(MODERN_VERSION, modern_only=True)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="modern") as client:
            await client.call_tool(PINNED_TOOL_NAME, {"a": 2, "b": 3})

    call = gateway.for_rpc("tools/call")[0]
    capabilities = call.meta.get("io.modelcontextprotocol/clientCapabilities", {})
    assert "io.modelcontextprotocol/tasks" not in capabilities.get("extensions", {})


@pytest.mark.asyncio
async def test_a_failed_task_surfaces_its_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A protocol-level task failure reaches the caller as an MCP error."""
    from mcp.shared.exceptions import MCPError

    endpoint = TaskServingServer(polls_before_done=0, terminal_status="failed")
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="modern") as client:
            with pytest.raises(MCPError, match="the task failed"):
                await client.call_tool(PINNED_TOOL_NAME, {"a": 2, "b": 3})


@pytest.mark.asyncio
async def test_the_poll_budget_bounds_a_task_that_never_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool must not silently consume the agent's whole run window."""
    monkeypatch.setenv(POLL_BUDGET_ENV_VAR, "0.3")
    endpoint = TaskServingServer(polls_before_done=1000, poll_interval_ms=100)
    gateway = RecordingGateway(endpoint.build_app())

    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="modern") as client:
            with pytest.raises(TimeoutError, match="task-1"):
                await client.call_tool(PINNED_TOOL_NAME, {"a": 2, "b": 3})

    # Bounded, and the task is left running server-side rather than cancelled.
    assert endpoint.poll_count <= 3
