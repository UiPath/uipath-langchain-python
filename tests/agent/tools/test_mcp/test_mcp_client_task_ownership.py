"""``McpClient`` teardown must not depend on which task opened the session.

The transport and ``ClientSession`` live in an anyio task group inside the
client's exit stack, so whoever exits it has to be the task that entered it,
and scopes on one task have to unwind LIFO. Neither holds for a client:
langgraph runs each tool call in its own task (``asyncio.gather`` in
``ToolNode``) while ``dispose()`` runs on the teardown task, and an agent with
several servers disposes them in creation order.

Before the connection moved onto its own task, the scope violation was caught
under ``except Exception`` and logged at DEBUG, so a test that only checked
state after ``dispose()`` passed anyway. These assert nothing was swallowed,
against a real server that records what reached it.
"""

import asyncio
import logging
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from uipath.agent.models.agent import AgentMcpResourceConfig, AgentMcpTool

from uipath_langchain.agent.tools.mcp import McpClient

MCP_CLIENT_LOGGER = "uipath_langchain.agent.tools.mcp.mcp_client"


@dataclass
class MathServer:
    """A running MCP server plus the HTTP methods it has seen, in order."""

    url: str
    methods: list[str] = field(default_factory=list)


@pytest.fixture
def math_server() -> Iterator[MathServer]:
    """Host a real ``FastMCP`` server on an ephemeral port, on its own thread."""
    import uvicorn
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("Math")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    running = MathServer(url=f"http://127.0.0.1:{port}/mcp")
    app = server.streamable_http_app()

    async def recording_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            running.methods.append(scope["method"])
        await app(scope, receive, send)

    uv = uvicorn.Server(
        uvicorn.Config(recording_app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not uv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert uv.started, "MCP test server did not start"
    try:
        yield running
    finally:
        uv.should_exit = True
        thread.join(5)


@contextmanager
def patched_sdk(url: str) -> Iterator[None]:
    """Point the client's lazy ``UiPath`` lookup at the local server."""

    class _Server:
        mcp_url = url
        slug = "math"
        folder_key = "folder-key"
        name = "Math"

    class _Mcp:
        async def retrieve_async(self, name: str, folder_path: str | None) -> _Server:
            return _Server()

    class _Config:
        secret = "test-token"

    class FakeUiPath:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.mcp = _Mcp()
            self._config = _Config()

    with patch("uipath.platform.UiPath", FakeUiPath):
        yield


def make_client() -> McpClient:
    return McpClient(
        config=AgentMcpResourceConfig(
            name="Math",
            description="Math MCP server",
            folder_path="Shared",
            slug="math",
            available_tools=[
                AgentMcpTool(
                    name="add",
                    description="Add two numbers",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                )
            ],
        )
    )


def swallowed_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Errors the client logged instead of raising."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == MCP_CLIENT_LOGGER and "error" in r.getMessage().lower()
    ]


@pytest.mark.asyncio
async def test_session_opened_on_a_tool_task_disposes_from_the_teardown_task(
    math_server: MathServer, caplog: pytest.LogCaptureFixture
) -> None:
    """Cached discovery connects on first tool call, i.e. inside a tool task."""
    caplog.set_level(logging.DEBUG, logger=MCP_CLIENT_LOGGER)
    with patched_sdk(math_server.url):
        client = make_client()

        # asyncio.gather wraps the coroutine in its own Task, the way
        # langgraph's ToolNode dispatches tool calls.
        (result,) = await asyncio.gather(client.call_tool("add", {"a": 2, "b": 3}))
        assert "5" in str(result)

        await client.dispose()

        assert not client.is_client_initialized

    assert swallowed_errors(caplog) == []
    # terminate_on_close still reaches the server through the task-owned teardown:
    # the HTTP client is entered before the transport, so it is open when the
    # transport exits and sends the session DELETE.
    assert "DELETE" in math_server.methods


@pytest.mark.asyncio
async def test_clients_dispose_cleanly_in_creation_order(
    math_server: MathServer, caplog: pytest.LogCaptureFixture
) -> None:
    """Dynamic discovery connects every server up front, on one task."""
    caplog.set_level(logging.DEBUG, logger=MCP_CLIENT_LOGGER)
    with patched_sdk(math_server.url):
        clients = [make_client() for _ in range(3)]
        for client in clients:
            await client.list_tools()

        # Oldest-first: the order a plain `for disposable in ...` loop uses.
        for client in clients:
            await client.dispose()

        assert all(not c.is_client_initialized for c in clients)

    assert swallowed_errors(caplog) == []


@pytest.mark.asyncio
async def test_cancelling_the_caller_mid_handshake_unwinds_promptly(
    math_server: MathServer, caplog: pytest.LogCaptureFixture
) -> None:
    """A caller cancelled during the handshake must not wait for it to finish.

    Signalling a connection task does nothing until the handshake returns, and
    the transport timeout is ten minutes. The task has to be cancelled.
    """
    caplog.set_level(logging.DEBUG, logger=MCP_CLIENT_LOGGER)
    with patched_sdk(math_server.url):
        client = make_client()
        handshake_started = asyncio.Event()

        async def slow_handshake() -> None:
            handshake_started.set()
            await asyncio.sleep(5)

        with patch.object(client, "_initialize_session", slow_handshake):
            caller = asyncio.create_task(client.call_tool("add", {"a": 2, "b": 3}))
            await asyncio.wait_for(handshake_started.wait(), timeout=5)

            loop = asyncio.get_running_loop()
            started = loop.time()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            elapsed = loop.time() - started

        assert elapsed < 1.0, f"cancellation waited {elapsed:.2f}s"
        assert client._connection_task is None
        assert not client.is_client_initialized

    assert swallowed_errors(caplog) == []


@pytest.mark.asyncio
async def test_a_connection_that_dies_on_its_own_is_rebuilt_on_the_next_call(
    math_server: MathServer, caplog: pytest.LogCaptureFixture
) -> None:
    """The connection task ending by itself must not poison the client.

    A transport child task failing after the handshake cancels the task
    group's scope, which ends the connection task. If that only cleared the
    session while leaving the client flagged as initialized, every later call
    would fail on a missing session with no retry. The next call has to
    rebuild the connection instead.
    """
    caplog.set_level(logging.DEBUG, logger=MCP_CLIENT_LOGGER)
    with patched_sdk(math_server.url):
        client = make_client()
        await client.call_tool("add", {"a": 1, "b": 1})

        task = client._connection_task
        assert task is not None
        task.cancel()  # what the transport's task group does when a child fails
        await asyncio.wait({task})

        result = await client.call_tool("add", {"a": 2, "b": 3})
        assert "5" in str(result)

        await client.dispose()

    assert swallowed_errors(caplog) == []
