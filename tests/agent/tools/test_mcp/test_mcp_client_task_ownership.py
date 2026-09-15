"""``McpClient`` teardown must not depend on which task opened the session.

The transport and ``ClientSession`` live in an anyio task group, so whoever
exits them has to be the task that entered them, and scopes on one task have to
unwind LIFO. Neither holds for a client: langgraph runs each tool call in its
own task (``asyncio.gather`` in ``ToolNode``), while ``dispose()`` runs on the
teardown task, and an agent with several servers disposes them in creation
order. Both shapes are covered here against a real server, and so is cancelling
a caller while the handshake is still in flight.

Before the connection moved onto its own task, the scope violation was caught
under ``except Exception`` and logged at DEBUG, so a test that only checked
state after ``dispose()`` passed anyway. These assert nothing was swallowed.
"""

import asyncio
import logging
from unittest.mock import patch

import pytest

from .real_server import (
    RecordingGateway,
    build_sdk_app,
    make_client,
    patched_sdk,
    serve,
)

MCP_CLIENT_LOGGER = "uipath_langchain.agent.tools.mcp.mcp_client"


def swallowed_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Errors the client logged instead of raising."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == MCP_CLIENT_LOGGER and "error" in record.getMessage().lower()
    ]


@pytest.mark.asyncio
async def test_session_opened_on_a_tool_task_disposes_from_the_teardown_task(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cached discovery connects on first tool call, i.e. inside a tool task."""
    caplog.set_level(logging.DEBUG, logger=MCP_CLIENT_LOGGER)
    gateway = RecordingGateway(build_sdk_app())
    async with serve(gateway) as url:
        with patched_sdk(url):
            client = make_client()

            # asyncio.gather wraps the coroutine in its own Task, the way
            # langgraph's ToolNode dispatches tool calls.
            (result,) = await asyncio.gather(client.call_tool("add", {"a": 2, "b": 3}))
            assert result is not None

            await client.dispose()

            assert not client.is_client_initialized

    assert swallowed_errors(caplog) == []
    assert "DELETE" in [r.http_method for r in gateway.records]


@pytest.mark.asyncio
async def test_clients_dispose_cleanly_in_creation_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dynamic discovery connects every server up front, on one task."""
    caplog.set_level(logging.DEBUG, logger=MCP_CLIENT_LOGGER)
    gateway = RecordingGateway(build_sdk_app())
    async with serve(gateway) as url:
        with patched_sdk(url):
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
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A caller cancelled during the handshake must not wait for it to finish.

    Signalling the connection task does nothing until the handshake returns,
    and the transport timeout is ten minutes. The task has to be cancelled.
    """
    caplog.set_level(logging.DEBUG, logger=MCP_CLIENT_LOGGER)
    gateway = RecordingGateway(build_sdk_app())
    async with serve(gateway) as url:
        with patched_sdk(url):
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
