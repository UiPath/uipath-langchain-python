"""Tests for McpClient against real MCP servers over streamable HTTP."""

import asyncio
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import httpx
import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCRequest
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.agent.models.agent import AgentMcpResourceConfig

from uipath_langchain.agent.tools.mcp import McpClient
from uipath_langchain.agent.tools.mcp.streamable_http import streamable_http_client

JSONRPC_MARKERS = (b"jsonrpc", b"tools/call", b"initialize")
SETTLE_SECONDS = 0.5
WAIT_SECONDS = 10

RedirectRules = dict[tuple[str, str], str]


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass
class RunningServer:
    port: int
    requests: list[RecordedRequest] = field(default_factory=list)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class RecordingApp:
    """ASGI wrapper that records requests and optionally redirects them.

    Redirect rules are keyed by (method, path); "*" matches any method.
    """

    def __init__(
        self,
        inner: Any,
        server: RunningServer,
        redirects: RedirectRules | None = None,
    ) -> None:
        self._inner = inner
        self._server = server
        self._redirects = redirects or {}

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._inner(scope, receive, send)
            return

        body = b""
        messages = []
        while True:
            message = await receive()
            messages.append(message)
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        self._server.requests.append(
            RecordedRequest(
                method=scope["method"],
                path=scope["path"],
                headers={k.decode().lower(): v.decode() for k, v in scope["headers"]},
                body=body,
            )
        )

        location = self._redirects.get(
            (scope["method"], scope["path"])
        ) or self._redirects.get(("*", scope["path"]))
        if location is not None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 307,
                    "headers": [(b"location", location.encode())],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return

        replay = iter(messages)

        async def replay_receive() -> Any:
            try:
                return next(replay)
            except StopIteration:
                return await receive()

        await self._inner(scope, replay_receive, send)


def _build_mcp_app() -> Any:
    mcp = FastMCP("real-http-test-server")

    @mcp.tool()
    def echo(text: str) -> str:
        """Echo the given text."""
        return f"echo:{text}"

    return mcp.streamable_http_app()


async def _not_found(scope: Any, receive: Any, send: Any) -> None:
    await send({"type": "http.response.start", "status": 404, "headers": []})
    await send({"type": "http.response.body", "body": b""})


class _ServerThread:
    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.port: int = self.sock.getsockname()[1]
        self.server: uvicorn.Server | None = None
        self.thread: threading.Thread | None = None

    def start(self, app: Any, lifespan: Literal["on", "off"] = "on") -> None:
        config = uvicorn.Config(
            app, log_level="warning", lifespan=lifespan, timeout_graceful_shutdown=2
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True
        )
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("uvicorn exited before it started")
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start in time")
            time.sleep(0.02)

    def stop(self) -> None:
        try:
            if self.server is not None:
                self.server.should_exit = True
            if self.thread is not None:
                self.thread.join(timeout=15)
                if self.thread.is_alive():
                    raise RuntimeError("uvicorn did not shut down in time")
        finally:
            self.sock.close()


@pytest.fixture
def start_server() -> Iterator[Callable[..., RunningServer]]:
    runners: list[_ServerThread] = []

    def _start(
        inner: Any,
        redirects: Callable[[RunningServer], RedirectRules] | None = None,
        lifespan: Literal["on", "off"] = "on",
    ) -> RunningServer:
        runner = _ServerThread()
        runners.append(runner)
        server = RunningServer(port=runner.port)
        rules = redirects(server) if redirects is not None else None
        runner.start(RecordingApp(inner, server, rules), lifespan=lifespan)
        return server

    try:
        yield _start
    finally:
        errors: list[Exception] = []
        for runner in runners:
            try:
                runner.stop()
            except RuntimeError as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


@pytest.fixture
def mcp_server(start_server: Callable[..., RunningServer]) -> RunningServer:
    return start_server(
        _build_mcp_app(),
        lambda server: {("*", "/mcp-moved"): f"{server.origin}/mcp"},
    )


@pytest.fixture
def redirecting_server(
    start_server: Callable[..., RunningServer], mcp_server: RunningServer
) -> RunningServer:
    return start_server(
        _not_found,
        lambda _: {("*", "/mcp"): f"{mcp_server.origin}/mcp"},
        lifespan="off",
    )


@pytest.fixture
def mcp_resource_config() -> AgentMcpResourceConfig:
    return AgentMcpResourceConfig(
        name="real_server",
        description="Real MCP server",
        folder_path="/Shared/TestFolder",
        slug="real-server",
        available_tools=[],
    )


def _sdk_for(url: str) -> MagicMock:
    sdk = MagicMock()
    server = MagicMock()
    server.mcp_url = url
    sdk.mcp.retrieve_async = AsyncMock(return_value=server)
    sdk._config.secret = "test-secret-token"
    return sdk


def _leaked_to(server: RunningServer) -> list[RecordedRequest]:
    return [
        r
        for r in server.requests
        if "mcp-session-id" in r.headers
        or "authorization" in r.headers
        or any(marker in r.body for marker in JSONRPC_MARKERS)
    ]


def _flatten(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for sub in error.exceptions for leaf in _flatten(sub)]
    return [error]


async def _wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        await asyncio.sleep(0.02)


async def _initialize_against(url: str) -> list[BaseException]:
    initialize = JSONRPCMessage(
        JSONRPCRequest(
            jsonrpc="2.0",
            id=1,
            method="initialize",
            params={
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            },
        )
    )
    kwargs = get_httpx_client_kwargs(
        headers={"Authorization": "Bearer test-secret-token"}
    )
    try:
        with anyio.fail_after(30):
            async with httpx.AsyncClient(**kwargs) as http_client:
                async with streamable_http_client(
                    url, http_client=http_client, terminate_on_close=False
                ) as (read_stream, write_stream):
                    await write_stream.send(SessionMessage(initialize))
                    await read_stream.receive()
    except BaseExceptionGroup as group:
        return _flatten(group)
    raise AssertionError("initialize unexpectedly succeeded")


class TestMcpClientRealHttp:
    """Test McpClient against real FastMCP servers over streamable HTTP."""

    async def test_round_trip_and_session_termination(
        self,
        mcp_resource_config: AgentMcpResourceConfig,
        mcp_server: RunningServer,
    ) -> None:
        client = McpClient(config=mcp_resource_config)

        with patch(
            "uipath.platform.UiPath",
            return_value=_sdk_for(f"{mcp_server.origin}/mcp"),
        ):
            tools = await client.list_tools()
            result = await client.call_tool("echo", {"text": "hi"})

        assert [t.name for t in tools.tools] == ["echo"]
        assert result.isError is False
        assert result.content[0].text == "echo:hi"  # type: ignore[union-attr]

        session_id = await client.get_session_id()
        assert session_id

        methods = [r.method for r in mcp_server.requests]
        assert "POST" in methods
        assert "DELETE" not in methods

        post_requests = [r for r in mcp_server.requests if r.method == "POST"]
        assert post_requests[0].headers.get("mcp-session-id") is None
        assert all(
            r.headers.get("mcp-session-id") == session_id for r in post_requests[1:]
        )
        assert all(
            r.headers.get("authorization") == "Bearer test-secret-token"
            for r in post_requests
        )

        await client.dispose()

        delete_requests = [r for r in mcp_server.requests if r.method == "DELETE"]
        assert len(delete_requests) == 1
        assert delete_requests[0].headers.get("mcp-session-id") == session_id

    async def test_cross_origin_post_redirect_is_not_followed(
        self,
        mcp_server: RunningServer,
        redirecting_server: RunningServer,
    ) -> None:
        errors = await _initialize_against(f"{redirecting_server.origin}/mcp")

        status_errors = [e for e in errors if isinstance(e, httpx.HTTPStatusError)]
        assert len(status_errors) == 1
        assert status_errors[0].response.status_code == 307
        assert [r.method for r in redirecting_server.requests] == ["POST"]
        assert _leaked_to(mcp_server) == []
        assert mcp_server.requests == []

    async def test_cross_origin_get_stream_redirect_is_not_followed(
        self,
        mcp_resource_config: AgentMcpResourceConfig,
        mcp_server: RunningServer,
        start_server: Callable[..., RunningServer],
    ) -> None:
        front = start_server(
            _build_mcp_app(),
            lambda _: {("GET", "/mcp"): f"{mcp_server.origin}/mcp"},
        )
        client = McpClient(config=mcp_resource_config)

        with patch(
            "uipath.platform.UiPath",
            return_value=_sdk_for(f"{front.origin}/mcp"),
        ):
            tools = await client.list_tools()
            await _wait_until(lambda: any(r.method == "GET" for r in front.requests))
            await asyncio.sleep(SETTLE_SECONDS)

        await client.dispose()

        assert [t.name for t in tools.tools] == ["echo"]
        assert _leaked_to(mcp_server) == []
        assert mcp_server.requests == []

    async def test_cross_origin_delete_redirect_is_not_followed(
        self,
        mcp_resource_config: AgentMcpResourceConfig,
        mcp_server: RunningServer,
        start_server: Callable[..., RunningServer],
    ) -> None:
        front = start_server(
            _build_mcp_app(),
            lambda _: {("DELETE", "/mcp"): f"{mcp_server.origin}/mcp"},
        )
        client = McpClient(config=mcp_resource_config)

        with patch(
            "uipath.platform.UiPath",
            return_value=_sdk_for(f"{front.origin}/mcp"),
        ):
            await client.list_tools()

        await client.dispose()

        assert [r.method for r in front.requests].count("DELETE") == 1
        assert _leaked_to(mcp_server) == []
        assert mcp_server.requests == []

    async def test_same_origin_redirect_is_followed(
        self,
        mcp_resource_config: AgentMcpResourceConfig,
        mcp_server: RunningServer,
    ) -> None:
        client = McpClient(config=mcp_resource_config)

        with patch(
            "uipath.platform.UiPath",
            return_value=_sdk_for(f"{mcp_server.origin}/mcp-moved"),
        ):
            result = await client.call_tool("echo", {"text": "redirected"})

        assert result.content[0].text == "echo:redirected"  # type: ignore[union-attr]
        assert any(r.path == "/mcp-moved" for r in mcp_server.requests)
        assert any(r.path == "/mcp" and r.method == "POST" for r in mcp_server.requests)

        await client.dispose()

    async def test_same_origin_get_stream_redirect_is_followed(
        self,
        mcp_resource_config: AgentMcpResourceConfig,
        mcp_server: RunningServer,
    ) -> None:
        client = McpClient(config=mcp_resource_config)

        with patch(
            "uipath.platform.UiPath",
            return_value=_sdk_for(f"{mcp_server.origin}/mcp-moved"),
        ):
            await client.list_tools()
            session_id = await client.get_session_id()

            def followed() -> bool:
                return any(
                    r.method == "GET" and r.path == "/mcp" for r in mcp_server.requests
                )

            await _wait_until(followed)

        await client.dispose()

        get_request = next(
            r for r in mcp_server.requests if r.method == "GET" and r.path == "/mcp"
        )
        assert get_request.headers.get("authorization") == "Bearer test-secret-token"
        assert get_request.headers.get("mcp-session-id") == session_id

    async def test_same_origin_delete_redirect_is_followed(
        self,
        mcp_resource_config: AgentMcpResourceConfig,
        start_server: Callable[..., RunningServer],
    ) -> None:
        front = start_server(
            _build_mcp_app(),
            lambda server: {("DELETE", "/mcp"): f"{server.origin}/mcp-deleted"},
        )
        client = McpClient(config=mcp_resource_config)

        with patch(
            "uipath.platform.UiPath",
            return_value=_sdk_for(f"{front.origin}/mcp"),
        ):
            await client.list_tools()
            session_id = await client.get_session_id()

        await client.dispose()

        followed = [
            r
            for r in front.requests
            if r.method == "DELETE" and r.path == "/mcp-deleted"
        ]
        assert len(followed) == 1
        assert followed[0].headers.get("authorization") == "Bearer test-secret-token"
        assert followed[0].headers.get("mcp-session-id") == session_id
