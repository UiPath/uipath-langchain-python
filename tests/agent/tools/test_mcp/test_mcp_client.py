"""Tests for the MCP 2 Streamable HTTP client integration."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import CONNECTION_CLOSED, INVALID_REQUEST, METHOD_NOT_FOUND
from uipath.agent.models.agent import AgentMcpResourceConfig, AgentMcpTool

from uipath_langchain.agent.tools.mcp import McpClient, SessionInfo, SessionInfoFactory


class LegacyMcpEndpoint:
    """Small Streamable HTTP endpoint used to exercise the real MCP SDK transport."""

    def __init__(
        self,
        protocol_version: str = "2025-11-25",
        *,
        failed_tool_calls: int = 0,
        failed_tool_message: str | None = None,
        block_initialize_on: int | None = None,
        fail_initialize_on: set[int] | None = None,
        rejected_session_ids: set[str] | None = None,
    ) -> None:
        self.protocol_version = protocol_version
        self.failed_tool_calls = failed_tool_calls
        self.failed_tool_message = failed_tool_message
        self.block_initialize_on = block_initialize_on
        self.fail_initialize_on = fail_initialize_on or set()
        self.rejected_session_ids = rejected_session_ids or set()
        self.initialize_blocked = asyncio.Event()
        self.release_initialize = asyncio.Event()
        self.methods: list[str] = []
        self.request_headers: list[tuple[str, httpx2.Headers]] = []
        self.initialize_count = 0
        self.tool_call_count = 0
        self.delete_count = 0
        self.transport = httpx2.MockTransport(self.handle)

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        """Return protocol-correct JSON responses for the MCP methods under test."""
        if request.method == "GET":
            return httpx2.Response(405)
        if request.method == "DELETE":
            self.delete_count += 1
            self.request_headers.append(("DELETE", request.headers))
            return httpx2.Response(204)

        body = json.loads(request.content)
        method = body["method"]
        self.methods.append(method)
        self.request_headers.append((method, request.headers))

        if method == "initialize":
            self.initialize_count += 1
            if self.initialize_count == self.block_initialize_on:
                self.initialize_blocked.set()
                await self.release_initialize.wait()
            if self.initialize_count in self.fail_initialize_on:
                return httpx2.Response(
                    400,
                    headers={"content-type": "application/json"},
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "error": {
                            "code": INVALID_REQUEST,
                            "message": "Replacement initialization failed",
                        },
                    },
                )
            return self._json_response(
                body["id"],
                {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "test-server", "version": "1.0.0"},
                },
                headers={"mcp-session-id": f"session-{self.initialize_count}"},
            )
        if method == "notifications/initialized":
            return httpx2.Response(202)
        if method == "ping":
            if request.headers.get("mcp-session-id") in self.rejected_session_ids:
                return httpx2.Response(
                    404,
                    headers={"content-type": "application/json"},
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "error": {
                            "code": INVALID_REQUEST,
                            "message": "Session not found",
                        },
                    },
                )
            if request.headers.get("mcp-protocol-version") != self.protocol_version:
                return httpx2.Response(
                    400,
                    headers={"content-type": "application/json"},
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "error": {
                            "code": INVALID_REQUEST,
                            "message": "Unsupported protocol version",
                        },
                    },
                )
            return self._json_response(body["id"], {})
        if method == "tools/list":
            return self._json_response(
                body["id"],
                {
                    "tools": [
                        {
                            "name": "test_tool",
                            "description": "A test tool",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                            "outputSchema": {
                                "type": "object",
                                "properties": {"result": {"type": "string"}},
                            },
                        }
                    ]
                },
            )
        if method == "tools/call":
            self.tool_call_count += 1
            if self.tool_call_count <= self.failed_tool_calls:
                if self.failed_tool_message is not None:
                    return httpx2.Response(
                        404,
                        headers={"content-type": "application/json"},
                        json={
                            "jsonrpc": "2.0",
                            "id": body["id"],
                            "error": {
                                "code": INVALID_REQUEST,
                                "message": self.failed_tool_message,
                            },
                        },
                    )
                return httpx2.Response(404)
            result = {"result": f"Success from {body['params']['name']}"}
            return self._json_response(
                body["id"],
                {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                    "structuredContent": result,
                    "isError": False,
                },
            )
        return httpx2.Response(
            404,
            json={
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": METHOD_NOT_FOUND, "message": "Method not found"},
            },
        )

    @staticmethod
    def _json_response(
        request_id: int,
        result: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx2.Response:
        response_headers = {"content-type": "application/json"}
        response_headers.update(headers or {})
        return httpx2.Response(
            200,
            headers=response_headers,
            json={"jsonrpc": "2.0", "id": request_id, "result": result},
        )

    def headers_for(self, method: str) -> list[httpx2.Headers]:
        """Return captured headers for one protocol or HTTP method."""
        return [headers for name, headers in self.request_headers if name == method]


@pytest.fixture
def mcp_resource_config() -> AgentMcpResourceConfig:
    """Create a minimal MCP resource config for testing."""
    return AgentMcpResourceConfig(
        name="test_server",
        description="Test MCP server",
        folder_path="/Shared/TestFolder",
        slug="test-server",
        available_tools=[
            AgentMcpTool(
                name="test_tool",
                description="A test tool",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
    )


@pytest.fixture
def mock_uipath_sdk() -> MagicMock:
    """Create a mock UiPath SDK and resolved MCP server."""
    sdk = MagicMock()
    server = MagicMock()
    server.mcp_url = "https://test.uipath.com/mcp"
    server.slug = "test-server"
    server.folder_key = "folder-key"
    sdk.mcp.retrieve_async = AsyncMock(return_value=server)
    sdk._config.secret = "test-secret-token"
    return sdk


@asynccontextmanager
async def configured_client(
    config: AgentMcpResourceConfig,
    sdk: MagicMock,
    endpoint: LegacyMcpEndpoint,
    **kwargs: Any,
) -> AsyncIterator[McpClient]:
    """Build an McpClient whose real HTTP client uses the mock transport."""
    client = McpClient(config=config, **kwargs)
    http_kwargs = {
        "headers": {"Authorization": "Bearer test-secret-token"},
        "transport": endpoint.transport,
        "follow_redirects": True,
    }
    with (
        patch("uipath.platform.UiPath", return_value=sdk),
        patch(
            "uipath_langchain.agent.tools.mcp.mcp_client.get_httpx_client_kwargs",
            return_value=http_kwargs,
        ),
    ):
        try:
            yield client
        finally:
            await client.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", ["2025-03-26", "2025-06-18", "2025-11-25"])
async def test_negotiates_supported_legacy_protocol_versions(
    protocol_version: str,
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """MCP 2's low-level initialize handshake remains compatible with 2025 servers."""
    endpoint = LegacyMcpEndpoint(protocol_version)
    async with configured_client(
        mcp_resource_config, mock_uipath_sdk, endpoint
    ) as client:
        result = await client.call_tool("test_tool", {"query": "test"})

        assert result.structured_content == {"result": "Success from test_tool"}
        assert endpoint.initialize_count == 1
        assert endpoint.tool_call_count == 1
        assert await client.get_session_id() == "session-1"
        assert endpoint.headers_for("tools/call")[0]["mcp-session-id"] == "session-1"
        assert (
            endpoint.headers_for("tools/call")[0]["mcp-protocol-version"]
            == protocol_version
        )


@pytest.mark.asyncio
async def test_replaces_transport_and_session_after_404(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """A terminated session gets a fresh handshake while reusing its HTTP client."""
    endpoint = LegacyMcpEndpoint(failed_tool_calls=1)
    async with configured_client(
        mcp_resource_config, mock_uipath_sdk, endpoint
    ) as client:
        result = await client.call_tool("test_tool", {"query": "test"})

        assert result.structured_content == {"result": "Success from test_tool"}
        assert endpoint.initialize_count == 2
        assert endpoint.tool_call_count == 2
        assert endpoint.delete_count == 1
        assert await client.get_session_id() == "session-2"
        assert [h["mcp-session-id"] for h in endpoint.headers_for("tools/call")] == [
            "session-1",
            "session-2",
        ]


@pytest.mark.asyncio
async def test_replaces_session_after_official_session_not_found_error(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """The SDK server's canonical expired-session response triggers recovery."""
    endpoint = LegacyMcpEndpoint(
        failed_tool_calls=1,
        failed_tool_message="Session not found",
    )
    async with configured_client(
        mcp_resource_config, mock_uipath_sdk, endpoint
    ) as client:
        result = await client.call_tool("test_tool", {"query": "test"})

        assert result.structured_content == {"result": "Success from test_tool"}
        assert endpoint.initialize_count == 2
        assert endpoint.tool_call_count == 2
        assert await client.get_session_id() == "session-2"


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", ["2025-03-26", "2025-06-18", "2025-11-25"])
async def test_persisted_session_is_reused_without_initialize(
    protocol_version: str,
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """The UiPath SessionInfo extension injects an externally restored session ID."""
    endpoint = LegacyMcpEndpoint(protocol_version)
    session_info = SessionInfo("persisted-session")

    class PersistedFactory(SessionInfoFactory):
        def create_session(self, mcp_server: Any) -> SessionInfo:
            return session_info

    async with configured_client(
        mcp_resource_config,
        mock_uipath_sdk,
        endpoint,
        session_info_factory=PersistedFactory(),
    ) as client:
        await client.call_tool("test_tool", {"query": "test"})

        assert endpoint.initialize_count == 0
        assert endpoint.headers_for("tools/call")[0]["mcp-session-id"] == (
            "persisted-session"
        )
        assert (
            endpoint.headers_for("tools/call")[0]["mcp-protocol-version"]
            == protocol_version
        )

    assert endpoint.delete_count == 1
    assert endpoint.headers_for("DELETE")[0]["mcp-session-id"] == "persisted-session"
    assert endpoint.headers_for("DELETE")[0]["mcp-protocol-version"] == protocol_version


@pytest.mark.asyncio
async def test_rejected_persisted_session_is_initialized_and_deleted_once(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """A stale restored ID is replaced and only the fresh SDK session is deleted."""
    endpoint = LegacyMcpEndpoint(rejected_session_ids={"expired-session"})
    session_info = SessionInfo("expired-session")

    class PersistedFactory(SessionInfoFactory):
        def create_session(self, mcp_server: Any) -> SessionInfo:
            return session_info

    async with configured_client(
        mcp_resource_config,
        mock_uipath_sdk,
        endpoint,
        session_info_factory=PersistedFactory(),
    ) as client:
        result = await client.call_tool("test_tool", {"query": "test"})

        assert result.structured_content == {"result": "Success from test_tool"}
        assert endpoint.initialize_count == 1
        assert await client.get_session_id() == "session-1"

    assert endpoint.delete_count == 1
    assert endpoint.headers_for("DELETE")[0]["mcp-session-id"] == "session-1"


@pytest.mark.asyncio
async def test_expired_persisted_session_falls_back_to_fresh_initialize(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """A 404 for an externally restored session is treated as session expiry."""
    endpoint = LegacyMcpEndpoint(failed_tool_calls=1)
    session_info = SessionInfo("expired-session")

    class PersistedFactory(SessionInfoFactory):
        def create_session(self, mcp_server: Any) -> SessionInfo:
            return session_info

    async with configured_client(
        mcp_resource_config,
        mock_uipath_sdk,
        endpoint,
        session_info_factory=PersistedFactory(),
    ) as client:
        await client.call_tool("test_tool", {"query": "test"})

        assert endpoint.initialize_count == 1
        assert endpoint.tool_call_count == 2
        assert await client.get_session_id() == "session-1"
        assert [h["mcp-session-id"] for h in endpoint.headers_for("tools/call")] == [
            "expired-session",
            "session-1",
        ]


@pytest.mark.asyncio
async def test_max_retries_exceeded_raises_mcp_error(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """Repeated session termination is surfaced after the configured retry."""
    endpoint = LegacyMcpEndpoint(failed_tool_calls=2)
    async with configured_client(
        mcp_resource_config, mock_uipath_sdk, endpoint, max_retries=1
    ) as client:
        with pytest.raises(MCPError) as exc_info:
            await client.call_tool("test_tool", {"query": "test"})

        assert exc_info.value.code == INVALID_REQUEST
        assert endpoint.initialize_count == 2
        assert endpoint.tool_call_count == 2


@pytest.mark.asyncio
async def test_concurrent_recovery_does_not_replace_a_new_session(
    mcp_resource_config: AgentMcpResourceConfig,
) -> None:
    """A late failure from an old session must not tear down its replacement."""
    client = McpClient(config=mcp_resource_config)
    failed_session = MagicMock()
    replacement_session = MagicMock()
    client._client_initialized = True
    client._session = replacement_session
    client._session_info = SessionInfo("replacement-id")
    open_connection = AsyncMock()

    with patch.object(client, "_open_connection", open_connection):
        await client._reinitialize_session(failed_session)

    assert client._session is replacement_session
    assert await client.get_session_id() == "replacement-id"
    open_connection.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_continues_when_failed_connection_cleanup_raises(
    mcp_resource_config: AgentMcpResourceConfig,
) -> None:
    """Closing the failed stack cannot mask recovery of the MCP connection."""
    client = McpClient(config=mcp_resource_config)
    failed_session = MagicMock()
    failed_stack = MagicMock()
    failed_stack.aclose = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    client._client_initialized = True
    client._session = failed_session
    client._connection_stack = failed_stack
    client._session_info = SessionInfo("failed-session")
    client._session_info.protocol_version = "2025-03-26"
    open_connection = AsyncMock()

    with patch.object(client, "_open_connection", open_connection):
        await client._reinitialize_session(failed_session)

    failed_stack.aclose.assert_awaited_once()
    open_connection.assert_awaited_once()
    assert await client.get_session_id() is None
    assert client._session_info.protocol_version is None


@pytest.mark.asyncio
async def test_concurrent_call_waits_for_recovery_initialization(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """A caller cannot use a replacement session before its handshake finishes."""
    endpoint = LegacyMcpEndpoint(
        failed_tool_calls=1,
        block_initialize_on=2,
    )

    async with configured_client(
        mcp_resource_config, mock_uipath_sdk, endpoint
    ) as client:
        recovery_call = asyncio.create_task(
            client.call_tool("test_tool", {"query": "recover"})
        )
        await asyncio.wait_for(endpoint.initialize_blocked.wait(), timeout=2)

        concurrent_list = asyncio.create_task(client.list_tools(force_refresh=True))
        await asyncio.sleep(0)
        assert not concurrent_list.done()

        endpoint.release_initialize.set()
        call_result, list_result = await asyncio.gather(
            recovery_call,
            concurrent_list,
        )

        assert call_result.structured_content == {"result": "Success from test_tool"}
        assert list_result.tools[0].name == "test_tool"


@pytest.mark.asyncio
async def test_later_call_recovers_after_replacement_initialization_failure(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """A failed replacement does not strand the client without a session."""
    endpoint = LegacyMcpEndpoint(
        failed_tool_calls=1,
        fail_initialize_on={2},
    )

    async with configured_client(
        mcp_resource_config, mock_uipath_sdk, endpoint
    ) as client:
        with pytest.raises(MCPError, match="Replacement initialization failed"):
            await client.call_tool("test_tool", {"query": "first"})

        assert client.is_client_initialized
        assert client._session is None

        result = await client.call_tool("test_tool", {"query": "second"})

        assert result.structured_content == {"result": "Success from test_tool"}
        assert endpoint.initialize_count == 3


@pytest.mark.asyncio
async def test_list_tools_cache_and_force_refresh(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """Tool discovery still caches normally across the SDK upgrade."""
    endpoint = LegacyMcpEndpoint()
    async with configured_client(
        mcp_resource_config, mock_uipath_sdk, endpoint
    ) as client:
        first = await client.list_tools()
        second = await client.list_tools()
        refreshed = await client.list_tools(force_refresh=True)

        assert first is second
        assert refreshed.tools[0].input_schema["required"] == ["query"]
        assert endpoint.methods.count("tools/list") == 2


@pytest.mark.asyncio
async def test_dispose_allows_client_reuse(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """Disposal closes resources and a later call builds a new client/session."""
    endpoint = LegacyMcpEndpoint()
    async with configured_client(
        mcp_resource_config, mock_uipath_sdk, endpoint
    ) as client:
        await client.call_tool("test_tool", {"query": "first"})
        await client.dispose()

        assert not client.is_client_initialized
        assert client._session is None
        assert await client.get_session_id() is None

        await client.call_tool("test_tool", {"query": "second"})
        assert endpoint.initialize_count == 2
        assert client.is_client_initialized


@pytest.mark.asyncio
async def test_raises_on_missing_mcp_url(
    mcp_resource_config: AgentMcpResourceConfig,
) -> None:
    """A server registration without an endpoint fails before allocating HTTP state."""
    sdk = MagicMock()
    server = MagicMock()
    server.mcp_url = None
    sdk.mcp.retrieve_async = AsyncMock(return_value=server)

    client = McpClient(config=mcp_resource_config)
    with patch("uipath.platform.UiPath", return_value=sdk):
        with pytest.raises(ValueError, match="has no URL configured"):
            await client.call_tool("test_tool", {"query": "test"})


@pytest.mark.asyncio
async def test_initialization_failure_cleans_state_and_allows_retry(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """A failed handshake releases both stacks and leaves the client reusable."""
    endpoint = LegacyMcpEndpoint()

    async with configured_client(
        mcp_resource_config, mock_uipath_sdk, endpoint
    ) as client:
        with patch.object(
            client,
            "_initialize_session",
            AsyncMock(side_effect=RuntimeError("initialize failed")),
        ):
            with pytest.raises(RuntimeError, match="initialize failed"):
                await client.call_tool("test_tool", {"query": "first"})

        assert client._stack is None
        assert client._connection_stack is None
        assert client._http_client is None
        assert client._session_info is None
        assert client._session is None
        assert not client.is_client_initialized

        result = await client.call_tool("test_tool", {"query": "second"})

        assert result.structured_content == {"result": "Success from test_tool"}
        assert endpoint.initialize_count == 1


def test_only_session_specific_invalid_request_is_retryable() -> None:
    """Ordinary INVALID_REQUEST errors must not be mislabeled as disconnects."""
    assert McpClient.is_session_error(
        MCPError(code=CONNECTION_CLOSED, message="Connection closed")
    )
    assert McpClient.is_session_error(
        MCPError(code=INVALID_REQUEST, message="Session terminated")
    )
    assert McpClient.is_session_error(
        MCPError(code=INVALID_REQUEST, message="Session not found")
    )
    assert not McpClient.is_session_error(
        MCPError(code=INVALID_REQUEST, message="Invalid request parameters")
    )
