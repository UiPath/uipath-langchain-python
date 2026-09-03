"""Tests for the MCP 2 Streamable HTTP client integration."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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
        failed_tool_code: int = INVALID_REQUEST,
        block_initialize_on: int | None = None,
        fail_initialize_on: set[int] | None = None,
        rejected_session_ids: set[str] | None = None,
        repeat_session_header: bool = False,
        known_session_ids: set[str] | None = None,
        mints_new_session_on_initialize: bool = False,
    ) -> None:
        self.protocol_version = protocol_version
        self.failed_tool_calls = failed_tool_calls
        self.failed_tool_message = failed_tool_message
        # JSON-RPC code for the injected failure. The SDK synthesizes
        # CONNECTION_CLOSED client-side when a transport dies, which a mock
        # transport cannot stage cleanly; returning the code in a body drives
        # the same recovery decision through the real transport.
        self.failed_tool_code = failed_tool_code
        self.block_initialize_on = block_initialize_on
        self.fail_initialize_on = fail_initialize_on or set()
        self.rejected_session_ids = rejected_session_ids or set()
        self.repeat_session_header = repeat_session_header
        # Sessions this endpoint will route to. Seed it to stand in for a session
        # a previous process established and persisted externally.
        self.known_session_ids = set(known_session_ids or ())
        # Real servers route by the session header and mint only when it is
        # absent. Set this to model a server that ignores the header instead.
        self.mints_new_session_on_initialize = mints_new_session_on_initialize
        self.initialize_blocked = asyncio.Event()
        self.release_initialize = asyncio.Event()
        self.methods: list[str] = []
        self.request_headers: list[tuple[str, httpx2.Headers]] = []
        self.initialize_count = 0
        self.session_mint_count = 0
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
            session_id = self._session_for_initialize(request)
            if session_id is None:
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
            return self._json_response(
                body["id"],
                {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "test-server", "version": "1.0.0"},
                },
                headers={"mcp-session-id": session_id},
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
                headers=self._repeated_session_header(),
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
                                "code": self.failed_tool_code,
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
                headers=self._repeated_session_header(),
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

    def _session_for_initialize(self, request: httpx2.Request) -> str | None:
        """Resolve the session an ``initialize`` belongs to, or None to reject it.

        Mirrors the SDK server: a request naming a live session is handled inside
        it, and a new session is minted only when no session header is present.
        An unknown or expired ID is rejected rather than silently replaced.
        """
        incoming = request.headers.get("mcp-session-id")
        if incoming is not None and not self.mints_new_session_on_initialize:
            if (
                incoming in self.rejected_session_ids
                or incoming not in self.known_session_ids
            ):
                return None
            return incoming
        self.session_mint_count += 1
        minted = f"session-{self.session_mint_count}"
        self.known_session_ids.add(minted)
        return minted

    def _repeated_session_header(self) -> dict[str, str] | None:
        if not self.repeat_session_header or self.session_mint_count == 0:
            return None
        return {"mcp-session-id": f"session-{self.session_mint_count}"}


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
async def test_legacy_httpx_timeout_is_normalized_for_final_client(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """The pre-upgrade public timeout type remains accepted by McpClient."""
    endpoint = LegacyMcpEndpoint()
    legacy_timeout = httpx.Timeout(20, connect=1, read=2, write=3, pool=4)

    async with configured_client(
        mcp_resource_config,
        mock_uipath_sdk,
        endpoint,
        timeout=legacy_timeout,
    ) as client:
        await client.call_tool("test_tool", {"query": "test"})

        assert client._http_client is not None
        final_timeout = client._http_client.timeout
        assert final_timeout.connect == 1
        assert final_timeout.read == 2
        assert final_timeout.write == 3
        assert final_timeout.pool == 4


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
async def test_dropped_connection_resumes_the_persisted_session(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """``CONNECTION_CLOSED`` reopens the transport but keeps the session.

    A dropped connection is not the server's verdict on the session, so the
    retry must resume it. A fresh handshake would start a cold session and, for
    a store-backed ``SessionInfo``, throw away the persisted one for nothing.
    """
    endpoint = LegacyMcpEndpoint(
        failed_tool_calls=1,
        failed_tool_message="Connection closed",
        failed_tool_code=CONNECTION_CLOSED,
    )
    async with configured_client(
        mcp_resource_config, mock_uipath_sdk, endpoint
    ) as client:
        result = await client.call_tool("test_tool", {"query": "test"})

        assert result.structured_content == {"result": "Success from test_tool"}
        # The handshake ran again on the new transport -- inside the same session.
        assert endpoint.initialize_count == 2
        assert endpoint.session_mint_count == 1
        assert await client.get_session_id() == "session-1"
        assert [h["mcp-session-id"] for h in endpoint.headers_for("tools/call")] == [
            "session-1",
            "session-1",
        ]


@pytest.mark.asyncio
async def test_auto_mode_does_not_offer_a_minted_id_to_a_legacy_handshake(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """The probe is pinned, but a legacy server never issued that ID.

    This endpoint refuses any ``initialize`` naming a session it did not mint, so
    a handshake still carrying the affinity ID would be rejected first and the
    session established only on the clean retry.
    """
    endpoint = LegacyMcpEndpoint()
    async with configured_client(
        mcp_resource_config, mock_uipath_sdk, endpoint, protocol_mode="auto"
    ) as client:
        result = await client.call_tool("test_tool", {"query": "test"})

        assert result.structured_content == {"result": "Success from test_tool"}
        assert endpoint.methods[0] == "server/discover"
        assert endpoint.headers_for("server/discover")[0].get("mcp-session-id")
        # Accepted on the first attempt: the minted ID was withdrawn before it.
        assert endpoint.initialize_count == 1
        assert endpoint.headers_for("initialize")[0].get("mcp-session-id") is None
        assert await client.get_session_id() == "session-1"


@pytest.mark.asyncio
async def test_persisted_session_replaced_when_server_ignores_the_header(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """A server that mints on every handshake loses the session but stays usable."""
    endpoint = LegacyMcpEndpoint(
        known_session_ids={"persisted-session"},
        mints_new_session_on_initialize=True,
    )
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
        result = await client.call_tool("test_tool", {"query": "test"})

        assert result.structured_content == {"result": "Success from test_tool"}
        assert endpoint.initialize_count == 1
        assert await client.get_session_id() == "session-1"
        assert endpoint.headers_for("tools/call")[0]["mcp-session-id"] == "session-1"


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
        # The refused handshake for the stale ID, then the clean one.
        assert endpoint.initialize_count == 2
        assert endpoint.session_mint_count == 1
        assert await client.get_session_id() == "session-1"

    assert endpoint.delete_count == 1
    assert endpoint.headers_for("DELETE")[0]["mcp-session-id"] == "session-1"


@pytest.mark.asyncio
async def test_only_an_initialize_response_assigns_a_session_id(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """A session ID arriving on any other response must not be adopted.

    The SDK's own transport reads the ID only from the handshake. Persisting it
    from any response would let a proxy echoing the header replace a stored ID
    mid-connection -- including a client-minted routing key in ``auto`` mode,
    whose probe runs on the legacy wire before the era is known.
    """

    class RecordingSessionInfo(SessionInfo):
        def __init__(self) -> None:
            super().__init__()
            self.persisted: list[str | None] = []

        async def set_session_id(self, session_id: str | None) -> None:
            self.persisted.append(session_id)
            await super().set_session_id(session_id)

    session_info = RecordingSessionInfo()

    class RecordingFactory(SessionInfoFactory):
        def create_session(self, mcp_server: Any) -> SessionInfo:
            return session_info

    # The endpoint stamps a *different* session ID onto every non-initialize
    # response, the way a session-rewriting proxy would.
    endpoint = LegacyMcpEndpoint(repeat_session_header=True)
    endpoint._repeated_session_header = lambda: {"mcp-session-id": "proxy-injected"}  # type: ignore[method-assign]

    async with configured_client(
        mcp_resource_config,
        mock_uipath_sdk,
        endpoint,
        session_info_factory=RecordingFactory(),
    ) as client:
        await client.call_tool("test_tool", {"query": "test"})

        assert session_info.persisted == ["session-1"]
        assert await client.get_session_id() == "session-1"


@pytest.mark.asyncio
async def test_repeated_session_headers_do_not_repeat_external_persistence(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """An unchanged response header is not persisted after every MCP call."""

    class CountingSessionInfo(SessionInfo):
        def __init__(self) -> None:
            super().__init__()
            self.persisted_values: list[str | None] = []

        async def set_session_id(self, session_id: str | None) -> None:
            self.persisted_values.append(session_id)
            await super().set_session_id(session_id)

    session_info = CountingSessionInfo()

    class CountingFactory(SessionInfoFactory):
        def create_session(self, mcp_server: Any) -> SessionInfo:
            return session_info

    endpoint = LegacyMcpEndpoint(repeat_session_header=True)
    async with configured_client(
        mcp_resource_config,
        mock_uipath_sdk,
        endpoint,
        session_info_factory=CountingFactory(),
    ) as client:
        await client.list_tools()
        await client.call_tool("test_tool", {"query": "first"})
        await client.call_tool("test_tool", {"query": "second"})

        assert session_info.persisted_values == ["session-1"]


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
    open_connection = AsyncMock()

    with patch.object(client, "_open_connection", open_connection):
        await client._reinitialize_session(
            failed_session, error=MCPError(INVALID_REQUEST, "Session terminated")
        )

    failed_stack.aclose.assert_awaited_once()
    open_connection.assert_awaited_once()
    assert await client.get_session_id() is None


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
