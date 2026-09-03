"""Per-era MCP protocol policy, and the two servers a real one cannot imitate.

Negotiation, affinity, retry semantics and wire identity are all driven against
real servers over real HTTP in ``test_mcp_client_real_http.py``. What is left
here is what a cooperative server cannot express:

* ``auto`` sending a restored ID before the era is resolved, against a server
  with no discovery endpoint.
* A proxy echoing ``mcp-session-id`` back in the modern era, where the client
  owns the value and nothing on the wire may overwrite it.

plus the pure-function policy matrices -- ``is_recoverable``, ``reset``, and
``build_protocol_strategy`` -- which need no server at all.
"""

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
from uipath_langchain.agent.tools.mcp.protocol_strategy import (
    AutoStrategy,
    LegacyHandshakeStrategy,
    ModernDiscoveryStrategy,
    build_protocol_strategy,
)
from uipath_langchain.agent.tools.mcp.streamable_http import MCP_SESSION_ID

MODERN_VERSION = "2026-07-28"
LEGACY_VERSION = "2025-11-25"

TOOL_SCHEMA = {
    "name": "test_tool",
    "description": "A test tool",
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    "outputSchema": {"type": "object", "properties": {"result": {"type": "string"}}},
}


class EraMcpEndpoint:
    """Streamable HTTP endpoint that can offer either era, or both.

    A modern-only server answers ``server/discover`` and has no ``initialize``
    endpoint at all; a legacy-only server is the reverse. ``auto`` has to pick
    correctly against either.
    """

    def __init__(
        self,
        *,
        supports_discover: bool = True,
        supports_initialize: bool = False,
        echo_session_id: str | None = None,
    ) -> None:
        self.supports_discover = supports_discover
        self.supports_initialize = supports_initialize
        # Stands in for a proxy that returns mcp-session-id even in the modern
        # era, where the client owns the value.
        self.echo_session_id = echo_session_id
        self.methods: list[str] = []
        self.request_headers: list[tuple[str, httpx2.Headers]] = []
        self.discover_count = 0
        self.initialize_count = 0
        self.tool_call_count = 0
        self.delete_count = 0
        self.transport = httpx2.MockTransport(self.handle)

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        """Answer the MCP methods under test for whichever era is enabled."""
        if request.method == "GET":
            return httpx2.Response(405)
        if request.method == "DELETE":
            self.delete_count += 1
            self.request_headers.append(("DELETE", request.headers))
            return httpx2.Response(204)

        body = json.loads(request.content)
        method = body["method"]
        params = body.get("params") or {}
        self.methods.append(method)
        self.request_headers.append((method, request.headers))

        if method == "server/discover":
            self.discover_count += 1
            if not self.supports_discover:
                return self._error(body["id"], METHOD_NOT_FOUND, "Not Found", 404)
            return self._result(
                body["id"],
                {
                    "supportedVersions": [MODERN_VERSION],
                    "capabilities": {"tools": {"listChanged": True}},
                    "resultType": "complete",
                },
                headers=(
                    {MCP_SESSION_ID: self.echo_session_id}
                    if self.echo_session_id
                    else None
                ),
            )
        if method == "initialize":
            self.initialize_count += 1
            if not self.supports_initialize:
                return self._error(body["id"], METHOD_NOT_FOUND, "Not Found", 404)
            return self._result(
                body["id"],
                {
                    "protocolVersion": LEGACY_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "test-server", "version": "1.0.0"},
                },
                headers={MCP_SESSION_ID: "server-session-1"},
            )
        if method == "notifications/initialized":
            return httpx2.Response(202)
        if method == "tools/list":
            # ``resultType`` is required on the 2026-07-28 wire, and a cacheable
            # result carries its cache directives too. Older peers ignore both.
            return self._result(
                body["id"],
                {
                    "tools": [TOOL_SCHEMA],
                    "resultType": "complete",
                    "ttlMs": 0,
                    "cacheScope": "private",
                },
            )
        if method == "tools/call":
            self.tool_call_count += 1
            result = {"result": f"Success from {params['name']}"}
            return self._result(
                body["id"],
                {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                    "structuredContent": result,
                    "isError": False,
                    "resultType": "complete",
                },
            )
        return self._error(body.get("id"), METHOD_NOT_FOUND, "Method not found", 404)

    @staticmethod
    def _result(
        request_id: Any,
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

    @staticmethod
    def _error(
        request_id: Any, code: int, message: str, status: int
    ) -> httpx2.Response:
        return httpx2.Response(
            status,
            headers={"content-type": "application/json"},
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            },
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
    endpoint: EraMcpEndpoint,
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


def _pinned_session_info(session_info: SessionInfo) -> SessionInfoFactory:
    class PinnedFactory(SessionInfoFactory):
        def create_session(self, mcp_server: Any) -> SessionInfo:
            return session_info

    return PinnedFactory()


@pytest.mark.asyncio
async def test_auto_mode_sends_a_restored_id_before_the_era_resolves(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """A persisted ID goes out on the shared header before the era is known.

    The transport opens before negotiation, and nothing distinguishes a
    server-minted session ID from a client-minted affinity ID. Because both eras
    use ``mcp-session-id``, no disambiguation is needed.
    """
    session_info = SessionInfo("ambiguous-id")
    endpoint = EraMcpEndpoint(supports_discover=False, supports_initialize=True)
    async with configured_client(
        mcp_resource_config,
        mock_uipath_sdk,
        endpoint,
        session_info_factory=_pinned_session_info(session_info),
        protocol_mode="auto",
    ) as client:
        await client.call_tool("test_tool", {"query": "test"})

        assert endpoint.headers_for("server/discover")[0][MCP_SESSION_ID] == (
            "ambiguous-id"
        )


@pytest.mark.asyncio
async def test_modern_mode_ignores_a_server_assigned_session_id(
    mcp_resource_config: AgentMcpResourceConfig,
    mock_uipath_sdk: MagicMock,
) -> None:
    """A response header must not overwrite the client-minted routing key.

    The two eras share ``mcp-session-id``, so a proxy or gateway echoing it back
    could otherwise replace the affinity ID mid-connection and scatter the
    remaining requests across instances.
    """
    session_info = SessionInfo("affinity-keep-me")
    endpoint = EraMcpEndpoint(echo_session_id="server-would-assign-this")
    async with configured_client(
        mcp_resource_config,
        mock_uipath_sdk,
        endpoint,
        session_info_factory=_pinned_session_info(session_info),
        protocol_mode="modern",
    ) as client:
        await client.call_tool("test_tool", {"query": "test"})

    assert await session_info.get_session_id() == "affinity-keep-me"
    sent = [
        headers.get(MCP_SESSION_ID)
        for method, headers in endpoint.request_headers
        if method != "DELETE"
    ]
    assert sent and all(value == "affinity-keep-me" for value in sent)
    # Filtering DELETE out above would hide a teardown, so count them directly:
    # an echoed ID adopted by the transport would surface here.
    assert endpoint.delete_count == 0


def _error(code: int, message: str) -> MCPError:
    return MCPError(code, message)


def test_legacy_recovers_from_session_loss_but_not_from_bad_requests() -> None:
    """Only a lost session justifies replacing a legacy connection."""
    strategy = LegacyHandshakeStrategy()

    assert strategy.is_recoverable(_error(CONNECTION_CLOSED, "Connection closed"), None)
    assert strategy.is_recoverable(_error(INVALID_REQUEST, "Session terminated"), None)
    assert strategy.is_recoverable(_error(INVALID_REQUEST, "Session not found"), None)
    assert not strategy.is_recoverable(_error(INVALID_REQUEST, "Bad params"), None)
    # A bare 404 is only a lost session while a restored ID is still in play.
    assert strategy.is_recoverable(_error(METHOD_NOT_FOUND, "Not Found"), "restored")
    assert not strategy.is_recoverable(_error(METHOD_NOT_FOUND, "Not Found"), None)


def test_modern_recovers_only_from_a_dropped_connection() -> None:
    """Every modern request is self-contained, so nothing else is retryable."""
    strategy = ModernDiscoveryStrategy()

    assert strategy.is_recoverable(_error(CONNECTION_CLOSED, "Connection closed"), None)
    assert not strategy.is_recoverable(
        _error(INVALID_REQUEST, "Session terminated"), "restored"
    )
    assert not strategy.is_recoverable(
        _error(METHOD_NOT_FOUND, "Not Found"), "restored"
    )


@pytest.mark.asyncio
async def test_modern_reset_keeps_the_affinity_id() -> None:
    """Discarding the routing ID on failure would abandon the warm instance."""
    strategy = ModernDiscoveryStrategy()
    session_info = SessionInfo("affinity-1")

    await strategy.reset(session_info)

    assert await session_info.get_session_id() == "affinity-1"


@pytest.mark.asyncio
async def test_legacy_reset_clears_the_stale_session_id() -> None:
    """A lost legacy session must not be re-announced on the next handshake."""
    strategy = LegacyHandshakeStrategy()
    session_info = SessionInfo("session-1")

    await strategy.reset(session_info)

    assert await session_info.get_session_id() is None


def test_auto_applies_the_legacy_policy_before_an_era_is_resolved() -> None:
    """A failure during the very first probe is judged conservatively."""
    strategy = AutoStrategy()

    assert strategy.is_recoverable(_error(INVALID_REQUEST, "Session terminated"), None)


@pytest.mark.asyncio
async def test_legacy_keeps_a_persisted_session_when_the_connection_drops() -> None:
    """A dead transport says nothing about whether the session is still valid.

    Clearing the ID here would destroy an externally persisted session --
    permanently, for a store-backed SessionInfo -- over a transient failure, and
    the retry would start a cold session instead of resuming the warm one.
    """
    strategy = LegacyHandshakeStrategy()
    session_info = SessionInfo("persisted-session")
    session = MagicMock()
    session.initialize = AsyncMock(
        side_effect=MCPError(CONNECTION_CLOSED, "Connection closed")
    )

    with pytest.raises(MCPError):
        await strategy.connect(session, session_info)

    assert await session_info.get_session_id() == "persisted-session"
    # One attempt only: no clean-session fallback on a dead transport.
    assert session.initialize.await_count == 1


@pytest.mark.asyncio
async def test_auto_does_not_carry_a_stale_era_through_a_failed_probe() -> None:
    """A failed negotiation must not leave the previous era deciding recovery.

    After resolving modern, a later probe that raises would otherwise keep the
    modern policy, which refuses to retry session errors -- so a legacy server
    reached on the retry would never recover.
    """
    strategy = AutoStrategy()
    session = MagicMock()

    with patch(
        "uipath_langchain.agent.tools.mcp.protocol_strategy.probe_modern_era",
        AsyncMock(return_value=True),
    ):
        await strategy.connect(session, SessionInfo())
    assert not strategy.is_recoverable(
        _error(INVALID_REQUEST, "Session terminated"), None
    )

    with patch(
        "uipath_langchain.agent.tools.mcp.protocol_strategy.probe_modern_era",
        AsyncMock(side_effect=MCPError(INVALID_REQUEST, "probe blew up")),
    ):
        with pytest.raises(MCPError):
            await strategy.connect(session, SessionInfo())

    assert strategy.is_recoverable(_error(INVALID_REQUEST, "Session terminated"), None)


def test_build_protocol_strategy_maps_every_mode() -> None:
    """The public ``protocol_mode`` values are the only accepted ones."""
    assert isinstance(build_protocol_strategy("legacy"), LegacyHandshakeStrategy)
    assert isinstance(build_protocol_strategy("modern"), ModernDiscoveryStrategy)
    assert isinstance(build_protocol_strategy("auto"), AutoStrategy)

    with pytest.raises(ValueError, match="Unknown MCP protocol mode"):
        build_protocol_strategy("2026-07-28")  # type: ignore[arg-type]


def test_legacy_is_the_default_mode(
    mcp_resource_config: AgentMcpResourceConfig,
) -> None:
    """Existing callers must keep the pre-2026 wire behavior untouched."""
    client = McpClient(config=mcp_resource_config)

    assert isinstance(client._strategy, LegacyHandshakeStrategy)
