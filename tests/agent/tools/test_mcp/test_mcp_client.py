"""Tests for McpClient class."""

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from uipath.agent.models.agent import AgentMcpResourceConfig, AgentMcpTool

from uipath_langchain.agent.tools.mcp import McpClient, SessionInfo, SessionInfoFactory

logger = logging.getLogger(__name__)


class TestMcpClient:
    """Test MCP client behavior with mocked HTTP."""

    @pytest.fixture
    def mcp_resource_config(self):
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
    def mock_uipath_sdk(self):
        """Create a mock UiPath SDK for patching."""
        mock_sdk = MagicMock()
        mock_server = MagicMock()
        mock_server.mcp_url = "https://test.uipath.com/mcp"
        mock_sdk.mcp.retrieve_async = AsyncMock(return_value=mock_server)
        mock_sdk._config = MagicMock()
        mock_sdk._config.secret = "test-secret-token"
        return mock_sdk

    def create_mock_stream_response(
        self,
        method_call_sequence: list[str],
        initialize_count: list[int],
        tool_call_count: list[int],
        session_guid_1: str = "test-session-first",
        session_guid_2: str = "test-session-retry",
        fail_first_tool_call: bool = False,
    ):
        """Create a MockStreamResponse class for testing.

        Args:
            method_call_sequence: List to track method calls.
            initialize_count: Mutable counter for initialize calls.
            tool_call_count: Mutable counter for tool calls.
            session_guid_1: Session ID for first initialization.
            session_guid_2: Session ID for retry initialization.
            fail_first_tool_call: If True, first tool call returns 404.
        """

        class MockStreamResponse:
            """Mock HTTP stream response for MCP protocol."""

            def __init__(self, method: str, url: str, **kwargs: Any):
                self.request_method = method
                self.url = url
                self.kwargs = kwargs

                if method == "GET":
                    self.status_code = 405
                    self.headers = {}
                    self._content = b""
                    return

                json_body = kwargs.get("json", {})
                request_headers = kwargs.get("headers", {})

                self.json_body = json_body
                self.method = json_body.get("method", "")
                self.request_headers = request_headers
                self.request_mcp_session_id = request_headers.get("mcp-session-id", "")

                logger.debug(
                    f"Responding to method {self.method} for session {self.request_mcp_session_id}"
                )
                method_call_sequence.append(self.method)

                status_code, response_json, headers = self._build_response()
                self.headers = headers or {}
                self._response_json = response_json
                self.status_code = status_code

                if response_json:
                    self._content = json.dumps(self._response_json).encode("utf-8")
                    self.headers["content-type"] = "application/json"
                else:
                    self._content = b""

            def _build_response(self) -> tuple[int, Any, dict[str, str] | None]:
                """Build JSON-RPC response based on method."""
                request_id = self.json_body.get("id")

                if self.method == "initialize":
                    initialize_count[0] += 1
                    session_id = (
                        session_guid_1 if initialize_count[0] == 1 else session_guid_2
                    )
                    logger.debug(f"MCP initializes new session {session_id}")
                    return (
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "protocolVersion": "2025-06-18",
                                "capabilities": {"tools": {}},
                                "serverInfo": {
                                    "name": "test-server",
                                    "version": "1.0.0",
                                },
                            },
                        },
                        {"mcp-session-id": session_id},
                    )

                elif self.method == "notifications/initialized":
                    return (204, None, {})

                elif self.method == "tools/list":
                    return (
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
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
                                            "properties": {
                                                "result": {"type": "string"}
                                            },
                                        },
                                    }
                                ],
                            },
                        },
                        {},
                    )

                elif self.method == "tools/call":
                    tool_call_count[0] += 1

                    if fail_first_tool_call and tool_call_count[0] == 1:
                        # Return HTTP 404 to trigger session re-initialization
                        return (404, None, None)

                    # Success response with structured content
                    params = self.json_body.get("params", {})
                    tool_name = params.get("name", "unknown")
                    structured_result = {"result": f"Success from {tool_name}"}

                    return (
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(structured_result),
                                    }
                                ],
                                "structuredContent": structured_result,
                                "isError": False,
                            },
                        },
                        {},
                    )

                else:
                    if request_id is None:
                        return (204, None, {})
                    return (
                        500,
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32601, "message": "Method not found"},
                        },
                        {},
                    )

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args: Any, **kwargs: Any):
                pass

            async def aread(self) -> bytes:
                """Return the response content."""
                return self._content

            def raise_for_status(self) -> None:
                """Check response status."""
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")

        return MockStreamResponse

    def create_mock_http_client(self, mock_stream_response_class: type) -> MagicMock:
        """Create a mock HTTP client that uses the given stream response class."""
        mock_client = MagicMock()
        mock_client.stream = lambda method, url, **kwargs: mock_stream_response_class(
            method, url, **kwargs
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        return mock_client

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_session_initializes_on_first_call(
        self, mock_async_client_class, mcp_resource_config, mock_uipath_sdk
    ):
        """Test that session is initialized lazily on first tool call."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(config=mcp_resource_config)

        # Session should not be initialized yet
        assert await session.get_session_id() is None
        assert not session.is_client_initialized

        # Call tool - should trigger initialization (with SDK mocked)
        with patch(
            "uipath.platform.UiPath",
            return_value=mock_uipath_sdk,
        ):
            result = await session.call_tool("test_tool", {"query": "test"})

        # Verify initialization happened
        assert initialize_count[0] == 1
        assert await session.get_session_id() == "test-session-first"
        assert session.is_client_initialized
        assert tool_call_count[0] == 1
        assert result is not None

        # Verify HTTP client was created once
        assert mock_async_client_class.call_count == 1

        # Verify method sequence
        assert "initialize" in method_call_sequence
        assert "notifications/initialized" in method_call_sequence
        assert "tools/call" in method_call_sequence

        await session.dispose()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_session_reused_across_calls(
        self, mock_async_client_class, mcp_resource_config, mock_uipath_sdk
    ):
        """Test that session is reused for multiple tool calls."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(config=mcp_resource_config)

        with patch(
            "uipath.platform.UiPath",
            return_value=mock_uipath_sdk,
        ):
            # First call
            await session.call_tool("test_tool", {"query": "first"})
            assert initialize_count[0] == 1

            # Second call - should reuse session
            await session.call_tool("test_tool", {"query": "second"})
            assert initialize_count[0] == 1  # Still only one initialization
            assert tool_call_count[0] == 2  # But two tool calls

        # HTTP client should still be created only once
        assert mock_async_client_class.call_count == 1

        await session.dispose()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_session_reinitializes_on_404_error(
        self, mock_async_client_class, mcp_resource_config, mock_uipath_sdk
    ):
        """Test that only session (not client) is reinitialized on 404 error.

        This verifies the key behavior: when a 404 error occurs, we should:
        - Keep the existing HTTP client (not create a new one)
        - Keep the existing streamable connection
        - Only call session.initialize() again to get a new session ID
        """
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence,
            initialize_count,
            tool_call_count,
            fail_first_tool_call=True,
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(config=mcp_resource_config)

        # Call tool - first call fails with 404, should retry
        with patch(
            "uipath.platform.UiPath",
            return_value=mock_uipath_sdk,
        ):
            result = await session.call_tool("test_tool", {"query": "test"})

        logger.info(f"Result: {result}")
        logger.info(f"Method sequence: {method_call_sequence}")
        logger.info(f"Initialize count: {initialize_count[0]}")
        logger.info(f"Tool call count: {tool_call_count[0]}")

        # Verify session was reinitialized (initialize called twice)
        assert initialize_count[0] == 2, (
            f"Expected 2 session initializations, got {initialize_count[0]}"
        )

        # Verify tool call was retried
        assert tool_call_count[0] == 2, (
            f"Expected 2 tool calls, got {tool_call_count[0]}"
        )

        # Verify session ID changed to the retry session
        assert await session.get_session_id() == "test-session-retry"
        assert result is not None

        # KEY ASSERTION: HTTP client should be created only ONCE
        # Session reinitialization reuses the existing client
        assert mock_async_client_class.call_count == 1, (
            f"Expected HTTP client to be created only once, "
            f"but was created {mock_async_client_class.call_count} times"
        )

        # Verify the expected method sequence
        expected_init_count = method_call_sequence.count("initialize")
        expected_tool_count = method_call_sequence.count("tools/call")
        assert expected_init_count == 2, (
            f"Expected 2 initialize calls, got {expected_init_count}"
        )
        assert expected_tool_count == 2, (
            f"Expected 2 tools/call, got {expected_tool_count}"
        )

        await session.dispose()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_max_retries_exceeded(
        self, mock_async_client_class, mcp_resource_config, mock_uipath_sdk
    ):
        """Test that exception is raised when max retries are exceeded."""
        from mcp.shared.exceptions import McpError

        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        # Create a response that always fails tool calls
        class AlwaysFailMockResponse:
            def __init__(self, method: str, url: str, **kwargs: Any):
                self.request_method = method
                if method == "GET":
                    self.status_code = 405
                    self.headers = {}
                    self._content = b""
                    return

                json_body = kwargs.get("json", {})
                self.method = json_body.get("method", "")
                method_call_sequence.append(self.method)
                request_id = json_body.get("id")

                if self.method == "initialize":
                    initialize_count[0] += 1
                    self.status_code = 200
                    self.headers = {"mcp-session-id": f"session-{initialize_count[0]}"}
                    self._response_json = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "test", "version": "1.0"},
                        },
                    }
                    self._content = json.dumps(self._response_json).encode()
                    self.headers["content-type"] = "application/json"
                elif self.method == "notifications/initialized":
                    self.status_code = 204
                    self.headers = {}
                    self._content = b""
                elif self.method == "tools/call":
                    tool_call_count[0] += 1
                    # Always return 404
                    self.status_code = 404
                    self.headers = {}
                    self._content = b""
                else:
                    self.status_code = 200
                    self.headers = {}
                    self._content = b""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args: Any):
                pass

            async def aread(self) -> bytes:
                return self._content

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")

        mock_http_client = self.create_mock_http_client(AlwaysFailMockResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(config=mcp_resource_config, max_retries=1)

        # Should raise McpError after retries exhausted
        with patch(
            "uipath.platform.UiPath",
            return_value=mock_uipath_sdk,
        ):
            with pytest.raises(McpError):
                await session.call_tool("test_tool", {"query": "test"})

        # Should have reinitialized session (2 initialize calls)
        assert initialize_count[0] == 2

        # Should have attempted tool call twice
        assert tool_call_count[0] == 2

        # HTTP client still created only once
        assert mock_async_client_class.call_count == 1

        await session.dispose()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_dispose_releases_resources(
        self, mock_async_client_class, mcp_resource_config, mock_uipath_sdk
    ):
        """Test that dispose() properly releases session resources."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(config=mcp_resource_config)

        # Initialize session
        with patch(
            "uipath.platform.UiPath",
            return_value=mock_uipath_sdk,
        ):
            await session.call_tool("test_tool", {"query": "test"})
        assert await session.get_session_id() is not None
        assert session.is_client_initialized

        # Close session
        await session.dispose()

        # Verify resources are released
        assert await session.get_session_id() is None
        assert session._session is None
        assert session._stack is None
        assert not session.is_client_initialized

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_client_initialized_property(
        self, mock_async_client_class, mcp_resource_config, mock_uipath_sdk
    ):
        """Test that is_client_initialized property reflects actual state."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(config=mcp_resource_config)

        # Before any call
        assert not session.is_client_initialized

        with patch(
            "uipath.platform.UiPath",
            return_value=mock_uipath_sdk,
        ):
            # After first call
            await session.call_tool("test_tool", {"query": "test"})
            assert session.is_client_initialized

        # After dispose
        await session.dispose()
        assert not session.is_client_initialized

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_session_can_be_reused_after_dispose(
        self, mock_async_client_class, mcp_resource_config, mock_uipath_sdk
    ):
        """Test that session can be reinitialized after dispose()."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(config=mcp_resource_config)

        with patch(
            "uipath.platform.UiPath",
            return_value=mock_uipath_sdk,
        ):
            # First use
            await session.call_tool("test_tool", {"query": "first"})
            assert await session.get_session_id() == "test-session-first"

            # Close
            await session.dispose()
            assert await session.get_session_id() is None

            # Reuse - should create new client and session
            # Note: mock returns "test-session-retry" for second initialize
            await session.call_tool("test_tool", {"query": "second"})
            assert await session.get_session_id() == "test-session-retry"
            assert session.is_client_initialized

        # HTTP client was created twice (once before dispose, once after)
        assert mock_async_client_class.call_count == 2

        await session.dispose()

    @pytest.mark.asyncio
    async def test_raises_on_missing_mcp_url(self, mcp_resource_config):
        """Test that ValueError is raised when MCP server has no URL configured."""
        mock_sdk = MagicMock()
        mock_server = MagicMock()
        mock_server.mcp_url = None  # No URL configured
        mock_sdk.mcp.retrieve_async = AsyncMock(return_value=mock_server)
        mock_sdk._config = MagicMock()
        mock_sdk._config.secret = "test-token"

        session = McpClient(config=mcp_resource_config)

        with patch(
            "uipath.platform.UiPath",
            return_value=mock_sdk,
        ):
            with pytest.raises(ValueError, match="has no URL configured"):
                await session.call_tool("test_tool", {"query": "test"})

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_custom_session_info_factory_is_used(
        self, mock_async_client_class, mcp_resource_config, mock_uipath_sdk
    ):
        """Test that a custom SessionInfoFactory is called during initialization."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )
        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        custom_session_info = SessionInfo()

        class TrackingFactory(SessionInfoFactory):
            called_with_server = None

            def create_session(self, mcp_server: Any) -> SessionInfo:
                TrackingFactory.called_with_server = mcp_server
                return custom_session_info

        factory = TrackingFactory()
        session = McpClient(
            config=mcp_resource_config,
            session_info_factory=factory,
        )

        with patch(
            "uipath.platform.UiPath",
            return_value=mock_uipath_sdk,
        ):
            await session.call_tool("test_tool", {"query": "test"})

        # Verify factory was called with the McpServer
        assert TrackingFactory.called_with_server is not None

        # Verify our custom SessionInfo instance is used by McpClient
        assert session._session_info is custom_session_info
        assert await session.get_session_id() == "test-session-first"

        await session.dispose()

    @pytest.mark.asyncio
    async def test_skips_initialize_when_session_info_has_id(self, mcp_resource_config):
        """Test that _initialize_session skips session.initialize() when SessionInfo has an ID."""
        session = McpClient(config=mcp_resource_config)

        # Simulate already-initialized client with pre-existing session ID
        session._session_info = SessionInfo(session_id="pre-existing-id")
        session._session = MagicMock()
        session._session.initialize = AsyncMock()

        await session._initialize_session()

        # initialize() should NOT be called because session_info already has an ID
        session._session.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_reinitialize_clears_session_info_before_init(
        self, mcp_resource_config
    ):
        """Test that _reinitialize_session clears session info then calls initialize."""
        session = McpClient(config=mcp_resource_config)

        # Simulate already-initialized client with a stale session ID
        session._client_initialized = True
        session._session_info = SessionInfo(session_id="stale-id")
        session._session = MagicMock()
        session._session.initialize = AsyncMock()

        await session._reinitialize_session()

        # Session info should have been cleared before re-initializing
        # (set_session_id(None) was called, then _initialize_session ran)
        session._session.initialize.assert_called_once()

        # After reinitialize, session_info.session_id is None because
        # the mocked initialize() doesn't set a new one
        assert await session.get_session_id() is None

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_list_tools_initializes_session_and_returns_result(
        self, mock_async_client_class, mcp_resource_config, mock_uipath_sdk
    ):
        """Test that list_tools lazily initializes session and returns tools."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        client = McpClient(config=mcp_resource_config)

        assert not client.is_client_initialized

        with patch(
            "uipath.platform.UiPath",
            return_value=mock_uipath_sdk,
        ):
            result = await client.list_tools()

        # Session should have been initialized
        assert initialize_count[0] == 1
        assert client.is_client_initialized

        # Should return the tools from the mock server
        assert result is not None
        assert len(result.tools) == 1
        assert result.tools[0].name == "test_tool"

        # Verify protocol flow includes tools/list
        assert "initialize" in method_call_sequence
        assert "tools/list" in method_call_sequence
        # tools/call should NOT have been called
        assert "tools/call" not in method_call_sequence

        await client.dispose()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_list_tools_reuses_session(
        self, mock_async_client_class, mcp_resource_config, mock_uipath_sdk
    ):
        """Test that list_tools reuses existing session on subsequent calls."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        client = McpClient(config=mcp_resource_config)

        with patch(
            "uipath.platform.UiPath",
            return_value=mock_uipath_sdk,
        ):
            await client.list_tools()
            assert initialize_count[0] == 1

            await client.list_tools()
            assert initialize_count[0] == 1  # Still only one initialization

        list_tools_count = method_call_sequence.count("tools/list")
        assert list_tools_count == 2

        await client.dispose()
