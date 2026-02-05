"""Tests for McpClient class."""

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uipath_langchain.agent.tools.mcp import McpClient

logger = logging.getLogger(__name__)


class TestMcpClient:
    """Test MCP client behavior with mocked HTTP."""

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
    async def test_session_initializes_on_first_call(self, mock_async_client_class):
        """Test that session is initialized lazily on first tool call."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(
            url="https://test.uipath.com/mcp",
            headers={"Authorization": "Bearer test-token"},
        )

        # Session should not be initialized yet
        assert session.session_id is None
        assert not session.is_client_initialized

        # Call tool - should trigger initialization
        result = await session.call_tool("test_tool", {"query": "test"})

        # Verify initialization happened
        assert initialize_count[0] == 1
        assert session.session_id == "test-session-first"
        assert session.is_client_initialized
        assert tool_call_count[0] == 1
        assert result is not None

        # Verify HTTP client was created once
        assert mock_async_client_class.call_count == 1

        # Verify method sequence
        assert "initialize" in method_call_sequence
        assert "notifications/initialized" in method_call_sequence
        assert "tools/call" in method_call_sequence

        await session.close()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_session_reused_across_calls(self, mock_async_client_class):
        """Test that session is reused for multiple tool calls."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(
            url="https://test.uipath.com/mcp",
            headers={"Authorization": "Bearer test-token"},
        )

        # First call
        await session.call_tool("test_tool", {"query": "first"})
        assert initialize_count[0] == 1

        # Second call - should reuse session
        await session.call_tool("test_tool", {"query": "second"})
        assert initialize_count[0] == 1  # Still only one initialization
        assert tool_call_count[0] == 2  # But two tool calls

        # HTTP client should still be created only once
        assert mock_async_client_class.call_count == 1

        await session.close()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_session_reinitializes_on_404_error(self, mock_async_client_class):
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

        session = McpClient(
            url="https://test.uipath.com/mcp",
            headers={"Authorization": "Bearer test-token"},
        )

        # Call tool - first call fails with 404, should retry
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
        assert session.session_id == "test-session-retry"
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

        await session.close()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_max_retries_exceeded(self, mock_async_client_class):
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

        session = McpClient(
            url="https://test.uipath.com/mcp",
            headers={"Authorization": "Bearer test-token"},
            max_retries=1,
        )

        # Should raise McpError after retries exhausted
        with pytest.raises(McpError):
            await session.call_tool("test_tool", {"query": "test"})

        # Should have reinitialized session (2 initialize calls)
        assert initialize_count[0] == 2

        # Should have attempted tool call twice
        assert tool_call_count[0] == 2

        # HTTP client still created only once
        assert mock_async_client_class.call_count == 1

        await session.close()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_close_releases_resources(self, mock_async_client_class):
        """Test that close() properly releases session resources."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(
            url="https://test.uipath.com/mcp",
            headers={"Authorization": "Bearer test-token"},
        )

        # Initialize session
        await session.call_tool("test_tool", {"query": "test"})
        assert session.session_id is not None
        assert session.is_client_initialized

        # Close session
        await session.close()

        # Verify resources are released
        assert session.session_id is None
        assert session._session is None
        assert session._stack is None
        assert not session.is_client_initialized

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_client_initialized_property(self, mock_async_client_class):
        """Test that is_client_initialized property reflects actual state."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(
            url="https://test.uipath.com/mcp",
            headers={"Authorization": "Bearer test-token"},
        )

        # Before any call
        assert not session.is_client_initialized

        # After first call
        await session.call_tool("test_tool", {"query": "test"})
        assert session.is_client_initialized

        # After close
        await session.close()
        assert not session.is_client_initialized

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_session_can_be_reused_after_close(self, mock_async_client_class):
        """Test that session can be reinitialized after close()."""
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )

        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        session = McpClient(
            url="https://test.uipath.com/mcp",
            headers={"Authorization": "Bearer test-token"},
        )

        # First use
        await session.call_tool("test_tool", {"query": "first"})
        assert session.session_id == "test-session-first"

        # Close
        await session.close()
        assert session.session_id is None

        # Reuse - should create new client and session
        # Note: mock returns "test-session-retry" for second initialize
        await session.call_tool("test_tool", {"query": "second"})
        assert session.session_id == "test-session-retry"
        assert session.is_client_initialized

        # HTTP client was created twice (once before close, once after)
        assert mock_async_client_class.call_count == 2

        await session.close()
