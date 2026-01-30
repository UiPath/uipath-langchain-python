"""Tests for mcp_tool.py metadata and functionality."""

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from uipath.agent.models.agent import AgentMcpResourceConfig, AgentMcpTool

from uipath_langchain.agent.tools.mcp.mcp_tool import create_mcp_tools_from_metadata

logger = logging.getLogger(__name__)


class TestMcpToolMetadata:
    """Test that MCP tool has correct metadata for observability."""

    @pytest.fixture
    def mcp_resource(self):
        """Create a minimal MCP tool resource config."""
        return AgentMcpResourceConfig(
            name="test_mcp_server",
            description="Test MCP server",
            folder_path="/Shared/MyFolder",
            slug="my-mcp-server",
            available_tools=[
                AgentMcpTool(
                    name="test_tool",
                    description="Test tool description",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                    output_schema={"type": "object", "properties": {}},
                )
            ],
        )

    @pytest.fixture
    def mock_uipath_sdk(self):
        """Create a mock UiPath SDK with MCP server."""
        mock_sdk = MagicMock()
        mock_server = MagicMock()
        mock_server.mcp_url = "https://test.uipath.com/mcp"
        mock_sdk.mcp.retrieve_async = AsyncMock(return_value=mock_server)
        return mock_sdk

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
    async def test_mcp_tool_has_metadata(
        self, mock_uipath_class, mcp_resource, mock_uipath_sdk
    ):
        """Test that MCP tool has metadata dict."""
        mock_uipath_class.return_value = mock_uipath_sdk

        tools = await create_mcp_tools_from_metadata(mcp_resource)

        assert len(tools) == 1
        tool = tools[0]
        assert tool.metadata is not None
        assert isinstance(tool.metadata, dict)

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
    async def test_mcp_tool_metadata_has_tool_type(
        self, mock_uipath_class, mcp_resource, mock_uipath_sdk
    ):
        """Test that metadata contains tool_type for span detection."""
        mock_uipath_class.return_value = mock_uipath_sdk

        tools = await create_mcp_tools_from_metadata(mcp_resource)

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "mcp"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
    async def test_mcp_tool_metadata_has_display_name(
        self, mock_uipath_class, mcp_resource, mock_uipath_sdk
    ):
        """Test that metadata contains display_name from tool name."""
        mock_uipath_class.return_value = mock_uipath_sdk

        tools = await create_mcp_tools_from_metadata(mcp_resource)

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "test_tool"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
    async def test_mcp_tool_metadata_has_folder_path(
        self, mock_uipath_class, mcp_resource, mock_uipath_sdk
    ):
        """Test that metadata contains folder_path for span attributes."""
        mock_uipath_class.return_value = mock_uipath_sdk

        tools = await create_mcp_tools_from_metadata(mcp_resource)

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["folder_path"] == "/Shared/MyFolder"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
    async def test_mcp_tool_metadata_has_slug(
        self, mock_uipath_class, mcp_resource, mock_uipath_sdk
    ):
        """Test that metadata contains slug for server identification."""
        mock_uipath_class.return_value = mock_uipath_sdk

        tools = await create_mcp_tools_from_metadata(mcp_resource)

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["slug"] == "my-mcp-server"


class TestMcpToolCreation:
    """Test MCP tool creation from metadata."""

    @pytest.fixture
    def mock_uipath_sdk(self):
        """Create a mock UiPath SDK with MCP server."""
        mock_sdk = MagicMock()
        mock_server = MagicMock()
        mock_server.mcp_url = "https://test.uipath.com/mcp"
        mock_sdk.mcp.retrieve_async = AsyncMock(return_value=mock_server)
        return mock_sdk

    @pytest.fixture
    def mcp_resource_multiple_tools(self):
        """Create MCP resource config with multiple tools."""
        return AgentMcpResourceConfig(
            name="multi_tool_server",
            description="Server with multiple tools",
            folder_path="/Shared",
            slug="multi-server",
            available_tools=[
                AgentMcpTool(
                    name="tool_one",
                    description="First tool",
                    input_schema={"type": "object", "properties": {}},
                ),
                AgentMcpTool(
                    name="tool_two",
                    description="Second tool",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "string"},
                ),
            ],
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
    async def test_creates_multiple_tools(
        self, mock_uipath_class, mcp_resource_multiple_tools, mock_uipath_sdk
    ):
        """Test that multiple tools are created from config."""
        mock_uipath_class.return_value = mock_uipath_sdk

        tools = await create_mcp_tools_from_metadata(mcp_resource_multiple_tools)

        assert len(tools) == 2
        assert tools[0].name == "tool_one"
        assert tools[1].name == "tool_two"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
    async def test_tool_has_correct_description(
        self, mock_uipath_class, mcp_resource_multiple_tools, mock_uipath_sdk
    ):
        """Test that tools have correct descriptions."""
        mock_uipath_class.return_value = mock_uipath_sdk

        tools = await create_mcp_tools_from_metadata(mcp_resource_multiple_tools)

        assert tools[0].description == "First tool"
        assert tools[1].description == "Second tool"

    @pytest.mark.asyncio
    async def test_disabled_config_returns_empty_list(self):
        """Test that disabled config returns no tools."""
        disabled_config = AgentMcpResourceConfig(
            name="disabled_server",
            description="Disabled server",
            folder_path="/Shared",
            slug="disabled",
            is_enabled=False,
            available_tools=[
                AgentMcpTool(
                    name="tool",
                    description="Tool",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
        )

        tools = await create_mcp_tools_from_metadata(disabled_config)

        assert tools == []


class TestMcpToolInvocation:
    """Test MCP tool invocation with mocked HTTP.

    This class tests the full flow of tool invocation without mocking the MCP SDK.
    Only httpx.AsyncClient is mocked, allowing the real MCP SDK to process messages.
    """

    def create_mock_stream_response(
        self,
        method_call_sequence: list[str],
        initialize_count: list[int],
        tool_call_count: list[int],
        session_guid: str = "test-session-12345",
    ):
        """Create a MockStreamResponse class for testing.

        Reuses the same pattern as test_mcp_client.py.
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
                self.json_body = json_body
                self.method = json_body.get("method", "")

                logger.debug(f"Responding to MCP method: {self.method}")
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
                        {"mcp-session-id": session_guid},
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
                                        "name": "search_tool",
                                        "description": "Search for information",
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
        # Mock the delete method for session termination (returns 204 No Content)
        mock_delete_response = MagicMock()
        mock_delete_response.status_code = 204
        mock_client.delete = AsyncMock(return_value=mock_delete_response)
        return mock_client

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
    @patch("httpx.AsyncClient")
    async def test_tool_invocation_initializes_session_and_returns_result(
        self,
        mock_async_client_class,
        mock_uipath_class,
    ):
        """Smoke test: verify tool invocation initializes MCP session and returns result.

        This test verifies the full integration between create_mcp_tools_from_metadata
        and McpClient without mocking any MCP SDK components.

        Expected behavior:
        - Session is initialized via MCP protocol (initialize + initialized notification)
        - Tool call is sent and result is returned
        - Only httpx.AsyncClient is mocked, real MCP SDK processes the messages
        """
        # Setup UiPath SDK mock
        mock_sdk = MagicMock()
        mock_uipath_class.return_value = mock_sdk
        mock_server = MagicMock()
        mock_server.mcp_url = "https://test.uipath.com/mcp"
        mock_sdk.mcp.retrieve_async = AsyncMock(return_value=mock_server)

        # Track MCP method calls
        method_call_sequence: list[str] = []
        initialize_count = [0]
        tool_call_count = [0]

        # Setup HTTP mock using pattern from test_mcp_client.py
        MockStreamResponse = self.create_mock_stream_response(
            method_call_sequence, initialize_count, tool_call_count
        )
        mock_http_client = self.create_mock_http_client(MockStreamResponse)
        mock_async_client_class.return_value = mock_http_client

        # Create resource config
        mcp_resource = AgentMcpResourceConfig(
            name="test_server",
            description="Test server",
            folder_path="/Shared/TestFolder",
            slug="test-server",
            available_tools=[
                AgentMcpTool(
                    name="search_tool",
                    description="Search for information",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                    },
                )
            ],
        )

        # Create tools from metadata
        tools = await create_mcp_tools_from_metadata(mcp_resource)
        assert len(tools) == 1

        tool = tools[0]
        assert tool.name == "search_tool"

        # Invoke tool
        result = await tool.ainvoke({"query": "test query"})

        # Verify session was initialized
        assert initialize_count[0] == 1, (
            f"Expected 1 initialize call, got {initialize_count[0]}"
        )

        # Verify tool was called
        assert tool_call_count[0] == 1, (
            f"Expected 1 tool call, got {tool_call_count[0]}"
        )

        # Verify result is returned (content attribute of CallToolResult)
        # Result is a list of TextContent objects from MCP SDK
        assert result is not None
        assert len(result) == 1
        # TextContent has .type and .text attributes (not dict subscript)
        assert result[0].type == "text"
        assert "Success from search_tool" in result[0].text

        # Verify MCP protocol flow
        assert "initialize" in method_call_sequence
        assert "notifications/initialized" in method_call_sequence
        assert "tools/call" in method_call_sequence

        logger.info(f"Method sequence: {method_call_sequence}")


class TestMcpToolNameSanitization:
    """Test that MCP tool names are properly sanitized."""

    @pytest.fixture
    def mock_uipath_sdk(self):
        """Create a mock UiPath SDK with MCP server."""
        mock_sdk = MagicMock()
        mock_server = MagicMock()
        mock_server.mcp_url = "https://test.uipath.com/mcp"
        mock_sdk.mcp.retrieve_async = AsyncMock(return_value=mock_server)
        return mock_sdk

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
    async def test_tool_name_with_spaces(self, mock_uipath_class, mock_uipath_sdk):
        """Test that tool names with spaces are sanitized."""
        mock_uipath_class.return_value = mock_uipath_sdk

        resource = AgentMcpResourceConfig(
            name="test_server",
            description="Test",
            folder_path="/Shared",
            slug="test",
            available_tools=[
                AgentMcpTool(
                    name="Search Tool With Spaces",
                    description="Search tool",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
        )

        tools = await create_mcp_tools_from_metadata(resource)

        assert " " not in tools[0].name

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
    async def test_tool_name_with_special_chars(
        self, mock_uipath_class, mock_uipath_sdk
    ):
        """Test that tool names with special characters are sanitized."""
        mock_uipath_class.return_value = mock_uipath_sdk

        resource = AgentMcpResourceConfig(
            name="test_server",
            description="Test",
            folder_path="/Shared",
            slug="test",
            available_tools=[
                AgentMcpTool(
                    name="search-tool@v1.0",
                    description="Search tool",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
        )

        tools = await create_mcp_tools_from_metadata(resource)

        # Tool name should be sanitized
        assert tools[0].name is not None
        assert len(tools[0].name) > 0
