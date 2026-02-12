"""Tests for mcp_tool.py metadata and functionality."""

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from uipath.agent.models.agent import (
    AgentMcpResourceConfig,
    AgentMcpTool,
    AgentResourceType,
    AgentSettings,
    LowCodeAgentDefinition,
)

from uipath_langchain.agent.tools.mcp import McpClient
from uipath_langchain.agent.tools.mcp.mcp_tool import (
    create_mcp_tools_from_agent,
    create_mcp_tools_from_metadata_for_mcp_server,
)

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
    def mock_mcp_client(self):
        """Create a mock McpClient."""
        return MagicMock(spec=McpClient)

    @pytest.mark.asyncio
    async def test_mcp_tool_has_metadata(self, mcp_resource, mock_mcp_client):
        """Test that MCP tool has metadata dict."""
        tools = await create_mcp_tools_from_metadata_for_mcp_server(
            mcp_resource, mock_mcp_client
        )

        assert len(tools) == 1
        tool = tools[0]
        assert tool.metadata is not None
        assert isinstance(tool.metadata, dict)

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_tool_type(self, mcp_resource, mock_mcp_client):
        """Test that metadata contains tool_type for span detection."""
        tools = await create_mcp_tools_from_metadata_for_mcp_server(
            mcp_resource, mock_mcp_client
        )

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "mcp"

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_display_name(
        self, mcp_resource, mock_mcp_client
    ):
        """Test that metadata contains display_name from tool name."""
        tools = await create_mcp_tools_from_metadata_for_mcp_server(
            mcp_resource, mock_mcp_client
        )

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_folder_path(
        self, mcp_resource, mock_mcp_client
    ):
        """Test that metadata contains folder_path for span attributes."""
        tools = await create_mcp_tools_from_metadata_for_mcp_server(
            mcp_resource, mock_mcp_client
        )

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["folder_path"] == "/Shared/MyFolder"

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_slug(self, mcp_resource, mock_mcp_client):
        """Test that metadata contains slug for server identification."""
        tools = await create_mcp_tools_from_metadata_for_mcp_server(
            mcp_resource, mock_mcp_client
        )

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["slug"] == "my-mcp-server"


class TestMcpToolCreation:
    """Test MCP tool creation from metadata."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock McpClient."""
        return MagicMock(spec=McpClient)

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
    async def test_creates_multiple_tools(
        self, mcp_resource_multiple_tools, mock_mcp_client
    ):
        """Test that multiple tools are created from config."""
        tools = await create_mcp_tools_from_metadata_for_mcp_server(
            mcp_resource_multiple_tools, mock_mcp_client
        )

        assert len(tools) == 2
        assert tools[0].name == "tool_one"
        assert tools[1].name == "tool_two"

    @pytest.mark.asyncio
    async def test_tool_has_correct_description(
        self, mcp_resource_multiple_tools, mock_mcp_client
    ):
        """Test that tools have correct descriptions."""
        tools = await create_mcp_tools_from_metadata_for_mcp_server(
            mcp_resource_multiple_tools, mock_mcp_client
        )

        assert tools[0].description == "First tool"
        assert tools[1].description == "Second tool"

    @pytest.mark.asyncio
    async def test_disabled_config_returns_empty_list(self, mock_mcp_client):
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

        tools = await create_mcp_tools_from_metadata_for_mcp_server(
            disabled_config, mock_mcp_client
        )

        assert tools == []


class TestCreateMcpToolsFromAgent:
    """Test create_mcp_tools_from_agent factory function."""

    @pytest.fixture
    def agent_with_mcp_resources(self):
        """Create an agent definition with MCP resources."""
        return LowCodeAgentDefinition(
            name="test_agent",
            description="Test agent",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            messages=[],
            settings=AgentSettings(
                engine="openai", model="gpt-4", max_tokens=1000, temperature=0.7
            ),
            resources=[
                AgentMcpResourceConfig(
                    resource_type=AgentResourceType.MCP,
                    name="mcp_server_1",
                    description="First MCP server",
                    folder_path="/Shared/Folder1",
                    slug="server-1",
                    is_enabled=True,
                    available_tools=[
                        AgentMcpTool(
                            name="tool_a",
                            description="Tool A",
                            input_schema={"type": "object", "properties": {}},
                        ),
                        AgentMcpTool(
                            name="tool_b",
                            description="Tool B",
                            input_schema={"type": "object", "properties": {}},
                        ),
                    ],
                ),
                AgentMcpResourceConfig(
                    resource_type=AgentResourceType.MCP,
                    name="mcp_server_2",
                    description="Second MCP server",
                    folder_path="/Shared/Folder2",
                    slug="server-2",
                    is_enabled=True,
                    available_tools=[
                        AgentMcpTool(
                            name="tool_c",
                            description="Tool C",
                            input_schema={"type": "object", "properties": {}},
                        ),
                    ],
                ),
            ],
        )

    @pytest.fixture
    def agent_with_disabled_mcp(self):
        """Create an agent with disabled MCP resource."""
        return LowCodeAgentDefinition(
            name="test_agent",
            description="Test agent",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            messages=[],
            settings=AgentSettings(
                engine="openai", model="gpt-4", max_tokens=1000, temperature=0.7
            ),
            resources=[
                AgentMcpResourceConfig(
                    resource_type=AgentResourceType.MCP,
                    name="enabled_server",
                    description="Enabled MCP server",
                    folder_path="/Shared",
                    slug="enabled",
                    is_enabled=True,
                    available_tools=[
                        AgentMcpTool(
                            name="enabled_tool",
                            description="Enabled tool",
                            input_schema={"type": "object", "properties": {}},
                        ),
                    ],
                ),
                AgentMcpResourceConfig(
                    resource_type=AgentResourceType.MCP,
                    name="disabled_server",
                    description="Disabled MCP server",
                    folder_path="/Shared",
                    slug="disabled",
                    is_enabled=False,
                    available_tools=[
                        AgentMcpTool(
                            name="disabled_tool",
                            description="Disabled tool",
                            input_schema={"type": "object", "properties": {}},
                        ),
                    ],
                ),
            ],
        )

    @pytest.fixture
    def agent_with_no_mcp(self):
        """Create an agent with no MCP resources."""
        return LowCodeAgentDefinition(
            name="test_agent",
            description="Test agent",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            messages=[],
            settings=AgentSettings(
                engine="openai", model="gpt-4", max_tokens=1000, temperature=0.7
            ),
            resources=[],
        )

    @pytest.mark.asyncio
    async def test_creates_tools_from_multiple_mcp_servers(
        self, agent_with_mcp_resources
    ):
        """Test that tools are created from all MCP servers in agent.

        Note: SDK is now called lazily inside McpClient, so no mocking needed
        for tool creation (only for tool invocation).
        """
        tools, clients = await create_mcp_tools_from_agent(agent_with_mcp_resources)

        # Should have 3 tools total (2 from server 1, 1 from server 2)
        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "tool_a" in tool_names
        assert "tool_b" in tool_names
        assert "tool_c" in tool_names

    @pytest.mark.asyncio
    async def test_returns_mcp_clients_for_each_server(self, agent_with_mcp_resources):
        """Test that McpClient instances are returned for each MCP server."""
        tools, clients = await create_mcp_tools_from_agent(agent_with_mcp_resources)

        # Should have 2 clients (one per MCP server)
        assert len(clients) == 2

    @pytest.mark.asyncio
    async def test_skips_disabled_mcp_resources(self, agent_with_disabled_mcp):
        """Test that disabled MCP resources are skipped."""
        tools, clients = await create_mcp_tools_from_agent(agent_with_disabled_mcp)

        # Only enabled server's tool should be created
        assert len(tools) == 1
        assert tools[0].name == "enabled_tool"

        # Only one client for enabled server
        assert len(clients) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_for_agent_without_mcp(self, agent_with_no_mcp):
        """Test that empty lists are returned for agent without MCP resources."""
        tools, clients = await create_mcp_tools_from_agent(agent_with_no_mcp)

        assert tools == []
        assert clients == []

    @pytest.mark.asyncio
    async def test_tools_have_correct_metadata(self, agent_with_mcp_resources):
        """Test that created tools have correct metadata."""
        tools, clients = await create_mcp_tools_from_agent(agent_with_mcp_resources)

        for tool in tools:
            assert tool.metadata is not None
            assert tool.metadata["tool_type"] == "mcp"
            assert "display_name" in tool.metadata
            assert "folder_path" in tool.metadata
            assert "slug" in tool.metadata


class TestMcpToolInvocation:
    """Test MCP tool invocation with mocked HTTP.

    This class tests the full flow of tool invocation without mocking the MCP SDK.
    Only httpx.AsyncClient is mocked, allowing the real MCP SDK to process messages.
    """

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
    @patch("httpx.AsyncClient")
    async def test_tool_invocation_initializes_session_and_returns_result(
        self,
        mock_async_client_class,
        mock_uipath_sdk,
    ):
        """Smoke test: verify tool invocation initializes MCP session and returns result.

        This test verifies the full integration between create_mcp_tools_from_metadata
        and McpClient without mocking any MCP SDK components.

        Expected behavior:
        - Session is initialized via MCP protocol (initialize + initialized notification)
        - Tool call is sent and result is returned
        - Only httpx.AsyncClient is mocked, real MCP SDK processes the messages
        """
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

        # Create McpClient and tools (SDK is called lazily on first tool call)
        mcp_client = McpClient(config=mcp_resource)
        tools = await create_mcp_tools_from_metadata_for_mcp_server(
            mcp_resource, mcp_client
        )
        assert len(tools) == 1

        tool = tools[0]
        assert tool.name == "search_tool"

        # Invoke tool (SDK is called here during initialization)
        with patch(
            "uipath.platform.UiPath",
            return_value=mock_uipath_sdk,
        ):
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
        # Result is a list of dicts (model_dump'd TextContent objects)
        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert "Success from search_tool" in result[0]["text"]

        # Verify MCP protocol flow
        assert "initialize" in method_call_sequence
        assert "notifications/initialized" in method_call_sequence
        assert "tools/call" in method_call_sequence

        logger.info(f"Method sequence: {method_call_sequence}")


class TestMcpToolResultSerialization:
    """Test that tool_fn properly serializes different result types."""

    @pytest.fixture
    def mcp_tool(self):
        return AgentMcpTool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {}},
        )

    @pytest.mark.asyncio
    async def test_single_object_with_model_dump(self, mcp_tool):
        """Test that a single result object with model_dump is serialized."""
        from uipath_langchain.agent.tools.mcp.mcp_tool import build_mcp_tool

        mock_content = MagicMock()
        mock_content.model_dump.return_value = {"type": "text", "text": "hello"}

        mock_result = MagicMock()
        mock_result.content = mock_content

        mock_client = MagicMock(spec=McpClient)
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        tool_fn = build_mcp_tool(mcp_tool, mock_client)
        result = await tool_fn()

        assert result == {"type": "text", "text": "hello"}
        mock_content.model_dump.assert_called_once_with(exclude_none=True)

    @pytest.mark.asyncio
    async def test_list_of_objects_with_model_dump(self, mcp_tool):
        """Test that a list of result objects with model_dump are serialized."""
        from uipath_langchain.agent.tools.mcp.mcp_tool import build_mcp_tool

        mock_item = MagicMock()
        mock_item.model_dump.return_value = {"type": "text", "text": "item1"}

        mock_result = MagicMock()
        mock_result.content = [mock_item]

        mock_client = MagicMock(spec=McpClient)
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        tool_fn = build_mcp_tool(mcp_tool, mock_client)
        result = await tool_fn()

        assert result == [{"type": "text", "text": "item1"}]

    @pytest.mark.asyncio
    async def test_plain_value_returned_as_is(self, mcp_tool):
        """Test that a plain value without model_dump is returned as-is."""
        from uipath_langchain.agent.tools.mcp.mcp_tool import build_mcp_tool

        mock_result = MagicMock()
        mock_result.content = "plain string"

        mock_client = MagicMock(spec=McpClient)
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        tool_fn = build_mcp_tool(mcp_tool, mock_client)
        result = await tool_fn()

        assert result == "plain string"


class TestMcpToolNameSanitization:
    """Test that MCP tool names are properly sanitized."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock McpClient."""
        return MagicMock(spec=McpClient)

    @pytest.mark.asyncio
    async def test_tool_name_with_spaces(self, mock_mcp_client):
        """Test that tool names with spaces are sanitized."""
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

        tools = await create_mcp_tools_from_metadata_for_mcp_server(
            resource, mock_mcp_client
        )

        assert " " not in tools[0].name

    @pytest.mark.asyncio
    async def test_tool_name_with_special_chars(self, mock_mcp_client):
        """Test that tool names with special characters are sanitized."""
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

        tools = await create_mcp_tools_from_metadata_for_mcp_server(
            resource, mock_mcp_client
        )

        # Tool name should be sanitized
        assert tools[0].name is not None
        assert len(tools[0].name) > 0
