"""Tests for mcp_tool.py metadata and functionality."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from uipath.agent.models.agent import AgentMcpResourceConfig, AgentMcpTool

if TYPE_CHECKING:
    pass

from uipath_langchain.agent.tools.mcp_tool import create_mcp_tools_from_metadata


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

    @pytest.mark.asyncio
    async def test_mcp_tool_has_metadata(self, mcp_resource):
        """Test that MCP tool has metadata dict."""
        tools = await create_mcp_tools_from_metadata(mcp_resource)

        assert len(tools) == 1
        tool = tools[0]
        assert tool.metadata is not None
        assert isinstance(tool.metadata, dict)

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_tool_type(self, mcp_resource):
        """Test that metadata contains tool_type for span detection."""
        tools = await create_mcp_tools_from_metadata(mcp_resource)

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "mcp"

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_display_name(self, mcp_resource):
        """Test that metadata contains display_name from tool name."""
        tools = await create_mcp_tools_from_metadata(mcp_resource)

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_folder_path(self, mcp_resource):
        """Test that metadata contains folder_path for span attributes."""
        tools = await create_mcp_tools_from_metadata(mcp_resource)

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["folder_path"] == "/Shared/MyFolder"

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_slug(self, mcp_resource):
        """Test that metadata contains slug for server identification."""
        tools = await create_mcp_tools_from_metadata(mcp_resource)

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["slug"] == "my-mcp-server"


class TestMcpToolCreation:
    """Test MCP tool creation from metadata."""

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
    async def test_creates_multiple_tools(self, mcp_resource_multiple_tools):
        """Test that multiple tools are created from config."""
        tools = await create_mcp_tools_from_metadata(mcp_resource_multiple_tools)

        assert len(tools) == 2
        assert tools[0].name == "tool_one"
        assert tools[1].name == "tool_two"

    @pytest.mark.asyncio
    async def test_tool_has_correct_description(self, mcp_resource_multiple_tools):
        """Test that tools have correct descriptions."""
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


class TestMcpToolFunctionality:
    """Test MCP tool function behavior with mocked HTTP."""

    @pytest.fixture
    def mcp_resource(self):
        """Create a minimal MCP tool resource config."""
        return AgentMcpResourceConfig(
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

    @pytest.fixture
    def mock_mcp_server(self):
        """Create a mock MCP server object."""
        mock_server = MagicMock()
        mock_server.mcp_url = (
            "https://test.uipath.com/agenthub_/mcp/Shared/TestFolder/test-server"
        )
        return mock_server

    @pytest.fixture
    def mock_session(self):
        """Create a mock ClientSession."""
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock()
        return mock_session

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp_tool.UiPath")
    @patch("mcp.ClientSession")
    @patch("httpx.AsyncClient")
    async def test_tool_invocation_calls_mcp_server(
        self,
        mock_async_client_class,
        mock_client_session_class,
        mock_uipath_class,
        mcp_resource,
        mock_mcp_server,
        mock_session,
    ):
        """Test that tool invocation creates session with GUID and calls MCP tool."""
        # Setup UiPath SDK mock
        mock_sdk = MagicMock()
        mock_uipath_class.return_value = mock_sdk
        mock_sdk._config.secret = "test-token"
        mock_sdk.mcp.retrieve_async = AsyncMock(return_value=mock_mcp_server)

        # Setup HTTP client with MCP session ID in init response headers
        session_guid = "a1b2c3d4-e5f6-4789-a012-3456789abcde"
        mock_http_client = MagicMock()

        # Mock init response with mcp-session-id header
        mock_init_response = MagicMock()
        mock_init_response.headers = {"mcp-session-id": session_guid}
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "serverInfo": {"name": "test-server", "version": "1.0.0"},
            },
        }

        # Mock tool call response (without session header since session already established)
        mock_tool_response = MagicMock()
        mock_tool_response.headers = {}
        mock_tool_response.status_code = 200
        mock_tool_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{"type": "text", "text": "Search result"}],
                "isError": False,
            },
        }

        # Setup post to return different responses
        mock_http_client.post = AsyncMock(
            side_effect=[mock_init_response, mock_tool_response]
        )

        # Mock stream() method for GET requests (SSE)
        async def mock_stream(*args, **kwargs):
            class MockSSEResponse:
                def __init__(self):
                    self.headers = {"mcp-session-id": session_guid}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

                async def aiter_lines(self):
                    return
                    yield  # Make it an async generator

            return MockSSEResponse()

        mock_http_client.stream = mock_stream
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock()
        mock_async_client_class.return_value = mock_http_client

        # Setup ClientSession mock
        mock_client_session_class.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session_class.return_value.__aexit__ = AsyncMock()

        # Setup tool result
        mock_result = MagicMock()
        mock_result.content = [{"type": "text", "text": "Search result"}]
        mock_session.call_tool.return_value = mock_result

        # Create tools
        tools = await create_mcp_tools_from_metadata(mcp_resource)
        tool = tools[0]

        # Invoke tool
        result = await tool.ainvoke({"query": "test query"})

        # Verify MCP server was retrieved
        mock_sdk.mcp.retrieve_async.assert_called_once_with(
            slug="test-server", folder_path="/Shared/TestFolder"
        )

        # Verify tool was called
        mock_session.call_tool.assert_called_once_with(
            "search_tool", arguments={"query": "test query"}
        )

        # Verify result
        assert result == [{"type": "text", "text": "Search result"}]

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp_tool.UiPath")
    @patch("mcp.ClientSession")
    @patch("httpx.AsyncClient")
    async def test_tool_returns_result_without_content_attribute(
        self,
        mock_async_client_class,
        mock_client_session_class,
        mock_uipath_class,
        mcp_resource,
        mock_mcp_server,
        mock_session,
    ):
        """Test that tool handles results without content attribute."""
        # Setup mocks
        mock_sdk = MagicMock()
        mock_uipath_class.return_value = mock_sdk
        mock_sdk._config.secret = "test-token"
        mock_sdk.mcp.retrieve_async = AsyncMock(return_value=mock_mcp_server)

        # Setup HTTP client with MCP session ID in init response
        session_guid = "f9e8d7c6-b5a4-4321-9876-543210fedcba"
        mock_http_client = MagicMock()

        # Mock init response with mcp-session-id header
        mock_init_response = MagicMock()
        mock_init_response.headers = {"mcp-session-id": session_guid}
        mock_init_response.status_code = 200
        mock_init_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "serverInfo": {"name": "test-server", "version": "1.0.0"},
            },
        }

        # Mock tool call response with direct result (no content attribute)
        mock_tool_response = MagicMock()
        mock_tool_response.headers = {}
        mock_tool_response.status_code = 200
        mock_tool_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"direct": "result"},
        }

        mock_http_client.post = AsyncMock(
            side_effect=[mock_init_response, mock_tool_response]
        )

        # Mock stream() for SSE
        async def mock_stream(*args, **kwargs):
            class MockSSEResponse:
                def __init__(self):
                    self.headers = {"mcp-session-id": session_guid}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

                async def aiter_lines(self):
                    return
                    yield

            return MockSSEResponse()

        mock_http_client.stream = mock_stream
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock()
        mock_async_client_class.return_value = mock_http_client

        # Setup ClientSession mock
        mock_client_session_class.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session_class.return_value.__aexit__ = AsyncMock()

        # Result without content attribute
        mock_result = {"direct": "result"}
        mock_session.call_tool.return_value = mock_result

        # Create and invoke tool
        tools = await create_mcp_tools_from_metadata(mcp_resource)
        tool = tools[0]
        result = await tool.ainvoke({"query": "test"})

        # Should return the result directly
        assert result == {"direct": "result"}


class TestMcpToolSessionReuse:
    """Test that MCP tool session is reused across multiple calls."""

    @pytest.fixture
    def mcp_resource_two_tools(self):
        """Create MCP resource with two tools for session reuse testing."""
        return AgentMcpResourceConfig(
            name="session_test_server",
            description="Server for session testing",
            folder_path="/Shared/SessionTest",
            slug="session-server",
            available_tools=[
                AgentMcpTool(
                    name="tool_a",
                    description="First tool",
                    input_schema={
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                    },
                ),
                AgentMcpTool(
                    name="tool_b",
                    description="Second tool",
                    input_schema={
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                    },
                ),
            ],
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp_tool.UiPath")
    @patch("httpx.AsyncClient")
    async def test_tools_share_same_session_id_variable(
        self,
        mock_async_client_class,
        mock_uipath_class,
        mcp_resource_two_tools,
    ):
        """Test that both tools use the same mcp-session-id by intercepting HTTP POST headers."""
        # Setup UiPath SDK mock
        mock_sdk = MagicMock()
        mock_uipath_class.return_value = mock_sdk
        mock_sdk._config.secret = "test-token"

        mock_server = MagicMock()
        mock_server.mcp_url = "https://test.uipath.com/mcp"
        mock_sdk.mcp.retrieve_async = AsyncMock(return_value=mock_server)

        # Track all stream calls (which include the POST requests)
        post_calls = []
        session_guid = "11111111-2222-3333-4444-555555555555"

        # Setup HTTP client mock - only mock the HTTP layer, let everything else run
        mock_http_client = MagicMock()

        # Mock stream() - returns JSON responses with proper content-type
        class MockStreamResponse:
            def __init__(self, json_body):
                import json

                self.json_body = json_body
                self.method = json_body.get("method", "")
                self.is_initialize = self.method == "initialize"

                # Set headers based on request type
                if self.is_initialize:
                    self.headers = {
                        "mcp-session-id": session_guid,
                        "content-type": "application/json",
                    }
                else:
                    self.headers = {"content-type": "application/json"}

                self._response_json = self._build_response()

                # For notifications (no response), return 202 Accepted
                if self._response_json is None:
                    self.status_code = 202
                    self._content = b""
                else:
                    self.status_code = 200
                    self._content = json.dumps(self._response_json).encode("utf-8")

            def _build_response(self):
                """Build JSON-RPC response based on method."""
                request_id = self.json_body.get("id")

                if self.method == "initialize":
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "test-server", "version": "1.0.0"},
                        },
                    }
                elif self.method == "notifications/initialized":
                    # Notifications don't have responses
                    return None
                elif self.method == "tools/list":
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"tools": []},
                    }
                elif self.method == "tools/call":
                    params = self.json_body.get("params", {})
                    tool_name = params.get("name", "unknown")
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {"type": "text", "text": f"Result from {tool_name}"}
                            ],
                            "isError": False,
                        },
                    }
                else:
                    if request_id is None:
                        # Notification - no response
                        return None
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "Method not found"},
                    }

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args, **kwargs):
                pass

            async def aread(self):
                """Return the response content."""
                return self._content

            def raise_for_status(self):
                """Check response status."""
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")

        def mock_stream(method, url, **kwargs):
            """Return a context manager for streaming JSON responses."""
            # Extract JSON body from kwargs and track the call
            json_body = kwargs.get("json", {})
            call_info = {
                "url": url,
                "headers": kwargs.get("headers", {}),
                "json": json_body,
            }
            post_calls.append(call_info)
            return MockStreamResponse(json_body)

        mock_http_client.stream = mock_stream
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock()
        mock_async_client_class.return_value = mock_http_client

        # Create tools
        tools = await create_mcp_tools_from_metadata(mcp_resource_two_tools)
        assert len(tools) == 2

        tool_a = tools[0]
        tool_b = tools[1]

        # Invoke first tool
        result_a = await tool_a.ainvoke({"input": "first"})

        # Invoke second tool
        result_b = await tool_b.ainvoke({"data": "second"})

        # Verify both tools returned results
        assert result_a is not None
        assert result_b is not None

        # Verify HTTP stream calls were captured
        # Expected: 1 initialize + 2 tool calls + notifications and other calls
        assert len(post_calls) >= 3, (
            f"Expected at least 3 calls (initialize + 2 tool calls), but got {len(post_calls)}"
        )

        # Find the initialize call
        initialize_calls = [
            call for call in post_calls if call["json"].get("method") == "initialize"
        ]
        assert len(initialize_calls) == 1, (
            f"Expected exactly 1 initialize call (session reuse working!), "
            f"but got {len(initialize_calls)}"
        )

        # Find tool call requests
        tool_calls = [
            call for call in post_calls if call["json"].get("method") == "tools/call"
        ]
        assert len(tool_calls) == 2, (
            f"Expected 2 tool call requests, but got {len(tool_calls)}"
        )

        # Verify both tool calls have the same mcp-session-id header
        # After initialization, subsequent requests should include the session ID
        tool_call_session_ids = [
            call["headers"].get("mcp-session-id") for call in tool_calls
        ]
        # Both should have the session ID
        assert all(sid is not None for sid in tool_call_session_ids), (
            f"Tool calls should include session ID header, but got {tool_call_session_ids}"
        )
        # Both should have the same session ID
        assert tool_call_session_ids[0] == tool_call_session_ids[1], (
            f"Tool calls should share the same session ID, "
            f"but got {tool_call_session_ids[0]} and {tool_call_session_ids[1]}"
        )
        # Session ID should match the one returned from initialize
        assert tool_call_session_ids[0] == session_guid, (
            f"Tool calls should use session ID {session_guid}, "
            f"but got {tool_call_session_ids[0]}"
        )

        # Verify tool names match what we invoked
        tool_names_called = [
            call["json"].get("params", {}).get("name") for call in tool_calls
        ]
        assert "tool_a" in tool_names_called, "tool_a was not called"
        assert "tool_b" in tool_names_called, "tool_b was not called"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.tools.mcp_tool.UiPath")
    @patch("mcp.ClientSession")
    @patch("httpx.AsyncClient")
    async def test_session_not_terminated_between_calls(
        self,
        mock_async_client_class,
        mock_client_session_class,
        mock_uipath_class,
        mcp_resource_two_tools,
    ):
        """Test that session is not terminated between tool calls."""
        # Setup UiPath SDK mock
        mock_sdk = MagicMock()
        mock_uipath_class.return_value = mock_sdk
        mock_sdk._config.secret = "test-token"

        mock_server = MagicMock()
        mock_server.mcp_url = "https://test.uipath.com/mcp"
        mock_sdk.mcp.retrieve_async = AsyncMock(return_value=mock_server)

        # Setup ClientSession mock
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=MagicMock(content="result"))

        mock_client_session_class.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_client_session_class.return_value.__aexit__ = AsyncMock()

        # Setup HTTP client mock to track delete calls (session termination)
        mock_http_client = MagicMock()
        mock_http_client.delete = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock()

        mock_async_client_class.return_value = mock_http_client

        # Create tools
        tools = await create_mcp_tools_from_metadata(mcp_resource_two_tools)
        tool_a = tools[0]
        tool_b = tools[1]

        # Invoke both tools
        await tool_a.ainvoke({"input": "first"})
        await tool_b.ainvoke({"data": "second"})

        # Verify DELETE was never called (session not terminated)
        mock_http_client.delete.assert_not_called()


class TestMcpToolNameSanitization:
    """Test that MCP tool names are properly sanitized."""

    @pytest.mark.asyncio
    async def test_tool_name_with_spaces(self):
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

        tools = await create_mcp_tools_from_metadata(resource)

        assert " " not in tools[0].name

    @pytest.mark.asyncio
    async def test_tool_name_with_special_chars(self):
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

        tools = await create_mcp_tools_from_metadata(resource)

        # Tool name should be sanitized
        assert tools[0].name is not None
        assert len(tools[0].name) > 0
