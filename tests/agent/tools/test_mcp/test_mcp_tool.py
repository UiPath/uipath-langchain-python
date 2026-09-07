"""Tests for mcp_tool.py metadata and functionality."""

import logging
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool
from mcp.shared.exceptions import MCPError
from mcp.types import ListToolsResult, Tool
from uipath.agent.models.agent import (
    AgentMcpResourceConfig,
    AgentMcpTool,
    AgentResourceType,
    CachedToolsConfig,
    DynamicToolsConfig,
    ToolsConfiguration,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.tools.mcp import McpClient
from uipath_langchain.agent.tools.mcp.mcp_tool import (
    _schema_change_message,
    build_mcp_tool,
    create_mcp_tools,
    create_mcp_tools_and_clients,
    open_mcp_tools,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)


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
        tools = await create_mcp_tools(mcp_resource, mock_mcp_client)

        assert len(tools) == 1
        tool = tools[0]
        assert tool.metadata is not None
        assert isinstance(tool.metadata, dict)

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_tool_type(self, mcp_resource, mock_mcp_client):
        """Test that metadata contains tool_type for span detection."""
        tools = await create_mcp_tools(mcp_resource, mock_mcp_client)

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "mcp"

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_display_name(
        self, mcp_resource, mock_mcp_client
    ):
        """Test that metadata contains display_name from tool name."""
        tools = await create_mcp_tools(mcp_resource, mock_mcp_client)

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["display_name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_folder_path(
        self, mcp_resource, mock_mcp_client
    ):
        """Test that metadata contains folder_path for span attributes."""
        tools = await create_mcp_tools(mcp_resource, mock_mcp_client)

        tool = tools[0]
        assert tool.metadata is not None
        assert tool.metadata["folder_path"] == "/Shared/MyFolder"

    @pytest.mark.asyncio
    async def test_mcp_tool_metadata_has_slug(self, mcp_resource, mock_mcp_client):
        """Test that metadata contains slug for server identification."""
        tools = await create_mcp_tools(mcp_resource, mock_mcp_client)

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
        tools = await create_mcp_tools(mcp_resource_multiple_tools, mock_mcp_client)

        assert len(tools) == 2
        assert tools[0].name == "tool_one"
        assert tools[1].name == "tool_two"

    @pytest.mark.asyncio
    async def test_tool_has_correct_description(
        self, mcp_resource_multiple_tools, mock_mcp_client
    ):
        """Test that tools have correct descriptions."""
        tools = await create_mcp_tools(mcp_resource_multiple_tools, mock_mcp_client)

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

        tools = await create_mcp_tools(disabled_config, mock_mcp_client)

        assert tools == []


class TestCreateMcpToolsFromAgent:
    """Test create_mcp_tools_and_clients factory function."""

    @pytest.fixture
    def mcp_resources(self):
        """Create a list of MCP resource configurations."""
        return [
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
        ]

    @pytest.fixture
    def mcp_resources_with_disabled(self):
        """Create MCP resources with one disabled."""
        return [
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
        ]

    @pytest.mark.asyncio
    async def test_creates_tools_from_multiple_mcp_servers(self, mcp_resources):
        """Test that tools are created from all MCP servers.

        Note: SDK is now called lazily inside McpClient, so no mocking needed
        for tool creation (only for tool invocation).
        """
        tools, clients = await create_mcp_tools_and_clients(mcp_resources)

        # Should have 3 tools total (2 from server 1, 1 from server 2)
        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "tool_a" in tool_names
        assert "tool_b" in tool_names
        assert "tool_c" in tool_names

    @pytest.mark.asyncio
    async def test_returns_mcp_clients_for_each_server(self, mcp_resources):
        """Test that McpClient instances are returned for each MCP server."""
        tools, clients = await create_mcp_tools_and_clients(mcp_resources)

        # Should have 2 clients (one per MCP server)
        assert len(clients) == 2

    @pytest.mark.asyncio
    async def test_skips_disabled_mcp_resources(self, mcp_resources_with_disabled):
        """Test that disabled MCP resources are skipped."""
        tools, clients = await create_mcp_tools_and_clients(mcp_resources_with_disabled)

        # Only enabled server's tool should be created
        assert len(tools) == 1
        assert tools[0].name == "enabled_tool"

        # Only one client for enabled server
        assert len(clients) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_resources(self):
        """Test that empty lists are returned for empty resource list."""
        tools, clients = await create_mcp_tools_and_clients([])

        assert tools == []
        assert clients == []

    @pytest.mark.asyncio
    async def test_tools_have_correct_metadata(self, mcp_resources):
        """Test that created tools have correct metadata."""
        tools, clients = await create_mcp_tools_and_clients(mcp_resources)

        for tool in tools:
            assert tool.metadata is not None
            assert tool.metadata["tool_type"] == "mcp"
            assert "display_name" in tool.metadata
            assert "folder_path" in tool.metadata
            assert "slug" in tool.metadata


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
    async def test_single_object_is_serialized_in_its_wire_shape(self, mcp_tool):
        """A lone content block keeps the camelCase spelling the model expects.

        Uses a real ``ImageContent`` rather than a mock: SDK 2.0 renamed the
        model attributes to snake case while keeping the camelCase names as
        serialization aliases, so only a real model exposes the difference. A
        mock asserting the call arguments would lock whichever call was written.
        """
        from mcp.types import ImageContent

        from uipath_langchain.agent.tools.mcp.mcp_tool import build_mcp_tool

        mock_result = MagicMock()
        mock_result.content = ImageContent(
            type="image", data="Zm9v", mimeType="image/png"
        )

        mock_client = MagicMock(spec=McpClient)
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        tool_fn = build_mcp_tool(mcp_tool, mock_client)
        result = await tool_fn()

        assert result == {"type": "image", "data": "Zm9v", "mimeType": "image/png"}

    @pytest.mark.asyncio
    async def test_list_of_blocks_keeps_camel_case_for_non_text_blocks(self, mcp_tool):
        """Every block in a list is serialized by alias, not just the first.

        Text blocks are byte-identical either way, so a text-only assertion
        cannot catch a snake_case regression -- the list mixes both kinds.
        """
        from mcp.types import (
            EmbeddedResource,
            ImageContent,
            TextContent,
            TextResourceContents,
        )

        from uipath_langchain.agent.tools.mcp.mcp_tool import build_mcp_tool

        mock_result = MagicMock()
        mock_result.content = [
            TextContent(type="text", text="item1"),
            ImageContent(type="image", data="Zm9v", mimeType="image/png"),
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///x.txt", mimeType="text/plain", text="hi"
                ),
            ),
        ]

        mock_client = MagicMock(spec=McpClient)
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        tool_fn = build_mcp_tool(mcp_tool, mock_client)
        result = await tool_fn()

        assert result == [
            {"type": "text", "text": "item1"},
            {"type": "image", "data": "Zm9v", "mimeType": "image/png"},
            {
                "type": "resource",
                "resource": {
                    "uri": "file:///x.txt",
                    "mimeType": "text/plain",
                    "text": "hi",
                },
            },
        ]

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


class TestMcpToolErrorHandling:
    """Test that protocol-level MCPErrors are mapped to categorized AgentRuntimeErrors."""

    @pytest.fixture
    def mcp_tool(self):
        return AgentMcpTool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {}},
        )

    def _mock_client(self, error: MCPError) -> MagicMock:
        client = MagicMock(spec=McpClient)
        client.server_slug = "my-mcp-server"
        client.call_tool = AsyncMock(side_effect=error)
        return client

    @pytest.mark.asyncio
    async def test_session_terminated_raises_system_error_with_retry_hint(
        self, mcp_tool
    ):
        error = MCPError(code=32600, message="Session terminated")
        client = self._mock_client(error)

        tool_fn = build_mcp_tool(mcp_tool, client)

        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool_fn()
        assert exc_info.value.error_info.category == UiPathErrorCategory.SYSTEM
        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.HTTP_ERROR
        )
        assert "my-mcp-server" in exc_info.value.error_info.detail
        assert "terminated" in exc_info.value.error_info.detail
        assert "retry" in exc_info.value.error_info.detail
        assert exc_info.value.__cause__ is error

    @pytest.mark.asyncio
    async def test_non_session_mcp_error_includes_server_message(self, mcp_tool):
        error = MCPError(code=-32601, message="Method not found")
        client = self._mock_client(error)

        tool_fn = build_mcp_tool(mcp_tool, client)

        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool_fn()
        assert exc_info.value.error_info.category == UiPathErrorCategory.SYSTEM
        assert "Method not found" in exc_info.value.error_info.detail
        assert "test_tool" in exc_info.value.error_info.detail

    @pytest.mark.asyncio
    async def test_non_mcp_error_propagates_unchanged(self, mcp_tool):
        client = MagicMock(spec=McpClient)
        client.server_slug = "my-mcp-server"
        client.call_tool = AsyncMock(side_effect=RuntimeError("boom"))

        tool_fn = build_mcp_tool(mcp_tool, client)

        with pytest.raises(RuntimeError, match="boom"):
            await tool_fn()


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

        tools = await create_mcp_tools(resource, mock_mcp_client)

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

        tools = await create_mcp_tools(resource, mock_mcp_client)

        # Tool name should be sanitized
        assert tools[0].name is not None
        assert len(tools[0].name) > 0


class TestToolsConfiguration:
    """Test tools_configuration behaviour (Cached vs Dynamic) in create_mcp_tools."""

    @pytest.fixture
    def server_tools(self):
        """MCP server tools returned by list_tools."""
        return [
            Tool(
                name="tool_a",
                description="Tool A from server",
                inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
                outputSchema={
                    "type": "object",
                    "properties": {"r": {"type": "string"}},
                },
            ),
            Tool(
                name="tool_b",
                description="Tool B from server",
                inputSchema={
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                },
            ),
            Tool(
                name="tool_c",
                description="Tool C from server",
                inputSchema={"type": "object", "properties": {}},
                outputSchema={"type": "string"},
            ),
        ]

    @pytest.fixture
    def mcp_resource_curated_dynamic(self):
        """Resource config with Dynamic discovery_mode + allow_all=False (curated subset)."""
        return AgentMcpResourceConfig(
            name="schema_server",
            description="Curated dynamic server",
            folder_path="/Shared",
            slug="schema-server",
            tools_configuration=ToolsConfiguration(
                discovery_mode=DynamicToolsConfig(allow_all=False)
            ),
            available_tools=[
                AgentMcpTool(
                    name="tool_a",
                    description="Tool A (stale)",
                    input_schema={"type": "object", "properties": {}},
                ),
                AgentMcpTool(
                    name="tool_b",
                    description="Tool B (stale)",
                    input_schema={"type": "object", "properties": {}},
                ),
            ],
        )

    @pytest.fixture
    def mcp_resource_dynamic(self):
        """Resource config with Dynamic discovery_mode + allow_all=True."""
        return AgentMcpResourceConfig(
            name="dynamic_server",
            description="Dynamic mode server",
            folder_path="/Shared",
            slug="dynamic-server",
            tools_configuration=ToolsConfiguration(
                discovery_mode=DynamicToolsConfig(allow_all=True)
            ),
            available_tools=[
                AgentMcpTool(
                    name="tool_a",
                    description="Tool A (stale)",
                    input_schema={"type": "object", "properties": {}},
                ),
            ],
        )

    @pytest.fixture
    def mock_mcp_client(self, server_tools):
        """McpClient mock that returns server_tools from list_tools."""
        client = MagicMock(spec=McpClient)
        client.list_tools = AsyncMock(return_value=ListToolsResult(tools=server_tools))
        return client

    @pytest.mark.asyncio
    async def test_curated_dynamic_calls_list_tools(
        self, mcp_resource_curated_dynamic, mock_mcp_client
    ):
        """Dynamic with allow_all=False still calls mcpClient.list_tools()."""
        await create_mcp_tools(mcp_resource_curated_dynamic, mock_mcp_client)

        mock_mcp_client.list_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_curated_dynamic_filters_to_available_tools(
        self, mcp_resource_curated_dynamic, mock_mcp_client
    ):
        """Dynamic with allow_all=False only includes tools listed in available_tools."""
        tools = await create_mcp_tools(mcp_resource_curated_dynamic, mock_mcp_client)

        tool_names = [t.name for t in tools]
        assert "tool_a" in tool_names
        assert "tool_b" in tool_names
        assert "tool_c" not in tool_names
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_curated_dynamic_uses_server_schemas_and_descriptions(
        self, mcp_resource_curated_dynamic, mock_mcp_client
    ):
        """Dynamic with allow_all=False uses input/output schemas and descriptions from the server."""
        tools = await create_mcp_tools(mcp_resource_curated_dynamic, mock_mcp_client)

        tool_a = next(t for t in tools if t.name == "tool_a")
        assert tool_a.description == "Tool A from server"
        assert isinstance(tool_a.args_schema, dict)
        assert "x" in tool_a.args_schema["properties"]

    @pytest.mark.asyncio
    async def test_curated_dynamic_warns_about_missing_allowed_tool(
        self, mock_mcp_client, caplog
    ):
        """Dynamic with allow_all=False logs a warning for allowlisted tools the server no longer has."""
        resource = AgentMcpResourceConfig(
            name="schema_server",
            description="Server with phantom in allowlist",
            folder_path="/Shared",
            slug="schema-server",
            tools_configuration=ToolsConfiguration(
                discovery_mode=DynamicToolsConfig(allow_all=False)
            ),
            available_tools=[
                AgentMcpTool(
                    name="tool_a",
                    description="Tool A (stale)",
                    input_schema={"type": "object", "properties": {}},
                ),
                AgentMcpTool(
                    name="phantom",
                    description="Tool that no longer exists on the server",
                    input_schema={"type": "object", "properties": {}},
                ),
            ],
        )

        with caplog.at_level(logging.WARNING):
            tools = await create_mcp_tools(resource, mock_mcp_client)

        tool_names = [t.name for t in tools]
        assert "tool_a" in tool_names
        assert "phantom" not in tool_names
        assert any(
            "'phantom' is in availableTools" in record.message
            and "schema-server" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_dynamic_calls_list_tools(
        self, mcp_resource_dynamic, mock_mcp_client
    ):
        """Test that Dynamic mode calls mcpClient.list_tools()."""
        await create_mcp_tools(mcp_resource_dynamic, mock_mcp_client)

        mock_mcp_client.list_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dynamic_returns_all_server_tools(
        self, mcp_resource_dynamic, mock_mcp_client
    ):
        """Test that Dynamic mode returns every tool from the server."""
        tools = await create_mcp_tools(mcp_resource_dynamic, mock_mcp_client)

        tool_names = [t.name for t in tools]
        assert "tool_a" in tool_names
        assert "tool_b" in tool_names
        assert "tool_c" in tool_names
        assert len(tools) == 3

    @pytest.mark.asyncio
    async def test_dynamic_ignores_available_tools(
        self, mcp_resource_dynamic, mock_mcp_client
    ):
        """Test that Dynamic mode ignores the available_tools list in the resource config."""
        # Resource only lists tool_a, but all 3 server tools should be returned
        tools = await create_mcp_tools(mcp_resource_dynamic, mock_mcp_client)

        assert len(tools) == 3

    @pytest.mark.asyncio
    async def test_dynamic_uses_server_schemas_and_descriptions(
        self, mcp_resource_dynamic, mock_mcp_client
    ):
        """Test that Dynamic mode uses schemas and descriptions from the server."""
        tools = await create_mcp_tools(mcp_resource_dynamic, mock_mcp_client)

        tool_a = next(t for t in tools if t.name == "tool_a")
        assert tool_a.description == "Tool A from server"
        assert isinstance(tool_a.args_schema, dict)
        assert "x" in tool_a.args_schema["properties"]

    @pytest.mark.asyncio
    async def test_cached_default_does_not_call_list_tools(self):
        """Test that Cached mode (default when tools_configuration is unset) does not call list_tools()."""
        resource = AgentMcpResourceConfig(
            name="default_server",
            description="Default (Cached) mode",
            folder_path="/Shared",
            slug="default",
            available_tools=[
                AgentMcpTool(
                    name="local_tool",
                    description="Local tool",
                    input_schema={"type": "object", "properties": {}},
                ),
            ],
        )

        client = MagicMock(spec=McpClient)
        client.list_tools = AsyncMock()

        tools = await create_mcp_tools(resource, client)

        client.list_tools.assert_not_awaited()
        assert len(tools) == 1
        assert tools[0].name == "local_tool"

    @pytest.mark.asyncio
    async def test_cached_uses_resource_schemas(self):
        """Test that Cached mode uses schemas from available_tools, not the server."""
        resource = AgentMcpResourceConfig(
            name="default_server",
            description="Default (Cached) mode",
            folder_path="/Shared",
            slug="default",
            available_tools=[
                AgentMcpTool(
                    name="my_tool",
                    description="My local description",
                    input_schema={
                        "type": "object",
                        "properties": {"local_param": {"type": "boolean"}},
                    },
                ),
            ],
        )

        client = MagicMock(spec=McpClient)

        tools = await create_mcp_tools(resource, client)

        assert tools[0].description == "My local description"
        assert isinstance(tools[0].args_schema, dict)
        assert "local_param" in tools[0].args_schema["properties"]


class TestCachedRefreshSchemaBeforeCall:
    """Test the refresh_schema_before_call behaviour for cached discovery mode."""

    @pytest.fixture
    def mcp_tool(self):
        return AgentMcpTool(
            name="tool_a",
            description="Tool A (cached)",
            input_schema={"type": "object", "properties": {}},
        )

    @pytest.fixture
    def server_tool(self):
        return Tool(
            name="tool_a",
            description="Tool A from server",
            inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
        )

    def _mock_client(self, server_tools, result_content="ok"):
        client = MagicMock(spec=McpClient)
        client.list_tools = AsyncMock(return_value=ListToolsResult(tools=server_tools))
        mock_result = MagicMock()
        mock_result.content = result_content
        client.call_tool = AsyncMock(return_value=mock_result)
        return client

    @pytest.mark.asyncio
    async def test_refresh_enabled_lists_tools_before_call(self, mcp_tool, server_tool):
        """With refresh on, list_tools runs before call_tool."""
        manager = MagicMock()
        client = self._mock_client([server_tool])
        manager.attach_mock(client.list_tools, "list_tools")
        manager.attach_mock(client.call_tool, "call_tool")

        tool_fn = build_mcp_tool(mcp_tool, client, refresh_schema_before_call=True)
        await tool_fn()

        client.list_tools.assert_awaited_once()
        client.call_tool.assert_awaited_once()
        called = [name for name, _, _ in manager.mock_calls]
        assert called.index("list_tools") < called.index("call_tool")

    @pytest.mark.asyncio
    async def test_refresh_disabled_skips_list_tools(self, mcp_tool, server_tool):
        """With refresh off, list_tools is not called at invocation."""
        client = self._mock_client([server_tool])

        tool_fn = build_mcp_tool(mcp_tool, client, refresh_schema_before_call=False)
        await tool_fn()

        client.list_tools.assert_not_awaited()
        client.call_tool.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_falls_back_when_list_tools_fails(self, mcp_tool):
        """A failing list_tools does not block the call; it falls back to the cached schema."""
        client = MagicMock(spec=McpClient)
        client.list_tools = AsyncMock(side_effect=RuntimeError("boom"))
        mock_result = MagicMock()
        mock_result.content = "ok"
        client.call_tool = AsyncMock(return_value=mock_result)

        tool_fn = build_mcp_tool(mcp_tool, client, refresh_schema_before_call=True)
        result = await tool_fn()

        client.call_tool.assert_awaited_once()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_refresh_returns_removed_message_when_tool_missing(
        self, mcp_tool, caplog
    ):
        """If the tool is gone from the live list, skip the call and tell the model."""
        other_tool = Tool(
            name="other_tool",
            description="A different tool",
            inputSchema={"type": "object", "properties": {}},
        )
        client = self._mock_client([other_tool])

        tool_fn = build_mcp_tool(mcp_tool, client, refresh_schema_before_call=True)
        with caplog.at_level(logging.WARNING):
            result = await tool_fn()

        client.call_tool.assert_not_awaited()
        assert "no longer available" in result
        assert mcp_tool.name in result
        assert any(
            "is no longer exposed by the MCP server" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_cached_default_enables_refresh(self):
        """A cached resource defaults to refresh on, so the tool lists before calling."""
        resource = AgentMcpResourceConfig(
            name="cached_server",
            description="Cached server",
            folder_path="/Shared",
            slug="cached",
            tools_configuration=ToolsConfiguration(discovery_mode=CachedToolsConfig()),
            available_tools=[
                AgentMcpTool(
                    name="tool_a",
                    description="Tool A",
                    input_schema={"type": "object", "properties": {}},
                ),
            ],
        )
        server_tool = Tool(
            name="tool_a",
            description="Tool A from server",
            inputSchema={"type": "object", "properties": {}},
        )
        client = self._mock_client([server_tool])

        tools = await create_mcp_tools(resource, client)
        # create_mcp_tools must not list tools for cached mode (refresh is per-call)
        client.list_tools.assert_not_awaited()

        await tools[0].ainvoke({})
        client.list_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cached_refresh_disabled_via_config(self):
        """refresh_schema_before_call=False on the cached config disables the per-call refresh."""
        resource = AgentMcpResourceConfig(
            name="cached_server",
            description="Cached server",
            folder_path="/Shared",
            slug="cached",
            tools_configuration=ToolsConfiguration(
                discovery_mode=CachedToolsConfig(refresh_schema_before_call=False)
            ),
            available_tools=[
                AgentMcpTool(
                    name="tool_a",
                    description="Tool A",
                    input_schema={"type": "object", "properties": {}},
                ),
            ],
        )
        client = self._mock_client([])

        tools = await create_mcp_tools(resource, client)
        await tools[0].ainvoke({})
        client.list_tools.assert_not_awaited()

    def _cached_resource(self, input_schema):
        return AgentMcpResourceConfig(
            name="cached_server",
            description="Cached server",
            folder_path="/Shared",
            slug="cached",
            tools_configuration=ToolsConfiguration(discovery_mode=CachedToolsConfig()),
            available_tools=[
                AgentMcpTool(
                    name="ask_question",
                    description="Ask a question",
                    input_schema=input_schema,
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_breaking_drift_heals_and_asks_retry(self):
        """On a breaking schema change the tool is not executed: it refreshes the bound
        schema and returns a retry instruction to the model."""
        resource = self._cached_resource(
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )
        live_tool = Tool(
            name="ask_question",
            description="Ask a question",
            inputSchema={
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        )
        client = self._mock_client([live_tool])

        tools = await create_mcp_tools(resource, client)
        tool = cast(StructuredToolWithArgumentProperties, tools[0])
        assert tool.coroutine is not None
        # Model issued the call using the stale cached parameter name.
        result = await tool.coroutine(query="What is X?")

        assert "ask_question" in result
        # the refreshed param list now includes the type hint
        assert "question (string)" in result
        client.call_tool.assert_not_awaited()
        # The schema bound to the model was healed to the live one.
        assert tool.args_schema == live_tool.input_schema

    def test_schema_change_message_lists_param_types(self):
        """The retry message lists each refreshed param with its type and optionality."""
        msg = _schema_change_message(
            "search",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "filter": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["query"],
            },
        )
        assert "query (string)" in msg
        assert "limit (integer, optional)" in msg
        # no simple type -> name (+ optional) only, never a misleading type
        assert "filter (optional)" in msg

    @pytest.mark.asyncio
    async def test_after_heal_next_call_executes(self):
        """After healing, a call matching the live schema executes normally."""
        resource = self._cached_resource(
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        )
        live_tool = Tool(
            name="ask_question",
            description="Ask a question",
            inputSchema={
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        )
        client = self._mock_client([live_tool])
        tools = await create_mcp_tools(resource, client)
        tool = cast(StructuredToolWithArgumentProperties, tools[0])
        assert tool.coroutine is not None

        # First call drifts and heals (the tool is not executed).
        await tool.coroutine(query="What is X?")
        client.call_tool.assert_not_awaited()

        # Second call matches the healed schema and executes.
        await tool.coroutine(question="What is X?")
        client.call_tool.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_nonbreaking_change_executes_without_retry(self):
        """An additive (non-breaking) schema change does not trigger a retry."""
        resource = self._cached_resource(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            }
        )
        live_tool = Tool(
            name="ask_question",
            description="Ask a question",
            inputSchema={
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a"],
            },
        )
        client = self._mock_client([live_tool])
        tools = await create_mcp_tools(resource, client)
        tool = cast(StructuredToolWithArgumentProperties, tools[0])
        assert tool.coroutine is not None

        result = await tool.coroutine(a="x")

        client.call_tool.assert_awaited_once()
        assert result == "ok"


class TestCreateMcpToolsFromConfig:
    @pytest.fixture
    def mcp_config(self):
        return AgentMcpResourceConfig(
            resource_type=AgentResourceType.MCP,
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
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_yields_tools_from_single_config(self, mcp_config):
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_client = AsyncMock(spec=McpClient)

        with patch(
            "uipath_langchain.agent.tools.mcp.mcp_tool.create_mcp_tools_and_clients",
            return_value=([mock_tool], [mock_client]),
        ):
            async with open_mcp_tools([mcp_config]) as tools:
                assert len(tools) == 1
                assert tools[0].name == "test_tool"

            mock_client.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_yields_tools_from_list_of_configs(self, mcp_config):
        mock_tool_1 = MagicMock(spec=BaseTool)
        mock_tool_1.name = "tool_1"
        mock_tool_2 = MagicMock(spec=BaseTool)
        mock_tool_2.name = "tool_2"
        mock_client_1 = AsyncMock(spec=McpClient)
        mock_client_2 = AsyncMock(spec=McpClient)

        with patch(
            "uipath_langchain.agent.tools.mcp.mcp_tool.create_mcp_tools_and_clients",
            return_value=([mock_tool_1, mock_tool_2], [mock_client_1, mock_client_2]),
        ):
            async with open_mcp_tools([mcp_config, mcp_config]) as tools:
                assert len(tools) == 2

            mock_client_1.dispose.assert_awaited_once()
            mock_client_2.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disposes_clients_on_exception(self, mcp_config):
        mock_tool = MagicMock(spec=BaseTool)
        mock_client = AsyncMock(spec=McpClient)

        with patch(
            "uipath_langchain.agent.tools.mcp.mcp_tool.create_mcp_tools_and_clients",
            return_value=([mock_tool], [mock_client]),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                async with open_mcp_tools([mcp_config]):
                    raise RuntimeError("boom")

            mock_client.dispose.assert_awaited_once()


class TestMcpToolArgumentProperties:
    """Test that argument_properties from config are applied to MCP tools."""

    @pytest.fixture
    def mock_mcp_client(self):
        return MagicMock(spec=McpClient)

    @pytest.mark.asyncio
    async def test_none_mode_passes_argument_properties_to_tool(self, mock_mcp_client):
        """In none mode, argument_properties from config must reach the built tool."""
        resource = AgentMcpResourceConfig(
            name="test_server",
            description="Test",
            folder_path="/Shared",
            slug="test",
            available_tools=[
                AgentMcpTool(
                    name="divide",
                    description="Divide two numbers",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                    argument_properties={
                        "$['a']": {
                            "variant": "static",
                            "value": 76,
                            "isSensitive": False,
                        }
                    },
                ),
            ],
        )

        tools = await create_mcp_tools(resource, mock_mcp_client)

        assert len(tools) == 1
        tool = tools[0]
        assert hasattr(tool, "argument_properties")
        assert "$['a']" in tool.argument_properties

    @pytest.mark.asyncio
    async def test_all_mode_carries_over_argument_properties_for_matching_tools(
        self, mock_mcp_client
    ):
        """In all mode, argument_properties from config must attach to matching server tools."""
        mock_mcp_client.list_tools = AsyncMock(
            return_value=ListToolsResult(
                tools=[
                    Tool(
                        name="divide",
                        description="Divide from server",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                        },
                    ),
                    Tool(
                        name="new_tool",
                        description="New tool not in config",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                ]
            )
        )

        resource = AgentMcpResourceConfig(
            name="test_server",
            description="Test",
            folder_path="/Shared",
            slug="test",
            tools_configuration=ToolsConfiguration(
                discovery_mode=DynamicToolsConfig(allow_all=True)
            ),
            available_tools=[
                AgentMcpTool(
                    name="divide",
                    description="Divide (stale)",
                    input_schema={"type": "object", "properties": {}},
                    argument_properties={
                        "$['a']": {
                            "variant": "static",
                            "value": 76,
                            "isSensitive": False,
                        }
                    },
                ),
            ],
        )

        tools = await create_mcp_tools(resource, mock_mcp_client)

        assert len(tools) == 2
        divide_tool = next(
            cast(StructuredToolWithArgumentProperties, t)
            for t in tools
            if t.name == "divide"
        )
        new_tool = next(
            cast(StructuredToolWithArgumentProperties, t)
            for t in tools
            if t.name == "new_tool"
        )

        assert "$['a']" in divide_tool.argument_properties
        assert not new_tool.argument_properties
