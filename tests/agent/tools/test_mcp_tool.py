"""Tests for mcp_tool.py — deduplication, filtering, and tool creation."""

from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool
from uipath.agent.models.agent import (
    AgentMcpResourceConfig,
    AgentMcpTool,
    AgentResourceType,
)

from uipath_langchain.agent.tools.mcp_tool import (
    _deduplicate_tools,
    _filter_tools,
    create_mcp_tools,
    create_mcp_tools_from_metadata,
)


def _make_tool(name: str) -> BaseTool:
    """Create a mock BaseTool with the given name."""
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    return tool


def _make_mcp_resource(
    available_tools: list[AgentMcpTool] | None = None,
    is_enabled: bool = True,
) -> AgentMcpResourceConfig:
    """Create an AgentMcpResourceConfig for testing."""
    if available_tools is None:
        available_tools = [
            AgentMcpTool(
                name="tool_a",
                description="Tool A",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
    return AgentMcpResourceConfig(
        name="test-mcp",
        description="Test MCP server",
        **{"$resourceType": AgentResourceType.MCP},
        folderPath="/Shared",
        slug="test-slug",
        availableTools=available_tools,
        isEnabled=is_enabled,
    )


class TestDeduplicateTools:
    """Tests for _deduplicate_tools."""

    def test_no_duplicates_unchanged(self) -> None:
        tools = [_make_tool("alpha"), _make_tool("beta"), _make_tool("gamma")]
        result = _deduplicate_tools(tools)
        assert [t.name for t in result] == ["alpha", "beta", "gamma"]

    def test_all_duplicates_get_suffix(self) -> None:
        tools = [_make_tool("search"), _make_tool("search"), _make_tool("search")]
        result = _deduplicate_tools(tools)
        assert [t.name for t in result] == ["search_1", "search_2", "search_3"]

    def test_partial_duplicates(self) -> None:
        tools = [
            _make_tool("search"),
            _make_tool("calc"),
            _make_tool("search"),
        ]
        result = _deduplicate_tools(tools)
        assert result[0].name == "search_1"
        assert result[1].name == "calc"
        assert result[2].name == "search_2"

    def test_empty_list(self) -> None:
        assert _deduplicate_tools([]) == []

    def test_single_tool(self) -> None:
        tools = [_make_tool("only")]
        result = _deduplicate_tools(tools)
        assert result[0].name == "only"


class TestFilterTools:
    """Tests for _filter_tools."""

    def test_filter_keeps_matching_tools(self) -> None:
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        cfg = _make_mcp_resource(
            available_tools=[
                AgentMcpTool(
                    name="a",
                    description="A",
                    inputSchema={"type": "object", "properties": {}},
                ),
                AgentMcpTool(
                    name="c",
                    description="C",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]
        )
        result = _filter_tools(tools, cfg)
        assert [t.name for t in result] == ["a", "c"]

    def test_filter_removes_all_when_none_match(self) -> None:
        tools = [_make_tool("x"), _make_tool("y")]
        cfg = _make_mcp_resource(
            available_tools=[
                AgentMcpTool(
                    name="z",
                    description="Z",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]
        )
        result = _filter_tools(tools, cfg)
        assert result == []

    def test_filter_empty_tools_list(self) -> None:
        cfg = _make_mcp_resource()
        result = _filter_tools([], cfg)
        assert result == []


class TestCreateMcpTools:
    """Tests for create_mcp_tools async context manager."""

    @pytest.mark.asyncio
    async def test_missing_uipath_url_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("UIPATH_URL", raising=False)
        monkeypatch.delenv("UIPATH_ACCESS_TOKEN", raising=False)

        cfg = _make_mcp_resource()
        with pytest.raises(ValueError, match="UIPATH_URL"):
            async with create_mcp_tools(cfg):
                pass

    @pytest.mark.asyncio
    async def test_missing_access_token_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UIPATH_URL", "https://example.com")
        monkeypatch.delenv("UIPATH_ACCESS_TOKEN", raising=False)

        cfg = _make_mcp_resource()
        with pytest.raises(ValueError, match="UIPATH_ACCESS_TOKEN"):
            async with create_mcp_tools(cfg):
                pass

    @pytest.mark.asyncio
    async def test_disabled_configs_yield_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UIPATH_URL", "https://example.com")
        monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "test-token")

        cfg = _make_mcp_resource(is_enabled=False)
        async with create_mcp_tools(cfg) as tools:
            assert tools == []

    @pytest.mark.asyncio
    async def test_all_disabled_in_list_yield_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UIPATH_URL", "https://example.com")
        monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "test-token")

        configs = [
            _make_mcp_resource(is_enabled=False),
            _make_mcp_resource(is_enabled=False),
        ]
        async with create_mcp_tools(configs) as tools:
            assert tools == []


class TestCreateMcpToolsFromMetadata:
    """Tests for create_mcp_tools_from_metadata."""

    @pytest.mark.asyncio
    async def test_creates_tools_from_available_tools(self) -> None:
        mcp_tools = [
            AgentMcpTool(
                name="get_weather",
                description="Get weather data",
                inputSchema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            ),
            AgentMcpTool(
                name="search_docs",
                description="Search documents",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        ]
        cfg = _make_mcp_resource(available_tools=mcp_tools)

        tools = await create_mcp_tools_from_metadata(cfg)

        assert len(tools) == 2
        assert tools[0].name == "get_weather"
        assert tools[1].name == "search_docs"
        assert tools[0].description == "Get weather data"

    @pytest.mark.asyncio
    async def test_tool_metadata_contains_expected_fields(self) -> None:
        mcp_tools = [
            AgentMcpTool(
                name="my_tool",
                description="A tool",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        cfg = _make_mcp_resource(available_tools=mcp_tools)

        tools = await create_mcp_tools_from_metadata(cfg)

        assert tools[0].metadata is not None
        assert tools[0].metadata["tool_type"] == "mcp"
        assert tools[0].metadata["display_name"] == "my_tool"
        assert tools[0].metadata["folder_path"] == "/Shared"
        assert tools[0].metadata["slug"] == "test-slug"

    @pytest.mark.asyncio
    async def test_empty_available_tools_returns_empty(self) -> None:
        cfg = _make_mcp_resource(available_tools=[])

        tools = await create_mcp_tools_from_metadata(cfg)

        assert tools == []

    @pytest.mark.asyncio
    async def test_tool_name_sanitized(self) -> None:
        mcp_tools = [
            AgentMcpTool(
                name="my tool with spaces!",
                description="A tool",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
        cfg = _make_mcp_resource(available_tools=mcp_tools)

        tools = await create_mcp_tools_from_metadata(cfg)

        # sanitize_tool_name replaces spaces with underscores and strips non-alphanum
        assert " " not in tools[0].name
        assert "!" not in tools[0].name
