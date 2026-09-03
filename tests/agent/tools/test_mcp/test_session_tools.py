"""Tests for binding active MCP sessions to LangChain tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from uipath_langchain.agent.tools.mcp import load_mcp_tools


@pytest.mark.asyncio
async def test_load_mcp_tools_binds_discovery_and_invocation() -> None:
    """Discovered MCP 2 schemas and results are usable as LangChain tools."""
    session = MagicMock()
    session.list_tools = AsyncMock(
        side_effect=[
            ListToolsResult(
                nextCursor="next-page",
                tools=[
                    Tool(
                        name="echo",
                        description="Echo a value",
                        inputSchema={
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                        },
                    )
                ],
            ),
            ListToolsResult(tools=[]),
        ]
    )
    session.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="hello")],
            isError=False,
        )
    )

    tools = await load_mcp_tools(session)
    result = await tools[0].ainvoke({"value": "hello"})

    assert result == [{"type": "text", "text": "hello"}]
    assert session.list_tools.await_count == 2
    assert session.list_tools.await_args_list[0].kwargs == {"params": None}
    assert session.list_tools.await_args_list[1].kwargs["params"].cursor == "next-page"
    session.call_tool.assert_awaited_once_with("echo", arguments={"value": "hello"})


@pytest.mark.asyncio
async def test_load_mcp_tools_maps_mcp_failures_to_tool_errors() -> None:
    """Protocol-level tool failures retain their server-provided message."""
    session = MagicMock()
    session.list_tools = AsyncMock(
        return_value=ListToolsResult(
            tools=[Tool(name="fail", inputSchema={"type": "object"})]
        )
    )
    session.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[TextContent(type="text", text="server rejected the call")],
            isError=True,
        )
    )

    tools = await load_mcp_tools(session)

    with pytest.raises(ToolException, match="server rejected the call"):
        await tools[0].ainvoke({})
