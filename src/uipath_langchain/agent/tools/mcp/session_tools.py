"""Convert tools from an active MCP SDK session into LangChain tools."""

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp import ClientSession
from mcp.types import CallToolResult, PaginatedRequestParams, Tool


def _content_blocks(result: CallToolResult) -> list[dict[str, Any]]:
    """Serialize MCP content blocks in their wire-compatible representation."""
    return [
        block.model_dump(by_alias=True, mode="json", exclude_none=True)
        for block in result.content
    ]


def _error_message(result: CallToolResult) -> str:
    """Build a readable LangChain tool error from MCP content blocks."""
    text = [
        block.text
        for block in result.content
        if getattr(block, "type", None) == "text" and hasattr(block, "text")
    ]
    return "\n".join(text) if text else str(_content_blocks(result))


def _convert_tool(session: ClientSession, tool: Tool) -> BaseTool:
    """Bind one discovered MCP tool to its active client session."""

    async def call_tool(**arguments: Any) -> list[dict[str, Any]]:
        result = await session.call_tool(tool.name, arguments=arguments)
        if result.is_error:
            raise ToolException(_error_message(result))
        return _content_blocks(result)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.input_schema,
        coroutine=call_tool,
    )


async def load_mcp_tools(session: ClientSession) -> list[BaseTool]:
    """Discover all tools from an active MCP session and bind them to LangChain.

    Args:
        session: An initialized MCP SDK ``ClientSession`` whose lifetime covers
            every invocation of the returned tools.

    Returns:
        LangChain tools backed by the supplied MCP session.
    """
    tools: list[Tool] = []
    cursor: str | None = None
    while True:
        params = PaginatedRequestParams(cursor=cursor) if cursor is not None else None
        page = await session.list_tools(params=params)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not cursor:
            break
    return [_convert_tool(session, tool) for tool in tools]
