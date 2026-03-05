import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncGenerator

from langchain_core.tools import BaseTool
from uipath.agent.models.agent import (
    AgentMcpResourceConfig,
    AgentMcpTool,
    DynamicToolsMode,
)
from uipath.eval.mocks import mockable

from uipath_langchain.agent.tools.base_uipath_structured_tool import (
    BaseUiPathStructuredTool,
)

from ..utils import sanitize_tool_name
from .mcp_client import McpClient, SessionInfoFactory

logger: logging.Logger = logging.getLogger(__name__)


@asynccontextmanager
async def open_mcp_tools(
    config: list[AgentMcpResourceConfig],
) -> AsyncGenerator[list[BaseTool], None]:
    """Connect to UiPath MCP server(s) via McpClient and yield LangChain-compatible tools.

    Wraps create_mcp_tools_and_clients() with automatic client lifecycle management.
    Tools are lazily initialized on first call via the UiPath SDK.

    Args:
        config: List of MCP resource configurations.

    Yields:
        List of BaseTool instances for all enabled MCP resources.
    """
    async with AsyncExitStack() as stack:
        tools, clients = await create_mcp_tools_and_clients(config)
        for client in clients:
            stack.push_async_callback(client.dispose)
        yield tools


async def create_mcp_tools(
    config: AgentMcpResourceConfig,
    mcpClient: McpClient,
) -> list[BaseTool]:
    """Create StructuredTool instances for each MCP tool in the resource config.

    Args:
        config: The MCP resource configuration containing available tools.
        mcpClient: The McpClient instance to use for tool invocations.

    Returns:
        List of BaseTool instances, one for each tool in the config.
        Returns empty list if config.is_enabled is False.

    Behavior depends on config.dynamic_tools:
        - none: Uses tool schemas from config.available_tools (default).
        - schema: Lists tools from the MCP server via mcpClient, but only
          includes tools whose names appear in config.available_tools.
        - all: Lists all tools from the MCP server via mcpClient, ignoring
          config.available_tools entirely.
    """

    if config.is_enabled is False:
        return []

    dynamic_tools = config.dynamic_tools
    logger.info(
        f"Loading MCP tools for server '{config.slug}' "
        f"(dynamic_tools={dynamic_tools.value})"
    )

    if dynamic_tools in (DynamicToolsMode.SCHEMA, DynamicToolsMode.ALL):
        logger.info(f"Fetching tools from MCP server '{config.slug}' via list_tools")
        result = await mcpClient.list_tools()
        server_tools = result.tools
        logger.info(
            f"MCP server '{config.slug}' returned {len(server_tools)} tools: "
            f"{[t.name for t in server_tools]}"
        )

        if dynamic_tools == DynamicToolsMode.SCHEMA:
            allowed_names = {t.name for t in config.available_tools}
            server_tool_names = {t.name for t in server_tools}
            missing = allowed_names - server_tool_names
            for name in missing:
                logger.warning(
                    f"Tool '{name}' is in availableTools for server "
                    f"'{config.slug}' but was not found on the MCP server"
                )
            server_tools = [t for t in server_tools if t.name in allowed_names]
            logger.info(
                f"Filtered to {len(server_tools)} tools matching availableTools"
            )

        mcp_tools = [
            AgentMcpTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
                output_schema=tool.outputSchema,
            )
            for tool in server_tools
        ]
    else:
        mcp_tools = config.available_tools
        logger.info(
            f"Using {len(mcp_tools)} tools from resource config for "
            f"server '{config.slug}'"
        )

    return [
        BaseUiPathStructuredTool(
            name=sanitize_tool_name(mcp_tool.name),
            description=mcp_tool.description,
            args_schema=mcp_tool.input_schema,
            coroutine=build_mcp_tool(mcp_tool, mcpClient),
            metadata={
                "tool_type": "mcp",
                "display_name": mcp_tool.name,
                "folder_path": config.folder_path,
                "slug": config.slug,
            },
        )
        for mcp_tool in mcp_tools
    ]


def build_mcp_tool(mcp_tool: AgentMcpTool, mcpClient: McpClient) -> Any:
    output_schema: Any
    if mcp_tool.output_schema:
        output_schema = mcp_tool.output_schema
    else:
        output_schema = {"type": "object", "properties": {}}

    @mockable(
        name=mcp_tool.name,
        description=mcp_tool.description,
        input_schema=mcp_tool.input_schema,
        output_schema=output_schema,
    )
    async def tool_fn(**kwargs: Any) -> Any:
        """Execute MCP tool call with ephemeral session.

        If a session disconnect error occurs (e.g., 404 or session terminated),
        the tool will retry once by re-initializing the session.
        """
        result = await mcpClient.call_tool(mcp_tool.name, arguments=kwargs)
        logger.info(f"Tool call successful for {mcp_tool.name}")
        content = result.content if hasattr(result, "content") else result
        if isinstance(content, list):
            return [
                item.model_dump(exclude_none=True)
                if hasattr(item, "model_dump")
                else item
                for item in content
            ]
        if hasattr(content, "model_dump"):
            return content.model_dump(exclude_none=True)
        return content

    return tool_fn


async def create_mcp_tools_and_clients(
    resources: list[AgentMcpResourceConfig],
    session_info_factory: SessionInfoFactory | None = None,
    terminate_on_close: bool = True,
) -> tuple[list[BaseTool], list[McpClient]]:
    """Create MCP tools from a list of MCP resource configurations.

    Iterates over all MCP resources and creates tools for each enabled MCP
    server. Each MCP server gets its own McpClient instance.

    The MCP server URL is loaded lazily on first tool call via the UiPath SDK,
    using environment variables (UIPATH_URL, UIPATH_ACCESS_TOKEN).

    Args:
        resources: List of MCP resource configurations.
        session_info_factory: Factory for creating SessionInfo instances.
            Defaults to the base ``SessionInfoFactory``.  Pass
            ``SessionInfoDebugStateFactory()`` for playground mode.
        terminate_on_close: Whether to terminate the MCP session on close.

    Returns:
        A tuple of (tools, mcp_clients) where:
        - tools: List of BaseTool instances for all MCP resources
        - mcp_clients: List of McpClient instances that need to be closed when done

    Note:
        The caller is responsible for closing the McpClient instances when done.
        Each McpClient manages its own session lifecycle with automatic 404 recovery.
    """
    tools: list[BaseTool] = []
    clients: list[McpClient] = []

    for resource in resources:
        if resource.is_enabled is False:
            logger.info(f"Skipping disabled MCP resource '{resource.name}'")
            continue

        logger.info(f"Creating MCP tools for resource '{resource.name}'")

        mcpClient = McpClient(
            config=resource,
            session_info_factory=session_info_factory,
            terminate_on_close=terminate_on_close,
        )
        clients.append(mcpClient)

        resource_tools = await create_mcp_tools(resource, mcpClient)
        tools.extend(resource_tools)
        logger.info(
            f"Created {len(resource_tools)} tools for MCP resource '{resource.name}'"
        )

    return tools, clients
