import copy
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncGenerator

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from uipath.agent.models.agent import (
    AgentMcpResourceConfig,
    AgentMcpTool,
    DynamicToolsMode,
)
from uipath.eval.mocks import mockable

from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.static_args import handle_static_args
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperReturnType

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

    config_tools_by_name = {t.name: t for t in config.available_tools}

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

        mcp_tools = []
        for tool in server_tools:
            config_tool = config_tools_by_name.get(tool.name)
            argument_properties = config_tool.argument_properties if config_tool else {}
            server_schema = tool.inputSchema
            if config_tool:
                server_schema = _merge_descriptions_from_config(
                    server_schema, config_tool.input_schema
                )
            mcp_tools.append(
                AgentMcpTool(
                    name=tool.name,
                    description=tool.description or "",
                    inputSchema=server_schema,
                    outputSchema=tool.outputSchema,
                    argumentProperties=argument_properties,
                )
            )
    else:
        mcp_tools = config.available_tools
        logger.info(
            f"Using {len(mcp_tools)} tools from resource config for "
            f"server '{config.slug}'"
        )

    tools: list[BaseTool] = []
    for mcp_tool in mcp_tools:
        tool = StructuredToolWithArgumentProperties(
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
            argument_properties=mcp_tool.argument_properties,
        )
        if mcp_tool.argument_properties:
            tool.set_tool_wrappers(awrapper=_mcp_tool_wrapper)
        tools.append(tool)
    return tools


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


async def _mcp_tool_wrapper(
    tool: BaseTool,
    call: ToolCall,
    state: AgentGraphState,
) -> ToolWrapperReturnType:
    """Wrapper that injects static argument values before MCP tool execution."""
    call["args"] = handle_static_args(tool, state, call["args"])
    return await tool.ainvoke(call)


def _merge_descriptions_from_config(
    server_schema: dict[str, Any],
    config_schema: dict[str, Any],
) -> dict[str, Any]:
    """Merge property descriptions from config schema into server schema.

    For each property that exists in both schemas, if the config schema
    has a description, it overwrites the server schema's description.
    This preserves prompt overrides that were configured in agent.json.

    Args:
        server_schema: The fresh schema from the MCP server.
        config_schema: The schema from the agent config with prompt overrides.

    Returns:
        A copy of the server schema with config descriptions merged in.
    """
    server_props = server_schema.get("properties", {})
    config_props = config_schema.get("properties", {})

    if not config_props or not server_props:
        return server_schema

    merged_schema = copy.deepcopy(server_schema)
    merged_props = merged_schema["properties"]

    for prop_name, config_prop in config_props.items():
        if prop_name not in merged_props:
            continue
        if "description" in config_prop:
            merged_props[prop_name]["description"] = config_prop["description"]

    return merged_schema
