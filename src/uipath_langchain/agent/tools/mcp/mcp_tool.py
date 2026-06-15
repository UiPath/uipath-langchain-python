import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator

from langchain_core.tools import BaseTool
from uipath.agent.models.agent import (
    AgentMcpResourceConfig,
    AgentMcpTool,
    CachedToolsConfig,
    DynamicToolsConfig,
)
from uipath.eval.mocks import mockable

from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from ..utils import sanitize_tool_name
from .jobs import (
    JOB_PROTOCOL_VERSION,
    JobStart,
    build_fetch_meta,
    build_start_meta,
    read_job_handle,
)
from .mcp_client import McpClient, SessionInfoFactory

if TYPE_CHECKING:
    from .jobs import McpJobExecutor, UiPathJobHandle

logger: logging.Logger = logging.getLogger(__name__)


def _normalize_tool_result(result: Any) -> Any:
    """Reduce a ``CallToolResult`` to the plain value the tool should return.

    Extracts ``.content`` and ``model_dump``s any structured blocks, matching the
    shape the LLM expects from an MCP tool call.
    """
    content = result.content if hasattr(result, "content") else result
    if isinstance(content, list):
        return [
            item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item
            for item in content
        ]
    if hasattr(content, "model_dump"):
        return content.model_dump(exclude_none=True)
    return content


async def _invoke_job_aware(
    mcp_tool: AgentMcpTool,
    mcpClient: McpClient,
    kwargs: dict[str, Any],
) -> Any:
    """Invoke a tool on a job-aware server through the injected ``McpJobExecutor``.

    Sends the START ``uipath.com/job`` ``_meta`` on the call; if the server starts a
    job it returns a handle and the executor suspends/awaits it, then FETCHes the
    result with a follow-up ``tools/call``. Non-job tools just return their result.
    """
    executor = mcpClient.job_executor
    assert executor is not None  # guarded by the caller
    version = min(JOB_PROTOCOL_VERSION, mcpClient.job_version or JOB_PROTOCOL_VERSION)

    async def start() -> JobStart:
        result = await mcpClient.call_tool(
            mcp_tool.name, arguments=kwargs, meta=build_start_meta(version)
        )
        handle = read_job_handle(result.meta)
        if handle is not None:
            logger.info(f"Tool '{mcp_tool.name}' started UiPath job {handle.job_key}")
            return JobStart(handle=handle)
        return JobStart(handle=None, result=_normalize_tool_result(result))

    async def fetch(handle: "UiPathJobHandle") -> Any:
        result = await mcpClient.call_tool(
            mcp_tool.name, arguments=None, meta=build_fetch_meta(handle)
        )
        return _normalize_tool_result(result)

    return await executor.run(start=start, fetch=fetch, tool_name=mcp_tool.name)


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

    Behavior depends on config.tools_configuration.discovery_mode:
        - Cached (default when unset): Uses the tools and schemas saved in
          config.available_tools.
        - Dynamic with allow_all=True: Lists all tools from the MCP
          server via mcpClient, ignoring config.available_tools as a source
          of truth.
        - Dynamic with allow_all=False: Lists tools from the MCP server,
          then filters by names present in config.available_tools (live
          schemas, curated tool set).
        Argument properties from the snapshot are carried over by matching tool name.
    """

    if config.is_enabled is False:
        return []

    discovery_mode = (
        config.tools_configuration.discovery_mode
        if config.tools_configuration is not None
        else CachedToolsConfig()
    )
    logger.info(
        f"Loading MCP tools for server '{config.slug}' "
        f"(discovery_mode={discovery_mode.type})"
    )

    config_tools_by_name = {t.name: t for t in config.available_tools}

    if isinstance(discovery_mode, DynamicToolsConfig):
        logger.info(f"Fetching tools from MCP server '{config.slug}' via list_tools")
        result = await mcpClient.list_tools()
        server_tools = result.tools
        logger.info(
            f"MCP server '{config.slug}' returned {len(server_tools)} tools: "
            f"{[t.name for t in server_tools]}"
        )

        if not discovery_mode.allow_all:
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
            mcp_tools.append(
                AgentMcpTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                    output_schema=tool.outputSchema,
                    argument_properties=argument_properties,
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
        tools.append(
            StructuredToolWithArgumentProperties(
                name=sanitize_tool_name(mcp_tool.name),
                description=mcp_tool.description,
                args_schema=mcp_tool.input_schema,
                coroutine=build_mcp_tool(mcp_tool, mcpClient),
                output_type=Any,
                metadata={
                    "tool_type": "mcp",
                    "display_name": mcp_tool.name,
                    "folder_path": config.folder_path,
                    "slug": config.slug,
                },
                argument_properties=mcp_tool.argument_properties,
            )
        )
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

        When the server advertised ``uipath.com/job`` support and an executor is
        configured, the call is routed through the executor so a long-running
        UiPath job suspends/resumes the agent instead of blocking.
        """
        if mcpClient.is_job_aware and mcpClient.job_executor is not None:
            return await _invoke_job_aware(mcp_tool, mcpClient, kwargs)

        result = await mcpClient.call_tool(mcp_tool.name, arguments=kwargs)
        logger.info(f"Tool call successful for {mcp_tool.name}")
        return _normalize_tool_result(result)

    return tool_fn


async def create_mcp_tools_and_clients(
    resources: list[AgentMcpResourceConfig],
    session_info_factory: SessionInfoFactory | None = None,
    terminate_on_close: bool = True,
    job_executor: "McpJobExecutor | None" = None,
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
        job_executor: Executor that awaits long-running UiPath jobs started behind
            a ``tools/call`` on a job-aware server (``uipath.com/job``). Defaults to
            :class:`LangGraphJobExecutor`, which suspends/resumes the agent — so a
            deployed LangGraph agent gains the behavior on package upgrade, with no
            ``agent.json`` change. Pass ``BlockingJobExecutor`` for non-LangGraph
            hosts.

    Returns:
        A tuple of (tools, mcp_clients) where:
        - tools: List of BaseTool instances for all MCP resources
        - mcp_clients: List of McpClient instances that need to be closed when done

    Note:
        The caller is responsible for closing the McpClient instances when done.
        Each McpClient manages its own session lifecycle with automatic 404 recovery.
    """
    if job_executor is None:
        # Lazy import: this package's default is LangGraph suspend/resume.
        from .job_executor import LangGraphJobExecutor

        job_executor = LangGraphJobExecutor()

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
            job_executor=job_executor,
        )
        clients.append(mcpClient)

        resource_tools = await create_mcp_tools(resource, mcpClient)
        tools.extend(resource_tools)
        logger.info(
            f"Created {len(resource_tools)} tools for MCP resource '{resource.name}'"
        )

    return tools, clients
