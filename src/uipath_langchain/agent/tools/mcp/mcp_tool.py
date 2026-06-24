import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncGenerator

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
from .mcp_client import McpClient, SessionInfoFactory

logger: logging.Logger = logging.getLogger(__name__)


def _breaking_schema_change(cached: dict[str, Any], live: dict[str, Any]) -> bool:
    """Whether the live input schema differs from the cached one in a way that would
    break a call the model built from the cached schema.

    Treated as breaking (the model cannot produce a valid call without seeing the new
    schema): a newly required parameter, a cached parameter the server dropped or
    renamed, or a type change on a shared parameter. Purely additive or cosmetic
    changes (new optional params, reworded descriptions) are not breaking, so the call
    proceeds against the cached schema.
    """
    cached_props = cached.get("properties") or {}
    live_props = live.get("properties") or {}
    cached_required = set(cached.get("required") or [])
    live_required = set(live.get("required") or [])

    if live_required - cached_required:
        return True
    if set(cached_props) - set(live_props):
        return True
    for name in set(cached_props) & set(live_props):
        if (cached_props.get(name) or {}).get("type") != (
            live_props.get(name) or {}
        ).get("type"):
            return True
    return False


def _describe_param(name: str, spec: Any, required: bool) -> str:
    """Format a parameter for the schema-change message, e.g. ``city (string, optional)``.

    Falls back to just the name (optionally with ``(optional)``) when the param has no
    simple JSON-schema ``type`` (unions, ``$ref``, etc.), to avoid a misleading hint.
    """
    type_hint = spec.get("type") if isinstance(spec, dict) else None
    attrs: list[str] = [type_hint] if isinstance(type_hint, str) else []
    if not required:
        attrs.append("optional")
    return f"{name} ({', '.join(attrs)})" if attrs else name


def _schema_change_message(name: str, schema: dict[str, Any]) -> str:
    """Instruction returned to the model when a cached tool's schema changed, telling
    it to re-issue the call against the refreshed schema."""
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if props:
        params = ", ".join(_describe_param(p, props[p], p in required) for p in props)
    else:
        params = "no parameters"
    return (
        f"The schema for tool '{name}' changed on the MCP server, so the tool was not "
        f"executed. Its parameters have been refreshed to: {params}. "
        f"Call '{name}' again using these parameters."
    )


def _tool_removed_message(name: str) -> str:
    """Instruction returned to the model when a cached tool is no longer exposed by the
    MCP server, telling it to stop calling that tool."""
    return (
        f"The tool '{name}' is no longer available on the MCP server, so it was not "
        f"executed. Do not call '{name}' again; use a different tool or tell the user "
        f"it is unavailable."
    )


async def _refresh_tool_schema(
    mcp_tool: AgentMcpTool,
    mcpClient: McpClient,
    tool_holder: dict[str, BaseTool] | None = None,
) -> str | None:
    """Fetch the live tool schema before invoking a cached tool and self-heal on drift.

    Lists the tools from the server and compares the live input schema with the cached
    one. If the change would break a call built from the cached schema, the cached
    snapshot and the schema the model is bound to are updated to the live schema and a
    retry instruction is returned; the caller must then NOT run the stale call. The
    ReAct loop re-binds tools on the next LLM turn, so the model re-issues the call
    against the refreshed schema.

    Returns None to let the call proceed against the cached schema when there is no
    breaking change, or, as a non-fatal fallback, when the live schema cannot be
    fetched. When the tool is missing from the live list (removed or renamed), returns
    a message telling the model the tool is gone so it does not retry a doomed call.

    Note: tools that carry static argument bindings (non-empty argument_properties) are
    re-bound each turn from a cached copy, so the rebind may not reach the model; such
    tools fall back to the server's own validation error rather than self-healing.
    """
    try:
        result = await mcpClient.list_tools()
    except Exception as exc:  # noqa: BLE001 - refresh is best effort
        logger.warning(
            f"Could not refresh schema for tool '{mcp_tool.name}' before call; "
            f"using cached schema. Error: {exc}"
        )
        return None

    fresh = next((t for t in result.tools if t.name == mcp_tool.name), None)
    if fresh is None:
        logger.warning(
            f"Tool '{mcp_tool.name}' is no longer exposed by the MCP server; "
            f"the model will be told to stop calling it"
        )
        return _tool_removed_message(mcp_tool.name)

    if not _breaking_schema_change(mcp_tool.input_schema, fresh.inputSchema):
        return None

    logger.warning(
        f"MCP tool '{mcp_tool.name}' schema changed on the server since design time; "
        f"refreshing the bound schema and asking the model to retry"
    )
    # Heal: update the cached baseline and the schema the model is bound to, so the
    # next LLM turn re-binds the live schema and the model can build a valid call.
    mcp_tool.input_schema = fresh.inputSchema
    mcp_tool.output_schema = fresh.outputSchema
    if fresh.description:
        mcp_tool.description = fresh.description
    tool = tool_holder.get("tool") if tool_holder else None
    if tool is not None:
        tool.args_schema = fresh.inputSchema
        if fresh.description:
            tool.description = fresh.description
    return _schema_change_message(mcp_tool.name, fresh.inputSchema)


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

    Behavior depends on config.tools_configuration.discovery_mode (defaults to
    Cached when tools_configuration is unset):
        - Cached: Uses the tools and schemas saved in config.available_tools. When
          the cached config has refresh_schema_before_call=True (the default), the
          live tool schema is fetched immediately before each tool invocation; if it
          changed in a breaking way, the bound schema is refreshed and the model is
          asked to retry the call against the live schema (self-healing).
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
    refresh_schema_before_call = (
        isinstance(discovery_mode, CachedToolsConfig)
        and discovery_mode.refresh_schema_before_call
    )
    logger.info(
        f"Loading MCP tools for server '{config.slug}' "
        f"(discovery_mode={discovery_mode.type}, "
        f"refresh_schema_before_call={refresh_schema_before_call})"
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
        # The holder gives the tool's coroutine a reference back to the tool wrapper so
        # it can refresh its own args_schema on schema drift (see _refresh_tool_schema).
        tool_holder: dict[str, BaseTool] = {}
        structured_tool = StructuredToolWithArgumentProperties(
            name=sanitize_tool_name(mcp_tool.name),
            description=mcp_tool.description,
            args_schema=mcp_tool.input_schema,
            coroutine=build_mcp_tool(
                mcp_tool, mcpClient, refresh_schema_before_call, tool_holder
            ),
            output_type=Any,
            metadata={
                "tool_type": "mcp",
                "display_name": mcp_tool.name,
                "folder_path": config.folder_path,
                "slug": config.slug,
            },
            argument_properties=mcp_tool.argument_properties,
        )
        tool_holder["tool"] = structured_tool
        tools.append(structured_tool)
    return tools


def _normalize_tool_result(result: Any) -> Any:
    """Normalize an MCP ``call_tool`` result into JSON-serializable content."""
    content = result.content if hasattr(result, "content") else result
    if isinstance(content, list):
        return [
            item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item
            for item in content
        ]
    if hasattr(content, "model_dump"):
        return content.model_dump(exclude_none=True)
    return content


def build_mcp_tool(
    mcp_tool: AgentMcpTool,
    mcpClient: McpClient,
    refresh_schema_before_call: bool = False,
    tool_holder: dict[str, BaseTool] | None = None,
) -> Any:
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

        When ``refresh_schema_before_call`` is set (cached discovery mode), the live
        tool schema is fetched first. If it changed in a breaking way, the tool is not
        executed: the bound schema is refreshed and a retry instruction is returned so
        the model re-issues the call against the live schema on the next turn.

        If a session disconnect error occurs (e.g., 404 or session terminated),
        the tool will retry once by re-initializing the session.
        """
        if refresh_schema_before_call:
            retry_message = await _refresh_tool_schema(mcp_tool, mcpClient, tool_holder)
            if retry_message is not None:
                return retry_message
        result = await mcpClient.call_tool(mcp_tool.name, arguments=kwargs)
        logger.info(f"Tool call successful for {mcp_tool.name}")
        return _normalize_tool_result(result)

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
