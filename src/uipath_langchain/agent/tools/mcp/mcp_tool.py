import json
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncGenerator

from langchain_core.tools import BaseTool
from uipath.agent.models.agent import (
    AgentMcpResourceConfig,
    AgentMcpTool,
    AgentMcpToolExecution,
    CachedToolsConfig,
    DynamicToolsConfig,
    McpToolTaskSupport,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import WaitJobRaw
from uipath.platform.orchestrator import Job, JobState

from uipath_langchain._utils.durable_interrupt import durable_interrupt
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from ..utils import sanitize_tool_name
from .mcp_client import McpClient, SessionInfoFactory

logger: logging.Logger = logging.getLogger(__name__)

# _meta keys AgentHubService stamps on a task result to mark it as a UiPath job (see PR adding
# uipath.com/* markers). The MCP client reads them to drive the job as a suspendable child job.
_UIPATH_SOURCE_META_KEY = "uipath.com/source"
_UIPATH_JOB_KEY_META_KEY = "uipath.com/jobKey"
_UIPATH_FOLDER_KEY_META_KEY = "uipath.com/folderKey"
_UIPATH_ORCHESTRATOR_SOURCE = "orchestrator"


def _execution_from_server_tool(tool: Any) -> AgentMcpToolExecution | None:
    """Map an MCP server Tool's ``execution.taskSupport`` into the snapshot model (dynamic mode)."""
    execution = getattr(tool, "execution", None)
    task_support = getattr(execution, "taskSupport", None) if execution else None
    if task_support is None:
        return None
    value = getattr(task_support, "value", task_support)
    return AgentMcpToolExecution(task_support=McpToolTaskSupport(value))


def _is_task_augmentable(mcp_tool: AgentMcpTool) -> bool:
    """Whether the tool advertises MCP task support (``optional`` / ``required``)."""
    execution = getattr(mcp_tool, "execution", None)
    return execution is not None and execution.task_support in (
        McpToolTaskSupport.OPTIONAL,
        McpToolTaskSupport.REQUIRED,
    )


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
                    execution=_execution_from_server_tool(tool),
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

    task_augmentable = _is_task_augmentable(mcp_tool)

    @mockable(
        name=mcp_tool.name,
        description=mcp_tool.description,
        input_schema=mcp_tool.input_schema,
        output_schema=output_schema,
    )
    async def tool_fn(**kwargs: Any) -> Any:
        """Execute an MCP tool call with an ephemeral session.

        When the tool supports MCP tasks, the call starts a UiPath job and suspends the
        agent until it completes (see :func:`_invoke_mcp_tool_as_job`). Otherwise the tool
        is called synchronously; a session disconnect (404) retries once.
        """
        if task_augmentable:
            return await _invoke_mcp_tool_as_job(mcp_tool, mcpClient, kwargs)

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


async def _invoke_mcp_tool_as_job(
    mcp_tool: AgentMcpTool,
    mcpClient: McpClient,
    arguments: dict[str, Any],
) -> Any:
    """Call a task-supporting MCP tool and suspend the agent job until the child completes.

    The task-augmented ``tools/call`` returns a ``CreateTaskResult`` whose ``_meta`` marks it
    as a UiPath Orchestrator job. We then ``interrupt`` with a ``WaitJobRaw`` (exactly like
    :func:`process_tool.create_process_tool`), so the runtime suspends the parent job and
    resumes it with the child job's output when it finishes.

    Args:
        mcp_tool: The MCP tool being invoked.
        mcpClient: The client used to start the task.
        arguments: The tool-call arguments.

    Returns:
        The completed child job's output (parsed JSON when possible).
    """

    @durable_interrupt
    async def start_mcp_job():
        create_result = await mcpClient.call_tool_as_task(
            mcp_tool.name, arguments=arguments
        )
        meta = create_result.meta or {}
        if meta.get(_UIPATH_SOURCE_META_KEY) != _UIPATH_ORCHESTRATOR_SOURCE:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
                title=f"Tool '{mcp_tool.name}' did not start a UiPath job",
                detail=(
                    "The MCP server returned a task that is not a UiPath Orchestrator job "
                    "(missing the uipath.com/source marker), which is not supported."
                ),
            )

        return WaitJobRaw(
            # The resume trigger keys off the job's GUID key (item_key = job.key) and re-fetches the
            # job on resume; the numeric id is required by the model but unused here, hence the 0.
            job=Job(
                id=0,
                key=meta.get(_UIPATH_JOB_KEY_META_KEY),
                folder_key=meta.get(_UIPATH_FOLDER_KEY_META_KEY),
            ),
            process_folder_key=meta.get(_UIPATH_FOLDER_KEY_META_KEY),
        )

    # First run: starts the job and suspends. On resume: returns the resolved Job.
    job = await start_mcp_job()

    if (getattr(job, "state", None) or "").lower() == JobState.FAULTED:
        return str(getattr(job, "info", None) or "Unknown error")

    output_str = await UiPath().jobs.extract_output_async(job)
    if output_str:
        try:
            return json.loads(output_str)
        except (json.JSONDecodeError, TypeError):
            return output_str
    return output_str


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
