import asyncio
import logging
import os
from collections import Counter, defaultdict
from contextlib import AsyncExitStack, asynccontextmanager
from itertools import chain
from typing import Any, AsyncGenerator

import httpx
from langchain_core.tools import BaseTool
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.agent.models.agent import (
    AgentMcpResourceConfig,
    AgentMcpTool,
    LowCodeAgentDefinition,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.orchestrator.mcp import McpServer

from uipath_langchain.agent.tools.base_uipath_structured_tool import (
    BaseUiPathStructuredTool,
)

from ..utils import sanitize_tool_name
from .mcp_client import McpClient

logger: logging.Logger = logging.getLogger(__name__)


def _deduplicate_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """Deduplicate tools by appending numeric suffix to duplicate names."""
    counts = Counter(tool.name for tool in tools)
    seen: defaultdict[str, int] = defaultdict(int)

    for tool in tools:
        if counts[tool.name] > 1:
            seen[tool.name] += 1
            tool.name = f"{tool.name}_{seen[tool.name]}"

    return tools


def _filter_tools(tools: list[BaseTool], cfg: AgentMcpResourceConfig) -> list[BaseTool]:
    """Filter tools to only include those in available_tools."""
    allowed = {t.name for t in cfg.available_tools}
    return [t for t in tools if t.name in allowed]


@asynccontextmanager
async def create_mcp_tools(
    config: AgentMcpResourceConfig | list[AgentMcpResourceConfig],
    max_concurrency: int = 5,
) -> AsyncGenerator[list[BaseTool], None]:
    """Connect to UiPath MCP server(s) and yield LangChain-compatible tools."""
    if not (base_url := os.getenv("UIPATH_URL")):
        raise ValueError("UIPATH_URL environment variable is not set")
    if not (access_token := os.getenv("UIPATH_ACCESS_TOKEN")):
        raise ValueError("UIPATH_ACCESS_TOKEN environment variable is not set")

    configs = config if isinstance(config, list) else [config]
    enabled = [c for c in configs if c.is_enabled is not False]

    if not enabled:
        yield []
        return

    base_url = base_url.rstrip("/")
    semaphore = asyncio.Semaphore(max_concurrency)

    default_client_kwargs = get_httpx_client_kwargs()
    client_kwargs = {
        **default_client_kwargs,
        "headers": {"Authorization": f"Bearer {access_token}"},
        "timeout": httpx.Timeout(60),
    }

    # Lazy import to improve cold start time
    from langchain_mcp_adapters.tools import load_mcp_tools
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def init_session(
        session: ClientSession, cfg: AgentMcpResourceConfig
    ) -> list[BaseTool]:
        async with semaphore:
            await session.initialize()
            tools = await load_mcp_tools(session)
            for tool in tools:
                tool.metadata = {"tool_type": "mcp", "display_name": tool.name}
            return _filter_tools(tools, cfg)

    async def create_session(
        stack: AsyncExitStack, cfg: AgentMcpResourceConfig
    ) -> ClientSession:
        url = f"{base_url}/agenthub_/mcp/{cfg.folder_path}/{cfg.slug}"
        http_client = await stack.enter_async_context(
            httpx.AsyncClient(**client_kwargs)
        )
        read, write, _ = await stack.enter_async_context(
            streamable_http_client(url=url, http_client=http_client)
        )
        return await stack.enter_async_context(ClientSession(read, write))

    async with AsyncExitStack() as stack:
        sessions = [(await create_session(stack, cfg), cfg) for cfg in enabled]
        results = await asyncio.gather(*[init_session(s, cfg) for s, cfg in sessions])
        yield _deduplicate_tools(list(chain.from_iterable(results)))


async def create_mcp_tools_from_metadata_for_mcp_server(
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
    """

    if config.is_enabled is False:
        return []

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
        for mcp_tool in config.available_tools
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
        return result.content if hasattr(result, "content") else result

    return tool_fn


async def create_mcp_tools_from_agent(
    agent: LowCodeAgentDefinition,
) -> tuple[list[BaseTool], list[McpClient]]:
    """Create MCP tools from a LowCodeAgentDefinition.

    Iterates over all MCP resources in the agent definition and creates tools
    for each enabled MCP server. Each MCP server gets its own McpClient instance.

    The UiPath SDK is lazily initialized inside this function using environment
    variables (UIPATH_URL, UIPATH_ACCESS_TOKEN).

    Args:
        agent: The agent definition containing MCP resources.

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
    sdk: UiPath = UiPath()  # Lazy initialization of SDK

    for resource in agent.resources:
        if not isinstance(resource, AgentMcpResourceConfig):
            continue

        if resource.is_enabled is False:
            logger.info(f"Skipping disabled MCP resource '{resource.name}'")
            continue

        logger.info(f"Creating MCP tools for resource '{resource.name}'")

        mcpServer: McpServer = await sdk.mcp.retrieve_async(
            slug=resource.slug, folder_path=resource.folder_path
        )
        if mcpServer.mcp_url is None:
            raise ValueError(f"MCP server '{resource.slug}' has no URL configured")

        mcpClient = McpClient(
            mcpServer.mcp_url,
            headers={"Authorization": f"Bearer {sdk._config.secret}"},
        )
        clients.append(mcpClient)

        resource_tools = await create_mcp_tools_from_metadata_for_mcp_server(
            resource, mcpClient
        )
        tools.extend(resource_tools)
        logger.info(
            f"Created {len(resource_tools)} tools for MCP resource '{resource.name}'"
        )

    return tools, clients
