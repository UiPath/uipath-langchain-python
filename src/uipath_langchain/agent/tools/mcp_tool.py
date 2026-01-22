import asyncio
import os
from collections import Counter, defaultdict
from contextlib import AsyncExitStack, asynccontextmanager
from itertools import chain
from typing import Any, AsyncGenerator

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.agent.models.agent import AgentMcpResourceConfig, AgentMcpTool
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.orchestrator.mcp import McpServer

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model

from .utils import sanitize_tool_name


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


async def create_mcp_tools_from_metadata(
    config: AgentMcpResourceConfig,
) -> list[BaseTool]:
    """Create individual StructuredTool instances for each MCP tool in the resource config.

    Each tool manages its own session lifecycle - creating, using, and cleaning up
    the MCP connection within the tool invocation.
    """

    if config.is_enabled is False:
        return []

    sdk = UiPath()
    mcpServer: McpServer = await sdk.mcp.retrieve_async(
        slug=config.slug, folder_path=config.folder_path
    )

    default_client_kwargs = get_httpx_client_kwargs()
    client_kwargs = {
        **default_client_kwargs,
        "headers": {"Authorization": f"Bearer {sdk._config.secret}"},
        "timeout": httpx.Timeout(600),
    }

    # Lazy import to improve cold start time
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    tools: list[BaseTool] = []

    for mcp_tool in config.available_tools:
        tool_name = sanitize_tool_name(mcp_tool.name)
        input_model: Any = create_model(mcp_tool.input_schema)

        def get_tool_coroutine(mcp_tool: AgentMcpTool, input_model: Any) -> Any:
            output_schema: Any
            if mcp_tool.output_schema:
                output_schema = create_model(mcp_tool.output_schema)
            else:
                output_schema = create_model({"type": "object", "properties": {}})

            @mockable(
                name=mcp_tool.name,
                description=mcp_tool.description,
                input_schema=input_model.model_json_schema(),
                output_schema=output_schema.model_json_schema(),
            )
            async def tool_fn(**kwargs: Any) -> Any:
                """Execute MCP tool call with ephemeral session."""
                async with AsyncExitStack() as stack:
                    # Create HTTP client
                    http_client = await stack.enter_async_context(
                        httpx.AsyncClient(**client_kwargs)
                    )

                    # Create streamable connection
                    read, write, _ = await stack.enter_async_context(
                        streamable_http_client(
                            url=f"{mcpServer.mcp_url}", http_client=http_client
                        )
                    )

                    # Create and initialize session
                    session = await stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()

                    # Call the tool
                    result = await session.call_tool(mcp_tool.name, arguments=kwargs)
                    return result.content if hasattr(result, "content") else result

            return tool_fn

        tool = StructuredTool(
            name=tool_name,
            description=mcp_tool.description,
            args_schema=input_model,
            coroutine=get_tool_coroutine(mcp_tool, input_model),
            metadata={
                "tool_type": "mcp",
                "display_name": mcp_tool.name,
                "folder_path": config.folder_path,
                "slug": config.slug,
            },
        )

        tools.append(tool)

    return tools
