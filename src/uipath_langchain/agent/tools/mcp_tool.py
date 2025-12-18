import asyncio
import os
from collections import Counter, defaultdict
from contextlib import AsyncExitStack, asynccontextmanager
from itertools import chain

import httpx
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.agent.models.agent import AgentMcpResourceConfig


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
):
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

    async def init_session(
        session: ClientSession, cfg: AgentMcpResourceConfig
    ) -> list[BaseTool]:
        async with semaphore:
            await session.initialize()
            tools = await load_mcp_tools(session)
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
