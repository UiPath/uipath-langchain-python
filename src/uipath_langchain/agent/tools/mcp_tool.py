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
    # Lazy import to improve cold start time
    import contextlib
    import logging
    from collections.abc import AsyncGenerator
    from contextlib import asynccontextmanager

    import anyio
    import httpx
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
    from mcp import ClientSession
    from mcp.client.streamable_http import GetSessionIdCallback, StreamableHTTPTransport
    from mcp.shared._httpx_utils import create_mcp_http_client
    from mcp.shared.message import SessionMessage

    logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def streamable_http_client(
        url: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        session_id: str | None,
        terminate_on_close: bool = False,
    ) -> AsyncGenerator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback,
        ],
        None,
    ]:
        """Client transport for StreamableHTTP.

        Args:
            url: The MCP server endpoint URL.
            http_client: Optional pre-configured httpx.AsyncClient. If None, a default
                client with recommended MCP timeouts will be created. To configure headers,
                authentication, or other HTTP settings, create an httpx.AsyncClient and pass it here.
            terminate_on_close: If True, send a DELETE request to terminate the session when the context exits.

        Yields:
            Tuple containing:
                - read_stream: Stream for reading messages from the server
                - write_stream: Stream for sending messages to the server
                - get_session_id_callback: Function to retrieve the current session ID

        Example:
            See examples/snippets/clients/ for usage patterns.
        """
        read_stream_writer, read_stream = anyio.create_memory_object_stream[
            SessionMessage | Exception
        ](0)
        write_stream, write_stream_reader = anyio.create_memory_object_stream[
            SessionMessage
        ](0)

        # Determine if we need to create and manage the client
        client_provided = http_client is not None
        client = http_client

        if client is None:
            # Create default client with recommended MCP timeouts
            client = create_mcp_http_client()

        transport = StreamableHTTPTransport(url)
        if session_id:
            transport.session_id = session_id

        async with anyio.create_task_group() as tg:
            try:
                logger.debug(f"Connecting to StreamableHTTP endpoint: {url}")

                async with contextlib.AsyncExitStack() as stack:
                    # Only manage client lifecycle if we created it
                    if not client_provided:
                        await stack.enter_async_context(client)

                    def start_get_stream() -> None:
                        tg.start_soon(
                            transport.handle_get_stream, client, read_stream_writer
                        )

                    tg.start_soon(
                        transport.post_writer,
                        client,
                        write_stream_reader,
                        read_stream_writer,
                        write_stream,
                        start_get_stream,
                        tg,
                    )

                    try:
                        yield (read_stream, write_stream, transport.get_session_id)
                    finally:
                        if transport.session_id and terminate_on_close:
                            await transport.terminate_session(client)
                        tg.cancel_scope.cancel()
            finally:
                await read_stream_writer.aclose()
                await write_stream.aclose()

    tools: list[BaseTool] = []
    session_id: str | None = None

    for mcp_tool in config.available_tools:
        tool_name = sanitize_tool_name(mcp_tool.name)

        def build_mcp_tool(mcp_tool: AgentMcpTool) -> Any:
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
                """Execute MCP tool call with ephemeral session."""
                async with AsyncExitStack() as stack:
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

                    # Create HTTP client
                    http_client = await stack.enter_async_context(
                        httpx.AsyncClient(**client_kwargs)
                    )

                    # Create streamable connection
                    nonlocal session_id
                    logger.info(f"Connecting to session {session_id}")
                    read, write, getSessionId = await stack.enter_async_context(
                        streamable_http_client(
                            url=f"{mcpServer.mcp_url}",
                            http_client=http_client,
                            session_id=session_id,
                        )
                    )

                    # Create and initialize session
                    session = await stack.enter_async_context(
                        ClientSession(read, write)
                    )

                    if not session_id:
                        await session.initialize()
                        session_id = getSessionId()
                        logger.info(f"session {session_id} created")

                    # Call the tool
                    result = await session.call_tool(mcp_tool.name, arguments=kwargs)
                    return result.content if hasattr(result, "content") else result

            return tool_fn

        tool = StructuredTool(
            name=tool_name,
            description=mcp_tool.description,
            args_schema=mcp_tool.input_schema,
            coroutine=build_mcp_tool(mcp_tool),
            metadata={
                "tool_type": "mcp",
                "display_name": mcp_tool.name,
                "folder_path": config.folder_path,
                "slug": config.slug,
            },
        )

        tools.append(tool)

    return tools
