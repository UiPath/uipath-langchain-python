"""MCP Session management for tool invocations.

This module provides a session class that manages the lifecycle of MCP connections,
including automatic reconnection on session disconnect errors.
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.client.streamable_http import (
    GetSessionIdCallback,
    streamable_http_client,
)
from mcp.shared.exceptions import McpError
from mcp.shared.message import SessionMessage
from mcp.types import CallToolResult
from uipath._utils._ssl_context import get_httpx_client_kwargs

logger = logging.getLogger(__name__)


class McpClient:
    """Manages an MCP session for tool invocations.

    This class handles the lifecycle of MCP connections with two distinct phases:

    1. **Client Initialization** (first call):
       - Creates HTTP client
       - Establishes streamable HTTP connection
       - Creates ClientSession
       - Calls session.initialize() to get session ID

    2. **Session Reinitialization** (on 404 error):
       - Reuses existing HTTP client and streamable connection
       - Calls session.initialize() again to get new session ID

    Thread-safety is ensured via asyncio.Lock for both phases.
    """

    # Error codes that indicate session disconnect/termination
    SESSION_ERROR_CODES = [32600, -32000]

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        max_retries: int = 1,
    ) -> None:
        """Initialize the MCP tool session.

        Args:
            url: The MCP server endpoint URL.
            headers: Optional headers to include in HTTP requests.
            timeout: Optional timeout configuration for HTTP requests.
            max_retries: Maximum number of retries on session disconnect errors.
        """
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout or httpx.Timeout(600)
        self._max_retries = max_retries

        # Lock for both client initialization and session reinitialization
        self._lock = asyncio.Lock()

        # Client state (created once, reused across session reinitializations)
        self._http_client: httpx.AsyncClient | None = None
        self._read_stream: (
            MemoryObjectReceiveStream[SessionMessage | Exception] | None
        ) = None
        self._write_stream: MemoryObjectSendStream[SessionMessage] | None = None
        self._get_session_id: GetSessionIdCallback | None = None
        self._stack: AsyncExitStack | None = None

        # Session state (can be reinitialized without recreating client)
        self._session: ClientSession | None = None
        self._session_id: str | None = None
        self._client_initialized: bool = False

    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._session_id

    @property
    def is_client_initialized(self) -> bool:
        """Check if the HTTP client and streamable connection are initialized."""
        return self._client_initialized

    async def _initialize_client(self) -> None:
        """Initialize the HTTP client and streamable connection.

        This is called once on first use. Creates:
        - httpx.AsyncClient
        - Streamable HTTP connection (read/write streams)
        - ClientSession

        Then calls _initialize_session() to complete the MCP handshake.
        """
        logger.debug("Initializing MCP client")

        # Create exit stack for resource management
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        # Create HTTP client with SSL, proxy, and redirect settings
        default_client_kwargs = get_httpx_client_kwargs()
        client_kwargs = {
            **default_client_kwargs,
            "headers": self._headers,
            "timeout": self._timeout,
        }
        self._http_client = await self._stack.enter_async_context(
            httpx.AsyncClient(**client_kwargs)
        )

        # Create streamable HTTP connection
        (
            self._read_stream,
            self._write_stream,
            self._get_session_id,
        ) = await self._stack.enter_async_context(
            streamable_http_client(
                url=self._url,
                http_client=self._http_client,
            )
        )

        # Create ClientSession (but don't initialize yet)
        # These are guaranteed to be set by the context manager above
        assert self._read_stream is not None
        assert self._write_stream is not None
        self._session = await self._stack.enter_async_context(
            ClientSession(self._read_stream, self._write_stream)
        )

        self._client_initialized = True
        logger.info("MCP client initialized")

        # Now initialize the MCP session
        await self._initialize_session()

    async def _initialize_session(self) -> None:
        """Initialize or reinitialize the MCP session.

        Calls session.initialize() to perform the MCP handshake and obtain
        a session ID from the server. Can be called multiple times on the
        same ClientSession to recover from session disconnects.

        Requires: Client must be initialized first (_initialize_client).
        """
        if self._session is None:
            raise RuntimeError("Cannot initialize session: client not initialized")

        logger.debug(f"Initializing MCP session (previous: {self._session_id})")

        await self._session.initialize()
        self._session_id = self._get_session_id()  # type: ignore[misc]

        logger.info(f"MCP session initialized: {self._session_id}")

    async def _ensure_session(self) -> ClientSession:
        """Ensure client and session are initialized, return the session.

        Thread-safe via lock. Only initializes once; subsequent calls
        return the existing session immediately.

        Returns:
            The initialized ClientSession.
        """
        if not self._client_initialized:
            async with self._lock:
                if not self._client_initialized:
                    await self._initialize_client()

        return self._session  # type: ignore[return-value]

    async def _reinitialize_session(self) -> None:
        """Reinitialize only the MCP session after a disconnect error.

        Thread-safe via lock. Reuses existing HTTP client and streamable
        connection; only performs a new MCP handshake.
        """
        async with self._lock:
            if not self._client_initialized:
                # Client not initialized, do full initialization
                await self._initialize_client()
            else:
                # Client exists, just reinitialize session
                await self._initialize_session()

    def _is_session_error(self, error: McpError) -> bool:
        """Check if an McpError indicates a session disconnect.

        Args:
            error: The McpError to check.

        Returns:
            True if the error indicates a session disconnect.
        """
        return (
            hasattr(error, "error")
            and hasattr(error.error, "code")
            and error.error.code in self.SESSION_ERROR_CODES
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Call an MCP tool with automatic retry on session disconnect.

        On first call, initializes the full client stack. On session
        disconnect (404/32600), reinitializes only the session and retries.

        Args:
            name: The name of the tool to call.
            arguments: Optional arguments to pass to the tool.

        Returns:
            The tool call result.

        Raises:
            McpError: If the tool call fails after all retries.
        """
        retry_count = 0

        while retry_count <= self._max_retries:
            try:
                session = await self._ensure_session()
                logger.debug(
                    f"Calling tool {name} (attempt {retry_count + 1}/{self._max_retries + 1})"
                )
                result = await session.call_tool(name, arguments=arguments)
                logger.info(f"Tool call successful: {name}")
                return result

            except McpError as e:
                logger.info(f"McpError during tool call: {e}")

                if self._is_session_error(e) and retry_count < self._max_retries:
                    logger.warning(
                        f"Session disconnected (error code: {e.error.code}), "
                        f"reinitializing session"
                    )
                    await self._reinitialize_session()
                    retry_count += 1
                    continue
                else:
                    if retry_count >= self._max_retries:
                        logger.error(f"Max retries reached after session error: {e}")
                    else:
                        logger.error(f"Non-retryable MCP error: {e}")
                    raise

        # Should not reach here, but just in case
        raise RuntimeError("Exited retry loop unexpectedly")

    async def close(self) -> None:
        """Close the session and release all resources.

        Releases the HTTP client, streamable connection, and ClientSession.
        After calling close(), the session can be reused - a new call_tool()
        will reinitialize everything.
        """
        async with self._lock:
            if self._stack is not None:
                try:
                    await self._stack.__aexit__(None, None, None)
                except Exception as e:
                    logger.debug(f"Error during cleanup: {e}")
                finally:
                    self._stack = None
                    self._session = None
                    self._session_id = None
                    self._http_client = None
                    self._read_stream = None
                    self._write_stream = None
                    self._get_session_id = None
                    self._client_initialized = False

            logger.info("MCP session closed")
