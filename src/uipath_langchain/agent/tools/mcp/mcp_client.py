"""MCP Session management for tool invocations.

This module provides a session class that manages the lifecycle of MCP connections,
including automatic reconnection on session disconnect errors.
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import httpx
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.shared.message import SessionMessage
from mcp.types import CallToolResult
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.runtime.base import UiPathDisposableProtocol

from .streamable_http import SessionInfo, streamable_http_client

if TYPE_CHECKING:
    from uipath.agent.models.agent import AgentMcpResourceConfig
    from uipath.platform.orchestrator.mcp import McpServer

logger = logging.getLogger(__name__)


class SessionInfoFactory:
    """Creates SessionInfo instances for MCP servers.

    The default implementation returns a plain ``SessionInfo``.
    Subclass and override ``create_session`` to customise behaviour
    (e.g. ``SessionInfoDebugStateFactory``).
    """

    def create_session(self, mcp_server: "McpServer") -> SessionInfo:
        """Create a SessionInfo for the given MCP server."""
        logger.info(
            f"Creating session for server '{mcp_server.slug}' "
            f"in folder '{mcp_server.folder_key}'"
        )
        return SessionInfo()


class McpClient(UiPathDisposableProtocol):
    """Manages an MCP session for tool invocations.

    This class handles the lifecycle of MCP connections with two distinct phases:

    1. **Client Initialization** (first call):
       - Instantiates UiPath SDK to retrieve MCP server URL
       - Creates HTTP client with authorization headers
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
        config: "AgentMcpResourceConfig",
        timeout: httpx.Timeout | None = None,
        max_retries: int = 1,
        session_info_factory: SessionInfoFactory | None = None,
        terminate_on_close: bool = True,
    ) -> None:
        """Initialize the MCP tool session.

        The MCP server URL and authorization headers are retrieved lazily
        from the UiPath SDK on first use, using the config's slug and folder_path.

        Args:
            config: The MCP resource configuration containing slug and folder_path.
            timeout: Optional timeout configuration for HTTP requests.
            max_retries: Maximum number of retries on session disconnect errors.
            session_info_factory: Factory for creating SessionInfo instances.
                Defaults to ``SessionInfoFactory`` which returns a plain SessionInfo.
        """
        self._config = config
        self._timeout = timeout or httpx.Timeout(600)
        self._max_retries = max_retries
        self._session_info_factory = session_info_factory or SessionInfoFactory()
        self._terminate_on_close = terminate_on_close

        # URL and headers are resolved lazily from SDK
        self._url: str | None = None
        self._headers: dict[str, str] = {}

        # Lock for both client initialization and session reinitialization
        self._lock = asyncio.Lock()

        # Client state (created once, reused across session reinitializations)
        self._http_client: httpx.AsyncClient | None = None
        self._read_stream: (
            MemoryObjectReceiveStream[SessionMessage | Exception] | None
        ) = None
        self._write_stream: MemoryObjectSendStream[SessionMessage] | None = None
        self._session_info: SessionInfo | None = None
        self._stack: AsyncExitStack | None = None

        # Session state (can be reinitialized without recreating client)
        self._session: ClientSession | None = None
        self._client_initialized: bool = False

    async def get_session_id(self) -> str | None:
        """Get the current session ID from the SessionInfo."""
        if self._session_info is None:
            return None
        return await self._session_info.get_session_id()

    @property
    def is_client_initialized(self) -> bool:
        """Check if the HTTP client and streamable connection are initialized."""
        return self._client_initialized

    async def _initialize_client(self) -> None:
        """Initialize the HTTP client and streamable connection.

        This is called once on first use. Creates:
        - UiPath SDK instance to retrieve MCP server URL
        - httpx.AsyncClient with authorization headers
        - Streamable HTTP connection (read/write streams)
        - ClientSession

        Then calls _initialize_session() to complete the MCP handshake.
        """
        logger.debug(
            f"Initializing MCP client for '{self._config.slug}' "
            f"in folder '{self._config.folder_path}'"
        )

        # Lazy import to improve cold start time
        from uipath.platform import UiPath

        # Retrieve MCP server URL from SDK
        sdk = UiPath()
        mcp_server = await sdk.mcp.retrieve_async(
            slug=self._config.slug, folder_path=self._config.folder_path
        )

        if mcp_server.mcp_url is None:
            raise ValueError(f"MCP server '{self._config.slug}' has no URL configured")

        self._url = mcp_server.mcp_url
        self._headers = {"Authorization": f"Bearer {sdk._config.secret}"}

        logger.debug(f"Retrieved MCP server URL: {self._url}")

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

        # Create session info for tracking session ID
        self._session_info = self._session_info_factory.create_session(mcp_server)

        # Load previously stored session ID (no-op for base SessionInfo,
        # triggers lazy load from debug state for SessionInfoDebugState)
        existing = await self._session_info.get_session_id()
        if existing:
            logger.info(f"Loaded existing session ID from session info: {existing}")

        # Create streamable HTTP connection
        (
            self._read_stream,
            self._write_stream,
        ) = await self._stack.enter_async_context(
            streamable_http_client(
                url=self._url,
                http_client=self._http_client,
                session_info=self._session_info,
                terminate_on_close=self._terminate_on_close,
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

        existing_session_id = (
            await self._session_info.get_session_id() if self._session_info else None
        )
        logger.info(
            f"Initializing MCP session (session_info id: {existing_session_id})"
        )

        if existing_session_id is None:
            await self._session.initialize()

            # The transport calls set_session_id during initialize,
            # so we just read the current value here.
            new_session_id = (
                await self._session_info.get_session_id()
                if self._session_info
                else None
            )
            logger.info(f"MCP session initialized with session ID: {new_session_id}")

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
        Clears the session info first so initialize() doesn't send a stale session ID.
        """
        async with self._lock:
            if not self._client_initialized:
                # Client not initialized, do full initialization
                await self._initialize_client()
            else:
                # Clear stale session ID before re-initializing
                if self._session_info:
                    await self._session_info.set_session_id(None)
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

    async def dispose(self) -> None:
        """Dispose of the client and release all resources.

        Implements UiPathDisposableProtocol.
        Releases the HTTP client, streamable connection, and ClientSession.
        After calling dispose(), the client can be reused - a new call_tool()
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
                    self._http_client = None
                    self._read_stream = None
                    self._write_stream = None
                    self._session_info = None
                    self._client_initialized = False

            logger.info("MCP client disposed")
