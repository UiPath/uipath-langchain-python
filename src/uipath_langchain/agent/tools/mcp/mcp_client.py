"""MCP Session management for tool invocations.

This module provides a session class that manages the lifecycle of MCP connections,
including automatic reconnection on session disconnect errors.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
import httpx2
from mcp import ClientSession
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CallToolResult,
    ListToolsResult,
)
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.runtime.base import UiPathDisposableProtocol

from uipath_langchain._utils import get_execution_folder_path

from .protocol_strategy import (
    ProtocolMode,
    ProtocolStrategy,
    build_protocol_strategy,
    is_session_rejected,
)
from .protocol_strategy import is_session_error as _is_session_error
from .streamable_http import SessionInfo, streamable_http_client

if TYPE_CHECKING:
    from uipath.agent.models.agent import AgentMcpResourceConfig
    from uipath.platform.orchestrator.mcp import McpServer

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _normalize_timeout(
    timeout: httpx.Timeout | httpx2.Timeout | float | None,
) -> httpx2.Timeout | float:
    """Convert the pre-upgrade HTTPX timeout type for the MCP 2 transport."""
    if timeout is None:
        return httpx2.Timeout(600)
    if isinstance(timeout, httpx.Timeout):
        return httpx2.Timeout(
            connect=timeout.connect,
            read=timeout.read,
            write=timeout.write,
            pool=timeout.pool,
        )
    return timeout


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

    2. **Session Reinitialization** (after a terminated session):
       - Reuses the existing HTTP client and persisted session store
       - Replaces the transport and ``ClientSession``
       - Performs a fresh legacy initialization handshake

    Thread-safety is ensured via asyncio.Lock for both phases.
    """

    def __init__(
        self,
        config: "AgentMcpResourceConfig",
        timeout: httpx.Timeout | httpx2.Timeout | float | None = None,
        max_retries: int = 1,
        session_info_factory: SessionInfoFactory | None = None,
        terminate_on_close: bool = True,
        *,
        protocol_mode: ProtocolMode = "legacy",
    ) -> None:
        """Initialize the MCP tool session.

        The MCP server URL and authorization headers are retrieved lazily
        from the UiPath SDK on first use, using the config's display name and
        folder_path.

        Args:
            config: The MCP resource configuration containing name and folder_path.
            timeout: Optional timeout configuration for HTTP requests.
            max_retries: Maximum number of retries on session disconnect errors.
            session_info_factory: Factory for creating SessionInfo instances.
                Defaults to ``SessionInfoFactory`` which returns a plain SessionInfo.
            terminate_on_close: Whether to terminate the server session on
                disposal. Already a no-op in the modern era, which has no session.
            protocol_mode: Which negotiation era to use. ``"legacy"`` (the
                default) sends only the ``initialize`` handshake, preserving
                pre-existing behavior. ``"modern"`` uses ``server/discover``
                only. ``"auto"`` probes for the modern era and falls back to the
                handshake.
        """
        self._config = config
        self._timeout = _normalize_timeout(timeout)
        self._max_retries = max_retries
        self._session_info_factory = session_info_factory or SessionInfoFactory()
        self._terminate_on_close = terminate_on_close
        self._strategy: ProtocolStrategy = build_protocol_strategy(protocol_mode)

        # URL and headers are resolved lazily from SDK
        self._url: str | None = None
        self._headers: dict[str, str] = {}

        # Lock for both client initialization and session reinitialization
        self._lock = asyncio.Lock()

        # Tool list cached in memory and fetched once per client lifetime, with its own
        # lock so a concurrent first call does not deadlock against ``_lock`` (held by
        # session initialization inside ``_execute_with_retry``).
        self._tools_lock = asyncio.Lock()
        self._tools_cache: ListToolsResult | None = None

        # Client state (created once, reused across session reinitializations)
        self._http_client: httpx2.AsyncClient | None = None
        self._session_info: SessionInfo | None = None
        self._stack: AsyncExitStack | None = None
        self._connection_stack: AsyncExitStack | None = None

        # Session state (replaced on recovery while the HTTP client is reused)
        self._session: ClientSession | None = None
        self._client_initialized: bool = False

    @property
    def server_slug(self) -> str:
        """Slug of the configured MCP server."""
        return self._config.slug

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
        - httpx2.AsyncClient with authorization headers
        - Streamable HTTP connection (read/write streams)
        - ClientSession

        Then calls _initialize_session() to complete the MCP handshake.
        """
        folder_path = get_execution_folder_path()
        logger.debug(
            f"Initializing MCP client for '{self._config.name}' "
            f"in folder '{folder_path}'"
        )

        # Lazy import to improve cold start time
        from uipath.platform import UiPath

        # Retrieve MCP server URL from SDK
        sdk = UiPath()
        mcp_server = await sdk.mcp.retrieve_async(
            name=self._config.name,
            folder_path=folder_path,
        )

        if mcp_server.mcp_url is None:
            raise ValueError(f"MCP server '{self._config.name}' has no URL configured")

        self._url = mcp_server.mcp_url
        self._headers = {"Authorization": f"Bearer {sdk._config.secret}"}

        logger.debug(f"Retrieved MCP server URL: {self._url}")

        stack = AsyncExitStack()
        await stack.__aenter__()
        self._stack = stack
        try:
            # Create HTTP client with SSL, proxy, and redirect settings
            client_kwargs = get_httpx_client_kwargs(headers=self._headers)
            client_kwargs["timeout"] = self._timeout
            self._http_client = await stack.enter_async_context(
                httpx2.AsyncClient(**client_kwargs)
            )

            # Create session info for tracking session ID
            self._session_info = self._session_info_factory.create_session(mcp_server)

            # Load a session ID persisted by the AgentHub debug-state integration.
            existing = await self._session_info.get_session_id()
            if existing:
                logger.info(f"Loaded existing session ID from session info: {existing}")

            await self._open_connection()
        except BaseException:
            await stack.aclose()
            self._stack = None
            self._http_client = None
            self._session_info = None
            raise

        self._client_initialized = True
        logger.info("MCP client initialized")

    async def _open_connection(self) -> None:
        """Open a fresh transport and ClientSession over the reusable HTTP client."""
        if self._url is None or self._http_client is None or self._session_info is None:
            raise RuntimeError(
                "Cannot open MCP connection: client prerequisites missing"
            )

        connection_stack = AsyncExitStack()
        await connection_stack.__aenter__()
        try:
            read_stream, write_stream = await connection_stack.enter_async_context(
                streamable_http_client(
                    url=self._url,
                    http_client=self._http_client,
                    session_info=self._session_info,
                    terminate_on_close=self._terminate_on_close,
                    identity=self._strategy.identity,
                )
            )
            self._session = await connection_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            self._connection_stack = connection_stack
            await self._initialize_session()
        except BaseException:
            await connection_stack.aclose()
            self._session = None
            self._connection_stack = None
            raise

    async def _initialize_session(self) -> None:
        """Negotiate the newly-created MCP session through the protocol strategy.

        The strategy owns the era-specific handshake: the legacy ``initialize``
        exchange, or the modern ``server/discover`` probe. MCP 2 makes
        ``ClientSession.initialize()`` idempotent per instance, so recovery
        creates a new session before calling this again.

        Requires: Client must be initialized first (_initialize_client).
        """
        if self._session is None or self._session_info is None:
            raise RuntimeError("Cannot initialize session: client not initialized")

        existing_session_id = await self._session_info.get_session_id()
        logger.info(
            f"Initializing MCP session (session_info id: {existing_session_id})"
        )
        await self._strategy.connect(self._session, self._session_info)

    async def _ensure_session(self) -> ClientSession:
        """Ensure client and session are initialized, return the session.

        Thread-safe via lock. Only initializes once; subsequent calls
        return the existing session immediately.

        Returns:
            The initialized ClientSession.
        """
        # Always cross the lifecycle lock. Recovery creates the replacement
        # ClientSession before its initialize handshake completes, so a lock-free
        # fast path could expose a half-initialized session to another operation.
        async with self._lock:
            if not self._client_initialized:
                await self._initialize_client()
            elif self._session is None:
                # A failed replacement leaves the reusable HTTP client intact.
                # Reopen on the next operation instead of poisoning the client.
                await self._open_connection()

            if self._session is None:
                raise RuntimeError("MCP client initialized without a session")
            return self._session

    async def _close_connection_for_recovery(self) -> None:
        """Detach and best-effort close the current connection stack."""
        connection_stack = self._connection_stack
        self._connection_stack = None
        self._session = None
        if connection_stack is None:
            return
        try:
            await connection_stack.aclose()
        except Exception as error:
            logger.debug("Error closing failed MCP connection: %s", error)

    async def _reinitialize_session(
        self,
        failed_session: ClientSession | None = None,
        error: MCPError | None = None,
    ) -> None:
        """Replace the transport/session after a disconnect and initialize again.

        MCP 2 makes ``ClientSession.initialize()`` idempotent, so recovery must
        create a fresh ClientSession rather than calling initialize on the old one.
        The HTTP client and external ``SessionInfo`` object are reused.

        The strategy is asked to discard persisted session state only when
        ``error`` is the server's verdict on the session -- see
        ``is_session_rejected``. A dropped transport is not: the ID is kept and
        the reconnect resumes the same session. With no ``error`` nothing is
        known against the session and it is likewise kept.
        """
        async with self._lock:
            if not self._client_initialized:
                # Client not initialized, do full initialization
                await self._initialize_client()
            else:
                if failed_session is not None and self._session is not failed_session:
                    logger.debug(
                        "MCP session was already replaced by another operation"
                    )
                    return
                await self._close_connection_for_recovery()
                if (
                    self._session_info is not None
                    and error is not None
                    and is_session_rejected(error)
                ):
                    await self._strategy.reset(self._session_info)
                await self._open_connection()

    @staticmethod
    def is_session_error(error: MCPError) -> bool:
        """Check if an MCPError indicates a session disconnect.

        Args:
            error: The MCPError to check.

        Returns:
            True if the error indicates a session disconnect.
        """
        return _is_session_error(error)

    async def _is_recoverable_session_error(self, error: MCPError) -> bool:
        """Ask the active protocol strategy whether a reconnect could fix ``error``.

        The answer is era-specific: a legacy session can be lost and re-established,
        while every modern request is self-contained, so only a dropped connection
        is worth retrying there.
        """
        restored_id = (
            await self._session_info.get_session_id() if self._session_info else None
        )
        return self._strategy.is_recoverable(error, restored_id)

    async def _execute_with_retry(
        self,
        operation: Callable[[ClientSession], Awaitable[T]],
        operation_name: str,
    ) -> T:
        """Execute a session operation with automatic retry on session disconnect.

        On first call, initializes the full client stack. On session
        disconnect, replaces the transport/session and retries up to
        ``_max_retries`` times.

        Args:
            operation: An async callable that receives the ``ClientSession``
                and returns the desired result.
            operation_name: A label used in log messages.

        Returns:
            The result of *operation*.

        Raises:
            MCPError: If the operation fails after all retries.
        """
        retry_count = 0

        while retry_count <= self._max_retries:
            session: ClientSession | None = None
            try:
                session = await self._ensure_session()
                logger.debug(
                    f"{operation_name} (attempt {retry_count + 1}/{self._max_retries + 1})"
                )
                return await operation(session)

            except MCPError as e:
                logger.info(f"MCPError during {operation_name}: {e}")

                recoverable = await self._is_recoverable_session_error(e)
                if recoverable and retry_count < self._max_retries:
                    logger.warning(
                        f"Session disconnected (error code: {e.code}), "
                        f"reinitializing session"
                    )
                    await self._reinitialize_session(session, error=e)
                    retry_count += 1
                    continue
                else:
                    if retry_count >= self._max_retries:
                        logger.error(f"Max retries reached after session error: {e}")
                    else:
                        logger.error(f"Non-retryable MCP error: {e}")
                    raise

        raise RuntimeError("Exited retry loop unexpectedly")

    async def list_tools(self, *, force_refresh: bool = False) -> ListToolsResult:
        """List available tools from the MCP server.

        The result is cached in memory on the first successful call and reused for the
        lifetime of this client. ``dispose()`` clears the cache, so a fresh client
        fetches the list again on its next call. Pass ``force_refresh=True`` to re-query
        the server and refresh the cache.

        Args:
            force_refresh: When True, re-query the server and refresh the cache.
        """
        if not force_refresh and self._tools_cache is not None:
            return self._tools_cache
        async with self._tools_lock:
            if not force_refresh and self._tools_cache is not None:
                return self._tools_cache
            result = await self._execute_with_retry(
                lambda session: session.list_tools(),
                "list_tools",
            )
            self._tools_cache = result
            return result

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Call an MCP tool by name.

        Args:
            name: The name of the tool to call.
            arguments: Optional arguments to pass to the tool.

        Returns:
            The tool call result.
        """
        return await self._execute_with_retry(
            lambda session: session.call_tool(name, arguments=arguments),
            f"call_tool({name})",
        )

    async def dispose(self) -> None:
        """Dispose of the client and release all resources.

        Implements UiPathDisposableProtocol.
        Releases the HTTP client, streamable connection, and ClientSession.
        After calling dispose(), the client can be reused - a new call_tool()
        will reinitialize everything.
        """
        # Acquire _tools_lock before _lock (the same order list_tools uses) so the tool
        # cache is cleared atomically with respect to an in-flight list_tools().
        async with self._tools_lock:
            self._tools_cache = None
            async with self._lock:
                if self._connection_stack is not None:
                    try:
                        await self._connection_stack.aclose()
                    except Exception as e:
                        logger.debug(f"Error during MCP connection cleanup: {e}")
                    finally:
                        self._connection_stack = None
                        self._session = None

                if self._stack is not None:
                    try:
                        await self._stack.aclose()
                    except Exception as e:
                        logger.debug(f"Error during cleanup: {e}")
                    finally:
                        self._stack = None
                        self._http_client = None
                        self._session_info = None
                        self._client_initialized = False

                logger.info("MCP client disposed")
