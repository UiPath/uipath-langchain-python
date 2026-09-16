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
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.shared.message import SessionMessage
from mcp.types import CallToolResult, ListToolsResult
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.runtime.base import UiPathDisposableProtocol

from uipath_langchain._utils import get_execution_folder_path

from .streamable_http import SessionInfo, streamable_http_client

if TYPE_CHECKING:
    from uipath.agent.models.agent import AgentMcpResourceConfig
    from uipath.platform.orchestrator.mcp import McpServer

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
        from the UiPath SDK on first use, using the config's display name and
        folder_path.

        Args:
            config: The MCP resource configuration containing name and folder_path.
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

        # Tool list cached in memory and fetched once per client lifetime, with its own
        # lock so a concurrent first call does not deadlock against ``_lock`` (held by
        # session initialization inside ``_execute_with_retry``).
        self._tools_lock = asyncio.Lock()
        self._tools_cache: ListToolsResult | None = None

        # Client state (created once, reused across session reinitializations)
        self._http_client: httpx.AsyncClient | None = None
        self._read_stream: (
            MemoryObjectReceiveStream[SessionMessage | Exception] | None
        ) = None
        self._write_stream: MemoryObjectSendStream[SessionMessage] | None = None
        self._session_info: SessionInfo | None = None
        self._stack: AsyncExitStack | None = None
        # The connection (HTTP client, transport, session) lives on its own task,
        # so its anyio cancel scopes are entered and exited by the same task.
        self._connection_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._close_requested: asyncio.Event | None = None

        # Session state (can be reinitialized without recreating client)
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
        """Resolve the server, then open the connection on a task that owns it.

        The HTTP client, the streamable HTTP transport and the ``ClientSession``
        all sit in one ``AsyncExitStack`` whose anyio cancel scopes must be exited
        by the task that entered them, in reverse order of entry. Callers satisfy
        neither: langgraph opens the connection inside a tool task while teardown
        runs on the main one, and an agent disposes its servers oldest-first.
        Keeping the whole stack on a dedicated task makes the caller's task and
        ordering irrelevant -- closing is a signal, not an unwind.
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

        # Create session info for tracking session ID
        session_info = self._session_info_factory.create_session(mcp_server)
        self._session_info = session_info

        # Load a session ID persisted by the AgentHub debug-state integration.
        existing = await session_info.get_session_id()
        if existing:
            logger.info(f"Loaded existing session ID from session info: {existing}")

        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._ready = ready
        self._close_requested = asyncio.Event()
        self._connection_task = asyncio.create_task(
            self._run_connection(
                ready, self._close_requested, self._url, self._headers, session_info
            ),
            name=f"mcp-connection-{self._config.slug}",
        )
        try:
            # Shielded so a cancelled caller does not cancel `ready` with it:
            # _close_connection reads it to tell a finished handshake (signal)
            # from one still in flight (cancel).
            await asyncio.shield(ready)
        except BaseException:
            await self._close_connection()
            self._session_info = None
            raise

        self._client_initialized = True
        logger.info("MCP client initialized")

    async def _run_connection(
        self,
        ready: asyncio.Future[None],
        close_requested: asyncio.Event,
        url: str,
        headers: dict[str, str],
        session_info: SessionInfo,
    ) -> None:
        """Hold the HTTP client, transport and session open until asked to close.

        Resolves *ready* once the session is negotiated, so ``_initialize_client``
        fails the same way it did when it opened the stack inline.
        """
        # Unwinding the transport's task group wraps whatever went wrong in a
        # BaseExceptionGroup. Callers used to see the original error, so keep it.
        setup_error: BaseException | None = None
        try:
            async with AsyncExitStack() as stack:
                self._stack = stack
                try:
                    client_kwargs = get_httpx_client_kwargs(headers=headers)
                    client_kwargs["timeout"] = self._timeout
                    self._http_client = await stack.enter_async_context(
                        httpx.AsyncClient(**client_kwargs)
                    )
                    streams = await stack.enter_async_context(
                        streamable_http_client(
                            url=url,
                            http_client=self._http_client,
                            session_info=session_info,
                            terminate_on_close=self._terminate_on_close,
                        )
                    )
                    self._read_stream, self._write_stream = streams
                    self._session = await stack.enter_async_context(
                        ClientSession(self._read_stream, self._write_stream)
                    )
                    await self._initialize_session()
                except BaseException as error:
                    setup_error = error
                    raise
                if not ready.done():
                    ready.set_result(None)
                await close_requested.wait()
        except BaseException as error:
            failure = setup_error if setup_error is not None else error
            if isinstance(failure, asyncio.CancelledError):
                # _close_connection cancelled us mid-handshake, or the loop is
                # going down. End as cancelled, not as failed.
                if not ready.done():
                    ready.cancel()
                if failure is error:
                    raise
                raise failure from error
            if not ready.done():
                ready.set_exception(failure)
            else:
                logger.debug("MCP connection ended with an error: %s", failure)
        finally:
            # Whatever ended the task, the client no longer has a connection.
            # Clearing the flag makes the next call rebuild it instead of
            # handing a missing session to the operation.
            self._client_initialized = False
            self._stack = None
            self._session = None
            self._read_stream = None
            self._write_stream = None
            self._http_client = None

    async def _close_connection(self) -> None:
        """Ask the connection task to unwind, and wait for it to finish.

        A task still in the handshake would not see the signal until the
        handshake returned -- up to the transport timeout -- so it is cancelled
        instead.
        """
        task = self._connection_task
        ready = self._ready
        close_requested = self._close_requested
        self._connection_task = None
        self._ready = None
        self._close_requested = None
        self._session = None
        if task is None:
            return

        if ready is not None and ready.done() and close_requested is not None:
            close_requested.set()
        else:
            task.cancel()

        # asyncio.wait rather than `await task`: the task's own cancellation
        # must not read as ours, while a real cancellation of this task still
        # propagates.
        await asyncio.wait({task})
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.debug("Error closing MCP connection: %s", error)

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

        session = self._session
        if session is None:
            raise RuntimeError("MCP client initialized without a session")
        return session

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

    async def _execute_with_retry(
        self,
        operation: Callable[[ClientSession], Awaitable[T]],
        operation_name: str,
    ) -> T:
        """Execute a session operation with automatic retry on session disconnect.

        On first call, initializes the full client stack. On session
        disconnect, reinitializes only the session and retries up to
        ``_max_retries`` times.

        Args:
            operation: An async callable that receives the ``ClientSession``
                and returns the desired result.
            operation_name: A label used in log messages.

        Returns:
            The result of *operation*.

        Raises:
            McpError: If the operation fails after all retries.
        """
        retry_count = 0

        while retry_count <= self._max_retries:
            try:
                session = await self._ensure_session()
                logger.debug(
                    f"{operation_name} (attempt {retry_count + 1}/{self._max_retries + 1})"
                )
                return await operation(session)

            except McpError as e:
                logger.info(f"McpError during {operation_name}: {e}")

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
                try:
                    await self._close_connection()
                finally:
                    # The connection fields are cleared by the task itself;
                    # this is the client-level state, and it must not survive
                    # a dispose() that gets cancelled halfway.
                    self._session_info = None
                    self._client_initialized = False
                logger.info("MCP client disposed")
