"""Real-HTTP MCP servers for driving ``McpClient`` over a socket.

``httpx2.MockTransport`` covers the wire faithfully but never exercises a real
server, a real ASGI stack, or the gateway hop UiPath puts in front of an MCP
endpoint. This module hosts genuine servers on an ephemeral port so
``McpClient`` -- not just the strategies underneath it -- can be driven end to
end.

Three kinds of endpoint are provided:

* :func:`build_sdk_app` hosts a genuine ``MCPServer`` over Streamable HTTP. It
  answers both eras, so it is the realistic leg for legacy, modern, and ``auto``.
* :class:`PinnedVersionServer` declares one specific ``protocolVersion``. The
  SDK server exposes no version knob, so pinning an older handshake version --
  or refusing the handshake the way a modern-only server would -- requires an
  endpoint that speaks the wire directly.
* :class:`RecordingGateway` is a pure ASGI middleware standing in for the
  AgentHub gateway: it records what every request carried and can inject a
  fault, so recovery paths are driven by a real HTTP response rather than a
  mocked transport.

``starlette.middleware.base.BaseHTTPMiddleware`` is deliberately **not** used.
It buffers through an inner task and breaks streaming responses here with
``ASGI callable returned without completing response``; a pure ASGI middleware
composes cleanly and can still read request bodies by wrapping ``receive``.
"""

import asyncio
import contextlib
import json
import socket
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterator,
    MutableMapping,
)
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types.version import HANDSHAKE_PROTOCOL_VERSIONS
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from uipath.agent.models.agent import AgentMcpResourceConfig, AgentMcpTool

from uipath_langchain.agent.tools.mcp import McpClient, SessionInfo, SessionInfoFactory

#: Newest handshake-era version; what a real SDK server settles on for legacy.
LEGACY_VERSION = "2025-11-25"

#: The only modern-era version; reached through ``server/discover``.
MODERN_VERSION = "2026-07-28"

#: Every handshake version SDK 2 still accepts, oldest first.
#: Every version reachable through the ``initialize`` handshake, taken from the
#: SDK rather than restated, so a version added upstream widens the matrix
#: instead of silently narrowing the "every handshake version" claim.
HANDSHAKE_VERSIONS: tuple[str, ...] = tuple(HANDSHAKE_PROTOCOL_VERSIONS)

MCP_SESSION_ID = "mcp-session-id"
MCP_PROTOCOL_VERSION = "mcp-protocol-version"

# JSON-RPC error codes the hand-written endpoints emit.
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601

PINNED_TOOL_NAME = "add"

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def build_sdk_app() -> Starlette:
    """Host a real SDK ``MCPServer`` with ``add`` and ``multiply`` tools.

    Returns:
        A Streamable HTTP ASGI app serving the server at ``/mcp``.
    """
    server = MCPServer("Math")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @server.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    return server.streamable_http_app(json_response=True)


class PinnedVersionServer:
    """Serve Streamable HTTP while declaring one fixed protocol version.

    Adapted from ``testcases/simple-http-mcp``. Modern-era results carry
    ``resultType``/``ttlMs``/``cacheScope``, which the ``2026-07-28`` wire
    requires and older peers ignore.

    Args:
        protocol_version: The version echoed from ``initialize`` (or advertised
            by ``server/discover`` in modern-only mode).
        modern_only: When True, serve ``server/discover`` and reject the legacy
            ``initialize`` handshake, the way a server that speaks only the
            modern protocol would. Requires a modern ``protocol_version``.
    """

    def __init__(self, protocol_version: str, *, modern_only: bool = False) -> None:
        self.protocol_version = protocol_version
        self.modern_only = modern_only
        self.session_ids: list[str] = []
        self.delete_count = 0
        self.initialize_count = 0
        self.discover_count = 0

    def build_app(self) -> Starlette:
        """Return an ASGI app exposing this endpoint at ``/mcp``."""
        return Starlette(
            routes=[Route("/mcp", self._handle, methods=["GET", "POST", "DELETE"])]
        )

    async def _handle(self, request: Request) -> Response:
        if request.method == "GET":
            # No server-initiated stream is needed for these tests.
            return Response(status_code=405)
        if request.method == "DELETE":
            self.delete_count += 1
            return Response(status_code=204)

        body = json.loads(await request.body())
        method = body.get("method")

        if method == "server/discover":
            return self._discover(body)
        if method == "initialize":
            return self._initialize(body)
        if method is not None and method.startswith("notifications/"):
            return Response(status_code=202)
        if method == "tools/list":
            return self._result(
                body["id"],
                {
                    "tools": [_pinned_tool_schema()],
                    "resultType": "complete",
                    "ttlMs": 0,
                    "cacheScope": "private",
                },
            )
        if method == "tools/call":
            return self._call_tool(body)
        return self._error(
            body.get("id"), METHOD_NOT_FOUND, f"Method not found: {method}", status=404
        )

    def _discover(self, body: dict[str, Any]) -> Response:
        """Answer ``server/discover``, but only for a modern endpoint."""
        if not self.modern_only:
            # A handshake-era server has no discovery method to land on.
            return self._error(
                body.get("id"),
                METHOD_NOT_FOUND,
                "server/discover is not supported; use the initialize handshake",
                status=404,
            )
        self.discover_count += 1
        return self._result(
            body["id"],
            {
                "supportedVersions": [self.protocol_version],
                "capabilities": {"tools": {"listChanged": True}},
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "private",
            },
        )

    def _initialize(self, body: dict[str, Any]) -> Response:
        if self.modern_only:
            # A modern-only server offers no legacy handshake at all.
            return self._error(
                body.get("id"),
                METHOD_NOT_FOUND,
                "Legacy initialize is not supported; use modern discovery",
                status=404,
            )
        self.initialize_count += 1
        session_id = f"session-{self.initialize_count}"
        self.session_ids.append(session_id)
        return self._result(
            body["id"],
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "pinned-server", "version": "1.0.0"},
            },
            headers={MCP_SESSION_ID: session_id},
        )

    def _call_tool(self, body: dict[str, Any]) -> Response:
        params = body.get("params") or {}
        if params.get("name") != PINNED_TOOL_NAME:
            return self._error(
                body.get("id"), INVALID_REQUEST, f"Unknown tool: {params.get('name')}"
            )
        arguments = params.get("arguments") or {}
        total = int(arguments.get("a", 0)) + int(arguments.get("b", 0))
        return self._result(
            body["id"],
            {
                "content": [{"type": "text", "text": str(total)}],
                "structuredContent": {"result": total},
                "isError": False,
                "resultType": "complete",
            },
        )

    @staticmethod
    def _result(
        request_id: Any,
        result: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> Response:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": request_id, "result": result}, headers=headers
        )

    @staticmethod
    def _error(
        request_id: Any, code: int, message: str, *, status: int = 400
    ) -> Response:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            },
            status_code=status,
        )


def _pinned_tool_schema() -> dict[str, Any]:
    """Describe the single tool the pinned endpoint exposes."""
    return {
        "name": PINNED_TOOL_NAME,
        "description": "Add two numbers",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    }


@dataclass
class RecordedRequest:
    """One HTTP request as the gateway saw it."""

    http_method: str
    """``POST``, ``DELETE``, ``GET`` -- so a session teardown is observable."""

    rpc_method: str | None
    """JSON-RPC ``method``, or ``None`` for a body-less request."""

    session_id: str | None
    """The ``mcp-session-id`` request header, or ``None`` when unpinned."""

    protocol_version: str | None
    """The ``mcp-protocol-version`` request header, stamped after negotiation."""

    meta: dict[str, Any] = field(default_factory=dict)
    """``params._meta`` as it arrived on the wire."""

    instance: str = ""
    """Backend the gateway would have routed to, derived from ``session_id``."""

    response_session_id: str | None = None
    """``mcp-session-id`` on the response, i.e. a server-issued session."""

    faulted: bool = False
    """True when the gateway answered this request itself with an injected fault."""


class RecordingGateway:
    """Pure ASGI middleware that records every request and can inject a fault.

    Stands in for the AgentHub gateway. It routes by ``mcp-session-id`` exactly
    as the real one does, so the instance spread it records is the observable
    consequence of the affinity identity.

    Args:
        app: The ASGI app to wrap.
        fault_on_tool_call: 1-based index of the ``tools/call`` to answer with a
            fault instead of forwarding, or ``None`` to forward everything.
        fault_message: JSON-RPC error message for the injected fault.
        fault_code: JSON-RPC error code for the injected fault.
        fault_status: HTTP status the injected fault is returned with.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        fault_on_tool_call: int | None = None,
        fault_message: str = "Session terminated",
        fault_code: int = INVALID_REQUEST,
        fault_status: int = 404,
    ) -> None:
        self.app = app
        self.fault_on_tool_call = fault_on_tool_call
        self.fault_message = fault_message
        self.fault_code = fault_code
        self.fault_status = fault_status
        self.records: list[RecordedRequest] = []
        self.instances: dict[str, str] = {}
        self._tool_calls = 0

    # --- observation helpers ------------------------------------------------

    def rpc_methods(self) -> list[str]:
        """Every JSON-RPC method seen, in order."""
        return [r.rpc_method for r in self.records if r.rpc_method is not None]

    def count(self, rpc_method: str) -> int:
        """How many times one JSON-RPC method was received."""
        return self.rpc_methods().count(rpc_method)

    def http_count(self, http_method: str) -> int:
        """How many requests used one HTTP method (``DELETE`` in particular)."""
        return sum(1 for r in self.records if r.http_method == http_method)

    def for_rpc(self, rpc_method: str) -> list[RecordedRequest]:
        """Every record for one JSON-RPC method, in order."""
        return [r for r in self.records if r.rpc_method == rpc_method]

    def server_session_ids(self) -> list[str]:
        """Session IDs the server assigned on a response header."""
        return [
            r.response_session_id
            for r in self.records
            if r.response_session_id is not None
        ]

    def unpinned(self) -> list[RecordedRequest]:
        """Requests that reached the gateway with no affinity/session header."""
        return [r for r in self.records if r.session_id is None]

    # --- ASGI ---------------------------------------------------------------

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Record, optionally fault, and otherwise forward one ASGI request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        http_method = str(scope.get("method", ""))

        body = b""
        forward_receive = receive
        if http_method == "POST":
            body, forward_receive = await _buffer_body(receive)

        payload = _parse_json(body)
        params = payload.get("params") if isinstance(payload, dict) else None
        session_id = headers.get(MCP_SESSION_ID)
        record = RecordedRequest(
            http_method=http_method,
            rpc_method=payload.get("method") if isinstance(payload, dict) else None,
            session_id=session_id,
            protocol_version=headers.get(MCP_PROTOCOL_VERSION),
            meta=dict((params or {}).get("_meta") or {})
            if isinstance(params, dict)
            else {},
            instance=self._instance_for(session_id),
        )
        self.records.append(record)

        if record.rpc_method == "tools/call":
            self._tool_calls += 1
            if self._tool_calls == self.fault_on_tool_call:
                record.faulted = True
                await self._send_fault(payload, send)
                return

        async def recording_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                for key, value in message.get("headers", ()):
                    if key.decode("latin-1").lower() == MCP_SESSION_ID:
                        record.response_session_id = value.decode("latin-1")
            await send(message)

        await self.app(scope, forward_receive, recording_send)

    def _instance_for(self, session_id: str | None) -> str:
        """Map an affinity/session ID onto the backend a gateway would pick.

        An unpinned request cannot be routed, so it gets an instance of its own
        -- which is what makes a missing affinity header visible as a spread.
        """
        key = session_id if session_id is not None else f"unpinned-{len(self.records)}"
        if key not in self.instances:
            self.instances[key] = f"instance-{len(self.instances) + 1}"
        return self.instances[key]

    async def _send_fault(self, payload: Any, send: Send) -> None:
        """Answer a request with a JSON-RPC error, as a failing gateway would."""
        request_id = payload.get("id") if isinstance(payload, dict) else None
        raw = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": self.fault_code, "message": self.fault_message},
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": self.fault_status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(raw)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": raw})


async def _buffer_body(receive: Receive) -> tuple[bytes, Receive]:
    """Read a request body fully and return it with a replaying ``receive``."""
    body = b""
    while True:
        message = await receive()
        if message["type"] != "http.request":
            break
        body += bytes(message.get("body", b""))
        if not message.get("more_body", False):
            break

    replayed = False

    async def replay() -> Message:
        nonlocal replayed
        if not replayed:
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return body, replay


def _parse_json(body: bytes) -> Any:
    """Decode a JSON body, returning ``None`` when there is nothing to decode."""
    if not body:
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _reset_sse_shutdown_latch() -> None:
    """Clear ``sse-starlette``'s process-global shutdown latch.

    ``sse_starlette`` runs a watcher that polls ``uvicorn.Server.should_exit``
    and latches a *module-global* ``AppStatus.should_exit`` when any server
    stops. Tests host one server per case, so from the second server onward
    every SSE stream would see the latch already set and end the instant it
    opened -- logging ``ASGI callable returned without completing response`` and
    sending the client into a reconnect loop. Clearing the latch (and the
    per-thread watcher bookkeeping, which is stranded on the previous event
    loop) gives every server the same first-server behaviour.

    Best-effort: these are private names, so a changed internal degrades to the
    noisy-but-working behaviour rather than breaking the harness.
    """
    try:
        from sse_starlette import sse as sse_module
    except ImportError:  # pragma: no cover - sse-starlette ships with mcp
        return
    try:
        sse_module.AppStatus.should_exit = False
        state = getattr(sse_module._thread_state, "shutdown_state", None)
        if state is not None:
            state.watcher_started = False
            state.events.clear()
    except AttributeError:  # pragma: no cover - upstream internals moved
        return


@asynccontextmanager
async def serve(app: ASGIApp) -> AsyncGenerator[str, None]:
    """Run *app* on an ephemeral port and yield its ``/mcp`` URL.

    Binding port 0 keeps parallel CI jobs from colliding, and hosting in-process
    means no child process is left behind when a test fails.

    Args:
        app: Any ASGI application.

    Yields:
        The ``http://127.0.0.1:<port>/mcp`` endpoint URL.
    """
    # Imported here so the module can be collected even if the optional dev
    # dependency is missing, and to keep import cost off the test session.
    import uvicorn

    _reset_sse_shutdown_latch()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    task: asyncio.Task[Any] | None = None
    try:
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if task.done():
                task.result()
                raise RuntimeError("MCP test server exited before startup")
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        if task is not None:
            with contextlib.suppress(BaseException):
                await task
        sock.close()


SERVER_NAME = "Math"
SERVER_SLUG = "math"
FOLDER_KEY = "folder-key"
FOLDER_PATH = "Shared"
ACCESS_TOKEN = "test-access-token"


@contextmanager
def patched_sdk(url: str) -> Iterator[None]:
    """Point ``McpClient``'s lazy SDK lookup at a locally hosted server.

    ``McpClient._initialize_client`` imports ``UiPath`` from ``uipath.platform``
    at call time, so replacing the module attribute is enough: no tenant,
    credentials, or network access to UiPath Cloud are involved, and the client
    still walks its real resolution path.

    Args:
        url: The MCP endpoint the fake registration should resolve to.
    """
    import uipath.platform as platform
    from uipath.platform.orchestrator.mcp import McpServer

    class _FakeMcpService:
        async def retrieve_async(
            self, name: str, folder_path: str | None = None
        ) -> McpServer:
            # Recorded so a test can assert *how* the server was resolved, not
            # just that it was: the client must look up by display name and pass
            # the execution folder through.
            SDK_LOOKUPS.append({"name": name, "folder_path": folder_path})
            return McpServer(
                id="mcp-server-id",
                name=name,
                slug=SERVER_SLUG,
                folderKey=FOLDER_KEY,
                mcpUrl=url,
            )

    class _FakeConfig:
        secret = ACCESS_TOKEN

    class _FakeUiPath:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._config = _FakeConfig()
            self.mcp = _FakeMcpService()

    SDK_LOOKUPS.clear()
    original = platform.UiPath
    platform.UiPath = _FakeUiPath  # type: ignore[misc,assignment]
    try:
        yield
    finally:
        platform.UiPath = original  # type: ignore[misc]


#: Every ``retrieve_async`` call made through :func:`patched_sdk`, in order.
#: Cleared by that context manager on entry.
SDK_LOOKUPS: list[dict[str, Any]] = []


def make_resource_config() -> AgentMcpResourceConfig:
    """Build the MCP resource config every real-HTTP test drives."""
    return AgentMcpResourceConfig(
        name=SERVER_NAME,
        description="Math MCP server",
        folder_path=FOLDER_PATH,
        slug=SERVER_SLUG,
        available_tools=[
            AgentMcpTool(
                name=PINNED_TOOL_NAME,
                description="Add two numbers",
                input_schema={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            )
        ],
    )


def pinned_session_factory(session_info: SessionInfo) -> SessionInfoFactory:
    """Return a factory handing every client the same ``SessionInfo``.

    Two clients sharing one store is how a persisted session (or affinity ID)
    survives across runs, so this is what a resume test drives.

    Args:
        session_info: The store every created client should use.
    """

    class _PinnedFactory(SessionInfoFactory):
        def create_session(self, mcp_server: Any) -> SessionInfo:
            return session_info

    return _PinnedFactory()


def make_client(**kwargs: Any) -> McpClient:
    """Create an ``McpClient`` for the shared resource config.

    The URL is resolved through :func:`patched_sdk`, so that context manager
    must be active when the returned client first connects.

    Args:
        **kwargs: Forwarded to ``McpClient`` (``protocol_mode``,
            ``session_info_factory``, ``terminate_on_close``, ...).
    """
    return McpClient(config=make_resource_config(), **kwargs)


@asynccontextmanager
async def connected_client(url: str, **kwargs: Any) -> AsyncGenerator[McpClient, None]:
    """Yield an ``McpClient`` wired to *url*, disposing it on exit.

    Args:
        url: The endpoint returned by :func:`serve`.
        **kwargs: Forwarded to ``McpClient``.
    """
    with patched_sdk(url):
        client = make_client(**kwargs)
        try:
            yield client
        finally:
            await client.dispose()
