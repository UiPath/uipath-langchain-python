"""MCP servers hosted over Streamable HTTP for the protocol-version matrix.

Two kinds of server are needed:

* :func:`build_sdk_app` hosts a genuine ``MCPServer`` over Streamable HTTP. It
  is the realistic leg, but it negotiates whatever the client asks for, so it
  always settles on the latest legacy handshake version.
* :class:`PinnedVersionServer` declares one specific ``protocolVersion``. The
  SDK server exposes no version knob, so pinning an older version -- or
  refusing the legacy handshake the way a modern-only server would -- requires
  a small endpoint that speaks the wire protocol directly.
"""

import asyncio
import contextlib
import json
import socket
from collections.abc import AsyncGenerator
from typing import Any

import uvicorn
from mcp.server.mcpserver import MCPServer
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

# JSON-RPC error codes used by the pinned endpoint.
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601

TOOL_NAME = "add"


def build_sdk_app() -> Starlette:
    """Host a real SDK ``MCPServer`` over Streamable HTTP."""
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

    Args:
        protocol_version: The version echoed from ``initialize``.
        modern_only: When True, reject the legacy ``initialize`` handshake the
            way a server supporting only modern discovery would. The MCP 2 SDK
            reaches modern versions through discovery rather than
            ``initialize``, so a legacy handshake has no endpoint to land on.
    """

    def __init__(self, protocol_version: str, *, modern_only: bool = False) -> None:
        self.protocol_version = protocol_version
        self.modern_only = modern_only
        self.session_ids: list[str] = []
        self.delete_count = 0
        self.initialize_count = 0

    def build_app(self) -> Starlette:
        """Return an ASGI app exposing this endpoint at ``/mcp``."""
        return Starlette(routes=[Route("/mcp", self._handle, methods=["GET", "POST", "DELETE"])])

    async def _handle(self, request: Request) -> Response:
        if request.method == "GET":
            # No server-initiated stream is needed for this testcase.
            return Response(status_code=405)
        if request.method == "DELETE":
            self.delete_count += 1
            return Response(status_code=204)

        body = json.loads(await request.body())
        method = body.get("method")

        if method == "initialize":
            return self._initialize(body)
        if method is not None and method.startswith("notifications/"):
            return Response(status_code=202)
        if method == "tools/list":
            return self._result(body["id"], {"tools": [_tool_schema()]})
        if method == "tools/call":
            return self._call_tool(body)
        return self._error(
            body.get("id"), METHOD_NOT_FOUND, f"Method not found: {method}", status=404
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
            headers={"mcp-session-id": session_id},
        )

    def _call_tool(self, body: dict[str, Any]) -> Response:
        params = body.get("params") or {}
        if params.get("name") != TOOL_NAME:
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
            },
        )

    @staticmethod
    def _result(
        request_id: Any, result: dict[str, Any], *, headers: dict[str, str] | None = None
    ) -> Response:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": request_id, "result": result}, headers=headers
        )

    @staticmethod
    def _error(
        request_id: Any, code: int, message: str, *, status: int = 400
    ) -> Response:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}},
            status_code=status,
        )


def _tool_schema() -> dict[str, Any]:
    """Describe the single tool the pinned endpoint exposes."""
    return {
        "name": TOOL_NAME,
        "description": "Add two numbers",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    }


@contextlib.asynccontextmanager
async def serve(app: Starlette) -> AsyncGenerator[str, None]:
    """Run *app* on an ephemeral port and yield its ``/mcp`` URL.

    Binding port 0 keeps parallel CI jobs from colliding, and hosting in-process
    means there is no child process left behind if a leg fails.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    task = None
    try:
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if task.done():
                task.result()
                raise RuntimeError("MCP test server exited before startup")
            await asyncio.sleep(0.02)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        if task is not None:
            with contextlib.suppress(BaseException):
                await task
        sock.close()


class DebugStateStore:
    """In-memory stand-in for the AgentHub debug-state endpoint.

    ``uipath-agents-python`` persists MCP session IDs by PUT/GET against
    ``{UIPATH_URL}/agenthub_/design/debugstate/{agent_id}/{key}``. Serving that
    route here keeps the downstream ``SessionInfo`` subclass on its real code
    path -- a legacy ``httpx.AsyncClient`` round trip -- instead of a stub.
    """

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.writes = 0
        self.reads = 0

    def attach(self, app: Starlette) -> None:
        """Mount the debug-state route onto an existing app."""
        app.router.routes.append(
            Route(
                "/agenthub_/design/debugstate/{agent_id}/{key:path}",
                self._handle,
                methods=["GET", "PUT"],
            )
        )

    async def _handle(self, request: Request) -> Response:
        key = request.path_params["key"]
        if request.method == "PUT":
            self.values[key] = (await request.body()).decode()
            self.writes += 1
            return Response(status_code=204)
        self.reads += 1
        value = self.values.get(key)
        if value is None:
            return Response(status_code=404)
        return PlainTextResponse(value)
