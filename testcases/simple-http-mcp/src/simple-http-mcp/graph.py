"""Verify the UiPath MCP adapter against HTTP-hosted servers on several protocol versions.

This testcase deliberately uses no LLM. It drives ``uipath_langchain``'s own
``streamable_http_client`` and ``SessionInfo`` -- not the raw SDK transport --
over real sockets, so it covers the adapter that the unit tests can only reach
through ``httpx2.MockTransport``.

Every leg runs through ``build_protocol_strategy``, so the matrix exercises the
same negotiation code ``McpClient`` uses rather than a parallel implementation.

* ``legacy`` against a genuine SDK ``MCPServer``, which negotiates ``2025-11-25``.
* ``legacy`` against an endpoint pinned to ``2025-06-18``, proving the server's
  counter-offer is honored.
* ``modern`` against the same real server, which negotiates ``2026-07-28`` and
  issues no session ID.
* ``modern`` against an endpoint that serves only ``server/discover`` and refuses
  the handshake, so the modern path cannot be passing by legacy fallback.
* ``auto`` against both a discovery-capable and a handshake-only server, checking
  it resolves to a different era for each.

A further leg drives a gateway stand-in to check that the UiPath affinity ID pins
one warm instance across separate clients -- the routing that ``mcp-session-id``
used to provide and ``2026-07-28`` removes.

A final leg pins the public API that ``uipath-agents-python`` consumes -- see
``agents_api.py`` -- so a break in the only known downstream consumer shows up
here rather than on its next dependency bump. That leg also drives ``McpClient``
itself on both eras, including a modern affinity pair whose two clients share one
``SessionInfo``.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Any

import httpx2
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession
from pydantic import BaseModel, Field

from uipath_langchain.agent.tools.mcp import load_mcp_tools
from uipath_langchain.agent.tools.mcp.protocol_strategy import build_protocol_strategy
from uipath_langchain.agent.tools.mcp.streamable_http import (
    SessionInfo,
    streamable_http_client,
)

logger = logging.getLogger(__name__)

MODERN_VERSION = "2026-07-28"


def _load_sibling(name: str) -> Any:
    """Load a sibling module by path.

    The source directory is not an importable package name, so a path-based
    load keeps this working however the graph module itself was loaded.
    """
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"mcp_testcase_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Cannot load MCP testcase module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


servers = _load_sibling("servers")
agents_api = _load_sibling("agents_api")
AgentsApiResult = agents_api.AgentsApiResult


class GraphInput(BaseModel):
    """Operands passed to the remote ``add`` tool on every supported leg."""

    a: int = Field(default=2, description="First operand for the add tool")
    b: int = Field(default=3, description="Second operand for the add tool")


class LegResult(BaseModel):
    """Outcome of one protocol-version leg."""

    label: str
    protocol_version: str
    server: str
    supported: bool
    mode: str = "legacy"
    era: str | None = None
    negotiated_version: str | None = None
    session_id_issued: bool = False
    server_session_id_seen: bool = False
    tools: list[str] = Field(default_factory=list)
    tool_result: str | None = None
    error_type: str | None = None
    error_code: int | None = None
    error_message: str | None = None


class AffinityResult(BaseModel):
    """Outcome of the gateway-routing leg."""

    affinity_ids: list[str] = Field(default_factory=list)
    instances: list[str] = Field(default_factory=list)
    requests: int = 0
    unpinned_requests: int = 0
    first_request_pinned: bool = False
    tool_results: list[str] = Field(default_factory=list)


class GraphOutput(BaseModel):
    """Full matrix result."""

    results: list[LegResult]
    supported_versions: list[str]
    unsupported_versions: list[str]
    agents_api: AgentsApiResult | None = None
    affinity: AffinityResult | None = None


class GraphState(BaseModel):
    """Workflow state."""

    a: int
    b: int
    results: list[LegResult] = Field(default_factory=list)
    agents_api: AgentsApiResult | None = None
    affinity: AffinityResult | None = None


def _unwrap_mcp_error(error: BaseException) -> BaseException:
    """Return the meaningful error inside nested ``ExceptionGroup`` wrappers.

    A failing handshake surfaces as an ``ExceptionGroup`` from the transport's
    task group; the useful ``MCPError`` sits several levels down.
    """
    current = error
    for _ in range(6):
        nested = getattr(current, "exceptions", None)
        if not nested:
            break
        current = nested[0]
    return current


def _first_text(blocks: Any) -> str | None:
    """Pull the first text block out of a LangChain tool result."""
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text"))
    return None if blocks is None else str(blocks)


async def _run_leg(
    label: str,
    protocol_version: str,
    server_kind: str,
    app: Any,
    operands: tuple[int, int],
    mode: str = "legacy",
) -> LegResult:
    """Connect to one HTTP-hosted server and record what the strategy negotiated."""
    session_info = SessionInfo()
    strategy = build_protocol_strategy(mode)  # type: ignore[arg-type]
    a, b = operands
    server_session_ids: list[str] = []

    async def watch_response(response: Any) -> None:
        """Note whether the server ever assigned a session ID on this leg."""
        if response.headers.get("mcp-session-id") is not None:
            server_session_ids.append(response.headers["mcp-session-id"])

    async with servers.serve(app) as url:
        logger.info("Probing %s at %s (mode=%s)", label, url, mode)
        client = httpx2.AsyncClient(
            follow_redirects=True, timeout=httpx2.Timeout(30, read=300)
        )
        client.event_hooks["response"].append(watch_response)
        try:
            async with client:
                async with streamable_http_client(
                    url,
                    http_client=client,
                    session_info=session_info,
                    identity=strategy.identity,
                ) as (read, write):
                    async with ClientSession(read, write) as session:
                        await strategy.connect(session, session_info)
                        tools = await load_mcp_tools(session)
                        add_tool = next(t for t in tools if t.name == "add")
                        blocks = await add_tool.ainvoke({"a": a, "b": b})
                        session_id = await session_info.get_session_id()
                        return LegResult(
                            label=label,
                            protocol_version=protocol_version,
                            server=server_kind,
                            supported=True,
                            mode=mode,
                            era=(
                                "modern"
                                if session.discover_result is not None
                                else "legacy"
                            ),
                            negotiated_version=str(session.protocol_version),
                            session_id_issued=session_id is not None,
                            server_session_id_seen=bool(server_session_ids),
                            tools=sorted(t.name for t in tools),
                            tool_result=_first_text(blocks),
                        )
        except BaseException as error:  # noqa: BLE001 - the failure IS the result
            inner = _unwrap_mcp_error(error)
            logger.info("%s is unsupported: %s", label, inner)
            session_id = await session_info.get_session_id()
            return LegResult(
                label=label,
                protocol_version=protocol_version,
                server=server_kind,
                supported=False,
                mode=mode,
                session_id_issued=session_id is not None,
                server_session_id_seen=bool(server_session_ids),
                error_type=type(inner).__name__,
                error_code=getattr(inner, "code", None),
                error_message=str(inner)[:300],
            )


async def run_matrix(state: GraphState) -> GraphState:
    """Run every protocol-version leg in turn."""
    operands = (state.a, state.b)
    results = [
        await _run_leg(
            "legacy-sdk-server",
            "2025-11-25",
            "real MCPServer over Streamable HTTP",
            servers.build_sdk_app(),
            operands,
            mode="legacy",
        ),
        await _run_leg(
            "legacy-pinned-2025-06-18",
            "2025-06-18",
            "endpoint pinned to 2025-06-18",
            servers.PinnedVersionServer("2025-06-18").build_app(),
            operands,
            mode="legacy",
        ),
        await _run_leg(
            "modern-sdk-server",
            MODERN_VERSION,
            "real MCPServer over Streamable HTTP",
            servers.build_sdk_app(),
            operands,
            mode="modern",
        ),
        await _run_leg(
            "modern-only-endpoint",
            MODERN_VERSION,
            "endpoint serving only server/discover",
            servers.PinnedVersionServer(MODERN_VERSION, modern_only=True).build_app(),
            operands,
            mode="modern",
        ),
        await _run_leg(
            "auto-sdk-server",
            MODERN_VERSION,
            "real MCPServer over Streamable HTTP",
            servers.build_sdk_app(),
            operands,
            mode="auto",
        ),
        await _run_leg(
            "auto-pinned-2025-06-18",
            "2025-06-18",
            "endpoint pinned to 2025-06-18",
            servers.PinnedVersionServer("2025-06-18").build_app(),
            operands,
            mode="auto",
        ),
    ]
    return GraphState(
        a=state.a,
        b=state.b,
        results=results,
        agents_api=state.agents_api,
        affinity=state.affinity,
    )


class _RoutingGateway:
    """Stand-in for AgentHub: pins a warm instance by ``mcp-session-id``.

    ``2026-07-28`` drops that header from the protocol, but UiPath keeps sending
    it with a client-minted value purely as a routing key -- so the gateway routes
    on exactly the header it already uses today, with no change. A request
    arriving without one cannot be pinned.

    Written as a pure ASGI middleware rather than a
    ``starlette.middleware.base.BaseHTTPMiddleware`` subclass: the latter
    buffers through an inner task and breaks the server's SSE stream with
    ``ASGI callable returned without completing response``.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.routed: list[tuple[str, bool]] = []
        self.instances: dict[str, str] = {}

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Record the instance this request would be routed to, then forward it."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token: str | None = None
        for key, value in scope.get("headers", ()):
            if key.decode("latin-1").lower() == "mcp-session-id":
                token = value.decode("latin-1")
        pinned = token is not None
        if token is None:
            # Unroutable: a real gateway would have to pick an instance blind, so
            # give every such request its own to make the spread visible.
            token = f"unpinned-{len(self.routed)}"
        if token not in self.instances:
            self.instances[token] = f"instance-{len(self.instances) + 1}"
        self.routed.append((self.instances[token], pinned))
        await self.app(scope, receive, send)


async def run_affinity(state: GraphState) -> GraphState:
    """Check the affinity ID pins one instance across separate modern clients.

    Two clients share one ``SessionInfo``, standing in for two runs of a
    playground agent whose session store survives the process.
    """
    gateway: dict[str, _RoutingGateway] = {}

    class _Capturing(_RoutingGateway):
        def __init__(self, app: Any) -> None:
            super().__init__(app)
            gateway["value"] = self

    app = servers.build_sdk_app()
    app.add_middleware(_Capturing)

    shared_info = SessionInfo()
    affinity_ids: list[str] = []
    tool_results: list[str] = []
    async with servers.serve(app) as url:
        for _ in range(2):
            strategy = build_protocol_strategy("modern")
            async with streamable_http_client(
                url, session_info=shared_info, identity=strategy.identity
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await strategy.connect(session, shared_info)
                    tools = await load_mcp_tools(session)
                    add_tool = next(t for t in tools if t.name == "add")
                    blocks = await add_tool.ainvoke({"a": state.a, "b": state.b})
                    tool_results.append(str(_first_text(blocks)))
            affinity_ids.append(str(await shared_info.get_session_id()))

    observed = gateway["value"].routed
    return GraphState(
        a=state.a,
        b=state.b,
        results=state.results,
        agents_api=state.agents_api,
        affinity=AffinityResult(
            affinity_ids=affinity_ids,
            instances=sorted({instance for instance, _ in observed}),
            requests=len(observed),
            unpinned_requests=sum(1 for _, pinned in observed if not pinned),
            # The client mints the ID before negotiating, so even the discovery
            # probe carries it -- a server-assigned session never could.
            first_request_pinned=bool(observed) and observed[0][1],
            tool_results=tool_results,
        ),
    )


async def run_agents_api(state: GraphState) -> GraphState:
    """Exercise the MCP API surface that ``uipath-agents-python`` consumes.

    Kept as its own node so a downstream-compatibility break is reported
    separately from a protocol-version regression.
    """
    app = servers.build_sdk_app()
    store = servers.DebugStateStore()
    store.attach(app)
    # Wrapping the app is the only way to tell a server-assigned session ID from
    # the client-minted affinity ID: both are opaque hex on the wire.
    recorder = servers.SessionHeaderRecorder(app)
    result = await agents_api.run_agents_api_leg(
        servers.serve, recorder, store, (state.a, state.b), recorder=recorder
    )
    return GraphState(
        a=state.a,
        b=state.b,
        results=state.results,
        agents_api=result,
        affinity=state.affinity,
    )


def build_output(state: GraphState) -> GraphOutput:
    """Summarize the matrix for assertions."""
    return GraphOutput(
        results=state.results,
        supported_versions=[r.protocol_version for r in state.results if r.supported],
        unsupported_versions=[
            r.protocol_version for r in state.results if not r.supported
        ],
        agents_api=state.agents_api,
        affinity=state.affinity,
    )


def _prepare(graph_input: GraphInput) -> GraphState:
    """Seed the workflow state from the graph input."""
    return GraphState(a=graph_input.a, b=graph_input.b)


builder = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphOutput)
builder.add_node("prepare", _prepare)
builder.add_node("run_matrix", run_matrix)
builder.add_node("run_affinity", run_affinity)
builder.add_node("run_agents_api", run_agents_api)
builder.add_node("summarize", build_output)
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "run_matrix")
builder.add_edge("run_matrix", "run_affinity")
builder.add_edge("run_affinity", "run_agents_api")
builder.add_edge("run_agents_api", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()
