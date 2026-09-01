"""Verify the UiPath MCP adapter against HTTP-hosted servers on several protocol versions.

This testcase deliberately uses no LLM. It drives ``uipath_langchain``'s own
``streamable_http_client`` and ``SessionInfo`` -- not the raw SDK transport --
over real sockets, so it covers the adapter that the unit tests can only reach
through ``httpx2.MockTransport``.

One leg per protocol version:

* ``2025-11-25`` against a genuine SDK ``MCPServer`` hosted over Streamable HTTP.
* ``2025-06-18`` against an endpoint pinned to that version.
* ``2026-07-28`` against a modern-only endpoint, which is expected to FAIL.
  The low-level ``ClientSession`` reaches only the legacy handshake versions, so
  a modern-only server is unreachable today. That limitation is asserted rather
  than skipped, so this leg flips to a success the day modern discovery lands.

A final leg pins the public API that ``uipath-agents-python`` consumes -- see
``agents_api.py`` -- so a break in the only known downstream consumer shows up
here rather than on its next dependency bump.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from mcp import ClientSession
from pydantic import BaseModel, Field

from uipath_langchain.agent.tools.mcp import load_mcp_tools
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
    negotiated_version: str | None = None
    session_id_issued: bool = False
    tools: list[str] = Field(default_factory=list)
    tool_result: str | None = None
    error_type: str | None = None
    error_code: int | None = None
    error_message: str | None = None


class GraphOutput(BaseModel):
    """Full matrix result."""

    results: list[LegResult]
    supported_versions: list[str]
    unsupported_versions: list[str]
    agents_api: AgentsApiResult | None = None


class GraphState(BaseModel):
    """Workflow state."""

    a: int
    b: int
    results: list[LegResult] = Field(default_factory=list)
    agents_api: AgentsApiResult | None = None


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
) -> LegResult:
    """Connect to one HTTP-hosted server and record what the adapter negotiated."""
    session_info = SessionInfo()
    a, b = operands
    async with servers.serve(app) as url:
        logger.info("Probing %s at %s", label, url)
        try:
            async with streamable_http_client(url, session_info=session_info) as (
                read,
                write,
            ):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    tools = await load_mcp_tools(session)
                    add_tool = next(t for t in tools if t.name == "add")
                    blocks = await add_tool.ainvoke({"a": a, "b": b})
                    session_id = await session_info.get_session_id()
                    return LegResult(
                        label=label,
                        protocol_version=protocol_version,
                        server=server_kind,
                        supported=True,
                        negotiated_version=str(init.protocol_version),
                        session_id_issued=session_id is not None,
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
                session_id_issued=session_id is not None,
                error_type=type(inner).__name__,
                error_code=getattr(inner, "code", None),
                error_message=str(inner)[:300],
            )


async def run_matrix(state: GraphState) -> GraphState:
    """Run every protocol-version leg in turn."""
    operands = (state.a, state.b)
    results = [
        await _run_leg(
            "sdk-server-2025-11-25",
            "2025-11-25",
            "real MCPServer over Streamable HTTP",
            servers.build_sdk_app(),
            operands,
        ),
        await _run_leg(
            "pinned-2025-06-18",
            "2025-06-18",
            "endpoint pinned to 2025-06-18",
            servers.PinnedVersionServer("2025-06-18").build_app(),
            operands,
        ),
        await _run_leg(
            f"modern-only-{MODERN_VERSION}",
            MODERN_VERSION,
            "modern-only endpoint (no legacy handshake)",
            servers.PinnedVersionServer(MODERN_VERSION, modern_only=True).build_app(),
            operands,
        ),
    ]
    return GraphState(
        a=state.a, b=state.b, results=results, agents_api=state.agents_api
    )


async def run_agents_api(state: GraphState) -> GraphState:
    """Exercise the MCP API surface that ``uipath-agents-python`` consumes.

    Kept as its own node so a downstream-compatibility break is reported
    separately from a protocol-version regression.
    """
    app = servers.build_sdk_app()
    store = servers.DebugStateStore()
    store.attach(app)
    result = await agents_api.run_agents_api_leg(
        servers.serve, app, store, (state.a, state.b)
    )
    return GraphState(a=state.a, b=state.b, results=state.results, agents_api=result)


def build_output(state: GraphState) -> GraphOutput:
    """Summarize the matrix for assertions."""
    return GraphOutput(
        results=state.results,
        supported_versions=[r.protocol_version for r in state.results if r.supported],
        unsupported_versions=[
            r.protocol_version for r in state.results if not r.supported
        ],
        agents_api=state.agents_api,
    )


def _prepare(graph_input: GraphInput) -> GraphState:
    """Seed the workflow state from the graph input."""
    return GraphState(a=graph_input.a, b=graph_input.b)


builder = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphOutput)
builder.add_node("prepare", _prepare)
builder.add_node("run_matrix", run_matrix)
builder.add_node("run_agents_api", run_agents_api)
builder.add_node("summarize", build_output)
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "run_matrix")
builder.add_edge("run_matrix", "run_agents_api")
builder.add_edge("run_agents_api", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()
