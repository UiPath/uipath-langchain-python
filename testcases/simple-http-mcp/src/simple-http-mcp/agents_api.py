"""Pin the MCP public API that ``uipath-agents-python`` consumes.

`uipath-agents-python` is the only known downstream consumer of the MCP tool
layer, and it pins ``uipath-langchain`` exactly, so a break there surfaces on
its next bump rather than in this repository's own tests. This leg exercises
the API it actually calls, over real sockets, against a real SDK ``MCPServer``:

* ``create_mcp_tools_and_clients(resources, session_info_factory=..., terminate_on_close=...)``
  -- both call shapes it uses: production (no factory, terminate on close) and
  playground (debug-state factory, session kept alive).
* ``SessionInfo`` subclassed with its own ``__init__`` and async
  ``get_session_id`` / ``set_session_id`` overrides, persisting through HTTP.
  This is the shape of ``SessionInfoDebugState``; MCP 2 added
  ``SessionInfo.protocol_version``, which the transport now reads directly, so
  such a subclass fails at the first request unless it calls ``super().__init__()``.
* ``SessionInfoFactory.create_session(mcp_server)`` reading ``McpServer.slug``
  and ``McpServer.folder_key``.
* Session resumption: a second client picking up the session ID the first one
  persisted, which is what playground mode relies on across runs.
* Disposal through ``McpClient.dispose()``, the way the caller drains its
  ``UiPathDisposableProtocol`` list.

The mirrored source lives in ``uipath_agents/agent_graph_builder/`` --
``graph.py`` and ``session_info_debug_state.py``. Keep this file in step with it.
"""

import contextlib
import logging
import os
from collections.abc import Iterator
from enum import Enum, auto
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.agent.models.agent import (
    AgentMcpResourceConfig,
    AgentMcpTool,
    CachedToolsConfig,
    DynamicToolsConfig,
    ToolsConfiguration,
)
from uipath.platform.orchestrator.mcp import McpServer

# Imported exactly as uipath-agents-python imports them. Dropping or renaming
# any of these turns this testcase into an ImportError at graph load.
from uipath_langchain.agent.tools.mcp import (
    McpClient,
    SessionInfo,
    SessionInfoFactory,
    create_mcp_tools_and_clients,
)

logger = logging.getLogger(__name__)

DOWNSTREAM_IMPORTS = [
    "McpClient",
    "SessionInfo",
    "SessionInfoFactory",
    "create_mcp_tools_and_clients",
]

AGENT_ID = "agent-under-test"
FOLDER_KEY = "folder-key"
FOLDER_PATH = "Shared"
SERVER_SLUG = "math"
SERVER_NAME = "Math"
ACCESS_TOKEN = "test-access-token"


# --- SessionInfoDebugState, mirrored from uipath-agents-python ---------------


class _SessionState(Enum):
    """State machine for the session ID lifecycle, as downstream defines it."""

    NOT_LOADED = auto()
    LOADED = auto()
    CLEARED = auto()


class SessionInfoDebugState(SessionInfo):
    """``SessionInfo`` subclass that persists session IDs over HTTP.

    Deliberately keeps downstream's structure: an ``__init__`` of its own that
    calls ``super().__init__()``, direct assignment to the inherited
    ``session_id`` attribute, and async overrides of both accessors.
    """

    def __init__(self, slug: str, folder_key: str, agent_id: str | None) -> None:
        super().__init__()
        self._slug = slug
        self._folder_key = folder_key
        self._agent_id = agent_id
        self._state = _SessionState.NOT_LOADED

    @property
    def key(self) -> str:
        """Debug-state key for this MCP resource."""
        return f"mcpsession:{self._folder_key}:{self._slug}"

    async def get_session_id(self) -> str | None:
        """Return the session ID, loading it from debug state on first call."""
        if self._state == _SessionState.NOT_LOADED:
            self._state = _SessionState.LOADED
            stored = await self._load_from_debug_state()
            if stored is not None:
                self.session_id = stored
        elif self._state == _SessionState.CLEARED:
            return None
        return self.session_id

    async def set_session_id(self, session_id: str | None) -> None:
        """Store the session ID locally and persist it to debug state."""
        if session_id is None:
            self.session_id = None
            self._state = _SessionState.CLEARED
        else:
            self.session_id = session_id
            self._state = _SessionState.LOADED
            await self._save_to_debug_state(session_id)

    def _debug_state_url(self) -> str | None:
        base_url = os.getenv("UIPATH_URL")
        if not base_url or not self._agent_id:
            return None
        encoded_key = quote(self.key, safe="")
        return f"{base_url}/agenthub_/design/debugstate/{self._agent_id}/{encoded_key}"

    def _auth_headers(self) -> dict[str, str]:
        token = os.getenv("UIPATH_ACCESS_TOKEN", "")
        return {"Authorization": f"Bearer {token}"}

    async def _load_from_debug_state(self) -> str | None:
        url = self._debug_state_url()
        if url is None:
            return None
        async with httpx.AsyncClient(
            headers=self._auth_headers(), **get_httpx_client_kwargs()
        ) as client:
            response = await client.get(url)
            if response.status_code == 200:
                return response.text
        return None

    async def _save_to_debug_state(self, session_id: str) -> None:
        url = self._debug_state_url()
        if url is None:
            return
        async with httpx.AsyncClient(
            headers=self._auth_headers(), **get_httpx_client_kwargs()
        ) as client:
            await client.put(
                url, content=session_id, headers={"Content-Type": "text/plain"}
            )


class SessionInfoDebugStateFactory(SessionInfoFactory):
    """Factory returning ``SessionInfoDebugState``, as downstream defines it."""

    def __init__(self, agent_id: str | None) -> None:
        self._agent_id = agent_id or os.getenv("UIPATH_PROJECT_ID")

    def create_session(self, mcp_server: McpServer) -> SessionInfoDebugState:
        """Create a SessionInfoDebugState from an McpServer."""
        return SessionInfoDebugState(
            slug=mcp_server.slug or "",
            folder_key=mcp_server.folder_key or "",
            agent_id=self._agent_id,
        )


# --- Stand-in for the UiPath SDK lookup McpClient performs lazily ------------


class _FakeMcpService:
    def __init__(self, url: str) -> None:
        self._url = url

    async def retrieve_async(
        self, name: str, folder_path: str | None = None
    ) -> McpServer:
        return McpServer(
            id="mcp-server-id",
            name=name,
            slug=SERVER_SLUG,
            folderKey=FOLDER_KEY,
            mcpUrl=self._url,
        )


class _FakeConfig:
    secret = ACCESS_TOKEN


@contextlib.contextmanager
def _patched_sdk(url: str) -> Iterator[None]:
    """Point ``McpClient``'s lazy SDK lookup at the local test server.

    ``McpClient._initialize_client`` imports ``UiPath`` from ``uipath.platform``
    at call time, so replacing the module attribute is enough; no tenant or
    network access to UiPath Cloud is involved.
    """
    import uipath.platform as platform

    class _FakeUiPath:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._config = _FakeConfig()
            self.mcp = _FakeMcpService(url)

    original = platform.UiPath
    platform.UiPath = _FakeUiPath  # type: ignore[misc]
    try:
        yield
    finally:
        platform.UiPath = original  # type: ignore[misc]


@contextlib.contextmanager
def _agenthub_env(base_url: str) -> Iterator[None]:
    """Set the environment ``SessionInfoDebugState`` reads its endpoint from."""
    previous = {
        key: os.environ.get(key) for key in ("UIPATH_URL", "UIPATH_ACCESS_TOKEN")
    }
    os.environ["UIPATH_URL"] = base_url
    os.environ["UIPATH_ACCESS_TOKEN"] = ACCESS_TOKEN
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --- Resource configs, as an agent definition would carry them ---------------


def _cached_resource() -> AgentMcpResourceConfig:
    """Design-time tool snapshot; the default cached discovery mode."""
    return AgentMcpResourceConfig(
        name=SERVER_NAME,
        description="Math MCP server",
        folderPath=FOLDER_PATH,
        slug=SERVER_SLUG,
        availableTools=[
            AgentMcpTool(
                name="add",
                description="Add two numbers",
                inputSchema={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            )
        ],
        toolsConfiguration=ToolsConfiguration(discoveryMode=CachedToolsConfig()),
    )


def _dynamic_resource() -> AgentMcpResourceConfig:
    """Live tool discovery; reads the SDK's snake_case ``Tool`` attributes."""
    return AgentMcpResourceConfig(
        name=SERVER_NAME,
        description="Math MCP server",
        folderPath=FOLDER_PATH,
        slug=SERVER_SLUG,
        availableTools=[],
        toolsConfiguration=ToolsConfiguration(
            discoveryMode=DynamicToolsConfig(allowAll=True)
        ),
    )


# --- Results ----------------------------------------------------------------


class LegSummary(BaseModel):
    """Outcome of one ``create_mcp_tools_and_clients`` call."""

    label: str
    tools: list[str] = Field(default_factory=list)
    tool_result: str | None = None
    session_id: str | None = None
    disposed: bool = False
    error_type: str | None = None
    error_message: str | None = None


class AgentsApiResult(BaseModel):
    """Everything the downstream compatibility leg asserts on."""

    imports: list[str]
    session_info_super_init: bool
    production: LegSummary
    playground: LegSummary
    playground_resumed: LegSummary
    debug_state_writes: int
    debug_state_reads: int
    persisted_session_id: str | None = None
    session_resumed: bool = False


def _first_text(blocks: Any) -> str | None:
    """Pull the first text block out of a normalized MCP tool result."""
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text"))
    return None if blocks is None else str(blocks)


async def _run_leg(
    label: str,
    resource: AgentMcpResourceConfig,
    session_info_factory: SessionInfoFactory | None,
    terminate_on_close: bool,
    operands: tuple[int, int],
) -> LegSummary:
    """Drive one downstream-shaped call from tool creation through disposal."""
    a, b = operands
    clients: list[McpClient] = []
    try:
        tools, clients = await create_mcp_tools_and_clients(
            [resource],
            session_info_factory=session_info_factory,
            terminate_on_close=terminate_on_close,
        )
        add_tool = next(tool for tool in tools if tool.name == "add")
        blocks = await add_tool.ainvoke({"a": a, "b": b})
        summary = LegSummary(
            label=label,
            tools=sorted(tool.name for tool in tools),
            tool_result=_first_text(blocks),
            session_id=await clients[0].get_session_id(),
        )
    except Exception as error:  # noqa: BLE001 - the failure IS the result
        logger.exception("Downstream-compatibility leg %s failed", label)
        summary = LegSummary(
            label=label,
            error_type=type(error).__name__,
            error_message=str(error)[:300],
        )

    # Mirrors how the caller drains its UiPathDisposableProtocol list. Disposal
    # is part of the contract, so a failure here is a leg failure.
    try:
        for client in clients:
            await client.dispose()
        summary.disposed = True
    except Exception as error:  # noqa: BLE001 - the failure IS the result
        summary.error_type = summary.error_type or type(error).__name__
        summary.error_message = summary.error_message or str(error)[:300]
    return summary


async def run_agents_api_leg(
    serve: Any, app: Any, store: Any, operands: tuple[int, int]
) -> AgentsApiResult:
    """Run every downstream-shaped call against one hosted MCP server.

    Args:
        serve: The ``servers.serve`` async context manager.
        app: The Starlette app hosting the MCP server and the debug-state route.
        store: The ``DebugStateStore`` mounted on *app*.
        operands: Operands handed to the remote ``add`` tool.
    """
    probe = SessionInfoDebugState(
        slug=SERVER_SLUG, folder_key=FOLDER_KEY, agent_id=None
    )
    # MCP 2 added this attribute and the transport reads it directly; a subclass
    # that skipped super().__init__() would raise AttributeError on first request.
    super_init_ok = getattr(probe, "protocol_version", "missing") is None

    async with serve(app) as url:
        base_url = url.removesuffix("/mcp")
        with _patched_sdk(url), _agenthub_env(base_url):
            production = await _run_leg(
                "production",
                _dynamic_resource(),
                session_info_factory=None,
                terminate_on_close=True,
                operands=operands,
            )

            factory = SessionInfoDebugStateFactory(agent_id=AGENT_ID)
            playground = await _run_leg(
                "playground",
                _cached_resource(),
                session_info_factory=factory,
                terminate_on_close=False,
                operands=operands,
            )
            persisted = store.values.get(probe.key)

            # A fresh factory stands in for the next playground run: it must pick
            # the persisted session ID back up rather than start a new session.
            resumed = await _run_leg(
                "playground-resumed",
                _cached_resource(),
                session_info_factory=SessionInfoDebugStateFactory(agent_id=AGENT_ID),
                terminate_on_close=False,
                operands=operands,
            )

    return AgentsApiResult(
        imports=DOWNSTREAM_IMPORTS,
        session_info_super_init=super_init_ok,
        production=production,
        playground=playground,
        playground_resumed=resumed,
        debug_state_writes=store.writes,
        debug_state_reads=store.reads,
        persisted_session_id=persisted,
        session_resumed=(persisted is not None and resumed.session_id == persisted),
    )
