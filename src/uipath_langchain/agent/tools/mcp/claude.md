# MCP Session Implementation Guide

> **CLAUDE: UPDATE THIS DOCUMENT**
>
> When you modify files in this module, you MUST update this document to reflect:
> - New or changed class attributes/methods (update Architecture section)
> - Changes to initialization phases (update Two-Phase Initialization section)
> - New error codes (update Session Error Codes table)
> - Protocol flow changes (update MCP Protocol Flow diagrams)
> - New guidelines or gotchas (update Guidelines for Changes section)
>
> Keep diagrams and code examples in sync with the actual implementation.

## Overview

This module implements MCP (Model Context Protocol) session management and tool
factory functions.  It connects LangGraph agents to UiPath MCP servers via
streamable HTTP transport and provides a factory pattern for session ID tracking.

## Module Structure

```
src/uipath_langchain/agent/tools/mcp/
├── __init__.py          # Public exports
├── mcp_client.py        # SessionInfoFactory, McpClient
├── mcp_tool.py          # Tool factory functions
└── streamable_http.py   # SessionInfo + thin adapter over the MCP SDK transport
```

### Public Exports (`__init__.py`)

```python
from .mcp_client import McpClient, SessionInfoFactory
from .mcp_tool import (
    create_mcp_tools_and_clients,
    open_mcp_tools,
    create_mcp_tools,
)
from .streamable_http import SessionInfo
```

`streamable_http_client` is intentionally **not** exported — it is an internal
transport helper used only by `McpClient`.

## Architecture

### streamable_http.py — Session-Aware SDK Transport Adapter

This file is a thin adapter around the **client-side** streamable HTTP transport
from MCP Python SDK 2.0. The SDK owns protocol parsing, SSE resumption,
cancellation, protocol headers, and session deletion; UiPath adds externally
persistable session ID tracking via `SessionInfo`.

**Source**: [`mcp.client.streamable_http`](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/streamable_http.py)

**Why an adapter is still needed:**

The SDK 2.0 transport owns its in-memory session ID but has no asynchronous hook
for loading and saving UiPath debug-state sessions. The adapter installs two
`httpx2` event hooks on the client used by the SDK:

1. Before every request, call `SessionInfo.get_session_id()` and set or remove
   the `mcp-session-id` header.
2. After every response, persist a returned `mcp-session-id` through
   `SessionInfo.set_session_id()`.

The hooks are removed when the adapter context exits. This keeps the UiPath
extension small and automatically picks up future SDK transport fixes instead
of maintaining another transport fork.

#### SessionInfo

Base class for MCP session ID tracking.  Lives in `streamable_http.py`.

```python
class SessionInfo:
    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id

    async def get_session_id(self) -> str | None: ...
    async def set_session_id(self, session_id: str | None) -> None: ...
```

The base implementation stores session ID in a plain attribute.  Async methods
exist so subclasses (e.g. `SessionInfoDebugState` in `uipath-agents`) can add
side-effects like HTTP persistence.

**Important:** The response hook calls `set_session_id` during `initialize()`
when the server assigns an ID. `McpClient._initialize_session` only reads the
stored value afterward. Passing `None` clears a stale session before recovery.

#### Upstream StreamableHTTPTransport

MCP SDK 2.0's transport handles POST requests, the optional GET SSE channel,
`Last-Event-ID` resumption, 2026 HTTP cancellation, protocol headers, structured
errors from non-2xx responses, and session termination via DELETE. None of those
internals are duplicated locally.

#### streamable_http_client (context manager)

Internal async context manager that attaches the session hooks and delegates to
the SDK context manager. Used by `McpClient._open_connection`.

```python
async with streamable_http_client(url, http_client=client, session_info=info) as (read, write):
    session = ClientSession(read, write)
```

The adapter yields the SDK's two transport streams unchanged. If no HTTP client
is supplied, it creates and owns an `httpx2.AsyncClient`; `McpClient` normally
supplies its long-lived authenticated client.

---

### SessionInfoFactory

Default factory in `mcp_client.py`.  Creates plain `SessionInfo` instances.

```python
class SessionInfoFactory:
    def create_session(self, mcp_server: McpServer) -> SessionInfo:
        logger.info(f"Creating session for server '{mcp_server.slug}' in folder '{mcp_server.folder_key}'")
        return SessionInfo()
```

Subclass this to provide custom `SessionInfo` implementations.  The factory
receives the full `McpServer` model (from `uipath.platform.orchestrator.mcp`)
so subclasses can extract slug, folder_key, or other metadata.

**Note:** `SessionInfoDebugState` and `SessionInfoDebugStateFactory` live in
`uipath-agents-python` at
`uipath_agents.agent_graph_builder.session_info_debug_state`, not in this
package.  They import `SessionInfo` and `SessionInfoFactory` from here.

### McpClient Class

`McpClient` implements `UiPathDisposableProtocol` and manages the lifecycle of
MCP connections for tool invocations with **two distinct initialization phases**:

1. **Client Initialization** (first call): Retrieves MCP server URL via SDK, creates the full stack
2. **Connection Reinitialization** (on session loss): Reuses the HTTP client,
   but replaces the transport and `ClientSession`

```
┌─────────────────────────────────────────────────────────────┐
│                     McpClient                               │
├─────────────────────────────────────────────────────────────┤
│  Configuration (immutable after __init__)                   │
│  ─────────────────────────────────────────                  │
│  _config: AgentMcpResourceConfig  # Contains slug, folder   │
│  _timeout: httpx2.Timeout | float | None                    │
│  _max_retries: int                                          │
│  _session_info_factory: SessionInfoFactory                  │
├─────────────────────────────────────────────────────────────┤
│  Lazy-Resolved State (set during _initialize_client)        │
│  ───────────────────────────────────────────────────        │
│  _url: str | None          # Retrieved from SDK             │
│  _headers: dict[str, str]  # Auth header from SDK           │
├─────────────────────────────────────────────────────────────┤
│  Synchronization                                            │
│  ───────────────                                            │
│  _lock: asyncio.Lock     # Protects both init phases        │
├─────────────────────────────────────────────────────────────┤
│  Client State (created once, reused on connection reinit)   │
│  ─────────────────────────────────────────────────────      │
│  _http_client: httpx2.AsyncClient | None                    │
│  _session_info: SessionInfo | None                          │
│  _stack: AsyncExitStack | None  # HTTP client               │
│  _client_initialized: bool                                  │
├─────────────────────────────────────────────────────────────┤
│  Connection State (replaced after session loss)             │
│  ──────────────────────────────────────────────             │
│  _connection_stack: AsyncExitStack | None                   │
│  _session: ClientSession | None                             │
├─────────────────────────────────────────────────────────────┤
│  Public Methods                                             │
│  ──────────────                                             │
│  + list_tools(force_refresh=False) -> ListToolsResult       │
│  + call_tool(name, arguments) -> CallToolResult             │
│  + dispose() -> None  # UiPathDisposableProtocol            │
│  + get_session_id() -> str | None                           │
│  + is_client_initialized: bool (property)                   │
├─────────────────────────────────────────────────────────────┤
│  Private Methods                                            │
│  ───────────────                                            │
│  - _initialize_client() -> None    # SDK + full init (once) │
│  - _open_connection() -> None      # transport + session    │
│  - _initialize_session() -> None   # legacy handshake       │
│  - _ensure_session() -> ClientSession                       │
│  - _reinitialize_session(failed_session) -> None            │
│  + is_session_error(error) -> bool                          │
└─────────────────────────────────────────────────────────────┘
```

#### Session ID Flow

During client initialization, `McpClient`:

1. Retrieves the `McpServer` from the UiPath SDK
2. Calls `self._session_info_factory.create_session(mcp_server)` to get a `SessionInfo`
3. Loads any existing session ID via `await session_info.get_session_id()`
4. Passes the `SessionInfo` to the local adapter, which opens the SDK transport
5. Creates a new `ClientSession` over those streams
6. If no ID was restored, calls `session.initialize()`; the response hook stores
   the server-assigned ID
7. Reads the new session ID via `await session_info.get_session_id()`

On recovery, the HTTP client and `SessionInfo` are reused, but the old connection
stack is closed and steps 4-7 run with a fresh transport and `ClientSession`.
This is required because SDK 2.0 makes `ClientSession.initialize()` idempotent
for the lifetime of one `ClientSession`.

### Tool Factory Functions

#### `create_mcp_tools_and_clients(agent, session_info_factory)` → `tuple[list[BaseTool], list[McpClient]]`

**Primary factory function** for creating MCP tools from a LowCodeAgentDefinition.

```python
async def create_mcp_tools_and_clients(
    agent: LowCodeAgentDefinition,
    session_info_factory: SessionInfoFactory | None = None,
) -> tuple[list[BaseTool], list[McpClient]]:
```

The `session_info_factory` parameter is optional.  When `None`, each `McpClient`
defaults to the base `SessionInfoFactory`.  Pass a custom factory (e.g.
`SessionInfoDebugStateFactory()`) to enable session persistence.

**Usage:**
```python
tools, clients = await create_mcp_tools_and_clients(agent, session_info_factory=factory)
try:
    # Use tools...
finally:
    for client in clients:
        await client.dispose()
```

#### `create_mcp_tools(config, mcpClient)` → `list[BaseTool]`

Creates tools for a single MCP resource config using an existing McpClient.

The discovery mode comes from `config.tools_configuration.discovery_mode`, defaulting
to cached when `tools_configuration` is unset.

**Cached mode + self-healing schema (`refresh_schema_before_call`):** `CachedToolsConfig`
carries a `refresh_schema_before_call` flag (default `True`). When set, each tool's
`tool_fn` calls `mcpClient.list_tools()` before `call_tool()` and compares
the live input schema with the cached one (`_refresh_tool_schema` + `_breaking_schema_change`):

- **No breaking change** (identical, or only additive/cosmetic): the call proceeds
  against the cached schema as normal.
- **Breaking change** (a newly required param, a dropped/renamed cached param, or a
  type change on a shared param): the tool is **not** executed. The cached snapshot
  (`mcp_tool`) and the model-facing `args_schema` are updated in place to the live
  schema, and `tool_fn` returns a retry instruction (`_schema_change_message`, which
  lists each refreshed param with its type and optionality). The ReAct loop re-binds
  tools on the next LLM turn (`llm_node.py` binds fresh every step),
  so the model re-issues the call against the live schema and it succeeds on retry.
- **Tool removed** (no longer in the live `list_tools()`): the tool is **not** executed;
  `tool_fn` returns a message (`_tool_removed_message`) telling the model the tool is
  gone, so it stops calling it instead of retrying a doomed call.

The tool wrapper reference needed to mutate `args_schema` is passed to `build_mcp_tool`
via a small `tool_holder` dict filled right after the tool is constructed. This keeps
the whole mechanism inside the MCP tool (the tool does not import or know about the
ReAct loop). `list_tools()` is **not** called at tool-creation time for cached mode.
The flag is read directly from the cached `discovery_mode.refresh_schema_before_call`
field (default `True`).

`McpClient.list_tools()` caches its result in memory, so the live list is fetched
**once per run** and reused; `dispose()` clears the cache, so a resumed run (fresh
client) fetches it again. The self-heal is evaluated against a tool list refreshed once
at the start of each run (and each resume), so a schema change that lands mid-run is
picked up on the next run or resume. `force_refresh=True` bypasses the cache for a live
re-query.

**Limitation:** tools with static argument bindings (non-empty `argument_properties`)
are re-bound each turn from a cached copy in `StaticArgsHandler`, so the `args_schema`
mutation may not reach the model; those tools fall back to the server's validation
error instead of self-healing.

#### `open_mcp_tools(config)` → Context Manager

Async context manager that wraps `create_mcp_tools_and_clients()` with automatic
client lifecycle management.  Yields a list of `BaseTool` instances and
disposes all `McpClient` instances on exit.

## Two-Phase Initialization

The key design principle is separating **client initialization** from **session initialization**:

```
Phase 1: Base Client Initialization (expensive, done once)
───────────────────────────────────────────────────────────
┌─────────────────┐
│ UiPath SDK      │ ─── Retrieves MCP server URL
│ mcp.retrieve()  │     and auth token (Bearer)
└─────────────────┘

┌─────────────────┐
│ SessionInfo     │ ─── Factory creates SessionInfo
│ Factory         │     (may load existing session ID)
└─────────────────┘

┌─────────────────┐
│httpx2.AsyncClient│ ─── Created once via the base AsyncExitStack
└─────────────────┘

Phase 2: Connection Initialization (repeated after session loss)
──────────────────────────────────────────────────────────────
┌─────────────────┐
│ SDK transport + │ ─── Fresh connection AsyncExitStack
│ ClientSession   │
└─────────────────┘

┌─────────────────┐
│ session.        │ ─── Sends initialize request
│ initialize()    │     Response hook calls set_session_id()
│                 │     (skipped when an ID was restored)
└─────────────────┘
┌─────────────────┐
│ McpClient reads │ ─── await session_info.get_session_id()
│ new session ID  │
└─────────────────┘
```

### Session Lifecycle

```
┌──────────────┐  first operation  ┌────────────────────┐
│   Created    │ ────────────────► │ Base client init   │
└──────────────┘                    │ SDK + HTTP client  │
                                    └─────────┬──────────┘
                                              │ open connection
                                              ▼
                                    ┌────────────────────┐
                              ┌────►│ Session init       │
                              │     │ transport + session│
                              │     └─────────┬──────────┘
                              │               │ initialize handshake
                              │               ▼
                              │     ┌────────────────────┐
                              │     │ Active session     │
                              │     └────┬──────────┬────┘
                              │          │          │ dispose()
               session error │          │          ▼
               close old +   └──────────┘  ┌──────────────┐
               clear ID                    │ Closed       │
                                           │ (can reuse)  │
                                           └──────────────┘
```

### MCP Protocol Flow

**First tool call (full initialization):**

```
Client                              Server
   │                                   │
   │──── initialize ──────────────────►│
   │◄─── result + session-id-1 ────────│  ← response hook calls set_session_id()
   │                                   │
   │──── notifications/initialized ───►│
   │◄─── 202 Accepted / 204 ───────────│
   │                                   │
   │──── tools/call ──────────────────►│
   │◄─── result ───────────────────────│
```

**On a terminated session (connection/session replacement):**

```
Client                              Server
   │                                   │
   │──── tools/call ──────────────────►│
   │◄─── 404 (session terminated) ─────│
   │                                   │
   │  [Closes old transport/session;   │
   │   clears stale SessionInfo;       │
   │   reuses existing HTTP client]    │
   │                                   │
   │──── initialize ──────────────────►│  ← new session
   │◄─── result + session-id-2 ────────│    (same HTTP client)
   │                                   │
   │──── notifications/initialized ───►│
   │◄─── 202 Accepted / 204 ───────────│
   │                                   │
   │──── tools/call ──────────────────►│  ← retry
   │◄─── result ───────────────────────│
```

### Session Error Codes

The following error codes trigger automatic session reinitialization:

| Code | Meaning | Source |
|------|---------|--------|
| `CONNECTION_CLOSED` | Transport connection closed | MCP SDK dispatcher/transport |
| `INVALID_REQUEST` (`-32600`) | Session terminated/expired/invalid | SDK 2 maps a bare session-bound HTTP 404 to this error |
| `32600` | Session terminated | Compatibility with the positive code emitted by the older local transport |

`INVALID_REQUEST` is retried only when its message explicitly identifies a
terminated, expired, or invalid session. An externally restored session is not
known inside a newly created SDK transport, so its first bare HTTP 404 appears
as `METHOD_NOT_FOUND`/`"Not Found"`; `McpClient` treats that exact shape as
recoverable only while `SessionInfo` still contains the restored ID. Structured
JSON-RPC method errors are not retried.

## Key Implementation Details

### 1. Lazy SDK Loading

The MCP server URL and authorization headers are loaded lazily on first tool call:

```python
async def _initialize_client(self) -> None:
    from uipath.platform import UiPath

    sdk = UiPath()
    mcp_server = await sdk.mcp.retrieve_async(
        slug=self._config.slug, folder_path=self._config.folder_path
    )
    self._url = mcp_server.mcp_url
    self._headers = {"Authorization": f"Bearer {sdk._config.secret}"}

    # Factory creates the right SessionInfo for this server
    self._session_info = self._session_info_factory.create_session(mcp_server)
```

**Why lazy loading is required:**

The `uipath debug` command loads resource bindings (which can override MCP server URLs)
**after** the LangGraph agent graph is built. This means bindings are only available at
execution time, not at graph construction time. By deferring the SDK call to the first
tool invocation, we ensure the bindings are properly loaded and applied.

### 2. HTTP Client Configuration

The HTTP client MUST use `get_httpx_client_kwargs()` for proper SSL/proxy configuration:

```python
from uipath._utils._ssl_context import get_httpx_client_kwargs

self._stack = AsyncExitStack()
await self._stack.__aenter__()
client_kwargs = get_httpx_client_kwargs(headers=self._headers)
client_kwargs["timeout"] = self._timeout
self._http_client = await self._stack.enter_async_context(
    httpx2.AsyncClient(**client_kwargs)
)
```

### 3. Single Lock for Both Phases

One `asyncio.Lock` protects both client initialization and session reinitialization:

```python
self._lock = asyncio.Lock()

async def _ensure_session(self) -> ClientSession:
    if not self._client_initialized:
        async with self._lock:
            if not self._client_initialized:
                await self._initialize_client()
    return self._session

async def _reinitialize_session(
    self, failed_session: ClientSession | None = None
) -> None:
    async with self._lock:
        if not self._client_initialized:
            await self._initialize_client()
        else:
            # Another failing operation may arrive after recovery completed.
            if failed_session is not None and self._session is not failed_session:
                return
            await self._connection_stack.aclose()
            await self._session_info.set_session_id(None)
            await self._open_connection()
```

### 4. No `with` Statement for AsyncExitStack

Manual lifecycle management:

```python
# Correct - manual management
self._stack = AsyncExitStack()
await self._stack.__aenter__()
# ... use stack ...
await self._stack.aclose()

# Wrong - exits too early
async with AsyncExitStack() as stack:
    ...  # Stack closes here!
```

### 5. Reinitialization Reuses the HTTP Client

On a recoverable session error, `_reinitialize_session()` closes the old
connection stack, clears the stale ID, and opens a fresh SDK transport and
`ClientSession`. The authenticated `httpx2.AsyncClient` and `SessionInfo`
instance are reused. The failed-session identity guard prevents a late failure
from a concurrent operation from tearing down a replacement session.

## Cross-Package Dependencies

```
uipath-langchain (this package)
├── streamable_http.py  → SessionInfo (base class)
├── mcp_client.py       → SessionInfoFactory (base factory)
└── mcp_tool.py         → create_mcp_tools_and_clients(session_info_factory=...)

uipath-agents (consumer)
├── session_info_debug_state.py
│   ├── SessionInfoDebugState(SessionInfo)     ← imports from uipath_langchain
│   └── SessionInfoDebugStateFactory(SessionInfoFactory)
└── graph.py
    └── Picks factory based on AgentExecutionType.PLAYGROUND
```

`SessionInfoDebugState` persists session IDs to the AgentHub debug state
endpoint (`GET/PUT agenthub_/design/debugstate/{agentId}/{key}`).  It lives
in `uipath-agents` because it depends on execution-type logic that belongs
in the agent layer, not in the langchain tools layer.

## Tests

Tests are in `tests/agent/tools/test_mcp/`.

For detailed test documentation, mocking strategies, and guidelines for adding new tests, see:
**`tests/agent/tools/test_mcp/claude.md`**

### Quick Reference

| Test File | Purpose |
|-----------|---------|
| `test_mcp_client.py` | Real SDK 2 transport, legacy negotiation, persisted sessions, recovery, caching, and disposal |
| `test_mcp_tool.py` | Tool factory, schema refresh, result serialization, and error mapping |
| `test_session_info.py` | SessionInfo and SessionInfoFactory contract |

### Key Test Classes

| Class | Tests |
|-------|-------|
| Module-level client tests | 2025 negotiation, persisted sessions, 404 retry, concurrency, client reuse |
| `TestMcpToolMetadata` | Tool metadata (tool_type, display_name, etc.) |
| `TestMcpToolCreation` | Multiple tools, descriptions, disabled config |
| `TestCreateMcpToolsFromAgent` | Agent factory function tests |
| `TestMcpToolNameSanitization` | Tool name sanitization |

### Key Assertion

The most important recovery test verifies a fresh session with HTTP client reuse:

```python
assert endpoint.initialize_count == 2
assert endpoint.tool_call_count == 2
assert await client.get_session_id() == "session-2"
assert [h["mcp-session-id"] for h in endpoint.headers_for("tools/call")] == [
    "session-1",
    "session-2",
]
```

## Guidelines for Changes

### Updating streamable_http.py

When the upstream MCP SDK changes its transport:

1. Keep delegating to [`mcp.client.streamable_http`](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/streamable_http.py); do not copy the transport again
2. Preserve the async `SessionInfo` request/response hooks and remove both hooks on context exit
3. Confirm the SDK still accepts a supplied `httpx2.AsyncClient` and yields two transport streams
4. Re-run the legacy-version, persisted-session, 404 recovery, and DELETE tests

### Adding New Factory Functions

1. Follow the pattern of existing functions
2. Always handle `is_enabled=False` case by returning empty list
3. Include proper metadata on created tools (`tool_type`, `display_name`, `folder_path`, `slug`)
4. Add tests for the new function

### Modifying Client Initialization

1. Changes go in `_initialize_client()`
2. All resources must be added to `_stack` via `enter_async_context()`
3. Set `_client_initialized = True` only after `_open_connection()` and the handshake succeed
4. Always use `get_httpx_client_kwargs()` for HTTP client
5. The `SessionInfo` is created via the factory — do not construct it directly

### Modifying Session Initialization

1. Changes go in `_initialize_session()`
2. It runs only on a newly created `ClientSession`; never call it again on the same SDK 2 session for recovery
3. Don't create HTTP resources here; `_open_connection()` owns the transport/session stack
4. The response hook handles `set_session_id` — `_initialize_session` only reads via `get_session_id`
5. Verify recovery creates two sessions while retaining one HTTP client

### Adding New Methods to McpClient

1. If the method accesses `_session`, use `_ensure_session()`:
   ```python
   async def new_method(self):
       session = await self._ensure_session()
       return await session.some_method()
   ```

2. If the method needs retry logic, follow the pattern in `call_tool()`

3. If the method modifies session state, acquire `_lock`

### Creating a New SessionInfo Subclass

1. Inherit from `SessionInfo` (imported from `uipath_langchain.agent.tools.mcp`)
2. Override `get_session_id` and/or `set_session_id` for custom behavior
3. Create a corresponding factory that inherits `SessionInfoFactory`
4. The factory receives `McpServer` — use its `slug`, `folder_key`, etc.
5. Pass the factory to `create_mcp_tools_and_clients(session_info_factory=...)`

## Related Files

| File | Package | Purpose |
|------|---------|---------|
| `streamable_http.py` | uipath-langchain | SessionInfo + thin SDK transport adapter |
| `mcp_client.py` | uipath-langchain | SessionInfoFactory + McpClient |
| `mcp_tool.py` | uipath-langchain | Tool factory functions |
| `__init__.py` | uipath-langchain | Public exports |
| `session_info_debug_state.py` | uipath-agents | SessionInfoDebugState + factory |
| `graph.py` | uipath-agents | Wires factory based on execution type |

## MCP SDK Reference

The implementation uses these MCP SDK components:

- `mcp.ClientSession` - MCP client session (`initialize()` is idempotent per instance)
- `mcp.shared.exceptions.MCPError` - Error handling
- `mcp.types.CallToolResult` - Tool call results
- `mcp.client.streamable_http.streamable_http_client` - Upstream transport context manager
- `httpx2.AsyncClient` - HTTP and SSE client used by MCP SDK 2

Key SDK behaviors:
- `ClientSession.initialize()` sends the latest legacy initialize request and initialized notification
- `ClientSession.call_tool()` calls `_validate_tool_result()` on success
- `_validate_tool_result()` calls `list_tools()` if output schema not cached
- A session-bound bare HTTP 404 is converted to `MCPError(INVALID_REQUEST, "Session terminated")`

SDK 2 accepts legacy handshake responses for `2024-11-05`, `2025-03-26`,
`2025-06-18`, and `2025-11-25`. This low-level UiPath path uses
`ClientSession.initialize()`, so a server that supports only modern
`2026-07-28` discovery is not supported here; the SDK high-level
`Client(mode="auto")` owns that probe/fallback behavior.

## Performance Considerations

Session reinitialization is efficient because:

1. **HTTP client reused**: No new TCP connections
2. **Connection state replaced**: A fresh transport/task group and `ClientSession`
3. **SessionInfo reused**: No new factory calls or debug state loads
4. **Only MCP handshake repeated**: Initialize + initialized notification before retry

This is significantly faster than full client reinitialization, which would require:
- Creating a new `httpx2.AsyncClient`
- Resolving the MCP registration and authorization again
- Re-running the `SessionInfoFactory` and any external debug-state load
