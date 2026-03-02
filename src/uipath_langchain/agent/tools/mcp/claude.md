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
└── streamable_http.py   # SessionInfo, StreamableHTTPTransport (copied from MCP SDK)
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

### streamable_http.py — Local Copy of MCP SDK Transport

This file is a local copy of the **client-side** streamable HTTP transport from
the MCP Python SDK, adapted for session ID tracking via `SessionInfo`.

**Source**: [`mcp.client.streamable_http`](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/streamable_http.py)

**Why a local copy?**

The upstream SDK transport has no hook for observing or injecting session IDs.
We need this to support session persistence (e.g. debug state for playground
mode).  The local copy adds a `SessionInfo` parameter that receives session ID
updates from the server.

**Key differences from the upstream SDK:**

1. **`SessionInfo` class added** — base class for session ID tracking, defined
   at the top of the file.  The transport delegates all session ID storage to
   this object via async methods.
2. **Transport does not own session state** — `StreamableHTTPTransport` has no
   `self.session_id`.  All reads/writes go through `self._session_info`.
3. **`_prepare_headers` is async** — because it calls
   `await self._session_info.get_session_id()`.
4. **`_maybe_extract_session_id_from_response` is async** — calls
   `await self._session_info.set_session_id()` so subclasses can persist.
5. **`RequestContext` has no `session_id` field** — it was unused upstream
   (headers are built from `_prepare_headers`, not from the context).
6. **`streamable_http_client` accepts `session_info` parameter** — passed
   through to the transport constructor.
7. **Returns 2 values, not 3** — yields `(read_stream, write_stream)` instead
   of the SDK's `(read_stream, write_stream, get_session_id_callback)`.

**What was kept identical:**

The overall request/response flow, SSE handling, reconnection logic, POST/GET
patterns, and error handling are structurally the same as the upstream SDK.
When updating, diff against the upstream source to understand what changed.

#### SessionInfo

Base class for MCP session ID tracking.  Lives in `streamable_http.py`.

```python
class SessionInfo:
    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id

    async def get_session_id(self) -> str | None: ...
    async def set_session_id(self, session_id: str) -> None: ...
```

The base implementation stores session ID in a plain attribute.  Async methods
exist so subclasses (e.g. `SessionInfoDebugState` in `uipath-agents`) can add
side-effects like HTTP persistence.

**Important:** The transport calls `set_session_id` during `initialize()` when
the server assigns a session ID.  `McpClient._initialize_session` then reads
the value via `get_session_id` — it does not call `set_session_id` again.

#### StreamableHTTPTransport

Handles the MCP streamable HTTP protocol: POST for requests, GET for
server-initiated SSE streams, reconnection with `Last-Event-ID`, and session
termination via DELETE.

Key methods:

| Method | Description |
|--------|-------------|
| `_prepare_headers()` | **async** — builds headers with session ID from `SessionInfo` |
| `_maybe_extract_session_id_from_response()` | **async** — extracts session ID from response, calls `set_session_id` |
| `_handle_post_request()` | POST with JSON or SSE response handling |
| `handle_get_stream()` | GET SSE listener with auto-reconnect |
| `_handle_reconnection()` | Recursive reconnect with `Last-Event-ID` |
| `post_writer()` | Main write loop, dispatches requests to server |
| `terminate_session()` | Sends DELETE to end the session |
| `get_session_id()` | **async** — delegates to `SessionInfo.get_session_id` |

#### streamable_http_client (context manager)

Internal async context manager that wires up `StreamableHTTPTransport` with
memory streams and a task group.  Used by `McpClient._initialize_client`.

```python
async with streamable_http_client(url, http_client=client, session_info=info) as (read, write):
    session = ClientSession(read, write)
```

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
2. **Session Reinitialization** (on 404): Lightweight, reuses existing client

```
┌─────────────────────────────────────────────────────────────┐
│                     McpClient                               │
├─────────────────────────────────────────────────────────────┤
│  Configuration (immutable after __init__)                   │
│  ─────────────────────────────────────────                  │
│  _config: AgentMcpResourceConfig  # Contains slug, folder   │
│  _timeout: httpx.Timeout                                    │
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
│  Client State (created once, reused on session reinit)      │
│  ─────────────────────────────────────────────────────      │
│  _http_client: httpx.AsyncClient | None                     │
│  _read_stream: MemoryObjectReceiveStream | None             │
│  _write_stream: MemoryObjectSendStream | None               │
│  _session_info: SessionInfo | None                          │
│  _stack: AsyncExitStack | None                              │
│  _client_initialized: bool                                  │
├─────────────────────────────────────────────────────────────┤
│  Session State (can be reinitialized without recreating)    │
│  ───────────────────────────────────────────────────────    │
│  _session: ClientSession | None                             │
│  _session_id: str | None                                    │
├─────────────────────────────────────────────────────────────┤
│  Public Methods                                             │
│  ──────────────                                             │
│  + call_tool(name, arguments) -> CallToolResult             │
│  + dispose() -> None  # UiPathDisposableProtocol            │
│  + session_id: str | None (property)                        │
│  + is_client_initialized: bool (property)                   │
├─────────────────────────────────────────────────────────────┤
│  Private Methods                                            │
│  ───────────────                                            │
│  - _initialize_client() -> None    # SDK + full init (once) │
│  - _initialize_session() -> None   # MCP handshake only     │
│  - _ensure_session() -> ClientSession                       │
│  - _reinitialize_session() -> None                          │
│  - _is_session_error(error) -> bool                         │
└─────────────────────────────────────────────────────────────┘
```

#### Session ID Flow

During client initialization, `McpClient`:

1. Retrieves the `McpServer` from the UiPath SDK
2. Calls `self._session_info_factory.create_session(mcp_server)` to get a `SessionInfo`
3. Loads any existing session ID via `await session_info.get_session_id()`
4. Passes the `SessionInfo` to the local `streamable_http_client`
5. Calls `session.initialize()` — the transport calls `set_session_id` internally
6. Reads the new session ID via `await session_info.get_session_id()`

On session reinitialization (404 retry), only steps 5-6 repeat.

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

#### `open_mcp_tools(config)` → Context Manager

Async context manager that wraps `create_mcp_tools_and_clients()` with automatic
client lifecycle management.  Yields a list of `BaseTool` instances and
disposes all `McpClient` instances on exit.

## Two-Phase Initialization

The key design principle is separating **client initialization** from **session initialization**:

```
Phase 1: Client Initialization (expensive, done once)
──────────────────────────────────────────────────────
┌─────────────────┐
│ UiPath SDK      │ ─── Retrieves MCP server URL
│ mcp.retrieve()  │     and auth token (Bearer)
└─────────────────┘

┌─────────────────┐
│ SessionInfo     │ ─── Factory creates SessionInfo
│ Factory         │     (may load existing session ID)
└─────────────────┘

┌─────────────────┐
│ httpx.AsyncClient │ ─┐
└─────────────────┘   │
                      │
┌─────────────────┐   │  Created once via
│ streamable_http │   ├─ AsyncExitStack
│   connection    │   │
└─────────────────┘   │
                      │
┌─────────────────┐   │
│  ClientSession  │ ─┘
└─────────────────┘

Phase 2: Session Initialization (lightweight, can repeat)
─────────────────────────────────────────────────────────
┌─────────────────┐
│ session.        │ ─── Sends initialize request
│ initialize()    │     Transport calls set_session_id()
└─────────────────┘
┌─────────────────┐
│ McpClient reads │ ─── await session_info.get_session_id()
│ new session ID  │
└─────────────────┘
```

### Session Lifecycle

```
           ┌──────────────┐
           │   Created    │
           │ (nothing     │
           │  initialized)│
           └──────┬───────┘
                  │ call_tool() [first time]
                  ▼
           ┌──────────────┐
           │   Client     │
           │ Initializing │
           │ (Phase 1)    │
           └──────┬───────┘
                  │ 1. UiPath SDK retrieves MCP URL
                  │ 2. Factory creates SessionInfo
                  │ 3. Creates HTTP client, streams, session
                  │ 4. Calls _initialize_session()
                  ▼
           ┌──────────────┐
           │   Session    │
           │ Initializing │◄────────────────┐
           │ (Phase 2)    │                 │
           └──────┬───────┘                 │
                  │ sends initialize,       │
                  │ transport calls         │ 404 error
                  │ set_session_id()        │ (only reinit
                  ▼                         │  session,
           ┌──────────────┐                 │  not client)
           │    Active    │─────────────────┘
           │   Session    │
           └──────┬───────┘
                  │ dispose()
                  ▼
           ┌──────────────┐
           │    Closed    │
           │ (can reuse)  │
           └──────────────┘
```

### MCP Protocol Flow

**First tool call (full initialization):**

```
Client                              Server
   │                                   │
   │──── initialize ──────────────────►│
   │◄─── result + session-id-1 ────────│  ← transport calls set_session_id()
   │                                   │
   │──── notifications/initialized ───►│
   │◄─── 204 No Content ───────────────│
   │                                   │
   │──── tools/call ──────────────────►│
   │◄─── result ───────────────────────│
```

**On 404 error (session reinitialization only):**

```
Client                              Server
   │                                   │
   │──── tools/call ──────────────────►│
   │◄─── 404 (session terminated) ─────│
   │                                   │
   │  [Reuses existing HTTP client     │
   │   and streamable connection]      │
   │                                   │
   │──── initialize ──────────────────►│  ← new session
   │◄─── result + session-id-2 ────────│    (same client)
   │                                   │
   │──── notifications/initialized ───►│
   │◄─── 204 No Content ───────────────│
   │                                   │
   │──── tools/call ──────────────────►│  ← retry
   │◄─── result ───────────────────────│
```

### Session Error Codes

The following error codes trigger automatic session reinitialization:

| Code | Meaning | Source |
|------|---------|--------|
| `32600` | Session terminated | HTTP 404 converted by transport |
| `-32000` | Server error | Can indicate session not found |

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

default_client_kwargs = get_httpx_client_kwargs()
self._http_client = await self._stack.enter_async_context(
    httpx.AsyncClient(
        **default_client_kwargs,
        headers=self._headers,
        timeout=self._timeout,
    )
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

async def _reinitialize_session(self) -> None:
    async with self._lock:
        if not self._client_initialized:
            await self._initialize_client()
        else:
            await self._initialize_session()  # Lightweight!
```

### 4. No `with` Statement for AsyncExitStack

Manual lifecycle management:

```python
# Correct - manual management
self._stack = AsyncExitStack()
await self._stack.__aenter__()
# ... use stack ...
await self._stack.__aexit__(None, None, None)

# Wrong - exits too early
async with AsyncExitStack() as stack:
    ...  # Stack closes here!
```

### 5. Reinitialization Reuses Client

On 404, only `_initialize_session()` is called — the HTTP client, streams,
and `SessionInfo` instance are all reused.

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
| `test_mcp_client.py` | McpClient session tests (7 tests) |
| `test_mcp_tool.py` | Tool factory tests (17 tests) |

### Key Test Classes

| Class | Tests |
|-------|-------|
| `TestMcpClient` | Session lifecycle, 404 retry, client reuse |
| `TestMcpToolMetadata` | Tool metadata (tool_type, display_name, etc.) |
| `TestMcpToolCreation` | Multiple tools, descriptions, disabled config |
| `TestCreateMcpToolsFromAgent` | Agent factory function tests |
| `TestMcpToolInvocation` | Full invocation flow smoke test |
| `TestMcpToolNameSanitization` | Tool name sanitization |

### Key Assertion

The most important test verifies client reuse on 404:

```python
# HTTP client created only ONCE (not recreated on retry)
assert mock_async_client_class.call_count == 1

# But session initialized TWICE
assert initialize_count[0] == 2
```

## Guidelines for Changes

### Updating streamable_http.py

When the upstream MCP SDK changes its transport:

1. Diff the upstream [`mcp/client/streamable_http.py`](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/streamable_http.py) against our local copy
2. Apply upstream changes while preserving our `SessionInfo` integration
3. Key areas to watch: `_prepare_headers` (must stay async), `_maybe_extract_session_id_from_response` (must use `set_session_id`), `streamable_http_client` (must accept `session_info` param)
4. The transport must never own session state directly — always delegate to `_session_info`

### Adding New Factory Functions

1. Follow the pattern of existing functions
2. Always handle `is_enabled=False` case by returning empty list
3. Include proper metadata on created tools (`tool_type`, `display_name`, `folder_path`, `slug`)
4. Add tests for the new function

### Modifying Client Initialization

1. Changes go in `_initialize_client()`
2. All resources must be added to `_stack` via `enter_async_context()`
3. Set `_client_initialized = True` before calling `_initialize_session()`
4. Always use `get_httpx_client_kwargs()` for HTTP client
5. The `SessionInfo` is created via the factory — do not construct it directly

### Modifying Session Initialization

1. Changes go in `_initialize_session()`
2. This should remain lightweight — just the MCP handshake
3. Don't create new HTTP resources here
4. The transport handles `set_session_id` — `_initialize_session` only reads via `get_session_id`
5. Verify tests still show `mock_async_client_class.call_count == 1` on retry

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
| `streamable_http.py` | uipath-langchain | SessionInfo + transport (local SDK copy) |
| `mcp_client.py` | uipath-langchain | SessionInfoFactory + McpClient |
| `mcp_tool.py` | uipath-langchain | Tool factory functions |
| `__init__.py` | uipath-langchain | Public exports |
| `session_info_debug_state.py` | uipath-agents | SessionInfoDebugState + factory |
| `graph.py` | uipath-agents | Wires factory based on execution type |

## MCP SDK Reference

The implementation uses these MCP SDK components:

- `mcp.ClientSession` - MCP client session (can call `initialize()` multiple times)
- `mcp.shared.exceptions.McpError` - Error handling
- `mcp.types.CallToolResult` - Tool call results
- `mcp.client._transport.TransportStreams` - Type alias used by `streamable_http_client`
- `mcp.shared._httpx_utils.create_mcp_http_client` - Default HTTP client factory
- `mcp.shared.message.SessionMessage` - Message wrapper for JSON-RPC

Key SDK behaviors:
- `ClientSession.initialize()` sends initialize request + initialized notification
- `ClientSession.call_tool()` calls `_validate_tool_result()` on success
- `_validate_tool_result()` calls `list_tools()` if output schema not cached
- HTTP 404 is converted to `McpError` with code `32600` by `StreamableHTTPTransport`

## Performance Considerations

Session reinitialization is efficient because:

1. **HTTP client reused**: No new TCP connections
2. **Streamable connection reused**: No new task groups or streams
3. **SessionInfo reused**: No new factory calls or debug state loads
4. **Only MCP handshake**: Just 2 HTTP requests (initialize + notification)

This is significantly faster than full client reinitialization, which would require:
- Creating new `httpx.AsyncClient`
- Creating new task groups
- Creating new memory streams
- Establishing new connections
