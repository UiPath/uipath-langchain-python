# MCP Session Implementation Guide

> **CLAUDE: UPDATE THIS DOCUMENT**
>
> When you modify `mcp_client.py` or `mcp_tool.py` (MCP-related code), you MUST update this document to reflect:
> - New or changed class attributes/methods (update Architecture section)
> - Changes to initialization phases (update Two-Phase Initialization section)
> - New error codes (update Session Error Codes table)
> - Protocol flow changes (update MCP Protocol Flow diagrams)
> - New guidelines or gotchas (update Guidelines for Changes section)
>
> Keep diagrams and code examples in sync with the actual implementation.

## Overview

This document describes the MCP (Model Context Protocol) session management implementation in `mcp_client.py` and tool factory functions in `mcp_tool.py`. Use this as a reference when modifying MCP-related code.

## Module Structure

```
src/uipath_langchain/agent/tools/mcp/
├── __init__.py          # Public exports
├── mcp_client.py        # McpClient class (session management)
└── mcp_tool.py          # Tool factory functions
```

### Public Exports (`__init__.py`)

```python
from .mcp_client import McpClient
from .mcp_tool import (
    create_mcp_tools,                           # Context manager for live sessions
    create_mcp_tools_from_agent,                # Factory from LowCodeAgentDefinition
    create_mcp_tools_from_metadata_for_mcp_server,  # Factory from config + McpClient
)
```

## Architecture

### McpClient Class

`McpClient` manages the lifecycle of MCP connections for tool invocations with **two distinct initialization phases**:

1. **Client Initialization** (first call): Full stack creation
2. **Session Reinitialization** (on 404): Lightweight, reuses existing client

```
┌─────────────────────────────────────────────────────────────┐
│                     McpClient                               │
├─────────────────────────────────────────────────────────────┤
│  Configuration (immutable after __init__)                   │
│  ─────────────────────────────────────────                  │
│  _url: str                                                  │
│  _headers: dict[str, str]                                   │
│  _timeout: httpx.Timeout                                    │
│  _max_retries: int                                          │
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
│  _get_session_id: GetSessionIdCallback | None               │
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
│  + close() -> None                                          │
│  + session_id: str | None (property)                        │
│  + is_client_initialized: bool (property)                   │
├─────────────────────────────────────────────────────────────┤
│  Private Methods                                            │
│  ───────────────                                            │
│  - _initialize_client() -> None    # Full init (once)       │
│  - _initialize_session() -> None   # MCP handshake only     │
│  - _ensure_session() -> ClientSession                       │
│  - _reinitialize_session() -> None                          │
│  - _is_session_error(error) -> bool                         │
└─────────────────────────────────────────────────────────────┘
```

### Tool Factory Functions

#### `create_mcp_tools_from_agent(agent)` → `tuple[list[BaseTool], list[McpClient]]`

**Primary factory function** for creating MCP tools from a LowCodeAgentDefinition.

```python
async def create_mcp_tools_from_agent(
    agent: LowCodeAgentDefinition,
) -> tuple[list[BaseTool], list[McpClient]]:
    """Create MCP tools from a LowCodeAgentDefinition.

    Iterates over all MCP resources in the agent definition and creates tools
    for each enabled MCP server. Each MCP server gets its own McpClient instance.

    The UiPath SDK is lazily initialized inside this function using environment
    variables (UIPATH_URL, UIPATH_ACCESS_TOKEN).

    Returns:
        A tuple of (tools, mcp_clients) where:
        - tools: List of BaseTool instances for all MCP resources
        - mcp_clients: List of McpClient instances that need to be closed when done

    Note:
        The caller is responsible for closing the McpClient instances when done.
    """
```

**Usage:**
```python
tools, clients = await create_mcp_tools_from_agent(agent)
try:
    # Use tools...
finally:
    for client in clients:
        await client.close()
```

#### `create_mcp_tools_from_metadata_for_mcp_server(config, mcpClient)` → `list[BaseTool]`

Creates tools for a single MCP resource config using an existing McpClient.

#### `create_mcp_tools(config)` → Context Manager

Async context manager that creates live MCP sessions (uses langchain-mcp-adapters).

### Two-Phase Initialization

The key design principle is separating **client initialization** from **session initialization**:

```
Phase 1: Client Initialization (expensive, done once)
──────────────────────────────────────────────────────
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
│ session.        │ ─── Just sends initialize request
│ initialize()    │     and stores new session_id
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
                  │ creates HTTP client, streams, session
                  │ then calls _initialize_session()
                  ▼
           ┌──────────────┐
           │   Session    │
           │ Initializing │◄────────────────┐
           │ (Phase 2)    │                 │
           └──────┬───────┘                 │
                  │ sends initialize,       │
                  │ gets session_id         │ 404 error
                  ▼                         │ (only reinit
           ┌──────────────┐                 │  session,
           │    Active    │─────────────────┘  not client)
           │   Session    │
           └──────┬───────┘
                  │ close()
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
   │◄─── result + session-id-1 ────────│
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
| `32600` | Session terminated | HTTP 404 converted by SDK |
| `-32000` | Server error | Can indicate session not found |

## Key Implementation Details

### 1. HTTP Client Configuration

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

### 2. Single Lock for Both Phases

One `asyncio.Lock` protects both client initialization and session reinitialization:

```python
self._lock = asyncio.Lock()

async def _ensure_session(self) -> ClientSession:
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

### 3. No `with` Statement for AsyncExitStack

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

### 4. Reinitialization Reuses Client

The key optimization - on 404, only `_initialize_session()` is called:

```python
async def _reinitialize_session(self) -> None:
    async with self._lock:
        if not self._client_initialized:
            await self._initialize_client()  # Full init if needed
        else:
            await self._initialize_session()  # Just handshake!
```

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
5. Update tests if the initialization sequence changes

### Modifying Session Initialization

1. Changes go in `_initialize_session()`
2. This should remain lightweight - just the MCP handshake
3. Don't create new HTTP resources here
4. Verify tests still show `mock_async_client_class.call_count == 1` on retry

### Adding New Methods to McpClient

1. If the method accesses `_session`, use `_ensure_session()`:
   ```python
   async def new_method(self):
       session = await self._ensure_session()
       return await session.some_method()
   ```

2. If the method needs retry logic, follow the pattern in `call_tool()`

3. If the method modifies session state, acquire `_lock`

## Related Files

| File | Purpose |
|------|---------|
| `mcp_client.py` | McpClient session management |
| `mcp_tool.py` | Tool factory functions |
| `__init__.py` | Public exports |
| `test_mcp_client.py` | Session tests |
| `test_mcp_tool.py` | Tool factory tests |

## MCP SDK Reference

The implementation uses these MCP SDK components:

- `mcp.ClientSession` - MCP client session (can call `initialize()` multiple times)
- `mcp.client.streamable_http.streamable_http_client` - HTTP transport
- `mcp.shared.exceptions.McpError` - Error handling
- `mcp.types.CallToolResult` - Tool call results

Key SDK behaviors:
- `ClientSession.initialize()` sends initialize request + initialized notification
- `ClientSession.call_tool()` calls `_validate_tool_result()` on success
- `_validate_tool_result()` calls `list_tools()` if output schema not cached
- HTTP 404 is converted to `McpError` with code `32600` by `StreamableHTTPTransport`
- `StreamableHTTPTransport.session_id` is updated on each initialize response

## Performance Considerations

Session reinitialization is efficient because:

1. **HTTP client reused**: No new TCP connections
2. **Streamable connection reused**: No new task groups or streams
3. **Only MCP handshake**: Just 2 HTTP requests (initialize + notification)

This is significantly faster than full client reinitialization, which would require:
- Creating new `httpx.AsyncClient`
- Creating new task groups
- Creating new memory streams
- Establishing new connections
