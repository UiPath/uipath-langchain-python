# MCP Session Implementation Guide

> **CLAUDE: UPDATE THIS DOCUMENT**
>
> When you modify `_mcp_client.py` or `mcp_tool.py` (MCP-related code), you MUST update this document to reflect:
> - New or changed class attributes/methods (update Architecture section)
> - Changes to initialization phases (update Two-Phase Initialization section)
> - New error codes (update Session Error Codes table)
> - Protocol flow changes (update MCP Protocol Flow diagrams)
> - New guidelines or gotchas (update Guidelines for Changes section)
>
> Keep diagrams and code examples in sync with the actual implementation.

## Overview

This document describes the MCP (Model Context Protocol) session management implementation in `_mcp_client.py` and its associated tests. Use this as a reference when modifying MCP-related code.

## Architecture

### McpClient Class

`McpClient` manages the lifecycle of MCP connections for tool invocations with **two distinct initialization phases**:

1. **Client Initialization** (first call): Full stack creation
2. **Session Reinitialization** (on 404): Lightweight, reuses existing client

```
┌─────────────────────────────────────────────────────────────┐
│                     McpClient                          │
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
   │                                   │
   │──── tools/list ──────────────────►│  ← SDK validates
   │◄─── tool definitions ─────────────│
```

### Session Error Codes

The following error codes trigger automatic session reinitialization:

| Code | Meaning | Source |
|------|---------|--------|
| `32600` | Session terminated | HTTP 404 converted by SDK |
| `-32000` | Server error | Can indicate session not found |

## Key Implementation Details

### 1. Single Lock for Both Phases

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

### 2. Why asyncio.Lock?

`asyncio.Lock` is the Pythonic choice for async code because:
- It's designed for async/await patterns
- It's non-blocking (doesn't block the event loop)
- It handles async context managers properly
- Standard library, no extra dependencies

### 3. Client vs Session Separation

```python
async def _initialize_client(self) -> None:
    """Phase 1: Create everything, then call Phase 2."""
    # Create HTTP client, streams, ClientSession...
    self._client_initialized = True
    await self._initialize_session()  # Chain to Phase 2

async def _initialize_session(self) -> None:
    """Phase 2: Just the MCP handshake."""
    await self._session.initialize()
    self._session_id = self._get_session_id()
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

Tests are in `tests/agent/tools/test_mcp/test__mcp_client.py`.

For detailed test documentation, mocking strategies, and guidelines for adding new tests, see:
**`tests/agent/tools/test_mcp/claude.md`**

### Quick Reference

| Test | Purpose |
|------|---------|
| `test_session_initializes_on_first_call` | Verifies lazy initialization |
| `test_session_reused_across_calls` | Verifies session reuse |
| `test_session_reinitializes_on_404_error` | **Key test**: client reused, only session reinit |
| `test_max_retries_exceeded` | Verifies exception after max retries |
| `test_close_releases_resources` | Verifies cleanup |
| `test_client_initialized_property` | Verifies property reflects state |
| `test_session_can_be_reused_after_close` | Verifies full reinit after close |

### Key Assertion

The most important test verifies client reuse on 404:

```python
# HTTP client created only ONCE (not recreated on retry)
assert mock_async_client_class.call_count == 1

# But session initialized TWICE
assert initialize_count[0] == 2
```

## Guidelines for Changes

### Adding New Error Codes for Retry

1. Add the error code to `SESSION_ERROR_CODES`:
   ```python
   SESSION_ERROR_CODES = [32600, -32000, NEW_CODE]
   ```

2. Add a test case that verifies the new code triggers retry
3. Ensure it only triggers session reinit, not full client reinit

### Modifying Client Initialization

1. Changes go in `_initialize_client()`
2. All resources must be added to `_stack` via `enter_async_context()`
3. Set `_client_initialized = True` before calling `_initialize_session()`
4. Update tests if the initialization sequence changes

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

### Modifying Tests

See `tests/agent/tools/test_mcp/claude.md` for detailed guidance on:
- Adding new MCP methods to `MockStreamResponse`
- Testing error scenarios
- Verifying client reuse
- Debugging failed tests

### SDK Changes

If the MCP SDK (`mcp` package) changes:

1. Check if `streamable_http_client` signature changed
2. Check if `ClientSession.initialize()` can still be called multiple times
3. Check if error codes changed
4. Update `_is_session_error()` if error structure changes

## Related Files

| File | Purpose |
|------|---------|
| `_mcp_client.py` | Session management implementation |
| `mcp_tool.py` | Tool creation (uses McpClient) |
| `test__mcp_client.py` | Session tests (7 tests including 404 retry) |
| `test_mcp_tool.py` | Tool creation and invocation tests (11 tests) |

## MCP SDK Reference

The implementation uses these MCP SDK components:

- `mcp.ClientSession` - MCP client session (can call `initialize()` multiple times)
- `mcp.client.streamable_http.streamable_http_client` - HTTP transport
- `mcp.shared.exceptions.McpError` - Error handling
- `mcp.types.CallToolResult` - Tool call results

SDK source: `/Users/eduard/work/agenthub/modelcontextprotocol/python-sdk/src/mcp/`

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
