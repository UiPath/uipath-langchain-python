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
├── __init__.py            # Public exports
├── mcp_client.py          # SessionInfoFactory, McpClient
├── mcp_tool.py            # Tool factory functions
├── protocol_strategy.py   # Per-era negotiation/recovery policy
├── session_tools.py       # MCP session -> LangChain tool conversion
└── streamable_http.py     # SessionInfo, session identity, SDK transport adapter
```

### Public Exports (`__init__.py`)

```python
from .mcp_client import McpClient, SessionInfoFactory
from .mcp_tool import (
    create_mcp_tools_and_clients,
    open_mcp_tools,
    create_mcp_tools,
)
from .session_tools import load_mcp_tools
from .streamable_http import SessionInfo
```

`streamable_http_client`, `build_protocol_strategy`, and the `SessionIdentity`
types are intentionally **not** exported — they are internal helpers used by
`McpClient` (and by the `simple-http-mcp` testcase, which imports them by module
path to drive the same code the client uses).

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

1. Before every request, call `SessionInfo.get_session_id()` and put the ID on
   the headers named by the active `SessionIdentityWire` (removing any it does
   not name).
2. After every response, persist an ID returned on the wire's
   `capture_response_header` through `SessionInfo.set_session_id()`.

The hooks are removed when the adapter context exits. This keeps the UiPath
extension small and automatically picks up future SDK transport fixes instead
of maintaining another transport fork.

#### SessionIdentityWire / SessionIdentity

The two protocol eras identify a connection differently, so the header behaviour
is data rather than code:

```python
@dataclass(frozen=True)
class SessionIdentityWire:
    request_headers: tuple[str, ...] = (MCP_SESSION_ID,)
    capture_response_header: str | None = MCP_SESSION_ID
```

The ID travels on the header only. Mirroring it into `params._meta` was tried
and removed: rewriting the request body needs private HTTPX internals, and
gateway routing needs nothing beyond the header.

| Wire | Sends on | Captures from | Used by |
|------|----------|---------------|---------|
| `LEGACY_IDENTITY` | `mcp-session-id` | `mcp-session-id` | legacy handshake; also `auto` before the era resolves |
| `MODERN_IDENTITY` | `mcp-session-id` | nothing (client mints) | `2026-07-28` |

**Both eras send the ID on `mcp-session-id`.** In the legacy era it is a real
session the server minted; in the modern era the protocol has no session, so
UiPath mints the value itself and the header carries it purely as a routing key,
which a modern server ignores. Reusing the header rather than inventing one means
the gateway needs no change — it keeps routing on the header it already uses.

The transport is opened **before** negotiation runs, so the era-specific wire
cannot be fixed at construction time. `streamable_http_client` therefore takes a
mutable `SessionIdentity` holder and reads `identity.wire` on every request; the
strategy narrows it once `connect()` resolves the era.

`auto` can safely open on the legacy wire: a stored ID is era-ambiguous, but both
eras send it on the same header, and a modern server never sends one back so
there is nothing to capture.

The eras differ only in `capture_response_header`, and that difference matters.
`MODERN_IDENTITY` sets it to `None` so a proxy or gateway echoing `mcp-session-id`
back cannot overwrite the client-minted routing key mid-connection and scatter
the remaining requests across instances.

#### SessionInfo

Base class for MCP session ID tracking.  Lives in `streamable_http.py`.

```python
class SessionInfo:
    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        self.protocol_version: str | None = None

    async def get_session_id(self) -> str | None: ...
    async def set_session_id(self, session_id: str | None) -> None: ...
    async def get_protocol_version(self) -> str | None: ...
    async def set_protocol_version(self, protocol_version: str | None) -> None: ...
```

The base implementation stores session ID in a plain attribute.  Async methods
exist so subclasses (e.g. `SessionInfoDebugState` in `uipath-agents`) can add
side-effects like HTTP persistence.

`SessionInfo` is deliberately era-agnostic: it stores **the ID we persist for
this MCP server**, whoever minted it. In the legacy era that is the server's
session ID; in the modern era it is the client-minted affinity ID. This is why
reaching `2026-07-28` required no change in `uipath-agents-python` —
`SessionInfoDebugState` persists either kind unmodified.

`protocol_version` holds **the version the stored session was negotiated at**,
and the legacy strategy both writes and reads it. It cannot be recovered from
the wire -- responses carry only the session ID -- so a store that persists the
ID should persist this too: with it, a resumed session needs no negotiation at
all (see below). A store that does not is not broken, only slower: `None` means
"not known", and the strategy falls back to re-running the handshake.

A subclass persisting externally should override all four accessors, so the ID
and its version are written and cleared together.

**Important:** The response hook calls `set_session_id` during `initialize()`
when the server assigns an ID. `McpClient._initialize_session` only reads the
stored value afterward. Passing `None` clears a stale session before recovery;
`LegacyHandshakeStrategy.reset` clears the version alongside it, so a
replacement session can never inherit a version it did not negotiate.

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

### protocol_strategy.py — Per-Era Session Lifecycle

MCP has two negotiation eras, and `ProtocolStrategy` is the seam between them.
Negotiation itself is one call either way; what genuinely differs is the session
*lifecycle*.

| Concern | Legacy (`2024-11-05`…`2025-11-25`) | Modern (`2026-07-28`) |
|---------|------------------------------------|------------------------|
| Negotiate | `ClientSession.initialize()` | `ClientSession.discover()` |
| Identity | `mcp-session-id`, server-minted | none in-protocol; UiPath-minted affinity ID |
| Resume | adopt the stored version locally; re-handshake only when it is unknown | nothing to resume |
| Terminate on close | `DELETE` | no-op (automatic; no session ID) |
| Recoverable errors | session lost → reopen and re-handshake | only `CONNECTION_CLOSED` |

```python
class ProtocolStrategy(Protocol):
    identity: SessionIdentity

    async def connect(self, session: ClientSession, info: SessionInfo) -> None: ...
    def is_recoverable(self, error: MCPError, restored_id: str | None) -> bool: ...
    async def reset(self, info: SessionInfo) -> None: ...
```

`McpClient` calls `reset` only when the error that triggered recovery is the
server's verdict on the session, which `is_session_rejected(error)` decides. A
dropped transport is not a verdict -- the server never rejected anything -- so
the ID is kept and the reconnect resumes the warm session; anything the server
answered *about the session* clears the ID before the fresh handshake.

The code alone cannot make that call. `CONNECTION_CLOSED` is JSON-RPC's
implementation-defined server-error code `-32000`, and the TypeScript SDK's
Streamable HTTP transport uses the same code to refuse a session it does not
know (`"Bad Request: No valid session ID provided"`). Treating every `-32000`
as a dropped connection would keep a session the server has just declared dead
and burn the retry resuming it, so a `-32000` whose message names a lost
session counts as a verdict.

Selected by `McpClient(protocol_mode=...)` via `build_protocol_strategy`:

- **`"legacy"` (default)** — `LegacyHandshakeStrategy`. Preserves the pre-2026
  wire behaviour exactly.
- **`"modern"`** — `ModernDiscoveryStrategy`. `server/discover` only.
- **`"auto"`** — `AutoStrategy`. Mints the affinity ID first so the probe is
  pinned, runs its own `server/discover` probe (`probe_modern_era`, built on the
  public `ClientSession.send_discover` / `adopt` seam), then delegates to
  whichever era won. Re-resolved on every `connect`, so a server upgraded
  mid-run is handled.

The default stays `"legacy"` on purpose. Defaulting to `"auto"` would silently
move any discovery-capable UiPath MCP server to stateless `2026-07-28` and stop
issuing session IDs, breaking the playground persistence `SessionInfoDebugState`
exists for.

#### Resuming a legacy session: adopt the version, do not renegotiate

A restored session ID is useless without the protocol version it was negotiated
at, and that version cannot be recovered from the wire — server responses carry
only the session ID. It is therefore stored *with* the ID, and a resume installs
it locally through `ClientSession.adopt`, which is documented to touch no wire:

```python
restored = await info.get_session_id()
if restored is None:
    await self._handshake(session, info)          # cold: negotiate and store the version
    return
if await self._adopt_restored_session(session, info, restored):
    return                                        # no request sent at all
try:
    result = await session.initialize()           # version unknown: lands inside the session
except MCPError as error:
    if error.code == CONNECTION_CLOSED:
        raise                                     # transport died; says nothing about the session
    await self.reset(info)                        # stale, or this server refuses a 2nd handshake
    await self._handshake(session, info)          # same transport: a refused request does not close it
```

**Why adopting beats re-handshaking.** Whether a server accepts a second
`initialize` inside a live session is implementation-defined. The Python SDK
does. The reference TypeScript implementation refuses it outright with
`-32600 "Invalid Request: Server already initialized"`, so a client that resumes
by re-handshaking loses the persisted session on **every** run against such a
server — falling back to a cold one, and with it the gateway affinity the
session ID exists to provide. Adopting asks the server nothing, so it works
either way, and it also restores the pre-SDK-2 wire shape: a resumed run sends
only ordinary requests carrying `mcp-session-id`.

The handshake path remains for a store written before versions were recorded
(`get_protocol_version()` returns `None`), and for a stored version this client
cannot speak on a legacy wire — a modern version, or one dropped upstream. It is
safe because the server routes purely by the session header and mints a new
session only when the header is **absent**
([`streamable_http_manager.py`](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/streamable_http_manager.py)).
A server that ignores the header instead mints a replacement; the strategy detects
that by comparing the ID before and after and continues with the new session.

Probing candidate versions with `send_ping` — the original approach — always
matched the *oldest* handshake version, because servers do not validate that
header against what the session negotiated. That silently downgraded every later
request and disabled the server's `2025-11-25` SSE resumability. Do not
reintroduce it: the version is remembered now, not guessed.

#### Modern-era instance affinity

`2026-07-28` removes `mcp-session-id` from the protocol, and AgentHub used it to
route to a warm serverless instance. `ModernDiscoveryStrategy` mints its own ID
and keeps sending it on that same header as an opaque routing key — off-spec for
the era, ignored by a modern server, and requiring **no gateway change**. Because
the client mints it *before* negotiating, it is present on the very first request
— `server/discover` included — which a server-assigned session ID never could be.

In `auto` mode the ID is minted *before* the probe as well, so `server/discover`
reaches the same instance the tool calls will. On a serverless gateway an
unpinned probe warms one instance and the first call lands on another, which is
exactly the scatter the affinity ID exists to prevent. A legacy server never
issued that ID, so when the probe falls back the freshly minted ID is cleared
before the handshake rather than offered as a session to resume -- a routing
server would refuse it. A *restored* ID is not cleared: it may be a live legacy
session, and the handshake resumes it (or replaces it when the server rejects it).

**Affinity is a hint, not a guarantee.** A fresh client has no ID and every
modern request is self-contained, so any instance must be able to answer any
request. Instance-local state is valid as a warm cache only; a server whose
instances hold state no peer can rebuild should stay on `protocol_mode="legacy"`.

---

### session_tools.py — Session-to-LangChain Tool Conversion

`load_mcp_tools(session)` paginates `tools/list` and returns `StructuredTool`s
bound to that session. It replaced the `langchain-mcp-adapters` dependency, which
imports `RequestContext` — removed in MCP 2.

It is **not** a drop-in replacement for that package's function of the same name:
it returns raw MCP content blocks (camelCase, via `model_dump(by_alias=True)`)
under the default `response_format="content"`, where the old one returned
LangChain content blocks plus a `structured_content` artifact.

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
│  _strategy: ProtocolStrategy   # from protocol_mode         │
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
│  - _initialize_session() -> None   # delegates to strategy  │
│  - _ensure_session() -> ClientSession                       │
│  - _reinitialize_session(failed_session, error) -> None     │
│  - _is_recoverable_session_error(error) -> bool             │
│  + is_session_error(error) -> bool  # legacy rules, public  │
└─────────────────────────────────────────────────────────────┘
```

#### Session ID Flow

During client initialization, `McpClient`:

1. Retrieves the `McpServer` from the UiPath SDK
2. Calls `self._session_info_factory.create_session(mcp_server)` to get a `SessionInfo`
3. Loads any existing session ID via `await session_info.get_session_id()`
4. Passes the `SessionInfo` to the local adapter, which opens the SDK transport
5. Creates a new `ClientSession` over those streams
6. Calls `strategy.connect(session, session_info)`, which negotiates for its era:
   - **legacy** — adopts a restored ID whose version is known, sending nothing;
     otherwise `session.initialize()`, whose response hook stores the
     server-assigned ID, and whose result is stored as the version
   - **modern** — mints an affinity ID if absent, then `session.discover()`
   - **auto** — mints an affinity ID if absent, then `probe_modern_era(session)`;
     on fallback it clears a freshly minted ID and runs the legacy `connect`;
     finally narrows `identity.wire` to the resolved era
7. Reads the current session ID via `await session_info.get_session_id()`

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
│                 │     (skipped when a stored ID *and* version
│                 │      are adopted instead)
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

Which errors trigger reinitialization is **era-specific**, decided by
`strategy.is_recoverable(error, restored_id)`.

**Legacy** (`LegacyHandshakeStrategy`):

| Code | Meaning | Source |
|------|---------|--------|
| `CONNECTION_CLOSED` (`-32000`) | Transport connection closed | MCP SDK dispatcher/transport |
| `INVALID_REQUEST` (`-32600`) | Session terminated/expired/invalid | SDK 2 maps a bare session-bound HTTP 404 to this error |
| `32600` | Session terminated | Compatibility with the positive code emitted by the older local transport |
| `METHOD_NOT_FOUND` (`-32601`) + `"Not Found"` | Restored session is invalid | Only while `SessionInfo` still holds the restored ID |

`CONNECTION_CLOSED` is recovered differently from the other three: the transport
dropped but the server never rejected the session, so `McpClient` skips `reset`
and the reconnect resumes the same session. The other codes are the server's verdict,
and `reset` clears the ID before the fresh handshake.

`-32000` is both `CONNECTION_CLOSED` and the code the TypeScript SDK's transport
refuses an unknown session with, so `is_session_rejected` reads the *message* to
tell a dropped socket from a verdict. See the `reset` discussion above.

`INVALID_REQUEST` is retried only when its message explicitly identifies a
terminated, expired, or invalid session. An externally restored session is not
known inside a newly created SDK transport, so its first bare HTTP 404 appears
as `METHOD_NOT_FOUND`/`"Not Found"`; that exact shape is treated as recoverable
only while the restored ID is still in play. Structured JSON-RPC method errors
are not retried.

**Modern** (`ModernDiscoveryStrategy`): `CONNECTION_CLOSED` **only**. Every
`2026-07-28` request is self-contained, so no server-side session can be lost;
retrying a session-shaped error would spend the retry budget on something a
reconnect cannot fix.

`McpClient.is_session_error` remains public and keeps the legacy rules — it is
called by `mcp_tool._map_mcp_error`.

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
    self,
    failed_session: ClientSession | None = None,
    error: MCPError | None = None,
) -> None:
    async with self._lock:
        if not self._client_initialized:
            await self._initialize_client()
        else:
            # Another failing operation may arrive after recovery completed.
            if failed_session is not None and self._session is not failed_session:
                return
            await self._close_connection_for_recovery()
            # Discard persisted session state only on the server's verdict;
            # a dropped connection keeps it so the reconnect resumes.
            if (
                self._session_info is not None
                and error is not None
                and is_session_rejected(error)
            ):
                await self._strategy.reset(self._session_info)
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
connection stack, calls `strategy.reset` to clear the ID unless
`is_session_rejected(error)` is False -- a dropped transport is not the server's
verdict, so the ID is kept and the reconnect resumes the same session -- and
opens a fresh SDK transport and `ClientSession`. The authenticated `httpx2.AsyncClient`
and `SessionInfo` instance are reused. The failed-session identity guard prevents a late failure
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
| `test_mcp_client_real_http.py` | **`McpClient` over real HTTP** against a real `MCPServer`: negotiation per mode, legacy resume, affinity, disposal, per-era retry, every handshake version |
| `real_server.py` | Harness for the above: `serve()`, `build_sdk_app()`, `PinnedVersionServer`, `RecordingGateway`, `patched_sdk()` |
| `test_mcp_client.py` | Pathological legacy servers and concurrency races, over `httpx2.MockTransport` |
| `test_protocol_strategy.py` | Pure per-era policy (`is_recoverable`, `reset`, `build_protocol_strategy`) plus the two server behaviours a cooperative server cannot produce |
| `test_protocol_version_support.py` | Tripwires on the SDK facts the strategies depend on |
| `test_mcp_tool.py` | Tool factory, schema refresh, result serialization, and error mapping |
| `test_session_tools.py` | `load_mcp_tools` discovery, invocation, and error mapping |
| `test_session_info.py` | SessionInfo and SessionInfoFactory contract |

**Where a behaviour belongs.** Anything a cooperative server can produce is
tested over real HTTP in `test_mcp_client_real_http.py`. `MockTransport` is kept
only for what a real server cannot express: concurrency races (blocking one
`initialize` mid-flight), pathological servers (minting a new session on every
handshake, echoing `mcp-session-id` back in the modern era, repeating the header
on every response), and pure-function matrices.

### Key Test Classes

| Class | Tests |
|-------|-------|
| `RecordingGateway` (`real_server.py`) | Pure ASGI middleware recording the JSON-RPC method, `mcp-session-id`, `mcp-protocol-version`, `params._meta` and HTTP method of every request, and optionally injecting a `Session terminated` fault on the Nth `tools/call` |
| Module-level client tests | Pathological session handling, 404 retry, concurrency, client reuse |
| `TestMcpToolMetadata` | Tool metadata (tool_type, display_name, etc.) |
| `TestMcpToolCreation` | Multiple tools, descriptions, disabled config |
| `TestCreateMcpToolsFromAgent` | Agent factory function tests |
| `TestMcpToolNameSanitization` | Tool name sanitization |

### Key Assertion

The regression guard lives in `test_legacy_resume_keeps_the_originally_negotiated_version`:
a resumed session keeping its ID is only half the contract — every request after
the resume must also carry the version that session was negotiated at. Probing
candidate versions instead settles on the *oldest* handshake version and
silently downgrades every later request, which no session-ID assertion catches.

```python
versions = {r.protocol_version for r in after_resume if r.protocol_version}
assert versions == {"2025-11-25"}
```

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

## Serializing content blocks: always `by_alias=True`

Any code turning an MCP model into a dict for the model or the wire **must** use:

```python
block.model_dump(by_alias=True, mode="json", exclude_none=True)
```

SDK 2.0 renamed the model attributes to snake case and kept the camelCase names
as *serialization aliases*. A plain `model_dump()` therefore silently rewrites
the shape: `mimeType` → `mime_type`, `_meta` → `meta`, for every image, audio,
resource-link and embedded-resource block.

**Text blocks are byte-identical either way**, which is what makes this hard to
catch — a text-only assertion passes against both. This bug shipped once
(`_normalize_tool_result` in `mcp_tool.py`) and was invisible to the whole suite
until a test used a real `ImageContent`.

Rules when touching serialization:

1. Use the call above. Both `session_tools._content_blocks` and
   `mcp_tool._dump_block` are correct references.
2. Never assert `model_dump` call arguments on a `MagicMock` — that locks in
   whichever call was written. Construct a real `mcp.types` model and assert the
   resulting keys.
3. Include at least one non-text block in any serialization test.

## Guidelines for Changes

### Updating streamable_http.py

When the upstream MCP SDK changes its transport:

1. Keep delegating to [`mcp.client.streamable_http`](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/streamable_http.py); do not copy the transport again
2. Preserve the async `SessionInfo` request/response hooks and remove both hooks on context exit
3. Confirm the SDK still accepts a supplied `httpx2.AsyncClient` and yields two transport streams
4. Re-run `test_mcp_client_real_http.py` first — it drives a real server, so an
   SDK transport change surfaces there before it surfaces in a mock
5. Then re-run the legacy-version, persisted-session, 404 recovery, and DELETE tests

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

1. Negotiation logic goes in a `ProtocolStrategy`, not in `_initialize_session()`,
   which only delegates
2. A strategy's `connect()` runs only on a newly created `ClientSession`; never
   call it again on the same SDK 2 session for recovery
3. Don't create HTTP resources there; `_open_connection()` owns the
   transport/session stack
4. The response hook handles `set_session_id` — strategies only read via
   `get_session_id`, except when minting an affinity ID or clearing a stale one
5. Verify recovery creates two sessions while retaining one HTTP client

### Adding a Protocol Era

1. Implement the three `ProtocolStrategy` methods plus a `SessionIdentity`
2. Add the mode to `ProtocolMode` and `build_protocol_strategy`
3. Decide what `is_recoverable` means for it — a stateless era should not retry
   session-shaped errors
4. Add real-HTTP `McpClient` tests for the era in
   `tests/agent/tools/test_mcp/test_mcp_client_real_http.py`, driving a real
   server through `real_server.py` — negotiation, resume, disposal, and retry
5. Add a leg to `testcases/simple-http-mcp` driving a real server on that era
6. Never change the default mode without a major version: it changes the wire
   for every existing caller

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
| `streamable_http.py` | uipath-langchain | SessionInfo, SessionIdentity, thin SDK transport adapter |
| `protocol_strategy.py` | uipath-langchain | Per-era negotiation, recovery, and identity policy |
| `session_tools.py` | uipath-langchain | `load_mcp_tools` session-to-LangChain conversion |
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
`2025-06-18`, and `2025-11-25`, and reaches `2026-07-28` through
`ClientSession.discover()`. **Both eras are reachable from this low-level path** —
the high-level `mcp.Client` is not required. `AutoStrategy` owns its era
negotiation in `probe_modern_era`, built on the public `ClientSession.send_discover`
/ `adopt` seam — the same two calls the SDK's private `mode="auto"` helper is
made of. That helper (`mcp.client._probe.negotiate_auto`) is deliberately **not**
imported: private surface can move in a patch release.
`test_protocol_version_support.py` pins the two public seams instead.

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
