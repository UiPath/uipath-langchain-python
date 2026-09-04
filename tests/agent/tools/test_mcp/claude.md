# MCP Session Tests Guide

> **CLAUDE: UPDATE THIS DOCUMENT**
>
> When you modify `real_server.py`, `test_mcp_client_real_http.py`,
> `test_mcp_client.py`, `test_protocol_strategy.py`, or `test_mcp_tool.py`, you
> MUST update this document to reflect:
> - New test cases (add to Test File Structure and create explanation section)
> - Changes to LegacyMcpEndpoint (update Handled MCP Methods table and examples)
> - New mocking patterns (add to Common Patterns section)
> - New assertion patterns (add to Guidelines for Adding New Tests)
> - Changes to test tracking variables (update Tracking Test State section)
>
> Keep code examples and test explanations in sync with actual test implementations.

## Overview

This document explains the testing strategy for MCP-related code. Use this as a reference when adding or modifying MCP-related tests.

## Testing Philosophy

There are **two tiers**, and which tier a behaviour belongs in is not a matter of
taste.

### Tier 1 - real HTTP (default)

`test_mcp_client_real_http.py` drives the public `McpClient` API against a
genuine `mcp.server.mcpserver.MCPServer` hosted in-process on an ephemeral port,
through a recording gateway. Nothing is simulated except the UiPath SDK lookup
that resolves the server URL. This tier:

- Runs a real ASGI server, a real socket, and the real SDK transport on both ends
- Exercises `McpClient` itself, not just the strategies underneath it
- Observes the wire (`mcp-session-id`, `mcp-protocol-version`, `params._meta`,
  HTTP method) exactly as a gateway would, so assertions survive refactors
- Catches server behaviour a hand-written mock silently gets wrong - session
  routing, `DELETE` on teardown, the optional GET SSE channel

**Anything a cooperative server can produce belongs here.**

### Tier 2 - `httpx2.MockTransport` (exceptions only)

`test_mcp_client.py` and `test_protocol_strategy.py` keep mocks only for
conditions a cooperative real server cannot express:

| Condition | Why a real server cannot do it |
|-----------|--------------------------------|
| Concurrency races | Requires blocking one `initialize` mid-flight and releasing it on cue |
| `mints_new_session_on_initialize` | A conforming server routes by the session header instead |
| `echo_session_id` in the modern era | A modern server has no session to echo |
| `repeat_session_header` | A conforming server sends the header once |
| `fail_initialize_on` | A deterministic handshake failure on the *second* attempt only |
| A bare, body-less HTTP 404 | The SDK server always returns a JSON-RPC body |
| Pure-function matrices | `is_recoverable`, `reset`, `build_protocol_strategy` need no server at all |

If you are adding a test and it does not fall into that table, write it in
`test_mcp_client_real_http.py`.

## Test File Structure

```
tests/agent/tools/test_mcp/
├── real_server.py             # Real-HTTP harness (no tests of its own)
│   ├── serve(app)             # ephemeral port + in-process uvicorn
│   ├── build_sdk_app()        # real MCPServer with add/multiply
│   ├── PinnedVersionServer    # one fixed protocolVersion, or modern-only
│   ├── RecordingGateway       # pure ASGI recorder + fault injector
│   ├── patched_sdk(url)       # redirects McpClient's lazy UiPath lookup
│   └── connected_client / make_client / pinned_session_factory
│
├── test_mcp_client_real_http.py   # McpClient over real HTTP  ← default tier
│   ├── negotiation per mode
│   │   ├── test_legacy_mode_negotiates_the_newest_handshake_version
│   │   ├── test_modern_mode_negotiates_without_any_server_session
│   │   ├── test_auto_mode_resolves_to_modern_against_a_real_server
│   │   ├── test_auto_mode_falls_back_to_legacy_against_a_handshake_only_server
│   │   └── test_modern_mode_works_against_a_server_that_refuses_the_handshake
│   ├── resume across clients
│   │   ├── test_legacy_resume_keeps_the_originally_negotiated_version  ← Key test
│   │   └── test_unknown_persisted_session_falls_back_to_a_fresh_session
│   ├── affinity and disposal
│   │   ├── test_modern_affinity_pins_one_instance_across_clients
│   │   ├── test_auto_mode_pins_the_first_request
│   │   ├── test_modern_disposal_does_not_delete_a_restored_affinity_id
│   │   └── test_legacy_disposal_deletes_a_restored_session
│   ├── retry semantics per era
│   │   ├── test_legacy_recovers_from_an_injected_session_termination
│   │   └── test_modern_does_not_retry_an_injected_session_termination
│   ├── version breadth
│   │   └── test_legacy_negotiates_every_supported_handshake_version[4 versions]
│   └── lifecycle
│       ├── test_list_tools_is_cached_until_force_refresh
│       └── test_dispose_then_reuse_reinitializes_the_client
│
├── test_mcp_client.py         # MockTransport: pathological servers + races
│   ├── LegacyMcpEndpoint      # httpx2.MockTransport request handler
│   ├── test_legacy_httpx_timeout_is_normalized_for_final_client
│   ├── test_replaces_transport_and_session_after_404  (bare, body-less 404)
│   ├── test_replaces_session_after_official_session_not_found_error
│   ├── test_dropped_connection_resumes_the_persisted_session  (CONNECTION_CLOSED keeps the ID)
│   ├── test_auto_mode_does_not_offer_a_minted_id_to_a_legacy_handshake
│   ├── test_persisted_session_replaced_when_server_ignores_the_header
│   ├── test_rejected_persisted_session_is_initialized_and_deleted_once
│   ├── test_repeated_session_headers_do_not_repeat_external_persistence
│   ├── test_max_retries_exceeded_raises_mcp_error
│   ├── test_concurrent_recovery_does_not_replace_a_new_session
│   ├── test_recovery_continues_when_failed_connection_cleanup_raises
│   ├── test_concurrent_call_waits_for_recovery_initialization
│   ├── test_later_call_recovers_after_replacement_initialization_failure
│   ├── test_raises_on_missing_mcp_url
│   ├── test_initialization_failure_cleans_state_and_allows_retry
│   └── test_only_session_specific_invalid_request_is_retryable
│
├── test_protocol_strategy.py  # Pure per-era policy + two odd server behaviours
│   ├── EraMcpEndpoint         # Serves either era, or both
│   ├── test_auto_mode_sends_a_restored_id_before_the_era_resolves
│   ├── test_modern_mode_ignores_a_server_assigned_session_id
│   ├── test_legacy_recovers_from_session_loss_but_not_from_bad_requests
│   ├── test_modern_recovers_only_from_a_dropped_connection
│   ├── test_modern_reset_keeps_the_affinity_id
│   ├── test_legacy_reset_clears_the_stale_session_id
│   ├── test_auto_applies_the_legacy_policy_before_an_era_is_resolved
│   ├── test_legacy_keeps_a_persisted_session_when_the_connection_drops
│   ├── test_auto_does_not_carry_a_stale_era_through_a_failed_probe
│   ├── test_build_protocol_strategy_maps_every_mode
│   └── test_legacy_is_the_default_mode
│
├── test_session_info.py       # SessionInfo + SessionInfoFactory contract
│
├── test_session_tools.py      # load_mcp_tools discovery/invocation/errors
│
├── test_protocol_version_support.py  # SDK facts the strategies depend on (tripwires)
│   ├── test_the_auto_probe_builds_on_public_session_methods
│   ├── test_the_low_level_session_reaches_the_modern_era
│   ├── test_initialize_cannot_choose_a_protocol_version
│   └── test_the_two_eras_share_no_protocol_version
│
└── test_mcp_tool.py           # Tool factory tests (17 tests)
    ├── TestMcpToolMetadata (class)
    │   ├── test_mcp_tool_has_metadata
    │   ├── test_mcp_tool_metadata_has_tool_type
    │   ├── test_mcp_tool_metadata_has_display_name
    │   ├── test_mcp_tool_metadata_has_folder_path
    │   └── test_mcp_tool_metadata_has_slug
    │
    ├── TestMcpToolCreation (class)
    │   ├── test_creates_multiple_tools
    │   ├── test_tool_has_correct_description
    │   └── test_disabled_config_returns_empty_list
    │
    ├── TestCreateMcpToolsFromAgent (class)  ← New!
    │   ├── test_creates_tools_from_multiple_mcp_servers
    │   ├── test_returns_mcp_clients_for_each_server
    │   ├── test_skips_disabled_mcp_resources
    │   ├── test_returns_empty_for_empty_resources
    │   ├── test_raises_on_missing_mcp_url
    │   └── test_tools_have_correct_metadata
    │
    ├── TestMcpToolResultSerialization (class)
    ├── TestMcpToolErrorHandling (class)
    │
    ├── TestMcpToolNameSanitization (class)
    │   ├── test_tool_name_with_spaces
    │   └── test_tool_name_with_special_chars
    │
    └── TestCachedRefreshSchemaBeforeCall (class)  ← refresh_schema_before_call + self-heal
        ├── test_refresh_enabled_lists_tools_before_call
        ├── test_refresh_disabled_skips_list_tools
        ├── test_refresh_falls_back_when_list_tools_fails
        ├── test_refresh_returns_removed_message_when_tool_missing
        ├── test_cached_default_enables_refresh
        ├── test_cached_refresh_disabled_via_config
        ├── test_breaking_drift_heals_and_asks_retry
        ├── test_after_heal_next_call_executes
        ├── test_nonbreaking_change_executes_without_retry
        └── test_schema_change_message_lists_param_types
```

### TestCachedRefreshSchemaBeforeCall

Covers the cached-mode `refresh_schema_before_call` flag (default `True`) and the
self-healing behaviour. Asserts ordering via `manager.attach_mock` (list_tools before
call_tool), graceful fallback when `list_tools` raises, a clear "tool removed" message
when the tool is gone from the server, that `create_mcp_tools` does not list tools at
creation time, and that
`refresh_schema_before_call=False` disables the refresh.

Self-heal cases: on a **breaking** schema change (cached `query` vs live `question`)
the tool is not executed (`call_tool` not awaited), `tool_fn` returns a retry message
listing each refreshed param with its type, and the wrapper's `args_schema` is healed
to the live schema; a subsequent call against
the healed schema executes; an **additive/non-breaking** change executes normally
without a retry. These tests invoke the tool's `coroutine` directly (not `ainvoke`)
because the stale arguments would fail `args_schema` validation before reaching the
tool.

`tool_fn` tests mock `mcpClient.list_tools` directly, so they exercise the refresh
logic per invocation independent of the client's caching. The once-per-run caching
itself lives in `McpClient.list_tools` and is covered over real HTTP in
`test_mcp_client_real_http.py` (`test_list_tools_is_cached_until_force_refresh`;
disposal/reuse is covered by `test_dispose_then_reuse_reinitializes_the_client`).

## The Real-HTTP Harness (`real_server.py`)

### serve(app)

Async context manager. Binds `("127.0.0.1", 0)` for an ephemeral port - parallel
CI jobs must never collide on a fixed one - starts uvicorn **in-process** (so a
failing test leaves no orphan), and yields the `/mcp` URL.

```python
gateway = RecordingGateway(build_sdk_app())
async with serve(gateway) as url:
    async with connected_client(url, protocol_mode="modern") as client:
        await client.call_tool("add", {"a": 2, "b": 3})
```

`serve()` also calls `_reset_sse_shutdown_latch()`. `sse-starlette` polls
`uvicorn.Server.should_exit` and latches a **module-global**
`AppStatus.should_exit` when any server stops; without the reset, every server
after the first would see the latch already set and kill its SSE stream the
instant it opened, logging `ASGI callable returned without completing response`
and sending the client into a reconnect loop. It is process-global state, not a
per-server flag - do not remove the reset.

### RecordingGateway - pure ASGI, never BaseHTTPMiddleware

Records per request: JSON-RPC `method`, the `mcp-session-id` and
`mcp-protocol-version` **request** headers, `params._meta`, the HTTP method (so
`DELETE` is observable), the `mcp-session-id` on the **response** (i.e. a
server-assigned session), and the instance a gateway routing on `mcp-session-id`
would have picked.

```python
gateway.count("tools/call")          # JSON-RPC method counts
gateway.http_count("DELETE")         # session teardown
gateway.for_rpc("tools/call")[0]     # one RecordedRequest
gateway.server_session_ids()         # [] in the modern era
gateway.unpinned()                   # requests with no affinity header
```

Fault injection replaces the Nth `tools/call` with a JSON-RPC `-32600
"Session terminated"` at HTTP 404, which is how the recovery paths are driven
without `MockTransport`:

```python
gateway = RecordingGateway(build_sdk_app(), fault_on_tool_call=1)
```

**Use a pure ASGI middleware (`async def __call__(self, scope, receive, send)`),
never `starlette.middleware.base.BaseHTTPMiddleware`.** The latter buffers
through an inner task and breaks streaming/SSE responses here with `ASGI
callable returned without completing response`. A pure ASGI middleware composes
cleanly and can still read request bodies by wrapping `receive` (see
`_buffer_body`, which replays the buffered body downstream).

### patched_sdk(url)

`McpClient._initialize_client` does `from uipath.platform import UiPath` at call
time, so replacing that module attribute is enough to point the client at a local
server. No tenant, credentials, or network access to UiPath Cloud is involved,
and the client still walks its real resolution path.

### Assert on the wire, not on internals

There is no public accessor for the negotiated version, so read it from the
`mcp-protocol-version` header the SDK stamps on every post-negotiation request:

```python
def negotiated_version(gateway, rpc_method):
    records = gateway.for_rpc(rpc_method)
    return records[0].protocol_version if records else None
```

## Mocking Strategy

### What We Mock

`LegacyMcpEndpoint` is an async handler installed on a real
`httpx2.AsyncClient` through `httpx2.MockTransport`:

```python
endpoint = LegacyMcpEndpoint(protocol_version="2025-06-18")
http_kwargs = {
    "headers": {"Authorization": "Bearer test-secret-token"},
    "transport": endpoint.transport,
    "follow_redirects": True,
}
with patch(
    "uipath_langchain.agent.tools.mcp.mcp_client.get_httpx_client_kwargs",
    return_value=http_kwargs,
):
    result = await client.call_tool("test_tool", {"query": "test"})
```

### What We DON'T Mock

- `mcp.ClientSession` - Real SDK session handling
- `mcp.client.streamable_http.streamable_http_client` - Real transport setup
- `mcp.shared.exceptions.MCPError` - Real error types
- UiPath's `streamable_http_client` event hooks - Real session persistence adapter

### Why This Approach?

```
┌─────────────────────────────────────────────────────────────┐
│                    Test Boundary                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │ McpClient   │ ──► │ MCP SDK 2   │ ──► │ MockTransport│  │
│  │             │     │ (real)      │     │ (mocked)    │   │
│  └─────────────┘     └─────────────┘     └─────────────┘   │
│        ▲                   │                   │            │
│        │                   │                   │            │
│        └───────────────────┴───────────────────┘            │
│              Real protocol flow, fake HTTP                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## LegacyMcpEndpoint Class

The core test endpoint simulates an MCP legacy Streamable HTTP server while
recording methods, headers, initialization count, tool calls, and DELETEs.

### Structure

```python
class LegacyMcpEndpoint:
    def __init__(
        self,
        protocol_version: str = "2025-11-25",
        *,
        failed_tool_calls: int = 0,
        known_session_ids: set[str] | None = None,
        mints_new_session_on_initialize: bool = False,
    ) -> None:
        self.protocol_version = protocol_version
        self.failed_tool_calls = failed_tool_calls
        self.known_session_ids = set(known_session_ids or ())
        self.mints_new_session_on_initialize = mints_new_session_on_initialize
        self.methods: list[str] = []
        self.request_headers: list[tuple[str, httpx2.Headers]] = []
        self.initialize_count = 0     # initialize requests handled
        self.session_mint_count = 0   # sessions actually created
        self.tool_call_count = 0
        self.delete_count = 0
        self.transport = httpx2.MockTransport(self.handle)

    async def handle(self, request: httpx2.Request) -> httpx2.Response: ...
```

**Session routing.** `_session_for_initialize` mirrors the SDK server: an
`initialize` naming a live session is handled *inside* it, and a new session is
minted only when no session header is present. An unknown or expired ID is
rejected with `"Session not found"` rather than silently replaced.

This fidelity matters — the legacy resume path re-runs `initialize` inside a
restored session, so an endpoint that minted a fresh ID on every handshake could
not express the behaviour under test.

- Seed `known_session_ids={"persisted-session"}` to stand in for a session a
  previous process established and persisted externally.
- Set `mints_new_session_on_initialize=True` to model a server that ignores the
  header instead.
- `initialize_count` counts requests; `session_mint_count` counts sessions
  created. A rejected handshake increments only the former.

### Handled MCP Methods

| Method | Response | Notes |
|--------|----------|-------|
| `initialize` | 200 + session ID, or 404 | Routes by `mcp-session-id`; mints only when absent, rejects unknown/expired IDs |
| `notifications/initialized` | 202 Accepted | Notification, no body |
| `tools/list` | 200 + tool definitions | For SDK output validation |
| `tools/call` | 200 + result OR bare 404 | Configurable via `failed_tool_calls` |
| GET requests | 405 | Server doesn't support GET streaming |
| DELETE requests | 204 | Records session termination |

### Response Format Examples

**Initialize response:**
```python
return httpx2.Response(
    200,
    headers={
        "content-type": "application/json",
        "mcp-session-id": f"session-{self.initialize_count}",
    },
    json={
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "test-server", "version": "1.0.0"},
        },
    },
)
```

**Tool call success:**
```python
return httpx2.Response(
    200,
    headers={"content-type": "application/json"},
    json={
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": json.dumps(structured_result)}],
            "structuredContent": structured_result,
            "isError": False,
        },
    },
)
```

**Tool call 404 (session terminated):**
```python
return httpx2.Response(404)
```

## Tracking Test State

Tests inspect state recorded directly on `LegacyMcpEndpoint`:

```python
assert endpoint.initialize_count == 2
assert endpoint.tool_call_count == 2
assert endpoint.delete_count == 1
assert endpoint.methods.count("tools/list") == 2
assert endpoint.headers_for("tools/call")[0]["mcp-session-id"] == "session-1"
```

## Test Cases Explained

### TestMcpClient Tests

#### test_replaces_transport_and_session_after_404 ⭐

The first tool request returns a bare HTTP 404. The test verifies two
initializations, two tool calls, one DELETE for the old session, a new session
ID, and correct session headers on both attempts. This catches the SDK 2
idempotent-`initialize()` breaking change: recovery must create a fresh
`ClientSession`, not call `initialize()` again on the old one.

```python
assert endpoint.initialize_count == 2
assert endpoint.tool_call_count == 2
assert endpoint.delete_count == 1
assert await client.get_session_id() == "session-2"
```

#### Persisted-session tests

Resuming a persisted session is covered over real HTTP
(`test_legacy_resume_keeps_the_originally_negotiated_version`,
`test_unknown_persisted_session_falls_back_to_a_fresh_session`,
`test_legacy_disposal_deletes_a_restored_session`). What stays here is the
behaviour a conforming server will not produce:

`test_persisted_session_replaced_when_server_ignores_the_header` covers a server
that mints on every handshake: the persisted session is lost, but the connection
stays usable and the client continues with the replacement ID.

`test_rejected_persisted_session_is_initialized_and_deleted_once` covers a server
that explicitly rejects one known-shaped session ID, and asserts only the fresh
SDK session is deleted on close.

#### Retry and concurrency tests

- `test_max_retries_exceeded_raises_mcp_error` expects the real `MCPError`
  after the configured retry is consumed.
- `test_concurrent_recovery_does_not_replace_a_new_session` verifies a late
  failure from an old `ClientSession` cannot tear down a replacement created by
  another operation.
- `test_only_session_specific_invalid_request_is_retryable` verifies an ordinary
  `INVALID_REQUEST` is not misclassified as a disconnect.

#### Cache, disposal, and configuration tests

Caching and disposal-then-reuse moved to real HTTP
(`test_list_tools_is_cached_until_force_refresh`,
`test_dispose_then_reuse_reinitializes_the_client`). What stays here:

- `test_raises_on_missing_mcp_url` verifies endpoint validation happens before
  HTTP resources are allocated.
- `test_initialization_failure_cleans_state_and_allows_retry` patches
  `_initialize_session` to fail, which no server response can cause.
- `test_legacy_httpx_timeout_is_normalized_for_final_client` pins the
  pre-upgrade public timeout type; its subject is `_normalize_timeout`, not the
  server.

### Real-HTTP tests explained

#### test_legacy_resume_keeps_the_originally_negotiated_version ⭐

Two `McpClient` instances share one `SessionInfo`, standing in for two runs of a
playground agent whose session store outlives the process. The first connects
with `terminate_on_close=False` and disposes; the second restores the ID.

The session ID surviving is only half the contract. **Every request after the
resume must also carry `mcp-protocol-version` equal to the version the session
was originally negotiated at.** Probing candidate versions instead — the
pre-existing approach — always matched the *oldest* handshake version, silently
downgrading every later request and disabling the server's `2025-11-25` SSE
resumability. No session-ID assertion catches that; this one does, and its
failure message says so.

#### Retry semantics per era

`test_legacy_recovers_from_an_injected_session_termination` and
`test_modern_does_not_retry_an_injected_session_termination` send the *identical*
injected `-32600 "Session terminated"` at HTTP 404 through both eras. Legacy
re-handshakes and retries (two `initialize`s, two `tools/call`s, different
session IDs); modern surfaces it immediately (one `tools/call`). Driving both
from one fault injector is what makes the contrast meaningful.

#### test_modern_disposal_does_not_delete_a_restored_affinity_id

A restored affinity ID looks exactly like a restored session to the transport.
Deleting it would reach the gateway as a teardown for a *live* instance on every
run after the first — this was a real bug. The paired
`test_legacy_disposal_deletes_a_restored_session` proves the legacy era still
does send the `DELETE`.

### TestCreateMcpToolsFromAgent Tests

Note: All tests use `patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")` to mock the SDK.

#### test_creates_tools_from_multiple_mcp_servers

**Purpose:** Verify tools are created from all MCP servers in agent

**Assertions:**
```python
with patch(..., return_value=mock_uipath_class):
    tools, clients = await create_mcp_tools_and_clients(agent)
assert len(tools) == 3  # 2 from server 1 + 1 from server 2
```

#### test_returns_mcp_clients_for_each_server

**Purpose:** Verify McpClient instances are returned for each server

**Assertions:**
```python
with patch(..., return_value=mock_uipath_class):
    tools, clients = await create_mcp_tools_and_clients(agent)
assert len(clients) == 2  # One per MCP server
```

#### test_skips_disabled_mcp_resources

**Purpose:** Verify disabled resources are not processed

**Assertions:**
```python
with patch(..., return_value=mock_uipath_class):
    tools, clients = await create_mcp_tools_and_clients(agent)
assert len(tools) == 1  # Only enabled server's tool
assert tools[0].name == "enabled_tool"
```

#### test_returns_empty_for_empty_resources

**Purpose:** Verify empty lists for agent without MCP resources

**Assertions:**
```python
with patch(..., return_value=mock_uipath_class):
    tools, clients = await create_mcp_tools_and_clients(agent)
assert tools == []
assert clients == []
```

#### test_raises_on_missing_mcp_url

**Purpose:** Verify ValueError when MCP server has no URL

**Assertions:**
```python
with patch(..., return_value=mock_sdk_no_url):
    with pytest.raises(ValueError, match="has no URL configured"):
        await create_mcp_tools_and_clients(agent)
```

#### test_tools_have_correct_metadata

**Purpose:** Verify all tools have correct metadata

**Assertions:**
```python
for tool in tools:
    assert tool.metadata["tool_type"] == "mcp"
    assert "display_name" in tool.metadata
    assert "folder_path" in tool.metadata
    assert "slug" in tool.metadata
```

## Guidelines for Adding New Tests

### 1. Start in the real-HTTP tier

Unless the behaviour is in the Tier 2 table above, write the test in
`test_mcp_client_real_http.py` against a real server:

```python
gateway = RecordingGateway(build_sdk_app())
async with serve(gateway) as url:
    async with connected_client(url, protocol_mode="legacy") as client:
        await client.call_tool("add", {"a": 2, "b": 3})

assert gateway.count("initialize") == 1
assert negotiated_version(gateway, "tools/call") == LEGACY_VERSION
```

Only if a cooperative server cannot produce the condition, fall back to
`LegacyMcpEndpoint`/`EraMcpEndpoint` with `configured_client`, which keep the
real SDK transport/session path over `httpx2.MockTransport`:

```python
endpoint = LegacyMcpEndpoint(
    protocol_version="2025-11-25",
    failed_tool_calls=0,
)
async with configured_client(config, mock_uipath_sdk, endpoint) as client:
    await client.call_tool("test_tool", {"query": "test"})
```

### 2. Add New MCP Methods to LegacyMcpEndpoint

If testing a new MCP method, add it to `handle()` and return an
`httpx2.Response` with wire-format JSON:

```python
if method == "resources/list":
    return httpx2.Response(
        200,
        headers={"content-type": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {"resources": [...]},
        },
    )
```

### 3. Always Verify Client Reuse

For retry tests, assert the base client is reused while connection/session state
is replaced. The endpoint counters and headers are the observable contract:

```python
assert endpoint.initialize_count == 2
assert endpoint.delete_count == 1
assert [h["mcp-session-id"] for h in endpoint.headers_for("tools/call")] == [
    "session-1",
    "session-2",
]
```

### 4. Track Method Sequences

For protocol flow tests, verify the sequence:

```python
assert endpoint.methods == [
    "initialize",
    "notifications/initialized",
    "tools/call",
    # ... expected sequence
]
```

### 5. Test Error Scenarios

When adding error tests:

```python
# Add a branch to LegacyMcpEndpoint.handle().
if method == "tools/call":
    return httpx2.Response(
        400,
        headers={"content-type": "application/json"},
        json={
            "jsonrpc": "2.0",
            "id": body["id"],
            "error": {"code": -32602, "message": "Invalid parameters"},
        },
    )
```

### 6. Clean Up After Tests

Prefer `configured_client`, which disposes in `finally`:

```python
async with configured_client(config, sdk, endpoint) as client:
    await client.call_tool("test_tool", {})
```

### 7. Use Proper AgentSettings

When creating `LowCodeAgentDefinition` in tests, use a real `AgentSettings`:

```python
from uipath.agent.models.agent import AgentSettings

settings = AgentSettings(
    engine="openai", model="gpt-4", max_tokens=1000, temperature=0.7
)
agent = LowCodeAgentDefinition(
    ...
    settings=settings,  # NOT MagicMock()
    ...
)
```

## Common Patterns

### Testing Different Session IDs

The endpoint returns `session-{initialize_count}`. Verify both the external
store and request headers:

```python
assert await client.get_session_id() == "session-2"
assert endpoint.headers_for("tools/call")[1]["mcp-session-id"] == "session-2"
```

### Testing Structured Content

The SDK validates `structuredContent` against `outputSchema`. Ensure mock returns
matching data. Wire JSON stays camelCase; SDK 2 Python attributes are snake_case
(`tool.input_schema`, `tool.output_schema`).

```python
# In tools/list response
"outputSchema": {
    "type": "object",
    "properties": {"result": {"type": "string"}},
}

# In tools/call response - must match schema!
"structuredContent": {"result": "some string value"}
```

### Testing with Multiple Tool Calls

Track counts to verify behavior:

```python
await session.call_tool("tool1", {...})
await session.call_tool("tool2", {...})
await session.call_tool("tool1", {...})

assert endpoint.tool_call_count == 3
assert endpoint.initialize_count == 1  # Session reused
```

### Testing create_mcp_tools_and_clients

The function uses lazy SDK initialization (`sdk = UiPath()`), so we patch the `UiPath` class:

```python
@pytest.fixture
def mock_uipath_class(self):
    """Create a mock UiPath class for patching."""
    mock_sdk = MagicMock()
    mock_server = MagicMock()
    mock_server.mcp_url = "https://test.uipath.com/mcp"
    mock_sdk.mcp.retrieve_async = AsyncMock(return_value=mock_server)
    mock_sdk._config = MagicMock()
    mock_sdk._config.secret = "test-secret-token"
    return mock_sdk

@pytest.mark.asyncio
async def test_example(self, agent_fixture, mock_uipath_class):
    with patch(
        "uipath_langchain.agent.tools.mcp.mcp_tool.UiPath",
        return_value=mock_uipath_class,
    ):
        tools, clients = await create_mcp_tools_and_clients(agent_fixture)
```

## Debugging Failed Tests

### Enable Logging

Run with logging to see MCP flow:

```bash
uv run pytest tests/agent/tools/test_mcp/ -v -s --log-cli-level=DEBUG
```

### Check Method Sequence

Print the sequence to understand what happened:

```python
logger.info(f"Method sequence: {endpoint.methods}")
# Output: ['initialize', 'notifications/initialized', 'tools/call', ...]
```

### Verify Mock Response

Add debug logging in mock:

```python
async def handle(self, request: httpx2.Request):
    body = json.loads(request.content)
    logger.debug(f"Building response for {body['method']}, id={body.get('id')}")
    # ...
```

## Related Files

| File | Purpose |
|------|---------|
| `real_server.py` | Real-HTTP harness: `serve`, `build_sdk_app`, `PinnedVersionServer`, `RecordingGateway`, `patched_sdk` |
| `test_mcp_client_real_http.py` | `McpClient` over real HTTP: negotiation per mode, resume, affinity, disposal, retry, every handshake version |
| `test_mcp_client.py` | MockTransport: pathological legacy servers, concurrency races, bare-404 mapping |
| `test_session_info.py` | Async session ID store and factory |
| `test_protocol_version_support.py` | Guards the SDK version constraints that dictate what `McpClient` can negotiate; each failure names the follow-up it unblocks |
| `test_mcp_tool.py` | Tool factories, schemas, result/error mapping, metadata |
| `src/.../mcp/mcp_client.py` | McpClient implementation |
| `src/.../mcp/mcp_tool.py` | Tool factory implementation |
| `src/.../mcp/claude.md` | Implementation documentation |
