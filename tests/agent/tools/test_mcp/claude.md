# MCP Session Tests Guide

> **CLAUDE: UPDATE THIS DOCUMENT**
>
> When you modify `test_mcp_client.py` or `test_mcp_tool.py`, you MUST update this document to reflect:
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

The client tests run the real MCP SDK 2.0 `ClientSession` and Streamable HTTP
transport over `httpx2.MockTransport`. Only the remote endpoint is simulated.
This approach:

- Tests the actual MCP protocol flow
- Validates error handling with real `MCPError` exceptions
- Verifies that recovery replaces an idempotently initialized `ClientSession`
- Exercises persisted-session request/response hooks
- Catches integration issues between our code and the SDK

## Test File Structure

```
tests/agent/tools/test_mcp/
├── test_mcp_client.py         # Real SDK 2 transport/session integration
│   ├── LegacyMcpEndpoint      # httpx2.MockTransport request handler
│   ├── test_negotiates_supported_legacy_protocol_versions
│   ├── test_replaces_transport_and_session_after_404  ← Key test
│   ├── test_persisted_session_is_reused_without_initialize
│   ├── test_expired_persisted_session_falls_back_to_fresh_initialize
│   ├── test_max_retries_exceeded_raises_mcp_error
│   ├── test_concurrent_recovery_does_not_replace_a_new_session
│   ├── test_list_tools_cache_and_force_refresh
│   ├── test_dispose_allows_client_reuse
│   ├── test_raises_on_missing_mcp_url
│   └── test_only_session_specific_invalid_request_is_retryable
│
├── test_session_info.py       # SessionInfo + SessionInfoFactory contract
│
├── test_protocol_version_support.py  # SDK protocol-version constraints (tripwires)
│   ├── test_low_level_session_cannot_choose_a_protocol_version
│   ├── test_sdk_client_rejects_a_handshake_era_mode_pin
│   ├── test_modern_versions_are_outside_the_handshake_set
│   └── test_restored_sessions_cover_every_handshake_version_but_the_oldest
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
    │   ├── test_returns_empty_for_agent_without_mcp
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
itself lives in `McpClient.list_tools` and is covered in `test_mcp_client.py`
(`test_list_tools_cache_and_force_refresh`; disposal/reuse is covered separately).

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
    ) -> None:
        self.protocol_version = protocol_version
        self.failed_tool_calls = failed_tool_calls
        self.methods: list[str] = []
        self.request_headers: list[tuple[str, httpx2.Headers]] = []
        self.initialize_count = 0
        self.tool_call_count = 0
        self.delete_count = 0
        self.transport = httpx2.MockTransport(self.handle)

    async def handle(self, request: httpx2.Request) -> httpx2.Response: ...
```

### Handled MCP Methods

| Method | Response | Notes |
|--------|----------|-------|
| `initialize` | 200 + session ID | Returns selected legacy version and a new ID |
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

#### test_negotiates_supported_legacy_protocol_versions

Parameterizes `2025-03-26`, `2025-06-18`, and `2025-11-25`. It verifies the
real SDK accepts each server-selected handshake version and stamps the selected
version plus session ID on the subsequent tool request.

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

`test_persisted_session_is_reused_without_initialize` verifies an ID restored
through a custom `SessionInfoFactory` is injected into `tools/call` without a
new handshake.

`test_expired_persisted_session_falls_back_to_fresh_initialize` verifies the
special bare-404 path: the SDK transport does not know an externally injected
ID, so UiPath recognizes `METHOD_NOT_FOUND`/`"Not Found"` while the ID is still
present, clears it, initializes a new session, and retries.

#### Retry and concurrency tests

- `test_max_retries_exceeded_raises_mcp_error` expects the real `MCPError`
  after the configured retry is consumed.
- `test_concurrent_recovery_does_not_replace_a_new_session` verifies a late
  failure from an old `ClientSession` cannot tear down a replacement created by
  another operation.
- `test_only_session_specific_invalid_request_is_retryable` verifies an ordinary
  `INVALID_REQUEST` is not misclassified as a disconnect.

#### Cache, disposal, and configuration tests

- `test_list_tools_cache_and_force_refresh` verifies normal caching and explicit
  refresh over the real protocol path.
- `test_dispose_allows_client_reuse` verifies disposal resets the state and a
  later call creates another HTTP client/session stack.
- `test_raises_on_missing_mcp_url` verifies endpoint validation happens before
  HTTP resources are allocated.

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

#### test_returns_empty_for_agent_without_mcp

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

### 1. Use the Shared Endpoint and Client Context

Use `LegacyMcpEndpoint` and `configured_client` so tests keep the real SDK
transport/session path:

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
| `test_mcp_client.py` | SDK 2 transport, legacy versions, session persistence/recovery, caching, disposal |
| `test_session_info.py` | Async session ID store and factory |
| `test_protocol_version_support.py` | Guards the SDK version constraints that dictate what `McpClient` can negotiate; each failure names the follow-up it unblocks |
| `test_mcp_tool.py` | Tool factories, schemas, result/error mapping, metadata |
| `src/.../mcp/mcp_client.py` | McpClient implementation |
| `src/.../mcp/mcp_tool.py` | Tool factory implementation |
| `src/.../mcp/claude.md` | Implementation documentation |
