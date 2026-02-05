# MCP Session Tests Guide

> **CLAUDE: UPDATE THIS DOCUMENT**
>
> When you modify `test_mcp_client.py` or `test_mcp_tool.py`, you MUST update this document to reflect:
> - New test cases (add to Test File Structure and create explanation section)
> - Changes to MockStreamResponse (update Handled MCP Methods table and examples)
> - New mocking patterns (add to Common Patterns section)
> - New assertion patterns (add to Guidelines for Adding New Tests)
> - Changes to test tracking variables (update Tracking Test State section)
>
> Keep code examples and test explanations in sync with actual test implementations.

## Overview

This document explains the testing strategy for MCP-related code. Use this as a reference when adding or modifying MCP-related tests.

## Testing Philosophy

The tests mock **only the HTTP layer** (`httpx.AsyncClient`), allowing the real MCP SDK to process messages. This approach:

- Tests the actual MCP protocol flow
- Validates error handling with real `McpError` exceptions
- Ensures `ClientSession.initialize()` behaves correctly when called multiple times
- Catches integration issues between our code and the SDK

## Test File Structure

```
tests/agent/tools/test_mcp/
├── test_mcp_client.py         # McpClient session tests (7 tests)
│   └── TestMcpClient (class)
│       ├── create_mock_stream_response()
│       ├── create_mock_http_client()
│       ├── test_session_initializes_on_first_call
│       ├── test_session_reused_across_calls
│       ├── test_session_reinitializes_on_404_error  ← Key test
│       ├── test_max_retries_exceeded
│       ├── test_dispose_releases_resources
│       ├── test_client_initialized_property
│       └── test_session_can_be_reused_after_dispose
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
    ├── TestMcpToolInvocation (class)
    │   └── test_tool_invocation_initializes_session_and_returns_result
    │
    └── TestMcpToolNameSanitization (class)
        ├── test_tool_name_with_spaces
        └── test_tool_name_with_special_chars
```

## Mocking Strategy

### What We Mock

Only `httpx.AsyncClient` is mocked at the module level:

```python
@patch("httpx.AsyncClient")
async def test_something(self, mock_async_client_class):
    # mock_async_client_class is the patched class
    # We configure it to return our mock client
    mock_http_client = self.create_mock_http_client(MockStreamResponse)
    mock_async_client_class.return_value = mock_http_client
```

### What We DON'T Mock

- `mcp.ClientSession` - Real SDK session handling
- `mcp.client.streamable_http.streamable_http_client` - Real transport setup
- `mcp.shared.exceptions.McpError` - Real error types

### Why This Approach?

```
┌─────────────────────────────────────────────────────────────┐
│                    Test Boundary                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │ McpClient   │ ──► │ MCP SDK     │ ──► │ HTTP Mock   │   │
│  │             │     │ (real)      │     │ (mocked)    │   │
│  └─────────────┘     └─────────────┘     └─────────────┘   │
│        ▲                   │                   │            │
│        │                   │                   │            │
│        └───────────────────┴───────────────────┘            │
│              Real protocol flow, fake HTTP                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## MockStreamResponse Class

The core of our mocking - simulates an MCP server's HTTP responses.

### Structure

```python
class MockStreamResponse:
    def __init__(self, method: str, url: str, **kwargs):
        # method: "GET" or "POST"
        # url: The endpoint URL
        # kwargs: Contains json (request body), headers, etc.

    def _build_response(self) -> tuple[int, Any, dict[str, str] | None]:
        # Returns: (status_code, json_body, headers)

    async def __aenter__(self): ...   # Context manager entry
    async def __aexit__(self, ...): ...  # Context manager exit
    async def aread(self) -> bytes: ...  # Read response body
    def raise_for_status(self): ...  # Check HTTP status
```

### Handled MCP Methods

| Method | Response | Notes |
|--------|----------|-------|
| `initialize` | 200 + session ID | Returns different IDs for each call |
| `notifications/initialized` | 204 No Content | Notification, no body |
| `tools/list` | 200 + tool definitions | For SDK output validation |
| `tools/call` | 200 + result OR 404 | Configurable via `fail_first_tool_call` |
| GET requests | 405 | Server doesn't support GET streaming |

### Response Format Examples

**Initialize response:**
```python
return (
    200,
    {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "test-server", "version": "1.0.0"},
        },
    },
    {"mcp-session-id": session_id},  # Header with session ID
)
```

**Tool call success:**
```python
return (
    200,
    {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": json.dumps(structured_result)}],
            "structuredContent": structured_result,
            "isError": False,
        },
    },
    {},
)
```

**Tool call 404 (session terminated):**
```python
return (404, None, None)
```

## Tracking Test State

Tests use mutable lists to track state across mock calls:

```python
method_call_sequence: list[str] = []  # Order of MCP methods called
initialize_count = [0]                 # How many times initialize was called
tool_call_count = [0]                  # How many times tools/call was called
```

Why lists? Because they're mutable and can be modified inside the mock class closure:

```python
def create_mock_stream_response(self, method_call_sequence, initialize_count, ...):
    class MockStreamResponse:
        def _build_response(self):
            if self.method == "initialize":
                initialize_count[0] += 1  # Modifies outer list
            method_call_sequence.append(self.method)  # Tracks call order
```

## Test Cases Explained

### TestMcpClient Tests

#### test_session_initializes_on_first_call

**Purpose:** Verify lazy initialization on first `call_tool()`

**Assertions:**
```python
assert session.session_id is None           # Before call
result = await session.call_tool(...)
assert session.session_id == "test-session-first"  # After call
assert session.is_client_initialized
assert mock_async_client_class.call_count == 1     # HTTP client created
```

#### test_session_reused_across_calls

**Purpose:** Verify session persists across multiple tool calls

**Assertions:**
```python
await session.call_tool(...)  # First call
assert initialize_count[0] == 1

await session.call_tool(...)  # Second call
assert initialize_count[0] == 1  # Still 1! No reinit
assert tool_call_count[0] == 2   # But 2 tool calls
```

#### test_session_reinitializes_on_404_error ⭐

**Purpose:** THE KEY TEST - verify client reuse on session reinit

**Setup:**
```python
MockStreamResponse = self.create_mock_stream_response(
    ...,
    fail_first_tool_call=True,  # First tools/call returns 404
)
```

**Critical Assertions:**
```python
# Session was reinitialized (initialize called twice)
assert initialize_count[0] == 2

# Tool call was retried
assert tool_call_count[0] == 2

# Session ID changed
assert session.session_id == "test-session-retry"

# KEY: HTTP client created only ONCE (not recreated)
assert mock_async_client_class.call_count == 1
```

#### test_max_retries_exceeded

**Purpose:** Verify `McpError` is raised after max retries

**Setup:** Custom mock that ALWAYS returns 404 for tool calls

**Assertions:**
```python
with pytest.raises(McpError):
    await session.call_tool(...)

assert initialize_count[0] == 2  # Tried to reinit
assert tool_call_count[0] == 2   # Tried twice
assert mock_async_client_class.call_count == 1  # Still only one client
```

#### test_dispose_releases_resources

**Purpose:** Verify `dispose()` cleans up properly

**Assertions:**
```python
await session.dispose()
assert session.session_id is None
assert session._session is None
assert session._stack is None
assert not session.is_client_initialized
```

#### test_client_initialized_property

**Purpose:** Verify `is_client_initialized` property accuracy

**Assertions:**
```python
assert not session.is_client_initialized  # Before
await session.call_tool(...)
assert session.is_client_initialized      # After call
await session.dispose()
assert not session.is_client_initialized  # After dispose
```

#### test_session_can_be_reused_after_dispose

**Purpose:** Verify session can be fully reinitialized after `dispose()`

**Assertions:**
```python
await session.call_tool(...)
await session.dispose()
await session.call_tool(...)  # Should work!

# HTTP client created TWICE (once before dispose, once after)
assert mock_async_client_class.call_count == 2
```

### TestCreateMcpToolsFromAgent Tests

Note: All tests use `patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")` to mock the SDK.

#### test_creates_tools_from_multiple_mcp_servers

**Purpose:** Verify tools are created from all MCP servers in agent

**Assertions:**
```python
with patch(..., return_value=mock_uipath_class):
    tools, clients = await create_mcp_tools_from_agent(agent)
assert len(tools) == 3  # 2 from server 1 + 1 from server 2
```

#### test_returns_mcp_clients_for_each_server

**Purpose:** Verify McpClient instances are returned for each server

**Assertions:**
```python
with patch(..., return_value=mock_uipath_class):
    tools, clients = await create_mcp_tools_from_agent(agent)
assert len(clients) == 2  # One per MCP server
```

#### test_skips_disabled_mcp_resources

**Purpose:** Verify disabled resources are not processed

**Assertions:**
```python
with patch(..., return_value=mock_uipath_class):
    tools, clients = await create_mcp_tools_from_agent(agent)
assert len(tools) == 1  # Only enabled server's tool
assert tools[0].name == "enabled_tool"
```

#### test_returns_empty_for_agent_without_mcp

**Purpose:** Verify empty lists for agent without MCP resources

**Assertions:**
```python
with patch(..., return_value=mock_uipath_class):
    tools, clients = await create_mcp_tools_from_agent(agent)
assert tools == []
assert clients == []
```

#### test_raises_on_missing_mcp_url

**Purpose:** Verify ValueError when MCP server has no URL

**Assertions:**
```python
with patch(..., return_value=mock_sdk_no_url):
    with pytest.raises(ValueError, match="has no URL configured"):
        await create_mcp_tools_from_agent(agent)
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

### 1. Use the Factory Methods

Always use the provided factory methods:

```python
MockStreamResponse = self.create_mock_stream_response(
    method_call_sequence,
    initialize_count,
    tool_call_count,
    fail_first_tool_call=False,  # Configure behavior
)
mock_http_client = self.create_mock_http_client(MockStreamResponse)
mock_async_client_class.return_value = mock_http_client
```

### 2. Add New MCP Methods to MockStreamResponse

If testing a new MCP method, add it to `_build_response()`:

```python
elif self.method == "resources/list":
    return (
        200,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"resources": [...]},
        },
        {},
    )
```

### 3. Always Verify Client Reuse

For any retry-related test, assert HTTP client count:

```python
# After retry logic
assert mock_async_client_class.call_count == 1, (
    "HTTP client should be created only once"
)
```

### 4. Track Method Sequences

For protocol flow tests, verify the sequence:

```python
assert method_call_sequence == [
    "initialize",
    "notifications/initialized",
    "tools/call",
    # ... expected sequence
]
```

### 5. Test Error Scenarios

When adding error tests:

```python
# Create custom mock for specific error
class CustomErrorMock:
    def _build_response(self):
        if self.method == "tools/call":
            return (
                200,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": "Custom error"},
                },
                {},
            )
```

### 6. Clean Up After Tests

Always dispose the session:

```python
try:
    # ... test logic ...
finally:
    await session.dispose()

# Or simply:
await session.dispose()  # At end of test
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

The mock returns different session IDs based on initialize count:

```python
session_id = (
    session_guid_1 if initialize_count[0] == 1 else session_guid_2
)
```

Use this to verify session ID changes:

```python
assert session.session_id == "test-session-first"   # After first init
# ... trigger reinit ...
assert session.session_id == "test-session-retry"   # After reinit
```

### Testing Structured Content

The SDK validates `structuredContent` against `outputSchema`. Ensure mock returns matching data:

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

assert tool_call_count[0] == 3
assert initialize_count[0] == 1  # Session reused
```

### Testing create_mcp_tools_from_agent

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
        tools, clients = await create_mcp_tools_from_agent(agent_fixture)
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
logger.info(f"Method sequence: {method_call_sequence}")
# Output: ['initialize', 'notifications/initialized', 'tools/call', ...]
```

### Verify Mock Response

Add debug logging in mock:

```python
def _build_response(self):
    logger.debug(f"Building response for {self.method}, id={request_id}")
    # ...
```

## Related Files

| File | Purpose |
|------|---------|
| `test_mcp_client.py` | McpClient session tests (7 tests) |
| `test_mcp_tool.py` | Tool factory tests (17 tests) |
| `src/.../mcp/mcp_client.py` | McpClient implementation |
| `src/.../mcp/mcp_tool.py` | Tool factory implementation |
| `src/.../mcp/claude.md` | Implementation documentation |
