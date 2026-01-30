# MCP Session Tests Guide

> **CLAUDE: UPDATE THIS DOCUMENT**
>
> When you modify `test__mcp_client.py` or add new MCP-related tests, you MUST update this document to reflect:
> - New test cases (add to Test File Structure and create explanation section)
> - Changes to MockStreamResponse (update Handled MCP Methods table and examples)
> - New mocking patterns (add to Common Patterns section)
> - New assertion patterns (add to Guidelines for Adding New Tests)
> - Changes to test tracking variables (update Tracking Test State section)
>
> Keep code examples and test explanations in sync with actual test implementations.

## Overview

This document explains the testing strategy for `McpClient` in `test__mcp_client.py`. Use this as a reference when adding or modifying MCP-related tests.

## Testing Philosophy

The tests mock **only the HTTP layer** (`httpx.AsyncClient`), allowing the real MCP SDK to process messages. This approach:

- Tests the actual MCP protocol flow
- Validates error handling with real `McpError` exceptions
- Ensures `ClientSession.initialize()` behaves correctly when called multiple times
- Catches integration issues between our code and the SDK

## Test File Structure

```
tests/agent/tools/test_mcp/test__mcp_client.py
├── TestMcpClient (class)
│   ├── create_mock_stream_response()    # Factory for mock responses
│   ├── create_mock_http_client()        # Creates mock httpx client
│   │
│   ├── test_session_initializes_on_first_call
│   ├── test_session_reused_across_calls
│   ├── test_session_reinitializes_on_404_error  ← Key test
│   ├── test_max_retries_exceeded
│   ├── test_close_releases_resources
│   ├── test_client_initialized_property
│   └── test_session_can_be_reused_after_close
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
│  │ McpTool     │ ──► │ MCP SDK     │ ──► │ HTTP Mock   │   │
│  │ Session     │     │ (real)      │     │ (mocked)    │   │
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

### test_session_initializes_on_first_call

**Purpose:** Verify lazy initialization on first `call_tool()`

**Assertions:**
```python
assert session.session_id is None           # Before call
result = await session.call_tool(...)
assert session.session_id == "test-session-first"  # After call
assert session.is_client_initialized
assert mock_async_client_class.call_count == 1     # HTTP client created
```

### test_session_reused_across_calls

**Purpose:** Verify session persists across multiple tool calls

**Assertions:**
```python
await session.call_tool(...)  # First call
assert initialize_count[0] == 1

await session.call_tool(...)  # Second call
assert initialize_count[0] == 1  # Still 1! No reinit
assert tool_call_count[0] == 2   # But 2 tool calls
```

### test_session_reinitializes_on_404_error ⭐

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

### test_max_retries_exceeded

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

### test_close_releases_resources

**Purpose:** Verify `close()` cleans up properly

**Assertions:**
```python
await session.close()
assert session.session_id is None
assert session._session is None
assert session._stack is None
assert not session.is_client_initialized
```

### test_client_initialized_property

**Purpose:** Verify `is_client_initialized` property accuracy

**Assertions:**
```python
assert not session.is_client_initialized  # Before
await session.call_tool(...)
assert session.is_client_initialized      # After call
await session.close()
assert not session.is_client_initialized  # After close
```

### test_session_can_be_reused_after_close

**Purpose:** Verify session can be fully reinitialized after `close()`

**Assertions:**
```python
await session.call_tool(...)
await session.close()
await session.call_tool(...)  # Should work!

# HTTP client created TWICE (once before close, once after)
assert mock_async_client_class.call_count == 2
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

Always close the session:

```python
try:
    # ... test logic ...
finally:
    await session.close()

# Or simply:
await session.close()  # At end of test
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

## Debugging Failed Tests

### Enable Logging

Run with logging to see MCP flow:

```bash
uv run pytest test__mcp_client.py -v -s --log-cli-level=DEBUG
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

## test_mcp_tool.py Tests

This file tests `create_mcp_tools_from_metadata` in `mcp_tool.py`.

### Test Classes

| Class | Purpose |
|-------|---------|
| `TestMcpToolMetadata` | Tests tool metadata (tool_type, display_name, etc.) |
| `TestMcpToolCreation` | Tests tool creation (multiple tools, descriptions, disabled config) |
| `TestMcpToolInvocation` | Smoke test for full tool invocation flow |
| `TestMcpToolNameSanitization` | Tests tool name sanitization |

### Key Test: test_tool_invocation_initializes_session_and_returns_result

This smoke test verifies the full integration between `create_mcp_tools_from_metadata` and `McpClient`:

```python
@pytest.mark.asyncio
@patch("uipath_langchain.agent.tools.mcp.mcp_tool.UiPath")
@patch("httpx.AsyncClient")
async def test_tool_invocation_initializes_session_and_returns_result(
    self,
    mock_async_client_class,
    mock_uipath_class,
):
```

**What it tests:**
- Tool creation from metadata config
- MCP session initialization via real MCP SDK
- Tool call execution and result retrieval
- Full MCP protocol flow (initialize → notifications/initialized → tools/call)

**What it mocks:**
- `UiPath` SDK (for `mcp.retrieve_async`)
- `httpx.AsyncClient` (HTTP layer only)

**What it does NOT mock:**
- MCP SDK (`mcp.ClientSession`, `streamable_http_client`, etc.)

**Reuses pattern from test__mcp_client.py:**
- `create_mock_stream_response()` factory
- `create_mock_http_client()` helper

### Mocking Strategy

The tests follow the same principle as `test__mcp_client.py`:

1. **Mock only `httpx.AsyncClient`** - The HTTP layer
2. **Mock `UiPath` SDK** - For `mcp.retrieve_async` to get MCP server URL
3. **Never mock MCP SDK** - Let real `ClientSession` process messages

This ensures:
- Real MCP protocol flow is tested
- Integration between `mcp_tool.py` and `_mcp_client.py` is validated
- HTTP layer behavior can be controlled for testing

### Adding New Tests

When adding tests to `test_mcp_tool.py`:

1. **For metadata/creation tests**: Just mock `UiPath` SDK
2. **For invocation tests**: Use the mock pattern from `TestMcpToolInvocation`
3. **Never mock** `mcp.ClientSession` or other MCP SDK components

## Related Files

| File | Purpose |
|------|---------|
| `test__mcp_client.py` | Session tests (McpClient class) |
| `test_mcp_tool.py` | Tool creation and invocation tests |
| `src/.../mcp/_mcp_client.py` | Session implementation |
| `src/.../mcp/mcp_tool.py` | Tool creation implementation |
| `src/.../mcp/claude.md` | Implementation documentation |
