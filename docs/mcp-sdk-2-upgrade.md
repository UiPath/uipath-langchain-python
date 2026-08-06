# MCP Python SDK 2.0 upgrade review

## Outcome

`uipath-langchain-python` now pins `mcp==2.0.0`, the latest stable MCP Python
SDK in the local upstream checkout. The lockfile resolves its new `mcp-types`
and `httpx2` dependencies. The unused `langchain-mcp-adapters==0.2.1` direct
dependency was removed; the core package does not import it. Standalone samples
that use the adapter declare it in their own `pyproject.toml` files.

The UiPath client continues to use the SDK's low-level `ClientSession` and
Streamable HTTP transport. Its externally persisted session-ID extension is
now a small adapter around the upstream transport rather than a private copy of
the complete transport.

## What UiPath changed in the old SDK 1.26 transport copy

The private `streamable_http.py` was introduced in commit `9c038fa2` and was
based on `mcp.client.streamable_http` from MCP Python SDK 1.26. Compared with
that upstream implementation, UiPath added:

- An asynchronous `SessionInfo` abstraction whose `get_session_id()` and
  `set_session_id()` methods can be overridden to load and save AgentHub debug
  state.
- Asynchronous request-header preparation so every request can load the latest
  externally stored session ID.
- Persistence of the `mcp-session-id` returned by an initialization response.
- A `session_info` argument on the local context manager, replacing the old
  transport's session-ID callback shape.
- Raw response-body logging for HTTP error responses.

The resulting file duplicated roughly 800 lines of SDK transport code. That
made fixes and new protocol behavior in upstream Streamable HTTP unavailable
without manually merging the copy.

## How Streamable HTTP evolved in SDK 2.0

SDK 2.0's upstream transport now owns substantially more behavior than the
1.26 copy, including:

- Legacy initialization and 2026 modern-protocol routing.
- `Mcp-Protocol-Version`, `Mcp-Method`, and `Mcp-Name` headers.
- Correct JSON-RPC errors from non-2xx response bodies and request-scoped
  fallback errors when a body is absent.
- SSE resumption with `Last-Event-ID` and bounded reconnection.
- 2026 HTTP cancellation by aborting the in-flight request POST.
- GET channel and DELETE session lifecycle handling.
- Per-request error delivery rather than transport-wide exception groups.

The UiPath adapter now delegates all of this to
`mcp.client.streamable_http.streamable_http_client`. Two `httpx2` event hooks
provide the UiPath-specific behavior:

1. Before a request, asynchronously load `SessionInfo` and set or remove
   `mcp-session-id`.
2. After a response, persist a returned `mcp-session-id` through `SessionInfo`.

This retains compatibility with `SessionInfoDebugState` in
`uipath-agents-python` without forking the transport again.

The old raw error-body logging was deliberately not recreated. SDK 2.0 now
parses a JSON-RPC error carried by a non-2xx response and surfaces its message
through `MCPError`; logging an arbitrary raw server body would add payload and
credential-leak risk without improving the structured error path.

## SDK 2.0 breaking changes relevant here

| SDK 1.x API | SDK 2.0 API / behavior | Upgrade action |
| --- | --- | --- |
| `McpError` | `MCPError(code, message, data)` | Updated imports, catches, construction, and tests. |
| Python model fields such as `inputSchema` and `outputSchema` | `input_schema` and `output_schema` | Updated all attribute reads. Wire JSON remains camelCase. |
| JSON-RPC root-model wrappers and `.root` | Plain discriminated message unions | The old copied transport was removed, eliminating these accesses locally. |
| `httpx` plus `httpx-sse` | `httpx2`, including SSE support | MCP connection and timeout types now use `httpx2`. |
| `timedelta` session timeout values | Seconds as `float` (or `None`) | The UiPath HTTP timeout is represented by `httpx2.Timeout`. |
| Transport `get_session_id` callback | No callback | Replaced with request/response event hooks. |
| `StreamableHTTPTransport.protocol_version` | Removed | Version handling is left to `ClientSession` and the transport. |
| Transport failures may surface through an `ExceptionGroup` | A request receives an `MCPError` | Retry logic catches `MCPError` directly. |
| Recalling `ClientSession.initialize()` could be used as local recovery logic | Initialization is idempotent per `ClientSession` | Recovery now replaces the transport and `ClientSession`, then performs a fresh handshake. |
| Experimental Tasks APIs | Removed | No UiPath code used them. |

## Protocol-version and backward-compatibility behavior

MCP SDK 2.0 declares these legacy handshake versions:

- `2024-11-05`
- `2025-03-26`
- `2025-06-18`
- `2025-11-25`

It also declares `2026-07-28` as a modern protocol version. The high-level SDK
`Client(mode="auto")` probes modern discovery and falls back to a legacy
initialization handshake.

UiPath currently uses low-level `ClientSession.initialize()`. That method sends
the latest legacy version (`2025-11-25`) and accepts any version in the legacy
handshake set returned by the server. Therefore:

| Server behavior | Current UiPath client |
| --- | --- |
| Negotiates `2025-03-26` | Supported and tested. |
| Negotiates `2025-06-18` | Supported and tested. |
| Negotiates `2025-11-25` | Supported and tested. |
| Supports 2026 but also accepts legacy initialize | Connects in legacy mode. |
| Supports only modern `2026-07-28` discovery | Not supported by the current low-level UiPath connection path. |

Supporting a 2026-only server would require adopting the high-level auto mode
or reproducing its discover/adopt flow. That is a separate behavior change from
this dependency upgrade.

## Session recovery details

For sessions initialized in the current process, SDK 2.0 maps a bare HTTP 404
to `MCPError(INVALID_REQUEST, "Session terminated")`. UiPath recognizes that
error and `CONNECTION_CLOSED`, closes the old connection stack, clears the
external session ID, and opens a fresh transport and `ClientSession` over the
same authenticated HTTP client.

An externally restored session ID is not stored inside the new transport; it is
injected by the request hook. Consequently, the transport initially maps a bare
404 to `METHOD_NOT_FOUND`. UiPath disambiguates that exact bare-404 error when
an external session ID was attached, clears the stale ID, initializes a new
session, and retries. JSON-RPC `METHOD_NOT_FOUND` errors with a response body
are not retried.

## Validation added

The MCP tests use the real SDK 2.0 `ClientSession` and Streamable HTTP transport
over `httpx2.MockTransport`. They cover:

- Negotiation with `2025-03-26`, `2025-06-18`, and `2025-11-25` servers.
- Session-header capture and reuse.
- Replacing the transport/session after HTTP 404 while reusing the HTTP client.
- Reuse of an externally persisted session without another initialization.
- Recovery from an expired externally persisted session.
- Retry exhaustion and non-session error classification.
- Tool listing cache/refresh, disposal/reuse, and full tool-call mapping.
