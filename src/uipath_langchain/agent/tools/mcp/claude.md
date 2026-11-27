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
├── mcp_tasks.py           # MCP Tasks extension: claim + poll loop
├── mcp_task_suspension.py # com.uipath/tasks: suspend the run on a long task
├── mcp_service_call.py    # call_mcp_tool: the same, for a coded agent
├── mcp_tool.py            # Tool factory functions
├── protocol_strategy.py   # Per-era negotiation/recovery policy
├── session_tools.py       # MCP session -> LangChain tool conversion
└── streamable_http.py     # SessionInfo, session identity, SDK transport adapter
```

### Public Exports (`__init__.py`)

```python
from .mcp_client import McpClient, SessionInfoFactory
from .mcp_service_call import call_mcp_tool
from .mcp_task_suspension import SuspensionSettings
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

Selected by `McpClient(protocol_mode=...)` via `build_protocol_strategy`
(`create_mcp_tools_and_clients` forwards the same kwarg to every client):

- **`"legacy"`** — `LegacyHandshakeStrategy`. Preserves the pre-2026
  wire behaviour exactly.
- **`"modern"`** — `ModernDiscoveryStrategy`. `server/discover` only.
- **`"auto"` (default)** — `AutoStrategy`. Mints the affinity ID first so the probe is
  pinned, runs its own `server/discover` probe (`probe_modern_era`, built on the
  public `ClientSession.send_discover` / `adopt` seam), then delegates to
  whichever era won. Re-resolved on every `connect`, so a server upgraded
  mid-run is handled.

The default is `"auto"`. On AgentHub, UiPath, Swagger and Platform servers
resolve to `2026-07-28`; Coded, Command and Remote servers fall back to the
handshake. Playground persistence survives the modern era: the client-minted
affinity ID is stored through `SessionInfo`, so `SessionInfoDebugState` persists
and restores it like a session ID. A server whose instances hold state no peer
can rebuild should pass `protocol_mode="legacy"` (see Modern-era instance
affinity below).

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

### mcp_tasks.py — The Tasks Extension (`io.modelcontextprotocol/tasks`)

A `tools/call` the server answers with `resultType: "task"` is polled to
completion inside `McpClient.call_tool`, so the caller always receives the
final `CallToolResult` and agent definitions never see task semantics.

It is built on the SDK's **extension/claim** seam rather than a hand-rolled
branch in the tool layer:

| Piece | Role |
|-------|------|
| `McpTasksExtension` | `ClientExtension` declaring the identifier and its one claim; also usable directly with the high-level `Client(extensions=[...])` |
| `McpTaskHandle` | The claimed `resultType: "task"` shape (flat, per ext-tasks) |
| `McpTaskSnapshot` | One `tasks/get` answer; `result`/`error` appear on terminal states |
| `_resolve_task` | The claim resolver: the poll loop |
| `session_extension_kwargs()` | Translates the extension into the two mappings `ClientSession` takes |
| `resolve_claimed_result()` | Finishes a claimed result; passes an ordinary one through |

**Why two seams instead of one.** The high-level `mcp.Client` resolves claims
for you; `McpClient` drives the low-level `ClientSession`, which *parses* a
claimed shape but raises `UnexpectedClaimedResult` unless the caller opts in.
So `call_tool` passes `allow_claimed=True` and calls `resolve_claimed_result`,
which mirrors what `Client.call_tool` does — including re-validating the output
schema on a non-`isError` result. Do not drop that revalidation: it is the only
thing keeping the claimed path's contract identical to the direct one.

**An ordinary result never touches the session again.** `call_tool` returns a
plain `CallToolResult` straight out of the retry, before `_ensure_session()`.
Reaching for a session first would let a call that already succeeded fail on a
concurrent session drop, for a session it has no further use for.

**Resolution must not re-run `tools/call`.** `resolve_claimed_result(...)`
runs *outside* `_execute_with_retry`. A task handle means the server already
accepted the work and a job is running, so a failure while polling must
propagate; recovery for polling belongs inside the poll loop, which keeps the
task id and the original deadline. With both in the retried closure, one
recoverable poll error re-ran `tools/call` and launched a **second job** for one
tool call, on a restarted poll budget.

Submission itself still goes through that retry, unchanged from before Tasks.
Worth knowing what it assumes: a session the server *rejected* proves the call
never ran, but a dropped transport proves only that the connection died — it may
have died after the server started the job, and then the resend runs the tool
twice. That is true of every tool call, not just task-shaped ones, and
`test_dropped_connection_resumes_the_persisted_session` pins the current
behavior deliberately, so narrowing it is its own change with its own
discussion.

**The era gate is free.** The SDK activates a claim (and advertises the
extension that owns it) only on protocol versions that can carry claimed
shapes. A legacy-era connection therefore neither advertises
`io.modelcontextprotocol/tasks` nor can receive a task — no version check is
written anywhere in this module, and
`test_a_legacy_connection_never_advertises_the_extension` pins it.

**Wire-shape caveat.** The models here mirror ext-tasks
(`ttlMs`/`pollIntervalMs`), *not* the SDK's built-in `mcp.types.Task`, which
still carries the superseded SEP-1686 names (`ttl`/`pollInterval`). Validating
our wire against `mcp.types.Task` fails on the missing `ttl`. The SDK's
`GetTaskRequest`/`GetTaskRequestParams` *are* reused — only the result models
are local.

**Rollout.** On by default, with two toggles on the SDK's shared
`uipath.core.feature_flags.FeatureFlags` registry — settable programmatically
by a host or injected as `UIPATH_FEATURE_<name>`:

| Flag | Default | Purpose |
|------|---------|---------|
| `DisableMcpTasksPolling` | `false` | Kill switch; `true` restores the pre-Tasks behavior exactly |
| `McpTasksPollBudgetSeconds` | `840` | Bounds one call's polling, inside the host's execution window |

The kill switch is phrased negatively to match the package's others
(`DisableBatchTransformFromAttachment`, `DisableDeepRagFromAttachments`), so
`tasks_polling_enabled()` negates it.

`DisableMcpTasksPolling` is an **emergency opt-out, not a rollout gate** — a
deployed runtime image carries a fixed package version, so there is no
per-deployment rollback here. Whether tasks happen at all is the server's call:
a server that does not advertise the extension can never answer with one. The
budget exists because a tool that used to fail fast on an HTTP timeout must not
silently consume the agent's whole run; on exhaustion the call raises
`MCPError(REQUEST_TIMEOUT)` and the task keeps running server-side. The budget
bounds **one call, not the run** — a run that starts its first task late, or
makes several task-shaped calls, can still reach the host's execution window,
and the host ends it there.

**Every failure out of the resolver is an `MCPError`.** `mcp_tool.tool_fn`
catches exactly that type and turns it into a categorized `AgentRuntimeError`;
anything else — a bare `TimeoutError`, a pydantic `ValidationError` from a
`completed` task carrying no usable result — walks straight past it and leaves
the tool as an uncategorized exception. That is why budget exhaustion is
`MCPError(REQUEST_TIMEOUT)` rather than `TimeoutError`, and why
`_completed_result` converts a validation failure.

**Cancellation is never sent.** Not on budget exhaustion, and not when the run
itself is cancelled or killed mid-poll: the task is a real Orchestrator job, and
the agent giving up on waiting is no reason to abort it. The cost is a dropped
result, plus a second job if the model retries the tool; the task id is in the
message so an operator can find the first one. Nothing here reaches
`tasks/cancel` — a caller that wants a task cancelled has to do it through
`McpService` in the platform SDK.

The budget is a **deadline, not a per-request allowance**, and one poll is also
capped at `MAX_POLL_REQUEST_SECONDS`: each `tasks/get` is sent with
`request_read_timeout_seconds` set to `min(remaining, 30)`, and the clock is
re-read *after* the inter-poll sleep as well as before. Without the deadline, a
stalled or slow server answers after the budget has expired — and an answer of
`completed` arriving then would be accepted. Without the cap, one black-holed
socket costs the whole call: bounding a poll by the entire remaining budget made
a fourteen-minute budget *less* resilient than a one-minute one, which is the
opposite of what a longer budget should buy.

A failed poll is the loop's to absorb, within limits: `INTERNAL_ERROR` (what a
gateway's non-JSON 5xx or 429 becomes on the client transport) and
`REQUEST_TIMEOUT` (one poll hitting its cap) are logged and re-polled on the
next tick — same task id, same deadline — up to `MAX_POLL_FAILURES` consecutive
times, after which the error itself is raised rather than buried by an eventual
timeout. Once the deadline has passed, any poll failure becomes the one
budget-exhaustion error instead. A `CONNECTION_CLOSED` is **not** absorbed: the
resolver cannot rebuild a dead transport, and the client's session recovery
deliberately no longer covers resolution, since that would resubmit the tool
call.

Terminal statuses map as: `completed` → the carried `CallToolResult`, or
`MCPError` when it carries nothing a `CallToolResult` can be built from;
`failed` → `MCPError` carrying the task's message **and its code in the text**,
raised as `INTERNAL_ERROR`; `cancelled` → `MCPError`; `input_required` →
`MCPError`, because the tool layer has no interactive party and parking would
burn the budget.

Why the failed task's own code is not re-raised as the error's code:
`_map_mcp_error` asks `McpClient.is_session_error(error)` whether to tell the
user the connection was terminated, and a tool is free to fail with a code that
answers yes — `-32000` among them. Reporting the code in the message keeps it
diagnosable without letting a tool failure impersonate a transport failure.

---

### mcp_task_suspension.py — Suspending the Run on a Long Task (`com.uipath/tasks`)

`com.uipath/tasks` is the Tasks protocol under a UiPath name: same
`resultType: "task"`, same `tasks/get`. A UiPath MCP server that serves it also
names, in the task's `_meta["com.uipath/task"]`, the Orchestrator job behind the
task (`jobKey`, `folderKey`). A run that is itself an Orchestrator job can then
suspend on a Job-type resume trigger instead of staying awake to poll.

`session_extension_kwargs()` declares the extension next to the standard one. It
registers **no claim** for it: the task it answers with is the one
`McpTasksExtension` already claims.

The declaration lists the resume-trigger kinds a run can suspend on,
`{"resumeTriggerTypes": ["Job"]}`, and the server names a job only to a client
whose list includes `Job`. In the other direction, `task_job()` reads an
annotation only at `version` 1 and only when `resumeTrigger.triggerType` is
`Job`. A task annotated with any other trigger kind (one the server may add
later, for a resume not implemented here) is polled, like a task with no
annotation.

**When it suspends.** Not at once: a resume costs a runtime start, so the task
is polled first and the run suspends only if it is still working. One setting,
on `SuspensionSettings`:

| Setting | Flag (seconds) | Default | Meaning |
|---------|----------------|---------|---------|
| `min_polling_time_before_suspension` | `McpTasksMinPollingTimeBeforeSuspensionSeconds` | unset | Polled this long, then suspended if still working |

The polling time is resolved per task by `polling_time_before_suspension()`:

1. the caller's own setting above, when it is set (zero included);
2. else the server's `suspendAfterMs` on the job annotation, bounded by the
   poll budget (`McpTasksPollBudgetSeconds`) so the server cannot keep a run
   awake longer than plain polling would;
3. else 60 seconds.

AgentHub fills `suspendAfterMs` from its own feature flag of the same name, so
the usual way to tune this is on the server, per tenant. The flag on this side
is an override for one run. A `suspendAfterMs` that is not a non-negative
number is ignored; the job is still suspended on.

A task with no annotation at all (a server without the extension) is polled
exactly as before. Deciding from an estimate of the job's run time — never
suspending on a short job, not polling a long one — is a follow-up
(PRODEV-2089); the server sends no estimate today. `DisableMcpTasksSuspension` is the kill switch; as with
polling, the rollout gate is the server's (`McpUiPathTasks` in AgentHub).

**How it survives the resume.** On resume the tool node runs again from its
first line, so `call_with_suspension` is a chain of `@durable_interrupt` steps:

1. `submit` — sends `tools/call` once. For a task it saves the **task handle**
   (task id, poll interval, the job annotation, the send time). That is all a
   new process on a new connection needs to go on polling.
2. `park` — only for a task that names a job. Polls for what is left of the
   polling time, takes one last look at the task, and if it is still working
   returns `WaitJobRaw` for the job. All of it is *inside* the step so that a
   replay skips it together with the suspend.
3. `collect` — polls the task to its end: after the resume (`poll_first=True`),
   or from the start for a task that names no job.

**Where the result is saved (`save_results`).** Whichever step ends the call
saves the result with it, through `TaskCall.dump_result` / `load_result`. A
coded node (`call_mcp_tool`) needs that: it can make this call and then suspend
on something else, the node is replayed from its first line, and the replay must
get the result back without sending the call or polling the task again — by
then the server may no longer keep the task. A tool node runs one call and
nothing after it, so `tool_fn` passes `save_results=False`: only the handle goes
into the checkpoint and a result, which can be large, is not stored a second
time. A replay that finds a step without its result fails with `STATE_ERROR`
rather than sending the call.

After the resume the task is read with `tasks/get`, `poll_first=True`. The
trigger only says when to look; the result still comes through the server and
its guardrail screening.

Rules to keep when changing it:

- **`WaitJobRaw`, not `WaitJob`.** The plain one reads the job's output on resume
  and faults the run when the job did not succeed. The result must come from
  `tasks/get`, which reports a faulted or stopped job the way the tool layer
  expects.
- **The schema check belongs to `submit`.** `_SessionTaskCall.submit` runs
  `_refresh_tool_schema` itself. Run before the step, a resumed call would check
  the schema again and could answer "retry" for a task whose job already ran.
- **The annotation is server input that reaches a resume trigger.** `task_job`
  accepts only a GUID `jobKey` *and* a GUID `folderKey`, on a `_meta` that is an
  object; anything else yields no job, which costs the suspend and never the
  call. The folder is required because the resume reads the job there; without
  it the read would go to the run's own folder. The platform SDK parses the same
  annotation as `McpTask.job` (from 0.3.19); switch to it when the floor allows.
- **It suspends only in the tool's own node.** `tool_fn` asks
  `runs_in_own_tool_node(tool)` (set by `UiPathToolNode`) and
  `durable_call_in_play()`.
  Reached from a bridge — the code interpreter runs tools inside the `eval`
  tool's node — the task is polled. That is why MCP tools do **not** carry
  `SUSPENDS_RUN` and stay available to such callers;
  `test_suspends_run_metadata.py` checks the guard instead of the stamp.
- **Step order is fixed.** `park` is reached only when the saved handle names a
  job, so a replay takes the same path as the first pass.
- **The kill switch never moves a call that is under way off these steps.**
  Callers ask `durable_call_in_play()`, not `suspension_enabled()`: with the
  switch on, a node being replayed with saved step values still goes through
  `call_with_suspension`, because one of those values may be this call's task.
  The plain path would send `tools/call` again for a job that is already
  running. Inside the flow the switch only stops `park` from starting a new
  suspend; it polls to the end instead.
- **The poller giving up is not the polling time being over.** `_resolve_task`
  raises as soon as its next poll would not fit — at once, when the server's
  poll interval is longer than the polling time. `park` waits out what is left
  and then looks once (`peek`) before it suspends.
- **A step value that is not ours fails the call; it is never a reason to send
  it.** `@durable_interrupt` counts its steps from the start of the node and
  LangGraph's own `interrupt()` is not in that count, so a coded node that calls
  `interrupt()` *before* `call_mcp_tool` cannot resume the call. It raises
  `AgentRuntimeError(STATE_ERROR)`. The limit is the decorator's, shared by every
  tool built on it, and is documented on `call_mcp_tool`.

#### A Remote server's task: suspending through a watch

A Remote MCP server's upstream can run tasks of its own. There is no
Orchestrator job behind one, so AgentHub annotates it with an `Api` trigger
(`resumeTrigger.triggerType == "Api"`, no `jobKey`) and the run suspends on an
**API trigger** that AgentHub fires. Opt-in: `EnableMcpTasksApiSuspension`
(default `false`), and only when the run is itself a job (`own_job()`: a GUID
`UIPATH_JOB_KEY` and `UIPATH_FOLDER_KEY`). `tasks_capability()` lists `Api` only
then, so a server never holds a reply for a caller that would not suspend.

The flow, after `submit`, for a task whose handle carries the `Api` annotation
(`task_watch_terms`), is a bounded loop of `park_on_watch` steps:

1. The first round polls for the polling time, as `park` does.
2. `TaskCall.watch(handle, job_key=…, folder_key=…)` sends one `tasks/get` whose
   `params._meta["com.uipath/task"].watch` names the run's own job. It answers
   the task's result (ended), a `TaskWatch(watch_id, expires_in)` (still
   working, and AgentHub is now watching it), or `None` (still working,
   unwatched).
3. On a `TaskWatch` the step returns a **list** of two interrupt values, which
   the runtime turns into sibling triggers: `{"com.uipath/task": {"watchId": …}}`
   (an API trigger; the server delivers only to a trigger whose request carries
   the watch id) and `WaitUntil(now + expires_in)`. The first to fire resumes.
4. After a resume the next round asks again: the result, or a new watch.

Rules to keep:

- **The timer is not optional.** A watch runs on credentials AgentHub captured
  from this run's own request and ends when they would (`expires_in`). The
  delivery can also fail. The timer is what guarantees the run wakes.
- **The resume value is never read.** Server payload or timer, the next round
  reads the task with `tasks/get`.
- **`None` means poll.** No watch (a server with guardrails, a relay, a job the
  caller cannot see) leaves the task to `collect`, exactly as without the
  feature. So does `MAX_WATCHES_PER_CALL` rounds, and the flag being switched
  off while suspended.
- **`watch` and `watch_terms` are optional on a `TaskCall`.** `_SessionTaskCall`
  has them. `_ServiceTaskCall` does not yet (`McpService.get_task` cannot send
  request `_meta`), so `call_mcp_tool` polls a Remote server's task.
- **Not in a conversational or a debug run.** Those runtimes treat an API
  trigger as a prompt for a person. Nothing here detects them; the flag is the
  guard, so leave it off for such agents.

`TaskCall` is the seam between the flow and the client that carries the call.
`_SessionTaskCall` (`mcp_tool.py`) drives `McpClient`; `_ServiceTaskCall`
(`mcp_service_call.py`) drives the platform SDK's stateless `McpService` for
`call_mcp_tool`, the coded-agent entry point. `McpService` declares
`com.uipath/tasks` from `uipath-platform` 0.3.19; with an older one the server
never names a job and `call_mcp_tool` polls.

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
│  _connection_task: asyncio.Task | None  # owns the scopes   │
│  _close_requested: asyncio.Event | None                     │
│  _session: ClientSession | None                             │
├─────────────────────────────────────────────────────────────┤
│  Public Methods                                             │
│  ──────────────                                             │
│  + list_tools(force_refresh=False) -> ListToolsResult       │
│  + call_tool(name, arguments) -> CallToolResult             │
│  + submit_tool_call(name, arguments) -> result | task handle│
│  + resolve_task(name, handle, budget_seconds, poll_first)   │
│  + peek_task(name, handle) -> CallToolResult | None         │
│  + watch_task(name, handle, job_key, folder_key)            │
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

**Tool naming:** the LLM-facing name is `mcp-{resource}-tool-{tool}`, built by
`mcp_tool_identity()`. MCP scopes tool names per server, so the resource is what
makes the name unique across the flat tool list the model is given; resource names
are unique within an agent definition, so the pair is unique by construction. A pair
past the provider's 64-character cap spends the cap on the tool name first, since that
is the part the model reads when choosing between tools, and shortens the resource to
a stub; a digest of the full pair keeps two shortened names apart. The tool's own name
stays on `metadata["display_name"]`, which is what span titles show.

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
│ SDK transport + │ ─── Opened on a dedicated connection task,
│ ClientSession   │     which holds them until dispose() signals
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

### 4. Two Stacks, Two Lifecycles

The base stack (HTTP client) outlives any single method, so it is managed by
hand:

```python
self._stack = AsyncExitStack()
await self._stack.__aenter__()
# ... reused across sessions ...
await self._stack.aclose()
```

The connection stack (transport + `ClientSession`) is different. anyio cancel
scopes must be exited by the task that entered them, in reverse order, and
neither is under the caller's control: langgraph opens the connection inside a
tool task, and an agent disposes several servers oldest-first. So
`_run_connection` owns the stack with a plain `async with` and stays inside it
until told to leave:

```python
async with AsyncExitStack() as connection_stack:
    ...  # enter transport and session
    ready.set_result(None)
    await close_requested.wait()  # dispose() sets this
```

`dispose()` never touches the connection stack. It signals the task, or cancels
it if the handshake has not finished yet, and waits for it to exit.

### 5. Reinitialization Reuses the HTTP Client

On a recoverable session error, `_reinitialize_session()` signals the old
connection task to close and waits for it, calls `strategy.reset` to clear the ID unless
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
