"""MCP Tasks extension: flag parsing and the claim resolver's poll loop."""

from typing import Any

import anyio
import pytest
from mcp.client.extension import ClaimContext
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CONNECTION_CLOSED,
    INTERNAL_ERROR,
    REQUEST_TIMEOUT,
    CallToolResult,
)
from uipath.core.feature_flags import FeatureFlags

from uipath_langchain.agent.tools.mcp import mcp_tasks
from uipath_langchain.agent.tools.mcp.mcp_client import McpClient
from uipath_langchain.agent.tools.mcp.mcp_tasks import (
    DEFAULT_POLL_BUDGET_SECONDS,
    MAX_POLL_FAILURES,
    MAX_POLL_REQUEST_SECONDS,
    POLL_BUDGET_FLAG,
    TASKS_POLLING_KILL_SWITCH,
    McpTaskHandle,
    McpTasksExtension,
    McpTaskSnapshot,
    _resolve_task,
    _task_ref,
    task_poll_budget_seconds,
    tasks_polling_enabled,
)


@pytest.fixture(autouse=True)
def clean_flags() -> Any:
    """The flag registry is a process-wide singleton; isolate every test."""
    FeatureFlags.reset_flags()
    yield
    FeatureFlags.reset_flags()


class FakeSession:
    """Answers tasks/get with a scripted sequence of snapshots.

    A snapshot may be an exception instance, which is raised instead — that is
    how a poll that outlives its own request bound is scripted.
    """

    def __init__(self, snapshots: list[Any]) -> None:
        self.snapshots = list(snapshots)
        self.requests: list[Any] = []
        self.request_bounds: list[float | None] = []

    async def send_request(
        self,
        request: Any,
        result_type: Any,
        request_read_timeout_seconds: float | None = None,
    ) -> Any:
        self.requests.append(request)
        self.request_bounds.append(request_read_timeout_seconds)
        answer = self.snapshots.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def handle(poll_interval_ms: int = 100) -> McpTaskHandle:
    return McpTaskHandle.model_validate(
        {
            "resultType": "task",
            "taskId": "task-1",
            "status": "working",
            "pollIntervalMs": poll_interval_ms,
        }
    )


def snapshot(status: str, **extra: Any) -> McpTaskSnapshot:
    return McpTaskSnapshot.model_validate(
        {"taskId": "task-1", "status": status, "pollIntervalMs": 100, **extra}
    )


def context(session: FakeSession) -> ClaimContext:
    return ClaimContext(
        session=session,  # type: ignore[arg-type]
        tool_name="run-invoice-process",
        read_timeout_seconds=None,
    )


# ── Flags ──────────────────────────────────────────────────


def test_tasks_polling_is_enabled_by_default() -> None:
    assert tasks_polling_enabled() is True


def test_tasks_polling_opt_out_via_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    # The injection path a host (SCP) uses: UIPATH_FEATURE_<flag>.
    monkeypatch.setenv(f"UIPATH_FEATURE_{TASKS_POLLING_KILL_SWITCH}", "true")
    assert tasks_polling_enabled() is False


def test_tasks_polling_opt_out_programmatically() -> None:
    FeatureFlags.configure_flags({TASKS_POLLING_KILL_SWITCH: True})
    assert tasks_polling_enabled() is False


def test_the_kill_switch_left_false_keeps_polling_on() -> None:
    FeatureFlags.configure_flags({TASKS_POLLING_KILL_SWITCH: False})
    assert tasks_polling_enabled() is True


def test_programmatic_configuration_wins_over_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(f"UIPATH_FEATURE_{TASKS_POLLING_KILL_SWITCH}", "true")
    FeatureFlags.configure_flags({TASKS_POLLING_KILL_SWITCH: False})
    assert tasks_polling_enabled() is True


@pytest.mark.parametrize("value", ["not-a-number", "-5", "0"])
def test_poll_budget_falls_back_on_bad_values(value: str) -> None:
    FeatureFlags.configure_flags({POLL_BUDGET_FLAG: value})
    assert task_poll_budget_seconds() == DEFAULT_POLL_BUDGET_SECONDS


def test_poll_budget_is_read_from_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "120.5")
    assert task_poll_budget_seconds() == 120.5


# ── The claim resolver ─────────────────────────────────────


async def test_resolve_polls_until_completed_and_returns_the_result() -> None:
    session = FakeSession(
        [
            snapshot("working"),
            snapshot(
                "completed",
                result={
                    "content": [{"type": "text", "text": "done"}],
                    "isError": False,
                },
            ),
        ]
    )

    result = await _resolve_task(handle(), context(session))

    assert isinstance(result, CallToolResult)
    assert result.content[0].text == "done"  # type: ignore[union-attr]
    assert len(session.requests) == 2
    assert session.requests[0].params.task_id == "task-1"


async def test_resolve_reports_the_tasks_own_error_on_failed() -> None:
    session = FakeSession(
        [snapshot("failed", error={"code": -32001, "message": "boom"})]
    )

    with pytest.raises(MCPError) as excinfo:
        await _resolve_task(handle(), context(session))

    # The task's message and code are both reported, but the code is carried in
    # the text rather than re-raised as this error's code -- see
    # test_a_failed_task_is_not_reported_as_a_lost_connection.
    assert excinfo.value.error.code == INTERNAL_ERROR
    assert "boom" in excinfo.value.error.message
    assert "-32001" in excinfo.value.error.message


async def test_resolve_raises_on_cancelled() -> None:
    session = FakeSession([snapshot("cancelled")])

    with pytest.raises(MCPError, match="cancelled"):
        await _resolve_task(handle(), context(session))


async def test_resolve_raises_on_input_required() -> None:
    # The tool layer has no interactive party; burning the budget on a parked
    # task would be worse than a clear error.
    session = FakeSession([snapshot("input_required")])

    with pytest.raises(MCPError, match="requires additional input"):
        await _resolve_task(handle(), context(session))


async def test_resolve_times_out_within_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A budget smaller than one poll interval must fail fast, before sleeping.
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "0.05")
    session = FakeSession([])

    with pytest.raises(MCPError, match="task-1") as raised:
        await _resolve_task(handle(), context(session))

    # An MCPError, not a TimeoutError: mcp_tool.tool_fn catches exactly this
    # type, and anything else leaves the tool as an uncategorized exception.
    assert raised.value.code == REQUEST_TIMEOUT
    assert session.requests == []


async def test_every_poll_is_bounded_by_what_is_left_of_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unbounded tasks/get could answer long after the budget expired."""
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "5")
    session = FakeSession(
        [
            snapshot("working"),
            snapshot("completed", result={"content": [], "isError": False}),
        ]
    )

    await _resolve_task(handle(), context(session))

    bounds = session.request_bounds
    assert len(bounds) == 2
    assert all(bound is not None and 0 < bound <= 5 for bound in bounds)
    # It shrinks: the budget is a deadline, not a per-request allowance.
    assert bounds[1] < bounds[0]  # type: ignore[operator]


async def test_a_stalled_poll_with_budget_left_is_polled_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One stalled connection costs a poll, not the call.

    Rewriting every ``REQUEST_TIMEOUT`` into budget exhaustion made a long
    budget *less* resilient than a short one: a single black-holed socket ended
    a fourteen-minute call that still had thirteen minutes to run.
    """
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "5")
    session = FakeSession(
        [
            MCPError(REQUEST_TIMEOUT, "Request 'tasks/get' timed out"),
            snapshot("completed", result={"content": [], "isError": False}),
        ]
    )

    result = await _resolve_task(handle(), context(session))

    assert isinstance(result, CallToolResult)
    assert len(session.requests) == 2


async def test_a_poll_is_capped_even_when_the_budget_is_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise the whole call rides on one socket."""
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "600")
    session = FakeSession([snapshot("completed", result={"content": []})])

    await _resolve_task(handle(), context(session))

    assert session.request_bounds == [MAX_POLL_REQUEST_SECONDS]


async def test_a_poll_failing_past_the_deadline_is_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever the poll failed with, out of budget is out of budget."""
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "5")
    clock = FakeClock()
    monkeypatch.setattr(mcp_tasks, "time", clock)

    async def sleep_then_overrun(seconds: float) -> None:
        clock.now += 1.0

    monkeypatch.setattr(anyio, "sleep", sleep_then_overrun)

    class SlowFailure(FakeSession):
        async def send_request(
            self,
            request: Any,
            result_type: Any,
            request_read_timeout_seconds: float | None = None,
        ) -> Any:
            clock.now += 10.0  # the request itself ran past the deadline
            return await super().send_request(
                request, result_type, request_read_timeout_seconds
            )

    session = SlowFailure([MCPError(INTERNAL_ERROR, "Server returned an error")])

    with pytest.raises(MCPError, match="task-1") as raised:
        await _resolve_task(handle(), context(session))

    assert raised.value.code == REQUEST_TIMEOUT


async def test_a_transient_poll_failure_is_absorbed_by_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gateway blip must not end a wait the server is still working on."""
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "5")
    session = FakeSession(
        [
            MCPError(INTERNAL_ERROR, "Server returned an error response"),
            snapshot("completed", result={"content": [], "isError": False}),
        ]
    )

    result = await _resolve_task(handle(), context(session))

    assert isinstance(result, CallToolResult)
    # Re-polled on the next tick, same task id, no resubmission.
    assert len(session.requests) == 2
    assert session.requests[1].params.task_id == "task-1"


async def test_a_server_answering_only_errors_stops_at_the_failure_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error reaches the caller instead of being buried by a timeout."""
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "30")
    session = FakeSession(
        [MCPError(INTERNAL_ERROR, "Server returned an error response")] * 5
    )

    with pytest.raises(MCPError, match="Server returned an error response"):
        await _resolve_task(handle(), context(session))

    assert len(session.requests) == MAX_POLL_FAILURES


async def test_a_dropped_connection_while_polling_is_not_absorbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead transport cannot be re-polled; only the client can rebuild it."""
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "30")
    session = FakeSession([MCPError(CONNECTION_CLOSED, "Connection closed")])

    with pytest.raises(MCPError, match="Connection closed"):
        await _resolve_task(handle(), context(session))

    assert len(session.requests) == 1


async def test_another_mcp_error_from_a_poll_is_not_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the request bound elapsing is rewritten; every other error stands."""
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "5")
    session = FakeSession([MCPError(-32602, "Unknown task id.")])

    with pytest.raises(MCPError, match="Unknown task id."):
        await _resolve_task(handle(), context(session))


async def test_the_deadline_is_rechecked_after_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sleep that overruns must stop the loop, not poll once more."""
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "5")
    clock = FakeClock()
    monkeypatch.setattr(mcp_tasks, "time", clock)

    async def oversleep(seconds: float) -> None:
        # A starved event loop: the 0.1s wait really took a minute.
        clock.now += 60.0

    monkeypatch.setattr(anyio, "sleep", oversleep)
    session = FakeSession([snapshot("completed", result={"content": []})])

    with pytest.raises(MCPError, match="task-1"):
        await _resolve_task(handle(), context(session))

    # The finding exactly: a stale success must not be fetched, let alone returned.
    assert session.requests == []


async def test_a_completed_task_with_no_result_is_an_mcp_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CallToolResult requires content, so an empty payload cannot validate.

    The ValidationError that raises is not what the tool layer catches, so it
    would leave the tool as an uncategorized exception instead of a mapped
    failure.
    """
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "5")
    session = FakeSession([snapshot("completed")])

    with pytest.raises(MCPError, match="without a usable result"):
        await _resolve_task(handle(), context(session))


async def test_a_failed_task_is_not_reported_as_a_lost_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The task's own code must not be mistaken for a transport verdict.

    ``mcp_tool._map_mcp_error`` reads the code to decide whether to tell the
    user the connection was terminated, and ``CONNECTION_CLOSED`` is a code a
    tool is free to fail with.
    """
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "5")
    session = FakeSession(
        [
            snapshot(
                "failed",
                error={"code": CONNECTION_CLOSED, "message": "the robot disconnected"},
            )
        ]
    )

    with pytest.raises(MCPError) as raised:
        await _resolve_task(handle(), context(session))

    assert raised.value.code == INTERNAL_ERROR
    assert "the robot disconnected" in raised.value.message
    assert str(CONNECTION_CLOSED) in raised.value.message
    assert not McpClient.is_session_error(raised.value)


# ── The task reference ─────────────────────────────────────

# Two real ids from a UiPath server: 368 characters, sharing their first 27.
REAL_ID_A = "AfDP2j-98rUt7wxBowad5FIlYGx" + "A" * 333 + "4VD53jN7yzE8"
REAL_ID_B = "AfDP2j-98rUt7wxBowad5FIlYGx" + "B" * 333 + "8eMkW1DIPfrp"


def test_two_different_tasks_do_not_read_the_same() -> None:
    """A prefix of these ids is a constant, not a name."""
    assert _task_ref(REAL_ID_A) != _task_ref(REAL_ID_B)


def test_a_long_id_is_shortened_at_both_ends() -> None:
    shortened = _task_ref(REAL_ID_A)

    assert len(shortened) < 40
    assert shortened.startswith(REAL_ID_A[:12])
    assert shortened.endswith(REAL_ID_A[-8:])


def test_a_short_id_is_left_alone() -> None:
    assert _task_ref("task-1") == "task-1"


async def test_the_poll_addresses_the_task_by_its_whole_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shortening is for text only: tasks/get carries the id as minted.

    A truncated id on the wire would address a different task, or none.
    """
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "5")
    session = FakeSession([snapshot("completed", result={"content": []})])
    long_handle = McpTaskHandle.model_validate(
        {
            "resultType": "task",
            "taskId": REAL_ID_A,
            "status": "working",
            "pollIntervalMs": 100,
        }
    )

    await _resolve_task(long_handle, context(session))

    assert session.requests[0].params.task_id == REAL_ID_A


async def test_the_budget_error_does_not_quote_the_whole_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This message is read by a model, so the handle does not belong in it."""
    monkeypatch.setenv(f"UIPATH_FEATURE_{POLL_BUDGET_FLAG}", "0.05")
    session = FakeSession([])
    long_handle = McpTaskHandle.model_validate(
        {
            "resultType": "task",
            "taskId": REAL_ID_A,
            "status": "working",
            "pollIntervalMs": 100,
        }
    )

    with pytest.raises(MCPError) as raised:
        await _resolve_task(long_handle, context(session))

    assert REAL_ID_A not in raised.value.message
    assert REAL_ID_A[:12] in raised.value.message


# ── The extension ──────────────────────────────────────────


def test_extension_claims_the_task_shape() -> None:
    claims = McpTasksExtension().claims()

    assert len(claims) == 1
    assert claims[0].result_type == "task"
    assert claims[0].model is McpTaskHandle
    assert claims[0].method == "tools/call"
