"""MCP Tasks extension: flag parsing and the claim resolver's poll loop."""

from typing import Any

import pytest
from mcp.client.extension import ClaimContext
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult
from uipath.core.feature_flags import FeatureFlags

from uipath_langchain.agent.tools.mcp.mcp_tasks import (
    DEFAULT_POLL_BUDGET_SECONDS,
    POLL_BUDGET_FLAG,
    TASKS_POLLING_FLAG,
    McpTaskHandle,
    McpTasksExtension,
    McpTaskSnapshot,
    _resolve_task,
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
    """Answers tasks/get with a scripted sequence of snapshots."""

    def __init__(self, snapshots: list[McpTaskSnapshot]) -> None:
        self.snapshots = list(snapshots)
        self.requests: list[Any] = []

    async def send_request(self, request: Any, result_type: Any) -> Any:
        self.requests.append(request)
        return self.snapshots.pop(0)


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
    monkeypatch.setenv(f"UIPATH_FEATURE_{TASKS_POLLING_FLAG}", "false")
    assert tasks_polling_enabled() is False


def test_tasks_polling_opt_out_programmatically() -> None:
    FeatureFlags.configure_flags({TASKS_POLLING_FLAG: False})
    assert tasks_polling_enabled() is False


def test_programmatic_configuration_wins_over_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(f"UIPATH_FEATURE_{TASKS_POLLING_FLAG}", "false")
    FeatureFlags.configure_flags({TASKS_POLLING_FLAG: True})
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


async def test_resolve_raises_the_tasks_own_error_on_failed() -> None:
    session = FakeSession(
        [snapshot("failed", error={"code": -32001, "message": "boom"})]
    )

    with pytest.raises(MCPError) as excinfo:
        await _resolve_task(handle(), context(session))

    assert excinfo.value.error.code == -32001
    assert "boom" in excinfo.value.error.message


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

    with pytest.raises(TimeoutError, match="task-1"):
        await _resolve_task(handle(), context(session))

    assert session.requests == []


# ── The extension ──────────────────────────────────────────


def test_extension_claims_the_task_shape() -> None:
    claims = McpTasksExtension().claims()

    assert len(claims) == 1
    assert claims[0].result_type == "task"
    assert claims[0].model is McpTaskHandle
    assert claims[0].method == "tools/call"
