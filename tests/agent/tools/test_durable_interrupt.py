"""Tests for the durable_interrupt module."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD

from uipath_langchain.agent.tools.durable_interrupt import (
    _durable_state,
    durable_interrupt,
)


class FakeScratchpad:
    """Minimal scratchpad with a resume list."""

    def __init__(self, resume: list[Any] | None = None) -> None:
        self.resume = resume or []


def _make_config(scratchpad: FakeScratchpad | None = None) -> dict[str, Any]:
    """Build a config dict matching get_config() shape."""
    if scratchpad is None:
        return {"configurable": {}}
    return {"configurable": {CONFIG_KEY_SCRATCHPAD: scratchpad}}


PATCH_GET_CONFIG = "uipath_langchain.agent.tools.durable_interrupt.get_config"
PATCH_INTERRUPT = "uipath_langchain.agent.tools.durable_interrupt.interrupt"


@pytest.fixture(autouse=True)
def _reset_durable_state() -> None:
    """Reset per-node counter between tests for isolation."""
    token = _durable_state.set(None)
    yield  # type: ignore[misc]
    _durable_state.reset(token)


class TestFirstExecution:
    """First execution (no resume values): action runs, interrupt with payload."""

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_runs_action(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)
        mock_interrupt.side_effect = lambda v: v

        action = AsyncMock(return_value="job-123")

        await durable_interrupt(action, lambda r: {"wait": r})

        action.assert_called_once()

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_interrupts_with_transformed_value(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)
        mock_interrupt.side_effect = lambda v: v

        action = AsyncMock(return_value="job-123")

        await durable_interrupt(action, lambda r: {"wait": r})

        mock_interrupt.assert_called_once_with({"wait": "job-123"})

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_returns_interrupt_result(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)
        mock_interrupt.return_value = "resume-value"

        action = AsyncMock(return_value="job-123")

        result = await durable_interrupt(action, lambda r: {"wait": r})

        assert result == "resume-value"


class TestResumeExecution:
    """Resume execution (resume value exists): action skipped, interrupt(None)."""

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_skips_action(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        scratchpad = FakeScratchpad(resume=["completed-job"])
        mock_get_config.return_value = _make_config(scratchpad)
        mock_interrupt.return_value = "completed-job"

        action = AsyncMock()

        await durable_interrupt(action, lambda r: r)

        action.assert_not_called()

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_interrupts_with_none(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        scratchpad = FakeScratchpad(resume=["completed-job"])
        mock_get_config.return_value = _make_config(scratchpad)
        mock_interrupt.return_value = "completed-job"

        action = AsyncMock()

        await durable_interrupt(action, lambda r: r)

        mock_interrupt.assert_called_once_with(None)

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_returns_resume_value(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        scratchpad = FakeScratchpad(resume=["completed-job"])
        mock_get_config.return_value = _make_config(scratchpad)
        mock_interrupt.return_value = "completed-job"

        action = AsyncMock()

        result = await durable_interrupt(action, lambda r: r)

        assert result == "completed-job"


class TestNoPregelContext:
    """No Pregel context (RuntimeError from get_config): action runs normally."""

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG, side_effect=RuntimeError("Not in Pregel context"))
    async def test_runs_action(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        mock_interrupt.side_effect = lambda v: v

        action = AsyncMock(return_value="result")

        await durable_interrupt(action, lambda r: {"payload": r})

        action.assert_called_once()

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG, side_effect=RuntimeError("Not in Pregel context"))
    async def test_interrupts_with_transformed_value(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        mock_interrupt.side_effect = lambda v: v

        action = AsyncMock(return_value="result")

        await durable_interrupt(action, lambda r: {"payload": r})

        mock_interrupt.assert_called_once_with({"payload": "result"})


class TestActionException:
    """Action exception propagates without calling interrupt."""

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_propagates_exception(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)

        action = AsyncMock(side_effect=ConnectionError("network fail"))

        with pytest.raises(ConnectionError, match="network fail"):
            await durable_interrupt(action, lambda r: r)

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_interrupt_not_called(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)

        action = AsyncMock(side_effect=ConnectionError("network fail"))

        with pytest.raises(ConnectionError):
            await durable_interrupt(action, lambda r: r)

        mock_interrupt.assert_not_called()


class TestMultipleCallsInSameNode:
    """Multiple durable_interrupt calls in one node execution."""

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_partial_resume_skips_first_runs_second(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        """One resume value: first call (idx 0) skips, second call (idx 1) runs."""
        scratchpad = FakeScratchpad(resume=["result-A"])
        mock_get_config.return_value = _make_config(scratchpad)
        mock_interrupt.side_effect = lambda v: v

        action_a = AsyncMock()
        action_b = AsyncMock(return_value="job-B")

        await durable_interrupt(action_a, lambda r: {"wait": r})
        await durable_interrupt(action_b, lambda r: {"wait": r})

        action_a.assert_not_called()
        action_b.assert_called_once()

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_partial_resume_interrupt_arguments(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        """First call: interrupt(None). Second call: interrupt(transformed value)."""
        scratchpad = FakeScratchpad(resume=["result-A"])
        mock_get_config.return_value = _make_config(scratchpad)
        mock_interrupt.side_effect = lambda v: v

        action_b = AsyncMock(return_value="job-B")

        await durable_interrupt(AsyncMock(), lambda r: {"wait": r})
        await durable_interrupt(action_b, lambda r: {"wait": r})

        calls = mock_interrupt.call_args_list
        assert calls[0].args == (None,)
        assert calls[1].args == ({"wait": "job-B"},)

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_full_resume_skips_all_actions(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        """Two resume values: both calls (idx 0, idx 1) skip their actions."""
        scratchpad = FakeScratchpad(resume=["result-A", "result-B"])
        mock_get_config.return_value = _make_config(scratchpad)
        mock_interrupt.side_effect = lambda v: v

        action_a = AsyncMock()
        action_b = AsyncMock()

        await durable_interrupt(action_a, lambda r: r)
        await durable_interrupt(action_b, lambda r: r)

        action_a.assert_not_called()
        action_b.assert_not_called()


class TestIndexResetOnScratchpadChange:
    """Index auto-resets when scratchpad identity changes (new node execution)."""

    @patch(PATCH_INTERRUPT)
    @patch(PATCH_GET_CONFIG)
    async def test_new_scratchpad_resets_index_to_zero(
        self, mock_get_config: MagicMock, mock_interrupt: MagicMock
    ) -> None:
        """After calls with one scratchpad, a different scratchpad restarts at idx 0.

        Without reset: counter stays at 2, so idx=2 >= len(resume)=1 -> runs action.
        With reset: counter resets to 0, so idx=0 < len(resume)=1 -> skips action.
        """
        mock_interrupt.side_effect = lambda v: v

        # Node 1: two calls advance internal counter to 2
        scratchpad_1 = FakeScratchpad(resume=["result-A"])
        mock_get_config.return_value = _make_config(scratchpad_1)
        await durable_interrupt(AsyncMock(), lambda r: r)  # idx=0, skipped
        await durable_interrupt(AsyncMock(return_value="x"), lambda r: r)  # idx=1, runs

        # Node 2: different scratchpad, index must reset to 0
        scratchpad_2 = FakeScratchpad(resume=["result-B"])
        mock_get_config.return_value = _make_config(scratchpad_2)

        action_new = AsyncMock()
        await durable_interrupt(action_new, lambda r: r)

        action_new.assert_not_called()
