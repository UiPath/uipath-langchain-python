"""Tests for the durable_task decorator."""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD

from uipath_langchain.agent.tools.durable_interrupt import (
    _durable_state,
    durable_task,
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


@pytest.fixture(autouse=True)
def _reset_durable_state() -> Generator[None]:
    """Reset per-node counter between tests for isolation."""
    token = _durable_state.set(None)
    yield
    _durable_state.reset(token)


class TestAsyncFirstExecution:
    """Async first execution (no resume values): body runs and returns result."""

    @patch(PATCH_GET_CONFIG)
    async def test_runs_function_body(self, mock_get_config: MagicMock) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)

        action = AsyncMock(return_value="job-123")

        @durable_task
        async def start_job() -> str:
            return await action()

        await start_job()

        action.assert_called_once()

    @patch(PATCH_GET_CONFIG)
    async def test_returns_function_result(self, mock_get_config: MagicMock) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)

        @durable_task
        async def start_job() -> dict[str, str]:
            return {"wait": "job-123"}

        result = await start_job()

        assert result == {"wait": "job-123"}


class TestAsyncResumeExecution:
    """Async resume execution (resume value exists): function skipped, returns None."""

    @patch(PATCH_GET_CONFIG)
    async def test_skips_function_body(self, mock_get_config: MagicMock) -> None:
        scratchpad = FakeScratchpad(resume=["completed-job"])
        mock_get_config.return_value = _make_config(scratchpad)

        action = AsyncMock()

        @durable_task
        async def start_job() -> Any:
            return await action()

        await start_job()

        action.assert_not_called()

    @patch(PATCH_GET_CONFIG)
    async def test_returns_none(self, mock_get_config: MagicMock) -> None:
        scratchpad = FakeScratchpad(resume=["completed-job"])
        mock_get_config.return_value = _make_config(scratchpad)

        @durable_task
        async def start_job() -> str:
            return "should-not-reach"

        result = await start_job()

        assert result is None


class TestSyncFirstExecution:
    """Sync first execution (no resume values): body runs and returns result."""

    @patch(PATCH_GET_CONFIG)
    def test_runs_function_body(self, mock_get_config: MagicMock) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)

        action = MagicMock(return_value="job-123")

        @durable_task
        def start_job() -> str:
            return action()

        start_job()

        action.assert_called_once()

    @patch(PATCH_GET_CONFIG)
    def test_returns_function_result(self, mock_get_config: MagicMock) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)

        @durable_task
        def start_job() -> dict[str, str]:
            return {"wait": "job-123"}

        result = start_job()

        assert result == {"wait": "job-123"}


class TestSyncResumeExecution:
    """Sync resume execution (resume value exists): function skipped, returns None."""

    @patch(PATCH_GET_CONFIG)
    def test_skips_function_body(self, mock_get_config: MagicMock) -> None:
        scratchpad = FakeScratchpad(resume=["completed-job"])
        mock_get_config.return_value = _make_config(scratchpad)

        action = MagicMock()

        @durable_task
        def start_job() -> Any:
            return action()

        start_job()

        action.assert_not_called()

    @patch(PATCH_GET_CONFIG)
    def test_returns_none(self, mock_get_config: MagicMock) -> None:
        scratchpad = FakeScratchpad(resume=["completed-job"])
        mock_get_config.return_value = _make_config(scratchpad)

        @durable_task
        def start_job() -> str:
            return "should-not-reach"

        result = start_job()

        assert result is None


class TestNoPregelContext:
    """No Pregel context (RuntimeError from get_config): function runs normally."""

    @patch(PATCH_GET_CONFIG, side_effect=RuntimeError("Not in Pregel context"))
    async def test_async_runs_function_body(self, mock_get_config: MagicMock) -> None:
        action = AsyncMock(return_value="result")

        @durable_task
        async def start_job() -> dict[str, str]:
            r = await action()
            return {"payload": r}

        result = await start_job()

        action.assert_called_once()
        assert result == {"payload": "result"}

    @patch(PATCH_GET_CONFIG, side_effect=RuntimeError("Not in Pregel context"))
    def test_sync_runs_function_body(self, mock_get_config: MagicMock) -> None:
        action = MagicMock(return_value="result")

        @durable_task
        def start_job() -> dict[str, str]:
            return {"payload": action()}

        result = start_job()

        action.assert_called_once()
        assert result == {"payload": "result"}


class TestFunctionException:
    """Function exception propagates normally."""

    @patch(PATCH_GET_CONFIG)
    async def test_async_propagates_exception(self, mock_get_config: MagicMock) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)

        @durable_task
        async def start_job() -> str:
            raise ConnectionError("network fail")

        with pytest.raises(ConnectionError, match="network fail"):
            await start_job()

    @patch(PATCH_GET_CONFIG)
    def test_sync_propagates_exception(self, mock_get_config: MagicMock) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)

        @durable_task
        def start_job() -> str:
            raise ConnectionError("network fail")

        with pytest.raises(ConnectionError, match="network fail"):
            start_job()


class TestMultipleCallsInSameNode:
    """Multiple @durable_task calls in one node execution."""

    @patch(PATCH_GET_CONFIG)
    async def test_partial_resume_skips_first_runs_second(
        self, mock_get_config: MagicMock
    ) -> None:
        """One resume value: first call (idx 0) skips, second call (idx 1) runs."""
        scratchpad = FakeScratchpad(resume=["result-A"])
        mock_get_config.return_value = _make_config(scratchpad)

        action_a = AsyncMock()
        action_b = AsyncMock(return_value="job-B")

        @durable_task
        async def task_a() -> Any:
            return await action_a()

        @durable_task
        async def task_b() -> dict[str, str]:
            r = await action_b()
            return {"wait": r}

        result_a = await task_a()
        result_b = await task_b()

        action_a.assert_not_called()
        action_b.assert_called_once()
        assert result_a is None
        assert result_b == {"wait": "job-B"}

    @patch(PATCH_GET_CONFIG)
    async def test_full_resume_skips_all_functions(
        self, mock_get_config: MagicMock
    ) -> None:
        """Two resume values: both calls (idx 0, idx 1) skip their functions."""
        scratchpad = FakeScratchpad(resume=["result-A", "result-B"])
        mock_get_config.return_value = _make_config(scratchpad)

        action_a = AsyncMock()
        action_b = AsyncMock()

        @durable_task
        async def task_a() -> Any:
            return await action_a()

        @durable_task
        async def task_b() -> Any:
            return await action_b()

        result_a = await task_a()
        result_b = await task_b()

        action_a.assert_not_called()
        action_b.assert_not_called()
        assert result_a is None
        assert result_b is None

    @patch(PATCH_GET_CONFIG)
    def test_sync_partial_resume_skips_first_runs_second(
        self, mock_get_config: MagicMock
    ) -> None:
        """Sync variant: one resume value, first skips, second runs."""
        scratchpad = FakeScratchpad(resume=["result-A"])
        mock_get_config.return_value = _make_config(scratchpad)

        action_a = MagicMock()
        action_b = MagicMock(return_value="job-B")

        @durable_task
        def task_a() -> Any:
            return action_a()

        @durable_task
        def task_b() -> str:
            return action_b()

        result_a = task_a()
        result_b = task_b()

        action_a.assert_not_called()
        action_b.assert_called_once()
        assert result_a is None
        assert result_b == "job-B"


class TestIndexResetOnScratchpadChange:
    """Index auto-resets when scratchpad identity changes (new node execution)."""

    @patch(PATCH_GET_CONFIG)
    async def test_new_scratchpad_resets_index_to_zero(
        self, mock_get_config: MagicMock
    ) -> None:
        """After calls with one scratchpad, a different scratchpad restarts at idx 0.

        Without reset: counter stays at 2, so idx=2 >= len(resume)=1 -> runs function.
        With reset: counter resets to 0, so idx=0 < len(resume)=1 -> skips function.
        """
        # Node 1: two calls advance internal counter to 2
        scratchpad_1 = FakeScratchpad(resume=["result-A"])
        mock_get_config.return_value = _make_config(scratchpad_1)

        @durable_task
        async def task_1a() -> str:
            return "skip"

        @durable_task
        async def task_1b() -> str:
            return "run"

        await task_1a()  # idx=0, skipped
        await task_1b()  # idx=1, runs

        # Node 2: different scratchpad, index must reset to 0
        scratchpad_2 = FakeScratchpad(resume=["result-B"])
        mock_get_config.return_value = _make_config(scratchpad_2)

        action_new = AsyncMock()

        @durable_task
        async def task_2() -> Any:
            return await action_new()

        await task_2()

        action_new.assert_not_called()


class TestDecoratorPreservesMetadata:
    """@durable_task preserves function name and docstring."""

    async def test_async_preserves_function_name(self) -> None:
        @durable_task
        async def my_special_function() -> str:
            return "value"

        assert my_special_function.__name__ == "my_special_function"

    async def test_async_preserves_docstring(self) -> None:
        @durable_task
        async def my_special_function() -> str:
            """This is the docstring."""
            return "value"

        assert my_special_function.__doc__ == "This is the docstring."

    def test_sync_preserves_function_name(self) -> None:
        @durable_task
        def my_special_function() -> str:
            return "value"

        assert my_special_function.__name__ == "my_special_function"

    def test_sync_preserves_docstring(self) -> None:
        @durable_task
        def my_special_function() -> str:
            """This is the docstring."""
            return "value"

        assert my_special_function.__doc__ == "This is the docstring."

    @patch(PATCH_GET_CONFIG)
    async def test_async_passes_arguments_through(
        self, mock_get_config: MagicMock
    ) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)

        received_args: dict[str, Any] = {}

        @durable_task
        async def start_job(name: str, count: int = 1) -> dict[str, Any]:
            received_args["name"] = name
            received_args["count"] = count
            return {"name": name, "count": count}

        await start_job("test", count=5)

        assert received_args == {"name": "test", "count": 5}

    @patch(PATCH_GET_CONFIG)
    def test_sync_passes_arguments_through(self, mock_get_config: MagicMock) -> None:
        scratchpad = FakeScratchpad(resume=[])
        mock_get_config.return_value = _make_config(scratchpad)

        received_args: dict[str, Any] = {}

        @durable_task
        def start_job(name: str, count: int = 1) -> dict[str, Any]:
            received_args["name"] = name
            received_args["count"] = count
            return {"name": name, "count": count}

        start_job("test", count=5)

        assert received_args == {"name": "test", "count": 5}


class TestWrapperTypeMatchesFunction:
    """@durable_task returns an async wrapper for async fns, sync for sync fns."""

    def test_async_function_produces_coroutine(self) -> None:
        import asyncio

        @durable_task
        async def my_async_fn() -> str:
            return "value"

        assert asyncio.iscoroutinefunction(my_async_fn)

    def test_sync_function_does_not_produce_coroutine(self) -> None:
        import asyncio

        @durable_task
        def my_sync_fn() -> str:
            return "value"

        assert not asyncio.iscoroutinefunction(my_sync_fn)
