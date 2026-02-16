"""Contract tests for LangGraph internals that durable_interrupt depends on.

These tests verify assumptions about LangGraph's interrupt/resume mechanism.
If any of these fail after a LangGraph upgrade, durable_interrupt.py likely
needs to be updated to match the new behavior.

Verified assumptions:
- CONFIG_KEY_SCRATCHPAD is the key for the scratchpad in configurable
- PregelScratchpad has a .resume list and .interrupt_counter callable
- interrupt() uses sequential indexing via interrupt_counter
- interrupt() returns resume[idx] when resume values exist
- interrupt() raises GraphInterrupt when no resume value is available
- get_config() raises RuntimeError outside a runnable context
"""

import dataclasses
import itertools
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.runnables.config import (
    RunnableConfig,
    var_child_runnable_config,
)
from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.config import get_config
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt


class TestScratchpadContract:
    """PregelScratchpad must have the fields durable_interrupt reads."""

    def test_config_key_scratchpad_value(self) -> None:
        """CONFIG_KEY_SCRATCHPAD is the interned string we expect."""
        assert CONFIG_KEY_SCRATCHPAD == "__pregel_scratchpad"

    def test_scratchpad_has_resume_field(self) -> None:
        """PregelScratchpad has a .resume attribute."""
        assert hasattr(PregelScratchpad, "resume")
        fields = {f.name for f in dataclasses.fields(PregelScratchpad)}
        assert "resume" in fields

    def test_scratchpad_has_interrupt_counter_field(self) -> None:
        """PregelScratchpad has an .interrupt_counter callable field."""
        fields = {f.name for f in dataclasses.fields(PregelScratchpad)}
        assert "interrupt_counter" in fields

    def test_scratchpad_resume_is_list(self) -> None:
        """Scratchpad.resume is a list (we index into it with idx < len(resume))."""
        sp = _make_scratchpad(resume=["val-0", "val-1"])
        assert isinstance(sp.resume, list)
        assert sp.resume[0] == "val-0"
        assert len(sp.resume) == 2

    def test_scratchpad_resume_empty_is_falsy(self) -> None:
        """Empty resume list is falsy (we rely on `bool(scratchpad.resume)`)."""
        sp = _make_scratchpad(resume=[])
        assert not sp.resume

    def test_scratchpad_resume_non_empty_is_truthy(self) -> None:
        """Non-empty resume list is truthy."""
        sp = _make_scratchpad(resume=["val"])
        assert sp.resume


class TestGetConfigContract:
    """get_config() must raise RuntimeError outside a runnable context."""

    def test_raises_runtime_error_outside_context(self) -> None:
        """get_config() raises RuntimeError when not inside a LangGraph node."""
        with pytest.raises(RuntimeError):
            get_config()

    def test_returns_config_inside_context(self) -> None:
        """get_config() returns config when var_child_runnable_config is set."""
        config: RunnableConfig = {"configurable": {"test_key": "test_value"}}
        token = var_child_runnable_config.set(config)
        try:
            result = get_config()
            assert result["configurable"]["test_key"] == "test_value"
        finally:
            var_child_runnable_config.reset(token)

    def test_scratchpad_accessible_via_config_key(self) -> None:
        """Scratchpad is stored under CONFIG_KEY_SCRATCHPAD in configurable."""
        sp = _make_scratchpad(resume=["value"])
        config: RunnableConfig = {"configurable": {CONFIG_KEY_SCRATCHPAD: sp}}
        token = var_child_runnable_config.set(config)
        try:
            result = get_config()
            retrieved = result["configurable"][CONFIG_KEY_SCRATCHPAD]
            assert retrieved is sp
            assert retrieved.resume == ["value"]
        finally:
            var_child_runnable_config.reset(token)


class TestInterruptResumeContract:
    """interrupt() must use sequential indexing and return resume[idx]."""

    def test_interrupt_raises_graph_interrupt_on_first_run(self) -> None:
        """First execution with no resume values raises GraphInterrupt."""
        sp = _make_scratchpad(resume=[])
        config = _make_runnable_config(sp)
        token = var_child_runnable_config.set(config)
        try:
            with pytest.raises(GraphInterrupt):
                interrupt("wait-value")
        finally:
            var_child_runnable_config.reset(token)

    def test_interrupt_returns_resume_value_by_index(self) -> None:
        """On resume, interrupt() returns resume[idx] for the matching index."""
        sp = _make_scratchpad(resume=["completed-job"])
        config = _make_runnable_config(sp)
        token = var_child_runnable_config.set(config)
        try:
            result = interrupt("ignored-on-resume")
            assert result == "completed-job"
        finally:
            var_child_runnable_config.reset(token)

    def test_interrupt_sequential_indexing_multiple_calls(self) -> None:
        """Multiple interrupt() calls consume resume values sequentially."""
        sp = _make_scratchpad(resume=["result-A", "result-B", "result-C"])
        config = _make_runnable_config(sp)
        token = var_child_runnable_config.set(config)
        try:
            assert interrupt("val-0") == "result-A"
            assert interrupt("val-1") == "result-B"
            assert interrupt("val-2") == "result-C"
        finally:
            var_child_runnable_config.reset(token)

    def test_interrupt_raises_after_resume_values_exhausted(self) -> None:
        """After all resume values consumed, next interrupt() raises GraphInterrupt."""
        sp = _make_scratchpad(resume=["result-A"])
        config = _make_runnable_config(sp)
        token = var_child_runnable_config.set(config)
        try:
            result = interrupt("val-0")
            assert result == "result-A"

            with pytest.raises(GraphInterrupt):
                interrupt("val-1")
        finally:
            var_child_runnable_config.reset(token)

    def test_interrupt_counter_starts_at_zero(self) -> None:
        """interrupt_counter starts at 0 for a fresh scratchpad."""
        counter = itertools.count(0).__next__
        assert counter() == 0
        assert counter() == 1
        assert counter() == 2


class TestDurableInterruptAlignment:
    """Verify that durable_interrupt's index stays in sync with interrupt()'s counter.

    durable_interrupt calls interrupt() internally. On resume it calls
    interrupt(None) which consumes the resume value at the matching index.
    The decorated function returns the resume value directly.
    """

    def test_single_durable_interrupt_returns_resume_value(self) -> None:
        """On resume, durable_interrupt returns the resume value directly."""
        from uipath_langchain.agent.tools.durable_interrupt import (
            _durable_state,
            durable_interrupt,
        )

        sp = _make_scratchpad(resume=["job-result"])
        config = _make_runnable_config(sp)

        state_token = _durable_state.set(None)
        config_token = var_child_runnable_config.set(config)
        try:

            @durable_interrupt
            def create_job() -> dict[str, str]:
                raise AssertionError("body should not run on resume")

            # durable_interrupt skips body, calls interrupt(None) internally,
            # returns resume[0]
            result = create_job()
            assert result == "job-result"
        finally:
            var_child_runnable_config.reset(config_token)
            _durable_state.reset(state_token)

    def test_two_durable_interrupts_return_sequential_resume_values(self) -> None:
        """Two @durable_interrupt calls return resume values by index."""
        from uipath_langchain.agent.tools.durable_interrupt import (
            _durable_state,
            durable_interrupt,
        )

        sp = _make_scratchpad(resume=["result-A", "result-B"])
        config = _make_runnable_config(sp)

        state_token = _durable_state.set(None)
        config_token = var_child_runnable_config.set(config)
        try:

            @durable_interrupt
            def task_a() -> str:
                raise AssertionError("task_a body should not run")

            @durable_interrupt
            def task_b() -> str:
                raise AssertionError("task_b body should not run")

            assert task_a() == "result-A"
            assert task_b() == "result-B"
        finally:
            var_child_runnable_config.reset(config_token)
            _durable_state.reset(state_token)

    def test_partial_resume_first_returns_value_second_raises(self) -> None:
        """One resume value: first returns it, second runs body and raises GraphInterrupt."""
        from uipath_langchain.agent.tools.durable_interrupt import (
            _durable_state,
            durable_interrupt,
        )

        sp = _make_scratchpad(resume=["result-A"])
        config = _make_runnable_config(sp)

        state_token = _durable_state.set(None)
        config_token = var_child_runnable_config.set(config)
        try:

            @durable_interrupt
            def task_a() -> str:
                raise AssertionError("task_a body should not run")

            @durable_interrupt
            def task_b() -> str:
                return "new-job-B"

            # First: resumed — returns resume value
            assert task_a() == "result-A"

            # Second: not resumed — runs body, interrupt("new-job-B") raises
            with pytest.raises(GraphInterrupt):
                task_b()
        finally:
            var_child_runnable_config.reset(config_token)
            _durable_state.reset(state_token)

    async def test_async_durable_interrupt_returns_resume_value(self) -> None:
        """Async variant: durable_interrupt returns resume value directly."""
        from uipath_langchain.agent.tools.durable_interrupt import (
            _durable_state,
            durable_interrupt,
        )

        sp = _make_scratchpad(resume=["async-result"])
        config = _make_runnable_config(sp)

        state_token = _durable_state.set(None)
        config_token = var_child_runnable_config.set(config)
        try:

            @durable_interrupt
            async def create_job() -> str:
                raise AssertionError("body should not run on resume")

            result = await create_job()
            assert result == "async-result"
        finally:
            var_child_runnable_config.reset(config_token)
            _durable_state.reset(state_token)


# -- Helpers ------------------------------------------------------------------


def _make_scratchpad(resume: list[Any] | None = None) -> PregelScratchpad:
    """Build a real PregelScratchpad with sensible defaults."""
    return PregelScratchpad(
        step=0,
        stop=1,
        call_counter=itertools.count(0).__next__,
        interrupt_counter=itertools.count(0).__next__,
        get_null_resume=lambda _: None,
        resume=resume or [],
        subgraph_counter=itertools.count(0).__next__,
    )


def _make_runnable_config(scratchpad: PregelScratchpad) -> RunnableConfig:
    """Build a RunnableConfig with scratchpad and required keys for interrupt()."""
    mock_send = MagicMock()
    return {
        "configurable": {
            CONFIG_KEY_SCRATCHPAD: scratchpad,
            "checkpoint_ns": "",
            "__pregel_send": mock_send,
        },
    }
