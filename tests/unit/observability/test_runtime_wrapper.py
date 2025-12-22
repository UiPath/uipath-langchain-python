"""Tests for TelemetryRuntimeWrapper."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from uipath_agents._observability.callback import UiPathTracingCallback
from uipath_agents._observability.runtime_wrapper import TelemetryRuntimeWrapper
from uipath_agents._observability.tracer import UiPathTracer


@pytest.fixture
def mock_delegate():
    """Create a mock delegate runtime."""
    delegate = AsyncMock()
    delegate.entrypoint = "test-agent"
    delegate.runtime_id = "test-id"
    delegate._get_trace_prompts = MagicMock(return_value=(None, None))
    return delegate


@pytest.fixture
def tracer():
    """Create a tracer."""
    return UiPathTracer()


@pytest.fixture
def callback(tracer):
    """Create a callback."""
    return UiPathTracingCallback(tracer)


class TestTelemetryRuntimeWrapper:
    """Test TelemetryRuntimeWrapper core functionality."""

    def test_init_stores_delegate_tracer_callback(
        self, mock_delegate, tracer, callback
    ):
        """Test initialization stores all dependencies."""
        wrapper = TelemetryRuntimeWrapper(mock_delegate, tracer, callback)

        assert wrapper.delegate is mock_delegate
        assert wrapper._tracer is tracer
        assert wrapper._callback is callback

    @pytest.mark.asyncio
    async def test_execute_calls_set_agent_span(self, mock_delegate, tracer, callback):
        """Test execute calls set_agent_span on callback before execution."""
        mock_result = MagicMock()
        mock_result.status.name = "FAILED"
        mock_delegate.execute.return_value = mock_result

        agent_span_set = None

        def capture_agent_span(span):
            nonlocal agent_span_set
            agent_span_set = span

        callback.set_agent_span = MagicMock(side_effect=capture_agent_span)

        wrapper = TelemetryRuntimeWrapper(mock_delegate, tracer, callback)
        await wrapper.execute({"input": "test"}, None)

        # set_agent_span should have been called with a span
        callback.set_agent_span.assert_called_once()
        assert agent_span_set is not None

    @pytest.mark.asyncio
    async def test_stream_calls_set_agent_span(self, mock_delegate, tracer, callback):
        """Test stream calls set_agent_span on callback."""
        agent_span_set = None

        def capture_agent_span(span):
            nonlocal agent_span_set
            agent_span_set = span

        callback.set_agent_span = MagicMock(side_effect=capture_agent_span)

        async def mock_stream(*args, **kwargs):
            yield "event1"

        mock_delegate.stream = mock_stream

        wrapper = TelemetryRuntimeWrapper(mock_delegate, tracer, callback)
        events = [e async for e in wrapper.stream({"input": "test"}, None)]

        assert events == ["event1"]
        callback.set_agent_span.assert_called_once()
        assert agent_span_set is not None

    @pytest.mark.asyncio
    async def test_get_schema_and_dispose_delegate(
        self, mock_delegate, tracer, callback
    ):
        """Test get_schema and dispose pass through to delegate."""
        mock_schema = MagicMock()
        mock_delegate.get_schema.return_value = mock_schema

        wrapper = TelemetryRuntimeWrapper(mock_delegate, tracer, callback)

        assert await wrapper.get_schema() is mock_schema
        await wrapper.dispose()
        mock_delegate.dispose.assert_called_once()

    def test_metadata_extraction(self, tracer, callback):
        """Test agent name and prompts extraction from delegate."""
        # With entrypoint
        mock_delegate = MagicMock()
        mock_delegate.entrypoint = "my-agent.json"
        mock_delegate._get_trace_prompts.return_value = ("system", "user")

        wrapper = TelemetryRuntimeWrapper(mock_delegate, tracer, callback)
        assert wrapper._get_agent_name() == "my-agent.json"
        assert wrapper._get_prompts() == ("system", "user")

        # Without entrypoint - falls back to unknown
        mock_delegate_empty = MagicMock(spec=[])
        wrapper_empty = TelemetryRuntimeWrapper(mock_delegate_empty, tracer, callback)
        assert wrapper_empty._get_agent_name() == "unknown"
        assert wrapper_empty._get_prompts() == (None, None)

    @pytest.mark.asyncio
    async def test_same_callback_used_across_executions(
        self, mock_delegate, tracer, callback
    ):
        """Same callback instance is used across multiple executions (debug/chat scenario)."""
        mock_result = MagicMock()
        mock_result.status.name = "FAILED"
        mock_delegate.execute.return_value = mock_result

        callback.set_agent_span = MagicMock()

        wrapper = TelemetryRuntimeWrapper(mock_delegate, tracer, callback)

        # Multiple executions (simulating debug/chat re-execution)
        await wrapper.execute({"input": "1"}, None)
        await wrapper.execute({"input": "2"}, None)
        await wrapper.execute({"input": "3"}, None)

        # set_agent_span called 3 times, once per execution
        assert callback.set_agent_span.call_count == 3

    @pytest.mark.asyncio
    async def test_cleanup_called_after_execution(
        self, mock_delegate, tracer, callback
    ):
        """Callback cleanup is called after each execution."""
        mock_result = MagicMock()
        mock_result.status.name = "FAILED"
        mock_delegate.execute.return_value = mock_result

        callback.cleanup = MagicMock()

        wrapper = TelemetryRuntimeWrapper(mock_delegate, tracer, callback)
        await wrapper.execute({"input": "test"}, None)

        callback.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_called_after_exception(
        self, mock_delegate, tracer, callback
    ):
        """Cleanup is called even when execution raises exception."""
        mock_delegate.execute.side_effect = RuntimeError("Test error")

        callback.cleanup = MagicMock()

        wrapper = TelemetryRuntimeWrapper(mock_delegate, tracer, callback)

        with pytest.raises(RuntimeError):
            await wrapper.execute({"input": "test"}, None)

        # Cleanup should still be called
        callback.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_executions_share_callback(
        self, mock_delegate, tracer, callback
    ):
        """Concurrent executions use the same callback instance."""
        mock_result = MagicMock()
        mock_result.status.name = "FAILED"

        call_count = 0

        async def delayed_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return mock_result

        mock_delegate.execute.side_effect = delayed_execute
        callback.set_agent_span = MagicMock()

        wrapper = TelemetryRuntimeWrapper(mock_delegate, tracer, callback)

        # Run concurrent executions
        await asyncio.gather(
            wrapper.execute({"input": "A"}, None),
            wrapper.execute({"input": "B"}, None),
            wrapper.execute({"input": "C"}, None),
        )

        assert call_count == 3
        # Same callback instance used for all
        assert callback.set_agent_span.call_count == 3


class TestCallbackPersistence:
    """Tests verifying callback persists across re-executions (debug/chat scenario)."""

    @pytest.mark.asyncio
    async def test_callback_persists_for_debug_reexecution(
        self, mock_delegate, tracer, callback
    ):
        """Simulate debug scenario: same runtime re-executed at breakpoints."""
        mock_result = MagicMock()
        mock_result.status.name = "FAILED"
        mock_delegate.execute.return_value = mock_result

        wrapper = TelemetryRuntimeWrapper(mock_delegate, tracer, callback)

        # Simulate debug: pause at breakpoint, resume, pause again
        spans_per_execution = []

        original_set_agent_span = callback.set_agent_span

        def track_spans(span):
            original_set_agent_span(span)
            spans_per_execution.append(span)

        callback.set_agent_span = track_spans

        # First execution (stopped at breakpoint)
        await wrapper.execute({"input": "step1"}, None)
        # Resume (re-execute)
        await wrapper.execute({"input": "step2"}, None)
        # Resume again
        await wrapper.execute({"input": "step3"}, None)

        # Each execution got its own agent span
        assert len(spans_per_execution) == 3
        # But all used the same callback instance (verified by the fact we're here)

    @pytest.mark.asyncio
    async def test_callback_persists_for_hitl_chat(
        self, mock_delegate, tracer, callback
    ):
        """Simulate chat HITL: tool needs approval, runtime re-executed after approval."""
        suspended_result = MagicMock()
        suspended_result.status.name = "SUSPENDED"

        success_result = MagicMock()
        success_result.status.name = "SUCCESSFUL"
        success_result.output = {"response": "done"}

        # First call suspends (needs HITL approval), second succeeds
        mock_delegate.execute.side_effect = [suspended_result, success_result]

        wrapper = TelemetryRuntimeWrapper(mock_delegate, tracer, callback)

        # Initial execution - suspends for HITL
        result1 = await wrapper.execute({"input": "initial"}, None)
        assert result1.status.name == "SUSPENDED"

        # User approves, re-execute with resume
        result2 = await wrapper.execute({"input": "approved"}, None)
        assert result2.status.name == "SUCCESSFUL"

        # Both executions used same wrapper (and thus same callback)
        assert mock_delegate.execute.call_count == 2
