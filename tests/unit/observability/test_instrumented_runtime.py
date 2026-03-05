"""Tests for InstrumentedRuntime."""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from uipath.agent.models.agent import (
    AgentDefinition,
    AgentMessage,
    AgentMessageRole,
    AgentMetadata,
    AgentSettings,
)
from uipath.runtime import UiPathRuntimeContext, UiPathRuntimeStatus

from uipath_agents._observability.instrumented_runtime import InstrumentedRuntime
from uipath_agents._observability.llmops.callback import LlmOpsInstrumentationCallback
from uipath_agents._observability.llmops.spans.span_factory import LlmOpsSpanFactory
from uipath_agents._observability.llmops.trace_context_storage import (
    PendingSpanData,
    TraceContextData,
)


@pytest.fixture
def mock_delegate():
    """Create a mock delegate runtime."""
    delegate = AsyncMock()
    delegate.entrypoint = "test-agent"
    delegate.runtime_id = "test-id"
    delegate._get_trace_prompts = MagicMock(return_value=(None, None))
    return delegate


@pytest.fixture
def mock_exporter():
    """Create a mock exporter for upsert tests."""
    from opentelemetry.sdk.trace.export import SpanExportResult

    exporter = MagicMock()
    exporter.upsert_span = MagicMock(return_value=SpanExportResult.SUCCESS)
    return exporter


@pytest.fixture
def tracer():
    """Create a tracer."""
    return LlmOpsSpanFactory()


@pytest.fixture
def tracer_with_exporter(span_exporter, mock_exporter):
    """Create a tracer with mock exporter and consistent global tracer state."""
    return LlmOpsSpanFactory(exporter=mock_exporter)


@pytest.fixture
def callback(tracer):
    """Create a callback."""
    return LlmOpsInstrumentationCallback(tracer)


@pytest.fixture
def callback_with_exporter(tracer_with_exporter):
    """Create a callback with exporter-enabled tracer."""
    return LlmOpsInstrumentationCallback(tracer_with_exporter)


@pytest.fixture
def mock_runtime_context():
    """Create a mock runtime context."""
    context = MagicMock(spec=UiPathRuntimeContext)
    context.command = "debug"
    context.org_id = "test-org-id"
    context.tenant_id = "test-tenant-id"
    context.job_id = "test-job-id"
    context.resume = False
    return context


class TestInstrumentedRuntime:
    """Test InstrumentedRuntime core functionality."""

    def test_init_stores_delegate_span_factory_callback(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Test initialization stores all dependencies."""
        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )

        assert instrumented_runtime.delegate is mock_delegate
        assert instrumented_runtime._span_factory is tracer
        assert instrumented_runtime._callback is callback

    @pytest.mark.asyncio
    async def test_execute_calls_set_agent_span(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Test execute calls set_agent_span on callback before execution."""
        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.FAULTED
        mock_delegate.execute.return_value = mock_result

        agent_span_set = None

        def capture_agent_span(span, run_id, prompts_captured=False):
            nonlocal agent_span_set
            agent_span_set = span

        callback.set_agent_span = MagicMock(side_effect=capture_agent_span)

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )
        await instrumented_runtime.execute({"input": "test"}, None)

        # set_agent_span should have been called with a span and run_id
        callback.set_agent_span.assert_called_once()
        assert agent_span_set is not None

    @pytest.mark.asyncio
    async def test_stream_calls_set_agent_span(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Test stream calls set_agent_span on callback."""
        agent_span_set = None

        def capture_agent_span(span, run_id, prompts_captured=False):
            nonlocal agent_span_set
            agent_span_set = span

        callback.set_agent_span = MagicMock(side_effect=capture_agent_span)

        async def mock_stream(*args, **kwargs):
            yield "event1"

        mock_delegate.stream = mock_stream

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )
        events = [e async for e in instrumented_runtime.stream({"input": "test"}, None)]

        assert events == ["event1"]
        callback.set_agent_span.assert_called_once()
        assert agent_span_set is not None

    @pytest.mark.asyncio
    async def test_get_schema_and_dispose_delegate(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Test get_schema and dispose pass through to delegate."""
        mock_schema = MagicMock()
        mock_delegate.get_schema.return_value = mock_schema

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )

        assert await instrumented_runtime.get_schema() is mock_schema
        await instrumented_runtime.dispose()
        mock_delegate.dispose.assert_called_once()

    def test_metadata_extraction(self, tracer, callback, mock_runtime_context):
        """Test agent name and prompts extraction from delegate."""
        # With agent_info provided
        mock_delegate = MagicMock()
        mock_delegate._get_trace_prompts.return_value = ("system", "user")

        agent_info = AgentDefinition(
            name="my-agent",
            messages=[],
            settings=AgentSettings(
                model="gpt-4o-2024-11-20", engine="v1", max_tokens=1000, temperature=0.7
            ),
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )
        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            agent_definition=agent_info,
        )
        assert instrumented_runtime._get_agent_name() == "my-agent"
        assert instrumented_runtime._get_prompts() == ("system", "user")
        assert instrumented_runtime._get_schemas() == (
            {"type": "object"},
            {"type": "string"},
        )

        # Without agent_info - falls back to runtime_id
        mock_delegate_with_id = MagicMock()
        mock_delegate_with_id.runtime_id = "test-runtime-id"
        instrumented_runtime_fallback = InstrumentedRuntime(
            mock_delegate_with_id, tracer, callback, mock_runtime_context
        )
        assert instrumented_runtime_fallback._get_agent_name() == "test-runtime-id"
        assert instrumented_runtime_fallback._get_schemas() == (None, None)

        # Without agent_info or runtime_id - falls back to unknown
        mock_delegate_empty = MagicMock(spec=[])
        instrumented_runtime_empty = InstrumentedRuntime(
            mock_delegate_empty, tracer, callback, mock_runtime_context
        )
        assert instrumented_runtime_empty._get_agent_name() == "unknown"
        assert instrumented_runtime_empty._get_prompts() == (None, None)

    def test_get_prompts_extracts_templates_from_messages(
        self, tracer, callback, mock_runtime_context
    ):
        mock_delegate = MagicMock(spec=[])  # No _get_trace_prompts method

        agent_info = AgentDefinition(
            name="template-agent",
            messages=[
                AgentMessage(
                    role=AgentMessageRole.SYSTEM,
                    content="You are a {{role}} assistant.",
                ),
                AgentMessage(
                    role=AgentMessageRole.USER,
                    content="Process this: {{input_string}}",
                ),
            ],
            settings=AgentSettings(
                model="gpt-4o-2024-11-20", engine="v1", max_tokens=1000, temperature=0.7
            ),
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )
        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            agent_definition=agent_info,
        )

        system_prompt, user_prompt = instrumented_runtime._get_prompts()

        assert system_prompt == "You are a {{role}} assistant."
        assert user_prompt == "Process this: {{input_string}}"

    @pytest.mark.asyncio
    async def test_same_callback_used_across_executions(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Same callback instance is used across multiple executions (debug/chat scenario)."""
        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.FAULTED
        mock_delegate.execute.return_value = mock_result

        callback.set_agent_span = MagicMock()

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )

        # Multiple executions (simulating debug/chat re-execution)
        await instrumented_runtime.execute({"input": "1"}, None)
        await instrumented_runtime.execute({"input": "2"}, None)
        await instrumented_runtime.execute({"input": "3"}, None)

        # set_agent_span called 3 times, once per execution
        assert callback.set_agent_span.call_count == 3

    @pytest.mark.asyncio
    async def test_cleanup_called_after_execution(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Callback cleanup is called after each execution."""
        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.FAULTED
        mock_delegate.execute.return_value = mock_result

        callback.cleanup = MagicMock()

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )
        await instrumented_runtime.execute({"input": "test"}, None)

        callback.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_called_after_exception(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Cleanup is called even when execution raises exception."""
        mock_delegate.execute.side_effect = RuntimeError("Test error")

        callback.cleanup = MagicMock()

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )

        with pytest.raises(RuntimeError):
            await instrumented_runtime.execute({"input": "test"}, None)

        # Cleanup should still be called
        callback.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_executions_share_callback(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Concurrent executions use the same callback instance."""
        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.FAULTED

        call_count = 0

        async def delayed_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return mock_result

        mock_delegate.execute.side_effect = delayed_execute
        callback.set_agent_span = MagicMock()

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )

        # Run concurrent executions
        await asyncio.gather(
            instrumented_runtime.execute({"input": "A"}, None),
            instrumented_runtime.execute({"input": "B"}, None),
            instrumented_runtime.execute({"input": "C"}, None),
        )

        assert call_count == 3
        # Same callback instance used for all
        assert callback.set_agent_span.call_count == 3


class TestCallbackPersistence:
    """Tests verifying callback persists across re-executions (debug/chat scenario)."""

    @pytest.mark.asyncio
    async def test_callback_persists_for_debug_reexecution(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Simulate debug scenario: same runtime re-executed at breakpoints."""
        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.FAULTED
        mock_delegate.execute.return_value = mock_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )

        # Simulate debug: pause at breakpoint, resume, pause again
        spans_per_execution = []

        original_set_agent_span = callback.set_agent_span

        def track_spans(span, run_id, prompts_captured=False):
            original_set_agent_span(span, run_id, prompts_captured)
            spans_per_execution.append(span)

        callback.set_agent_span = track_spans

        # First execution (stopped at breakpoint)
        await instrumented_runtime.execute({"input": "step1"}, None)
        # Resume (re-execute)
        await instrumented_runtime.execute({"input": "step2"}, None)
        # Resume again
        await instrumented_runtime.execute({"input": "step3"}, None)

        # Each execution got its own agent span
        assert len(spans_per_execution) == 3
        # But all used the same callback instance (verified by the fact we're here)

    @pytest.mark.asyncio
    async def test_callback_persists_for_hitl_chat(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Simulate chat HITL: tool needs approval, runtime re-executed after approval."""
        suspended_result = MagicMock()
        suspended_result.status = UiPathRuntimeStatus.SUSPENDED

        success_result = MagicMock()
        success_result.status = UiPathRuntimeStatus.SUCCESSFUL
        success_result.output = {"response": "done"}

        # First call suspends (needs HITL approval), second succeeds
        mock_delegate.execute.side_effect = [suspended_result, success_result]

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )

        # Initial execution - suspends for HITL
        result1 = await instrumented_runtime.execute({"input": "initial"}, None)
        assert result1.status == UiPathRuntimeStatus.SUSPENDED

        # User approves, re-execute with resume
        result2 = await instrumented_runtime.execute({"input": "approved"}, None)
        assert result2.status == UiPathRuntimeStatus.SUCCESSFUL

        # Both executions used same instrumented_runtime (and thus same callback)
        assert mock_delegate.execute.call_count == 2


@pytest.fixture
def mock_trace_context_storage():
    """Create a mock trace context storage with async methods."""
    storage = MagicMock()
    storage.load_trace_context = AsyncMock(return_value=None)
    storage.save_trace_context = AsyncMock()
    storage.clear_trace_context = AsyncMock()
    return storage


class TestInterruptibleTraceContext:
    """Tests for interruptible process trace context preservation."""

    @pytest.mark.asyncio
    async def test_init_accepts_trace_context_storage(
        self,
        mock_delegate,
        tracer,
        callback,
        mock_trace_context_storage,
        mock_runtime_context,
    ):
        """Test instrumented_runtime accepts optional trace_context_storage parameter."""
        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        assert instrumented_runtime._trace_context_storage is mock_trace_context_storage

    @pytest.mark.asyncio
    async def test_init_without_storage_works(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Test instrumented_runtime works without trace context storage (backward compatibility)."""
        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )

        assert instrumented_runtime._trace_context_storage is None

    @pytest.mark.asyncio
    async def test_suspended_saves_trace_context(
        self,
        mock_delegate,
        tracer,
        callback,
        mock_trace_context_storage,
        mock_runtime_context,
    ):
        """Test SUSPENDED result saves trace context for re-parenting."""
        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUSPENDED
        mock_delegate.execute.return_value = mock_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        await instrumented_runtime.execute({"input": "test"}, None)

        # Context should be saved for re-parenting on resume
        mock_trace_context_storage.save_trace_context.assert_called_once()
        call_args = mock_trace_context_storage.save_trace_context.call_args
        assert call_args[0][0] == "test-id"  # runtime_id
        saved_context = call_args[0][1]
        assert "trace_id" in saved_context
        assert "span_id" in saved_context

    @pytest.mark.asyncio
    async def test_successful_clears_trace_context(
        self,
        mock_delegate,
        tracer,
        callback,
        mock_trace_context_storage,
        mock_runtime_context,
    ):
        """Test SUCCESSFUL result clears saved trace context."""
        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUCCESSFUL
        mock_result.output = {"result": "done"}
        mock_delegate.execute.return_value = mock_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        await instrumented_runtime.execute({"input": "test"}, None)

        mock_trace_context_storage.clear_trace_context.assert_called_once_with(
            "test-id"
        )

    @pytest.mark.asyncio
    async def test_faulted_clears_trace_context(
        self,
        mock_delegate,
        tracer,
        callback,
        mock_trace_context_storage,
        mock_runtime_context,
    ):
        """Test FAULTED result clears saved trace context."""
        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.FAULTED
        mock_delegate.execute.return_value = mock_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        await instrumented_runtime.execute({"input": "test"}, None)

        mock_trace_context_storage.clear_trace_context.assert_called_once_with(
            "test-id"
        )

    @pytest.mark.asyncio
    async def test_resume_loads_saved_context(
        self,
        mock_delegate,
        tracer,
        callback,
        mock_trace_context_storage,
        mock_runtime_context,
    ):
        """Test resume execution loads saved trace context."""
        saved_context = TraceContextData(
            trace_id="abc123def456789012345678901234567890",
            span_id="1234567890123456",
            parent_span_id=None,
            name="Agent run - test-agent",
            start_time="2024-01-15T10:30:00Z",
            start_time_ns=0,
            attributes={"agentId": "test-id"},
            pending_tool_span_id=None,
            pending_process_span_id=None,
            pending_tool_name=None,
            pending_tool_span=None,
            pending_process_span=None,
            pending_escalation_span=None,
            pending_guardrail_hitl_evaluation_span=None,
            pending_guardrail_hitl_container_span=None,
            pending_llm_span=None,
        )
        mock_trace_context_storage.load_trace_context = AsyncMock(
            return_value=saved_context
        )

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUCCESSFUL
        mock_result.output = {"result": "done"}
        mock_delegate.execute.return_value = mock_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        await instrumented_runtime.execute({"input": "resume"}, None)

        # Should have loaded context
        mock_trace_context_storage.load_trace_context.assert_called_once_with("test-id")

    @pytest.mark.asyncio
    async def test_full_suspend_resume_flow(
        self, mock_delegate, mock_exporter, span_exporter, mock_runtime_context
    ):
        """Test complete suspend → resume → complete flow with re-parenting.

        Verifies referenceId from the original run propagates into the
        resumed agent span upsert (spans go through upsert, not span.end()).
        """
        from opentelemetry.sdk.trace.export import SpanExportResult

        mock_exporter.upsert_span = MagicMock(return_value=SpanExportResult.SUCCESS)
        tracer_with_exp = LlmOpsSpanFactory(exporter=mock_exporter)
        callback_with_exp = LlmOpsInstrumentationCallback(tracer_with_exp)

        agent_def = AgentDefinition(
            id="test-agent-id-123",
            name="test-agent",
            messages=[],
            settings=AgentSettings(
                model="gpt-4", engine="v1", max_tokens=1000, temperature=0.7
            ),
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )

        # Simulate storage behavior
        stored_context = None

        async def mock_save(runtime_id, context):
            nonlocal stored_context
            stored_context = context

        async def mock_load(runtime_id):
            return stored_context

        async def mock_clear(runtime_id):
            nonlocal stored_context
            stored_context = None

        storage = MagicMock()
        storage.save_trace_context = AsyncMock(side_effect=mock_save)
        storage.load_trace_context = AsyncMock(side_effect=mock_load)
        storage.clear_trace_context = AsyncMock(side_effect=mock_clear)

        # First execution: SUSPENDED
        suspended_result = MagicMock()
        suspended_result.status = UiPathRuntimeStatus.SUSPENDED

        # Second execution: SUCCESSFUL
        success_result = MagicMock()
        success_result.status = UiPathRuntimeStatus.SUCCESSFUL
        success_result.output = {"result": "done"}

        mock_delegate.execute.side_effect = [suspended_result, success_result]

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer_with_exp,
            callback_with_exp,
            mock_runtime_context,
            agent_definition=agent_def,
            trace_context_storage=storage,
        )

        # First execute - suspends
        result1 = await instrumented_runtime.execute({"input": "initial"}, None)
        assert result1.status == UiPathRuntimeStatus.SUSPENDED

        # Context should be saved for re-parenting
        assert stored_context is not None
        assert "trace_id" in stored_context

        # Second execute - resume (will re-parent to original trace)
        result2 = await instrumented_runtime.execute({"input": "resume"}, None)
        assert result2.status == UiPathRuntimeStatus.SUCCESSFUL

        # Context should be cleared after success
        assert stored_context is None

        # Verify referenceId propagates into the resumed agent span upsert.
        # Suspended/resumed spans go through upsert_span (not span.end()),
        # so we check the mock exporter calls instead of finished spans.
        upsert_calls = mock_exporter.upsert_span.call_args_list
        agent_upserts = [
            c
            for c in upsert_calls
            if hasattr(c[0][0], "attributes")
            and c[0][0].attributes
            and "referenceId" in c[0][0].attributes
        ]
        assert len(agent_upserts) >= 1, "Expected at least one upsert with referenceId"
        for call in agent_upserts:
            assert call[0][0].attributes["referenceId"] == "test-agent-id-123"

    @pytest.mark.asyncio
    async def test_no_storage_still_works(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Test instrumented_runtime works without storage (no upsert/save, just normal execution)."""
        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUSPENDED
        mock_delegate.execute.return_value = mock_result

        # No storage provided
        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )

        # Should not raise
        result = await instrumented_runtime.execute({"input": "test"}, None)
        assert result.status == UiPathRuntimeStatus.SUSPENDED

    @pytest.mark.asyncio
    async def test_stream_suspended_saves_context(
        self,
        mock_delegate,
        tracer,
        callback,
        mock_trace_context_storage,
        mock_runtime_context,
    ):
        """Test stream with SUSPENDED result saves trace context."""
        from uipath.runtime import UiPathRuntimeResult

        suspended_result = MagicMock(spec=UiPathRuntimeResult)
        suspended_result.status = UiPathRuntimeStatus.SUSPENDED

        async def mock_stream(*args, **kwargs):
            yield "event1"
            yield suspended_result

        mock_delegate.stream = mock_stream

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        events = [e async for e in instrumented_runtime.stream({"input": "test"}, None)]

        assert len(events) == 2
        mock_trace_context_storage.save_trace_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_reference_id_persists_across_suspend_resume(
        self, mock_delegate, tracer, callback, mock_runtime_context, span_exporter
    ):
        """Test that reference_id ContextVar persists across suspend/resume cycles."""
        # Set up storage
        stored_context = None

        async def mock_save(runtime_id, context):
            nonlocal stored_context
            stored_context = context

        async def mock_load(runtime_id):
            return stored_context

        async def mock_clear(runtime_id):
            nonlocal stored_context
            stored_context = None

        storage = MagicMock()
        storage.save_trace_context = AsyncMock(side_effect=mock_save)
        storage.load_trace_context = AsyncMock(side_effect=mock_load)
        storage.clear_trace_context = AsyncMock(side_effect=mock_clear)

        # Set up agent definition with agent_id
        agent_def = AgentDefinition(
            name="TestAgent",
            id="test-agent-id-123",
            messages=[],
            settings=AgentSettings(
                model="gpt-4",
                engine="v1",
                max_tokens=1000,
                temperature=0.7,
            ),
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )

        # First execution: SUSPENDED
        suspended_result = MagicMock()
        suspended_result.status = UiPathRuntimeStatus.SUSPENDED

        # Second execution: SUCCESSFUL
        success_result = MagicMock()
        success_result.status = UiPathRuntimeStatus.SUCCESSFUL
        success_result.output = {"result": "done"}

        mock_delegate.execute.side_effect = [suspended_result, success_result]

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            agent_definition=agent_def,
            trace_context_storage=storage,
        )

        # First execute - suspends
        result1 = await instrumented_runtime.execute({"input": "initial"}, None)
        assert result1.status == UiPathRuntimeStatus.SUSPENDED

        # Verify reference_id was saved in trace context
        assert stored_context is not None
        assert "attributes" in stored_context
        assert stored_context["attributes"]["referenceId"] == "test-agent-id-123"

        # Second execute - resume
        # This should restore the reference_id ContextVar
        result2 = await instrumented_runtime.execute({"input": "resume"}, None)
        assert result2.status == UiPathRuntimeStatus.SUCCESSFUL

        # The suspended agent span is intentionally not ended (no span.end()),
        # so it won't appear in finished spans. Verify reference_id was correctly
        # saved in trace context and resume completed successfully.
        assert stored_context is None  # cleared after successful resume


class TestSuspendedUsesResumedDataForNonRecordingSpans:
    """Tests that _handle_suspended prefers already-stored resumed span data
    over re-extracting from NonRecordingSpan instances (which lack rich data).
    """

    @pytest.mark.asyncio
    async def test_suspended_uses_resumed_container_data_instead_of_non_recording_span(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """After a resume, a second suspend should preserve the original rich
        guardrail container span data rather than extracting empty data from
        the NonRecordingSpan."""
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.trace import SpanContext, TraceFlags

        stored_context = None

        async def mock_save(runtime_id, context):
            nonlocal stored_context
            stored_context = context

        async def mock_load(runtime_id):
            return stored_context

        async def mock_clear(runtime_id):
            nonlocal stored_context
            stored_context = None

        storage = MagicMock()
        storage.save_trace_context = AsyncMock(side_effect=mock_save)
        storage.load_trace_context = AsyncMock(side_effect=mock_load)
        storage.clear_trace_context = AsyncMock(side_effect=mock_clear)

        # First execution: SUSPENDED
        suspended_result = MagicMock()
        suspended_result.status = UiPathRuntimeStatus.SUSPENDED
        mock_delegate.execute.return_value = suspended_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            trace_context_storage=storage,
        )

        # Set up escalation + guardrail HITL spans on callback to trigger
        # the escalation code path in _handle_suspended
        mock_escalation_span = MagicMock(spec=ReadableSpan)
        mock_escalation_span.get_span_context.return_value = MagicMock(
            span_id=0xAAAA, trace_id=0xBBBB
        )
        mock_escalation_span.name = "escalation"
        mock_escalation_span.attributes = {"escalationType": "Escalate"}
        mock_escalation_span.start_time = 1000
        mock_escalation_span.parent = None

        mock_eval_span = MagicMock(spec=ReadableSpan)
        mock_eval_span.get_span_context.return_value = MagicMock(
            span_id=0xCCCC, trace_id=0xBBBB
        )
        mock_eval_span.name = "guardrail_eval"
        mock_eval_span.attributes = {"guardrailName": "pii_guard"}
        mock_eval_span.start_time = 2000
        mock_eval_span.parent = None

        mock_container_span = MagicMock(spec=ReadableSpan)
        mock_container_span.get_span_context.return_value = MagicMock(
            span_id=0xDDDD, trace_id=0xBBBB
        )
        mock_container_span.name = "guardrail_container"
        mock_container_span.attributes = {
            "containerAttr": "rich_value",
            "guardrailName": "pii_guard",
        }
        mock_container_span.start_time = 3000
        mock_container_span.parent = MagicMock(span_id=0xEEEE)

        callback.get_pending_escalation = MagicMock(return_value=mock_escalation_span)
        callback.get_pending_guardrail_hitl_evaluation = MagicMock(
            return_value=mock_eval_span
        )
        callback.get_pending_guardrail_hitl_container = MagicMock(
            return_value=mock_container_span
        )
        callback.get_current_llm = MagicMock(return_value=None)
        callback.get_current_tool = MagicMock(return_value=None)
        callback.get_resumed_tool_data = MagicMock(return_value=None)
        callback.get_resumed_hitl_guardrail_container_data = MagicMock(
            return_value=None
        )
        callback.get_resumed_llm_data = MagicMock(return_value=None)

        # First suspension — extracts rich data from ReadableSpan
        await instrumented_runtime.execute({"input": "initial"}, None)

        assert stored_context is not None
        first_container_data = stored_context["pending_guardrail_hitl_container_span"]
        assert first_container_data is not None
        assert first_container_data["name"] == "guardrail_container"
        assert first_container_data["attributes"]["containerAttr"] == "rich_value"
        assert first_container_data["start_time_ns"] == 3000

        # Now simulate second suspension after a resume: the container span is a
        # NonRecordingSpan (no rich attributes), but resumed data is available.
        non_recording_ctx = SpanContext(
            trace_id=0xBBBB,
            span_id=0xDDDD,
            is_remote=True,
            trace_flags=TraceFlags(0x01),
        )
        non_recording_container = otel_trace.NonRecordingSpan(non_recording_ctx)

        callback.get_pending_guardrail_hitl_container = MagicMock(
            return_value=non_recording_container
        )
        # Simulate the resumed data being available from the first cycle
        callback.get_resumed_hitl_guardrail_container_data = MagicMock(
            return_value=first_container_data
        )

        # Reset storage load so it's treated as a fresh execution (not a resume)
        storage.load_trace_context = AsyncMock(return_value=None)

        # Second suspension
        await instrumented_runtime.execute({"input": "second_suspend"}, None)

        assert stored_context is not None
        second_container_data = stored_context["pending_guardrail_hitl_container_span"]
        assert second_container_data is not None
        # Should have the ORIGINAL rich data, not empty NonRecordingSpan data
        assert second_container_data["name"] == "guardrail_container"
        assert second_container_data["attributes"]["containerAttr"] == "rich_value"
        assert second_container_data["start_time_ns"] == 3000

    @pytest.mark.asyncio
    async def test_suspended_uses_resumed_llm_data_instead_of_non_recording_span(
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """After a resume, a second suspend should preserve the original rich
        LLM span data rather than extracting empty data from the NonRecordingSpan."""
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.trace import SpanContext, TraceFlags

        stored_context = None

        async def mock_save(runtime_id, context):
            nonlocal stored_context
            stored_context = context

        async def mock_load(runtime_id):
            return stored_context

        async def mock_clear(runtime_id):
            nonlocal stored_context
            stored_context = None

        storage = MagicMock()
        storage.save_trace_context = AsyncMock(side_effect=mock_save)
        storage.load_trace_context = AsyncMock(side_effect=mock_load)
        storage.clear_trace_context = AsyncMock(side_effect=mock_clear)

        # First execution: SUSPENDED
        suspended_result = MagicMock()
        suspended_result.status = UiPathRuntimeStatus.SUSPENDED
        mock_delegate.execute.return_value = suspended_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            trace_context_storage=storage,
        )

        # Set up escalation + guardrail HITL + LLM spans
        mock_escalation_span = MagicMock(spec=ReadableSpan)
        mock_escalation_span.get_span_context.return_value = MagicMock(
            span_id=0xAAAA, trace_id=0xBBBB
        )
        mock_escalation_span.name = "escalation"
        mock_escalation_span.attributes = {}
        mock_escalation_span.start_time = 1000
        mock_escalation_span.parent = None

        mock_eval_span = MagicMock(spec=ReadableSpan)
        mock_eval_span.get_span_context.return_value = MagicMock(
            span_id=0xCCCC, trace_id=0xBBBB
        )
        mock_eval_span.name = "guardrail_eval"
        mock_eval_span.attributes = {}
        mock_eval_span.start_time = 2000
        mock_eval_span.parent = None

        mock_container_span = MagicMock(spec=ReadableSpan)
        mock_container_span.get_span_context.return_value = MagicMock(
            span_id=0xDDDD, trace_id=0xBBBB
        )
        mock_container_span.name = "container"
        mock_container_span.attributes = {}
        mock_container_span.start_time = 3000
        mock_container_span.parent = None

        mock_llm_span = MagicMock(spec=ReadableSpan)
        mock_llm_span.get_span_context.return_value = MagicMock(
            span_id=0xFFFF, trace_id=0xBBBB
        )
        mock_llm_span.name = "llm_call"
        mock_llm_span.attributes = {
            "llm.model": "gpt-4o",
            "llm.token_count": 150,
        }
        mock_llm_span.start_time = 4000
        mock_llm_span.parent = MagicMock(span_id=0x1111)

        callback.get_pending_escalation = MagicMock(return_value=mock_escalation_span)
        callback.get_pending_guardrail_hitl_evaluation = MagicMock(
            return_value=mock_eval_span
        )
        callback.get_pending_guardrail_hitl_container = MagicMock(
            return_value=mock_container_span
        )
        callback.get_current_llm = MagicMock(return_value=mock_llm_span)
        callback.get_current_tool = MagicMock(return_value=None)
        callback.get_resumed_tool_data = MagicMock(return_value=None)
        callback.get_resumed_hitl_guardrail_container_data = MagicMock(
            return_value=None
        )
        callback.get_resumed_llm_data = MagicMock(return_value=None)

        # First suspension — extracts rich data from ReadableSpan
        await instrumented_runtime.execute({"input": "initial"}, None)

        assert stored_context is not None
        first_llm_data = stored_context["pending_llm_span"]
        assert first_llm_data is not None
        assert first_llm_data["name"] == "llm_call"
        assert first_llm_data["attributes"]["llm.model"] == "gpt-4o"
        assert first_llm_data["attributes"]["llm.token_count"] == 150
        assert first_llm_data["start_time_ns"] == 4000

        # Now simulate second suspension: LLM span is NonRecordingSpan
        non_recording_ctx = SpanContext(
            trace_id=0xBBBB,
            span_id=0xFFFF,
            is_remote=True,
            trace_flags=TraceFlags(0x01),
        )
        non_recording_llm = otel_trace.NonRecordingSpan(non_recording_ctx)

        callback.get_current_llm = MagicMock(return_value=non_recording_llm)
        # Simulate resumed data available from first cycle
        callback.get_resumed_llm_data = MagicMock(return_value=first_llm_data)

        # Reset storage load so it's treated as a fresh execution
        storage.load_trace_context = AsyncMock(return_value=None)

        # Second suspension
        await instrumented_runtime.execute({"input": "second_suspend"}, None)

        assert stored_context is not None
        second_llm_data = stored_context["pending_llm_span"]
        assert second_llm_data is not None
        # Should have the ORIGINAL rich data, not empty NonRecordingSpan data
        assert second_llm_data["name"] == "llm_call"
        assert second_llm_data["attributes"]["llm.model"] == "gpt-4o"
        assert second_llm_data["attributes"]["llm.token_count"] == 150
        assert second_llm_data["start_time_ns"] == 4000


class TestUpsertSpanOnSuspend:
    """Tests for upsert span calls during suspend."""

    @pytest.mark.asyncio
    async def test_handle_suspended_without_pending_spans_upserts_agent_span(
        self,
        mock_delegate,
        tracer_with_exporter,
        callback_with_exporter,
        mock_exporter,
        mock_trace_context_storage,
        mock_runtime_context,
    ):
        """Test _handle_suspended upserts agent span even without pending tool spans."""
        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUSPENDED
        mock_delegate.execute.return_value = mock_result

        # Mock get_pending_tool_info to return no pending spans
        callback_with_exporter.get_pending_tool_info = MagicMock(
            return_value=(None, None, None)
        )

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer_with_exporter,
            callback_with_exporter,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        await instrumented_runtime.execute({"input": "test"}, None)

        # 2 upserts: agent start (UNSET) + agent suspended (UNSET)
        # No final OK upsert — suspended spans must stay open
        assert mock_exporter.upsert_span.call_count == 2
        calls = mock_exporter.upsert_span.call_args_list
        assert calls[0][1]["status_override"] == 0  # UNSET (start)
        assert calls[1][1]["status_override"] == 0  # UNSET (suspended)

    @pytest.mark.asyncio
    async def test_handle_suspended_upserts_pending_tool_and_process_spans(
        self,
        mock_delegate,
        tracer_with_exporter,
        callback_with_exporter,
        mock_exporter,
        mock_trace_context_storage,
        mock_runtime_context,
    ):
        """Test _handle_suspended upserts pending process, tool, and agent spans."""
        from opentelemetry.sdk.trace import ReadableSpan

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUSPENDED
        mock_delegate.execute.return_value = mock_result

        mock_tool_span = MagicMock(spec=ReadableSpan)
        mock_tool_span.get_span_context.return_value = MagicMock(
            span_id=0x1234, trace_id=0xABCD
        )
        mock_tool_span.name = "tool"
        mock_tool_span.attributes = {}
        mock_tool_span.start_time = 0
        mock_tool_span.parent = None

        mock_process_span = MagicMock(spec=ReadableSpan)
        mock_process_span.get_span_context.return_value = MagicMock(
            span_id=0x5678, trace_id=0xABCD
        )
        mock_process_span.name = "process"
        mock_process_span.attributes = {}
        mock_process_span.start_time = 0
        mock_process_span.parent = None

        callback_with_exporter.get_pending_tool_info = MagicMock(
            return_value=("escalate_tool", mock_tool_span, mock_process_span)
        )

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer_with_exporter,
            callback_with_exporter,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        await instrumented_runtime.execute({"input": "test"}, None)

        # 4 upserts: agent start + process suspended + tool suspended + agent suspended
        # No final agent end — suspended spans must stay open
        assert mock_exporter.upsert_span.call_count == 4


class TestGetAgentModel:
    """Tests for get_agent_model delegation."""

    def test_get_agent_model_delegates_to_runtime(
        self, tracer, callback, mock_runtime_context
    ):
        """Test get_agent_model delegates to the wrapped runtime."""
        mock_delegate = MagicMock()
        mock_delegate.get_agent_model.return_value = "gpt-4o-2024-11-20"

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )
        model = instrumented_runtime.get_agent_model()

        assert model == "gpt-4o-2024-11-20"
        mock_delegate.get_agent_model.assert_called_once()

    def test_get_agent_model_returns_none_when_delegate_lacks_method(
        self, tracer, callback, mock_runtime_context
    ):
        """Test get_agent_model returns None when delegate doesn't have the method."""
        mock_delegate = MagicMock(spec=[])  # Empty spec, no get_agent_model

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )
        model = instrumented_runtime.get_agent_model()

        assert model is None

    def test_get_agent_model_returns_none_when_delegate_returns_none(
        self, tracer, callback, mock_runtime_context
    ):
        """Test get_agent_model returns None when delegate returns None."""
        mock_delegate = MagicMock()
        mock_delegate.get_agent_model.return_value = None

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )
        model = instrumented_runtime.get_agent_model()

        assert model is None


class TestRegisterLicensing:
    """Tests for the register_licensing_async helper."""

    @pytest.fixture
    def agent_definition(self) -> AgentDefinition:
        return AgentDefinition(
            name="cost-agent",
            messages=[],
            settings=AgentSettings(
                model="gpt-4o",
                engine="v1",
                max_tokens=1000,
                temperature=0.7,
            ),
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )

    @pytest.mark.asyncio
    async def test_registers_with_model_and_job_key(
        self,
        agent_definition: AgentDefinition,
    ) -> None:
        from uipath_agents._services import register_licensing_async

        mock_service = AsyncMock()

        with (
            patch("uipath.platform.UiPath") as mock_uipath_cls,
            patch(
                "uipath_agents._services.licensing_service.LicensingService",
                return_value=mock_service,
            ),
        ):
            mock_uipath_cls.return_value = MagicMock()
            await register_licensing_async(agent_definition, job_key="job-123")

        mock_service.register_consumption_async.assert_awaited_once_with(
            "gpt-4o", job_key="job-123"
        )

    @pytest.mark.asyncio
    async def test_failure_does_not_raise(
        self,
        agent_definition: AgentDefinition,
    ) -> None:
        from uipath_agents._services import register_licensing_async

        mock_service = AsyncMock()
        mock_service.register_consumption_async.side_effect = RuntimeError("network")

        with (
            patch("uipath.platform.UiPath", return_value=MagicMock()),
            patch(
                "uipath_agents._services.licensing_service.LicensingService",
                return_value=mock_service,
            ),
        ):
            await register_licensing_async(agent_definition, job_key="job-123")

    @pytest.mark.asyncio
    async def test_skips_when_no_agent_definition(self) -> None:
        from uipath_agents._services import register_licensing_async

        mock_service_cls = MagicMock()

        with patch(
            "uipath_agents._services.licensing_service.LicensingService",
            mock_service_cls,
        ):
            await register_licensing_async(None)

        mock_service_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_no_model(self) -> None:
        from uipath_agents._services import register_licensing_async

        agent_def = AgentDefinition(
            name="no-model-agent",
            messages=[],
            settings=AgentSettings(
                model="",
                engine="v1",
                max_tokens=1000,
                temperature=0.7,
            ),
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )

        mock_service_cls = MagicMock()

        with patch(
            "uipath_agents._services.licensing_service.LicensingService",
            mock_service_cls,
        ):
            await register_licensing_async(agent_def)

        mock_service_cls.assert_not_called()


class TestConversationalAgentSuppressesUserPrompt:
    """Tests that conversational agents suppress the dummy user_prompt template."""

    @staticmethod
    def _make_delegate() -> MagicMock:
        delegate = MagicMock(spec=[])
        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.FAULTED
        delegate.execute = AsyncMock(return_value=mock_result)
        return delegate

    @staticmethod
    def _make_agent_definition(
        *,
        name: str = "chat-agent",
        is_conversational: bool = True,
        messages: list[AgentMessage] | None = None,
    ) -> AgentDefinition:
        kwargs: Dict[str, Any] = dict(
            name=name,
            messages=messages or [],
            settings=AgentSettings(
                model="gpt-4o", engine="v1", max_tokens=1000, temperature=0.7
            ),
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )
        if is_conversational:
            kwargs["metadata"] = AgentMetadata(
                is_conversational=True, storage_version="1.0"
            )
        return AgentDefinition(**kwargs)

    @pytest.mark.asyncio
    async def test_conversational_agent_nulls_user_prompt_on_span(
        self, tracer, callback, mock_runtime_context, span_exporter
    ) -> None:
        agent_info = self._make_agent_definition(
            messages=[
                AgentMessage(
                    role=AgentMessageRole.SYSTEM,
                    content="You are a helpful assistant.",
                ),
                AgentMessage(
                    role=AgentMessageRole.USER,
                    content="{{input_string}}",
                ),
            ],
        )

        instrumented_runtime = InstrumentedRuntime(
            self._make_delegate(),
            tracer,
            callback,
            mock_runtime_context,
            agent_definition=agent_info,
        )
        await instrumented_runtime.execute({"input": "hello"}, None)

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "agent run" in s.name.lower()]
        assert len(agent_spans) >= 1

        attrs = dict(agent_spans[0].attributes)
        assert "userPrompt" not in attrs
        assert attrs["systemPrompt"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_conversational_agent_sets_prompts_captured_true(
        self, tracer, callback, mock_runtime_context
    ) -> None:
        """Prevents LLM instrumentor from re-adding userPrompt via _capture_interpolated_prompts."""
        agent_info = self._make_agent_definition()

        captured_prompts_flag = None

        def capture_set_agent_span(span, run_id, prompts_captured=False):
            nonlocal captured_prompts_flag
            captured_prompts_flag = prompts_captured

        callback.set_agent_span = MagicMock(side_effect=capture_set_agent_span)

        instrumented_runtime = InstrumentedRuntime(
            self._make_delegate(),
            tracer,
            callback,
            mock_runtime_context,
            agent_definition=agent_info,
        )
        await instrumented_runtime.execute({"input": "hello"}, None)

        assert captured_prompts_flag is True

    @pytest.mark.asyncio
    async def test_non_conversational_agent_preserves_user_prompt(
        self, tracer, callback, mock_runtime_context, span_exporter
    ) -> None:
        agent_info = self._make_agent_definition(
            name="standard-agent",
            is_conversational=False,
            messages=[
                AgentMessage(
                    role=AgentMessageRole.SYSTEM,
                    content="You are a helper.",
                ),
                AgentMessage(
                    role=AgentMessageRole.USER,
                    content="Process: {{input_string}}",
                ),
            ],
        )

        instrumented_runtime = InstrumentedRuntime(
            self._make_delegate(),
            tracer,
            callback,
            mock_runtime_context,
            agent_definition=agent_info,
        )
        await instrumented_runtime.execute({"input": "test"}, None)

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "Agent run" in s.name]
        assert len(agent_spans) >= 1

        attrs = dict(agent_spans[0].attributes)
        assert attrs["userPrompt"] == "Process: {{input_string}}"


class TestNestedAgentResumeFixes:
    """Tests for PR #288 fixes: trace errors on nested agent raise_error scenarios."""

    @pytest.mark.asyncio
    async def test_resumed_trace_id_cleared_after_upsert(
        self,
        mock_delegate,
        tracer,
        callback,
        mock_runtime_context,
    ) -> None:
        """After _upsert_resumed_spans_on_completion, resumed_trace_id must be None
        so resumed_spans_completed() returns True and OK spans aren't flipped to ERROR.
        """
        from uipath_agents._observability.llmops.instrumentors.base import (
            InstrumentationState,
        )
        from uipath_agents._observability.llmops.instrumentors.tool_instrumentor import (
            ToolSpanInstrumentor,
        )

        mock_span_factory = MagicMock()
        state = InstrumentationState(span_factory=mock_span_factory)
        state.resumed_trace_id = "trace-abc-123"
        state.resumed_process_span_data = None
        state.resumed_tool_span_data = {"attributes": {}}

        instrumentor = ToolSpanInstrumentor(state=state, close_container=MagicMock())

        instrumentor._upsert_resumed_spans_on_completion({"answer": 42}, None)

        assert state.resumed_trace_id is None
        assert callback.resumed_spans_completed()

    @pytest.mark.asyncio
    async def test_faulted_result_sets_agent_span_error(
        self,
        mock_delegate,
        tracer,
        callback,
        mock_runtime_context,
        span_exporter,
    ) -> None:
        """FAULTED result must set the error attribute on the root agentRun span
        even though no exception is raised.
        """
        import json

        from uipath.runtime.errors import UiPathErrorContract

        faulted_result = MagicMock()
        faulted_result.status = UiPathRuntimeStatus.FAULTED
        faulted_result.error = UiPathErrorContract(
            title="Child agent failed", detail="timeout", code="AGENT_FAULTED"
        )
        mock_delegate.execute.return_value = faulted_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate, tracer, callback, mock_runtime_context
        )
        await instrumented_runtime.execute({"input": "test"}, None)

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "Agent run" in s.name]
        assert len(agent_spans) >= 1

        agent_span = agent_spans[0]
        # _set_agent_span_error sets the "error" attribute with JSON error details
        error_attr = agent_span.attributes.get("error")
        assert error_attr is not None
        error_data = json.loads(error_attr)
        assert error_data["message"] == "Child agent failed"
        assert error_data["type"] == "AgentRuntimeError"
        assert error_data["detail"] == "timeout"

    @pytest.mark.asyncio
    async def test_faulted_does_not_retroactively_error_tool_spans(
        self,
        mock_delegate,
        tracer,
        callback,
        mock_runtime_context,
    ) -> None:
        """When child agent FAULTs on resume, already-OK tool/agentTool spans
        must not be retroactively marked ERROR.
        """
        from uipath_agents._observability.llmops.instrumentors.base import (
            InstrumentationState,
        )
        from uipath_agents._observability.llmops.instrumentors.tool_instrumentor import (
            ToolSpanInstrumentor,
        )

        mock_span_factory = MagicMock()
        state = InstrumentationState(span_factory=mock_span_factory)
        state.resumed_trace_id = "trace-xyz"
        state.resumed_process_span_data = {"attributes": {"type": "processTool"}}
        state.resumed_tool_span_data = {"attributes": {}}

        instrumentor = ToolSpanInstrumentor(state=state, close_container=MagicMock())

        # Complete the resumed spans with a normal result (OK)
        instrumentor._upsert_resumed_spans_on_completion({"result": "ok"}, None)

        # Verify resumed_trace_id is cleared — this is the gate that
        # prevents subsequent error propagation from flipping these spans
        assert state.resumed_trace_id is None
        assert state.resumed_process_span_data is None
        # resumed_tool_span_data is intentionally kept alive for potential
        # HITL guardrail suspend; resumed_trace_id=None is the authoritative
        # signal that upserts are done.
        assert state.resumed_tool_span_data is not None

        # The 2 upserts (process + tool) should have been called with OK data
        assert mock_span_factory.upsert_span_complete_by_data.call_count == 2

    @pytest.mark.asyncio
    async def test_resumed_agent_span_upserted_with_correct_timing(
        self,
        mock_delegate,
        tracer,
        callback,
        mock_trace_context_storage,
        mock_runtime_context,
        span_exporter,
    ) -> None:
        """On resume completion, the agent span must be upserted with
        start_time_ns from saved context (not epoch) and a fresh EndTime.
        """
        from opentelemetry.sdk.trace.export import SpanExportResult

        mock_exporter = MagicMock()
        mock_exporter.upsert_span = MagicMock(return_value=SpanExportResult.SUCCESS)
        tracer_with_exp = LlmOpsSpanFactory(exporter=mock_exporter)
        callback_with_exp = LlmOpsInstrumentationCallback(tracer_with_exp)

        start_ns = 1708672800_000_000_000
        saved_context = TraceContextData(
            trace_id="aabbccdd11223344aabbccdd11223344",
            span_id="1122334455667788",
            parent_span_id=None,
            name="Agent run - TestAgent",
            start_time="2024-02-23T10:00:00Z",
            start_time_ns=start_ns,
            attributes={"agentId": "test-id", "type": "agentRun"},
            pending_tool_span_id=None,
            pending_process_span_id=None,
            pending_tool_name=None,
            pending_tool_span=None,
            pending_process_span=None,
            pending_escalation_span=None,
            pending_guardrail_hitl_evaluation_span=None,
            pending_guardrail_hitl_container_span=None,
            pending_llm_span=None,
        )
        mock_trace_context_storage.load_trace_context = AsyncMock(
            return_value=saved_context
        )

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUCCESSFUL
        mock_result.output = {"result": "done"}
        mock_delegate.execute.return_value = mock_result

        mock_runtime_context.resume = True

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer_with_exp,
            callback_with_exp,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        await instrumented_runtime.execute({"input": "resume"}, None)

        # The final upsert_resumed_agent_span should have fired
        assert mock_exporter.upsert_span.call_count >= 1

    @pytest.mark.asyncio
    async def test_uipath_source_context_importable_from_public_path(self) -> None:
        """uipath_source_context must be importable from the public llmops __init__
        chain, not just from internal base module.
        """
        from uipath_agents._observability.llmops import uipath_source_context
        from uipath_agents._observability.llmops.spans import (
            uipath_source_context as from_spans,
        )
        from uipath_agents._observability.llmops.spans.spans_schema import (
            uipath_source_context as from_schema,
        )

        # All three should resolve to the same ContextVar object
        assert uipath_source_context is from_spans
        assert uipath_source_context is from_schema

        # Should be usable: set and reset
        token = uipath_source_context.set(1)
        assert uipath_source_context.get() == 1
        uipath_source_context.reset(token)

    @pytest.mark.asyncio
    async def test_suspended_skips_final_ok_upsert(
        self,
        mock_delegate,
        tracer,
        callback,
        mock_trace_context_storage,
        mock_runtime_context,
        span_exporter,
    ) -> None:
        """When agent suspends (nested agent running), the root span must NOT
        get a final OK upsert that overwrites the UNSET suspended state.
        """

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUSPENDED
        mock_delegate.execute.return_value = mock_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        result = await instrumented_runtime.execute({"input": "test"}, None)
        assert result.status == UiPathRuntimeStatus.SUSPENDED

        # Suspended agent spans are intentionally NOT ended — span.end() is
        # never called, so LiveTrackingSpanProcessor.on_end() never fires.
        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "Agent run" in s.name]
        assert len(agent_spans) == 0


class TestMultipleSuspensionFixes:
    """Tests for multi-suspension bugs: start_time_ns loss and premature OK upsert."""

    @pytest.mark.asyncio
    async def test_re_suspension_preserves_start_time_ns(
        self,
        mock_delegate: AsyncMock,
        tracer: LlmOpsSpanFactory,
        callback: LlmOpsInstrumentationCallback,
        mock_runtime_context: MagicMock,
    ) -> None:
        """suspend → resume → suspend again must retain original start_time_ns."""
        stored_context: TraceContextData | None = None

        async def mock_save(runtime_id: str, context: TraceContextData) -> None:
            nonlocal stored_context
            stored_context = context

        async def mock_load(runtime_id: str) -> TraceContextData | None:
            return stored_context

        async def mock_clear(runtime_id: str) -> None:
            nonlocal stored_context
            stored_context = None

        storage = MagicMock()
        storage.save_trace_context = AsyncMock(side_effect=mock_save)
        storage.load_trace_context = AsyncMock(side_effect=mock_load)
        storage.clear_trace_context = AsyncMock(side_effect=mock_clear)

        suspended_result = MagicMock()
        suspended_result.status = UiPathRuntimeStatus.SUSPENDED

        success_result = MagicMock()
        success_result.status = UiPathRuntimeStatus.SUCCESSFUL
        success_result.output = {"result": "done"}

        mock_delegate.execute.side_effect = [
            suspended_result,  # first run: suspend
            suspended_result,  # resume: re-suspend
            success_result,  # second resume: complete
        ]

        agent_def = AgentDefinition(
            id="agent-multi-suspend",
            name="MultiSuspendAgent",
            messages=[],
            settings=AgentSettings(
                model="gpt-4o", engine="v1", max_tokens=1000, temperature=0.7
            ),
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            agent_definition=agent_def,
            trace_context_storage=storage,
        )

        # First run → SUSPENDED
        result1 = await instrumented_runtime.execute({"input": "first"}, None)
        assert result1.status == UiPathRuntimeStatus.SUSPENDED
        assert stored_context is not None

        first_start_ns = stored_context["start_time_ns"]
        first_name = stored_context["name"]
        first_attrs = stored_context["attributes"]
        # The first span is a real ReadableSpan, so start_time_ns should be > 0
        assert first_start_ns != 0, "First suspension must capture real start_time_ns"
        assert first_attrs.get("referenceId") == "agent-multi-suspend"

        # Resume → re-SUSPENDED (agent_span is NonRecordingSpan)
        result2 = await instrumented_runtime.execute({"input": "second"}, None)
        assert result2.status == UiPathRuntimeStatus.SUSPENDED
        assert stored_context is not None

        # start_time_ns must be carried forward from the first suspension
        assert stored_context["start_time_ns"] == first_start_ns
        assert stored_context["name"] == first_name
        assert stored_context["attributes"].get("referenceId") == "agent-multi-suspend"

    @pytest.mark.asyncio
    async def test_re_suspension_skips_ok_upsert(
        self,
        mock_delegate: AsyncMock,
        mock_runtime_context: MagicMock,
    ) -> None:
        """Resume that re-suspends must NOT fire _upsert_resumed_agent_span."""
        from opentelemetry.sdk.trace.export import SpanExportResult

        mock_exporter = MagicMock()
        mock_exporter.upsert_span = MagicMock(return_value=SpanExportResult.SUCCESS)
        tracer_with_exp = LlmOpsSpanFactory(exporter=mock_exporter)
        callback_with_exp = LlmOpsInstrumentationCallback(tracer_with_exp)

        original_start_ns = 1708672800_000_000_000

        saved_context = TraceContextData(
            trace_id="aabbccdd11223344aabbccdd11223344",
            span_id="1122334455667788",
            parent_span_id=None,
            name="Agent run - TestAgent",
            start_time="2024-02-23T10:00:00Z",
            start_time_ns=original_start_ns,
            attributes={"agentId": "test-id", "type": "agentRun"},
            pending_tool_span_id=None,
            pending_process_span_id=None,
            pending_tool_name=None,
            pending_tool_span=None,
            pending_process_span=None,
            pending_escalation_span=None,
            pending_guardrail_hitl_evaluation_span=None,
            pending_guardrail_hitl_container_span=None,
            pending_llm_span=None,
        )

        stored_context: TraceContextData | None = dict(saved_context)  # type: ignore[assignment]

        async def mock_save(runtime_id: str, context: TraceContextData) -> None:
            nonlocal stored_context
            stored_context = context

        async def mock_load(runtime_id: str) -> TraceContextData | None:
            return stored_context

        async def mock_clear(runtime_id: str) -> None:
            nonlocal stored_context
            stored_context = None

        storage = MagicMock()
        storage.save_trace_context = AsyncMock(side_effect=mock_save)
        storage.load_trace_context = AsyncMock(side_effect=mock_load)
        storage.clear_trace_context = AsyncMock(side_effect=mock_clear)

        suspended_result = MagicMock()
        suspended_result.status = UiPathRuntimeStatus.SUSPENDED
        mock_delegate.execute.return_value = suspended_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer_with_exp,
            callback_with_exp,
            mock_runtime_context,
            trace_context_storage=storage,
        )

        # Resume execution that re-suspends
        await instrumented_runtime.execute({"input": "resume-suspend"}, None)

        # Collect all upsert calls and check none have OK/ERROR status
        # (the only upserts should be UNSET from _handle_suspended)
        upsert_calls = mock_exporter.upsert_span.call_args_list
        for call in upsert_calls:
            status_override = call[1].get("status_override")
            # 0 = UNSET (in-progress), which is correct for suspended spans
            assert status_override == 0, (
                f"Expected UNSET (0) status on re-suspension, got {status_override}"
            )

    @pytest.mark.asyncio
    async def test_re_suspension_preserves_pending_tool_name(
        self,
        mock_delegate: AsyncMock,
        mock_runtime_context: MagicMock,
    ) -> None:
        """Resume → re-suspend must carry forward pending_tool_name so the
        next resume arms the skip logic and avoids duplicate tool spans."""
        from opentelemetry.sdk.trace.export import SpanExportResult

        mock_exporter = MagicMock()
        mock_exporter.upsert_span = MagicMock(return_value=SpanExportResult.SUCCESS)
        tracer_with_exp = LlmOpsSpanFactory(exporter=mock_exporter)
        callback_with_exp = LlmOpsInstrumentationCallback(tracer_with_exp)

        original_tool_span_data: PendingSpanData = {
            "span_id": "aabb000000001234",
            "parent_span_id": "1122334455667788",
            "name": "Tool call - DeepRAG",
            "start_time_ns": 1708672800_000_000_000,
            "attributes": {"tool.name": "DeepRAG", "type": "toolCall"},
        }
        original_process_span_data: PendingSpanData = {
            "span_id": "aabb000000005678",
            "parent_span_id": "aabb000000001234",
            "name": "DeepRAG",
            "start_time_ns": 1708672800_100_000_000,
            "attributes": {"type": "contextGroundingTool"},
        }

        saved_context = TraceContextData(
            trace_id="aabbccdd11223344aabbccdd11223344",
            span_id="1122334455667788",
            parent_span_id=None,
            name="Agent run - TestAgent",
            start_time="2024-02-23T10:00:00Z",
            start_time_ns=1708672800_000_000_000,
            attributes={"agentId": "test-id", "type": "agentRun"},
            pending_tool_span_id="aabb000000001234",
            pending_process_span_id="aabb000000005678",
            pending_tool_name="DeepRAG",
            pending_tool_span=original_tool_span_data,
            pending_process_span=original_process_span_data,
            pending_escalation_span=None,
            pending_guardrail_hitl_evaluation_span=None,
            pending_guardrail_hitl_container_span=None,
            pending_llm_span=None,
        )

        stored_context: TraceContextData | None = dict(saved_context)  # type: ignore[assignment]

        async def mock_save(runtime_id: str, context: TraceContextData) -> None:
            nonlocal stored_context
            stored_context = context

        async def mock_load(runtime_id: str) -> TraceContextData | None:
            return stored_context

        async def mock_clear(runtime_id: str) -> None:
            nonlocal stored_context
            stored_context = None

        storage = MagicMock()
        storage.save_trace_context = AsyncMock(side_effect=mock_save)
        storage.load_trace_context = AsyncMock(side_effect=mock_load)
        storage.clear_trace_context = AsyncMock(side_effect=mock_clear)

        suspended_result = MagicMock()
        suspended_result.status = UiPathRuntimeStatus.SUSPENDED
        mock_delegate.execute.return_value = suspended_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer_with_exp,
            callback_with_exp,
            mock_runtime_context,
            trace_context_storage=storage,
        )

        # Resume execution that re-suspends (tool suspends again)
        await instrumented_runtime.execute({"input": "resume-suspend"}, None)

        assert stored_context is not None
        # pending_tool_name must be carried forward from previous context
        assert stored_context["pending_tool_name"] == "DeepRAG", (
            "pending_tool_name must survive re-suspend so next resume arms skip logic"
        )

    @pytest.mark.asyncio
    async def test_re_suspension_preserves_pending_tool_and_process_spans(
        self,
        mock_delegate: AsyncMock,
        mock_runtime_context: MagicMock,
    ) -> None:
        """Resume → re-suspend must carry forward pending_tool_span and
        pending_process_span so the next resume can complete the original spans."""
        from opentelemetry.sdk.trace.export import SpanExportResult

        mock_exporter = MagicMock()
        mock_exporter.upsert_span = MagicMock(return_value=SpanExportResult.SUCCESS)
        tracer_with_exp = LlmOpsSpanFactory(exporter=mock_exporter)
        callback_with_exp = LlmOpsInstrumentationCallback(tracer_with_exp)

        original_tool_span_data: PendingSpanData = {
            "span_id": "aabb000000001234",
            "parent_span_id": "1122334455667788",
            "name": "Tool call - DeepRAG",
            "start_time_ns": 1708672800_000_000_000,
            "attributes": {"tool.name": "DeepRAG", "type": "toolCall"},
        }
        original_process_span_data: PendingSpanData = {
            "span_id": "aabb000000005678",
            "parent_span_id": "aabb000000001234",
            "name": "DeepRAG",
            "start_time_ns": 1708672800_100_000_000,
            "attributes": {"type": "contextGroundingTool"},
        }

        saved_context = TraceContextData(
            trace_id="aabbccdd11223344aabbccdd11223344",
            span_id="1122334455667788",
            parent_span_id=None,
            name="Agent run - TestAgent",
            start_time="2024-02-23T10:00:00Z",
            start_time_ns=1708672800_000_000_000,
            attributes={"agentId": "test-id", "type": "agentRun"},
            pending_tool_span_id="aabb000000001234",
            pending_process_span_id="aabb000000005678",
            pending_tool_name="DeepRAG",
            pending_tool_span=original_tool_span_data,
            pending_process_span=original_process_span_data,
            pending_escalation_span=None,
            pending_guardrail_hitl_evaluation_span=None,
            pending_guardrail_hitl_container_span=None,
            pending_llm_span=None,
        )

        stored_context: TraceContextData | None = dict(saved_context)  # type: ignore[assignment]

        async def mock_save(runtime_id: str, context: TraceContextData) -> None:
            nonlocal stored_context
            stored_context = context

        async def mock_load(runtime_id: str) -> TraceContextData | None:
            return stored_context

        async def mock_clear(runtime_id: str) -> None:
            nonlocal stored_context
            stored_context = None

        storage = MagicMock()
        storage.save_trace_context = AsyncMock(side_effect=mock_save)
        storage.load_trace_context = AsyncMock(side_effect=mock_load)
        storage.clear_trace_context = AsyncMock(side_effect=mock_clear)

        suspended_result = MagicMock()
        suspended_result.status = UiPathRuntimeStatus.SUSPENDED
        mock_delegate.execute.return_value = suspended_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer_with_exp,
            callback_with_exp,
            mock_runtime_context,
            trace_context_storage=storage,
        )

        # Resume execution that re-suspends
        await instrumented_runtime.execute({"input": "resume-suspend"}, None)

        assert stored_context is not None

        # pending_tool_span must be carried forward with original span_id
        tool_span = stored_context.get("pending_tool_span")
        assert tool_span is not None
        assert tool_span["span_id"] == "aabb000000001234"
        assert tool_span["name"] == "Tool call - DeepRAG"

        # pending_process_span must also be carried forward
        process_span = stored_context.get("pending_process_span")
        assert process_span is not None
        assert process_span["span_id"] == "aabb000000005678"
        assert process_span["name"] == "DeepRAG"

    @pytest.mark.asyncio
    async def test_triple_suspend_preserves_tool_context_throughout(
        self,
        mock_delegate: AsyncMock,
        mock_runtime_context: MagicMock,
    ) -> None:
        """suspend → resume → re-suspend → resume → re-suspend must preserve
        tool name and span data across all cycles so the final resume works."""
        from opentelemetry.sdk.trace.export import SpanExportResult

        mock_exporter = MagicMock()
        mock_exporter.upsert_span = MagicMock(return_value=SpanExportResult.SUCCESS)
        tracer_with_exp = LlmOpsSpanFactory(exporter=mock_exporter)
        callback_with_exp = LlmOpsInstrumentationCallback(tracer_with_exp)

        agent_def = AgentDefinition(
            id="agent-triple-suspend",
            name="TripleSuspendAgent",
            messages=[],
            settings=AgentSettings(
                model="gpt-4o", engine="v1", max_tokens=1000, temperature=0.7
            ),
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )

        stored_context: TraceContextData | None = None

        async def mock_save(runtime_id: str, context: TraceContextData) -> None:
            nonlocal stored_context
            stored_context = context

        async def mock_load(runtime_id: str) -> TraceContextData | None:
            return stored_context

        async def mock_clear(runtime_id: str) -> None:
            nonlocal stored_context
            stored_context = None

        storage = MagicMock()
        storage.save_trace_context = AsyncMock(side_effect=mock_save)
        storage.load_trace_context = AsyncMock(side_effect=mock_load)
        storage.clear_trace_context = AsyncMock(side_effect=mock_clear)

        suspended_result = MagicMock()
        suspended_result.status = UiPathRuntimeStatus.SUSPENDED

        mock_delegate.execute.return_value = suspended_result

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer_with_exp,
            callback_with_exp,
            mock_runtime_context,
            agent_definition=agent_def,
            trace_context_storage=storage,
        )

        # --- First execution: suspend (creates real spans) ---
        # Simulate a pending tool via the callback state
        from opentelemetry.sdk.trace import ReadableSpan

        mock_tool_span = MagicMock(spec=ReadableSpan)
        mock_tool_span.get_span_context.return_value = MagicMock(
            span_id=0xAA11, trace_id=0xBB22
        )
        mock_tool_span.name = "Tool call - DeepRAG"
        mock_tool_span.attributes = {"tool.name": "DeepRAG", "type": "toolCall"}
        mock_tool_span.start_time = 1708672800_000_000_000
        mock_tool_span.parent = None

        mock_process_span = MagicMock(spec=ReadableSpan)
        mock_process_span.get_span_context.return_value = MagicMock(
            span_id=0xCC33, trace_id=0xBB22
        )
        mock_process_span.name = "DeepRAG"
        mock_process_span.attributes = {"type": "contextGroundingTool"}
        mock_process_span.start_time = 1708672800_100_000_000
        mock_process_span.parent = MagicMock(span_id=0xAA11)

        with patch.object(
            callback_with_exp,
            "get_pending_tool_info",
            return_value=("DeepRAG", mock_tool_span, mock_process_span),
        ):
            await instrumented_runtime.execute({"input": "first"}, None)
        assert stored_context is not None
        assert stored_context["pending_tool_name"] == "DeepRAG"
        first_tool = stored_context["pending_tool_span"]
        first_process = stored_context["pending_process_span"]
        assert first_tool is not None
        assert first_process is not None
        first_tool_span_id = first_tool["span_id"]
        first_process_span_id = first_process["span_id"]

        # --- Second execution: resume → re-suspend ---
        # On resume, no new pending tool is set (tool start was skipped)

        await instrumented_runtime.execute({"input": "second"}, None)
        assert stored_context is not None
        assert stored_context["pending_tool_name"] == "DeepRAG", (
            "pending_tool_name must survive first re-suspend"
        )
        second_tool = stored_context["pending_tool_span"]
        second_process = stored_context["pending_process_span"]
        assert second_tool is not None
        assert second_tool["span_id"] == first_tool_span_id
        assert second_process is not None
        assert second_process["span_id"] == first_process_span_id

        # --- Third execution: resume → re-suspend again ---
        await instrumented_runtime.execute({"input": "third"}, None)
        assert stored_context is not None
        assert stored_context["pending_tool_name"] == "DeepRAG", (
            "pending_tool_name must survive second re-suspend"
        )
        third_tool = stored_context["pending_tool_span"]
        third_process = stored_context["pending_process_span"]
        assert third_tool is not None
        assert third_tool["span_id"] == first_tool_span_id
        assert third_process is not None
        assert third_process["span_id"] == first_process_span_id


class TestResumeErrorDoesNotOverwriteApprovedEscalation:
    """When a block guardrail fires after an approved escalation,
    _handle_resume_error must not overwrite the Review task span."""

    @pytest.mark.asyncio
    async def test_handle_resume_error_skips_escalation_when_callback_completed_it(
        self,
        mock_delegate,
        mock_runtime_context,
    ) -> None:
        """_handle_resume_error must not upsert the escalation span when
        the callback already completed it (approved)."""
        from opentelemetry.sdk.trace.export import SpanExportResult

        from uipath_agents._observability.llmops.trace_context_storage import (
            TraceContextData,
        )

        mock_exporter = MagicMock()
        mock_exporter.upsert_span = MagicMock(return_value=SpanExportResult.SUCCESS)
        tracer = LlmOpsSpanFactory(exporter=mock_exporter)
        callback = LlmOpsInstrumentationCallback(tracer)

        saved_context = TraceContextData(
            trace_id="aabbccdd11223344aabbccdd11223344",
            span_id="1122334455667788",
            parent_span_id=None,
            name="Agent run - TestAgent",
            start_time="2024-02-23T10:00:00Z",
            start_time_ns=1708672800_000_000_000,
            attributes={"agentId": "test-id", "type": "agentRun"},
            pending_tool_span_id=None,
            pending_process_span_id=None,
            pending_tool_name=None,
            pending_tool_span=None,
            pending_process_span=None,
            pending_escalation_span={
                "name": "Review task",
                "span_id": "aaaa1111bbbb2222",
                "attributes": {
                    "type": "guardrailEscalation",
                    "reviewStatus": "waiting",
                },
            },
            pending_guardrail_hitl_evaluation_span={
                "name": "Guardrail_3",
                "span_id": "cccc3333dddd4444",
                "attributes": {},
            },
            pending_guardrail_hitl_container_span={
                "name": "Tool input guardrail check",
                "span_id": "eeee5555ffff6666",
                "attributes": {},
            },
            pending_llm_span=None,
        )

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.FAULTED
        mock_result.error = None
        mock_delegate.execute = AsyncMock(
            side_effect=Exception("Blocked by guardrail [Guardrail_1]")
        )

        mock_trace_context_storage = MagicMock()
        mock_trace_context_storage.load_trace_context = AsyncMock(
            return_value=saved_context
        )
        mock_trace_context_storage.clear_trace_context = AsyncMock()
        mock_runtime_context.resume = True

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        # Simulate: callback already completed the escalation span (approved)
        callback._state.resumed_escalation_span_data = None

        with pytest.raises(Exception, match="Blocked by guardrail"):
            await instrumented_runtime.execute({"input": "resume"}, None)

        # The escalation span must NOT have been upserted with ERROR
        escalation_upserts = [
            c
            for c in mock_exporter.upsert_span.call_args_list
            if hasattr(c[0][0], "name") and c[0][0].name == "Review task"
        ]
        assert len(escalation_upserts) == 0, (
            "Review task span should not be re-upserted when callback already completed it"
        )

    @pytest.mark.asyncio
    async def test_handle_resume_error_upserts_escalation_when_callback_did_not_complete_it(
        self,
        mock_delegate,
        mock_runtime_context,
    ) -> None:
        """_handle_resume_error must still upsert the escalation span with ERROR
        when the callback did NOT complete it (e.g., escalation was rejected or
        never processed)."""
        from opentelemetry.sdk.trace.export import SpanExportResult

        from uipath_agents._observability.llmops.trace_context_storage import (
            TraceContextData,
        )

        mock_exporter = MagicMock()
        mock_exporter.upsert_span = MagicMock(return_value=SpanExportResult.SUCCESS)
        tracer = LlmOpsSpanFactory(exporter=mock_exporter)
        callback = LlmOpsInstrumentationCallback(tracer)

        saved_context = TraceContextData(
            trace_id="aabbccdd11223344aabbccdd11223344",
            span_id="1122334455667788",
            parent_span_id=None,
            name="Agent run - TestAgent",
            start_time="2024-02-23T10:00:00Z",
            start_time_ns=1708672800_000_000_000,
            attributes={"agentId": "test-id", "type": "agentRun"},
            pending_tool_span_id=None,
            pending_process_span_id=None,
            pending_tool_name="Sentence_Analyzer",
            pending_tool_span={
                "name": "Tool call - Sentence_Analyzer",
                "span_id": "7777888899990000",
                "start_time_ns": 1000,
                "attributes": {"type": "toolCall"},
            },
            pending_process_span=None,
            pending_escalation_span={
                "name": "Review task",
                "span_id": "aaaa1111bbbb2222",
                "attributes": {
                    "type": "guardrailEscalation",
                    "reviewStatus": "waiting",
                },
            },
            pending_guardrail_hitl_evaluation_span={
                "name": "Guardrail_3",
                "span_id": "cccc3333dddd4444",
                "attributes": {},
            },
            pending_guardrail_hitl_container_span={
                "name": "Tool input guardrail check",
                "span_id": "eeee5555ffff6666",
                "attributes": {},
            },
            pending_llm_span=None,
        )

        mock_delegate.execute = AsyncMock(
            side_effect=Exception("Blocked by guardrail [Guardrail_1]")
        )

        mock_trace_context_storage = MagicMock()
        mock_trace_context_storage.load_trace_context = AsyncMock(
            return_value=saved_context
        )
        mock_trace_context_storage.clear_trace_context = AsyncMock()
        mock_runtime_context.resume = True

        instrumented_runtime = InstrumentedRuntime(
            mock_delegate,
            tracer,
            callback,
            mock_runtime_context,
            trace_context_storage=mock_trace_context_storage,
        )

        with pytest.raises(Exception, match="Blocked by guardrail"):
            await instrumented_runtime.execute({"input": "resume"}, None)

        # The escalation span SHOULD have been upserted with ERROR
        escalation_upserts = [
            c
            for c in mock_exporter.upsert_span.call_args_list
            if hasattr(c[0][0], "name") and c[0][0].name == "Review task"
        ]
        assert len(escalation_upserts) == 1, (
            "Review task span should be upserted with ERROR when callback didn't complete it"
        )
