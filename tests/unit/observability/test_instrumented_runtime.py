"""Tests for InstrumentedRuntime."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from uipath.agent.models.agent import (
    AgentDefinition,
    AgentMessage,
    AgentMessageRole,
    AgentSettings,
)
from uipath.runtime import UiPathRuntimeContext, UiPathRuntimeStatus

from uipath_agents._observability.instrumented_runtime import InstrumentedRuntime
from uipath_agents._observability.llmops.callback import LlmOpsInstrumentationCallback
from uipath_agents._observability.llmops.spans.span_factory import LlmOpsSpanFactory
from uipath_agents._observability.llmops.trace_context_storage import TraceContextData


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
        self, mock_delegate, tracer, callback, mock_runtime_context
    ):
        """Test complete suspend → resume → complete flow with re-parenting."""
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
            tracer,
            callback,
            mock_runtime_context,
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

        # Verify that spans created during resume have reference_id
        spans = span_exporter.get_finished_spans()

        # Filter to agent run spans
        agent_spans = [s for s in spans if "Agent run" in s.name]
        assert len(agent_spans) >= 1

        # All agent spans should have the same reference_id
        for span in agent_spans:
            if hasattr(span, "attributes") and "referenceId" in span.attributes:
                assert span.attributes["referenceId"] == "test-agent-id-123"


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

        # 3 upserts: agent start (UNSET) + agent suspended (UNSET) + agent end (OK)
        assert mock_exporter.upsert_span.call_count == 3
        calls = mock_exporter.upsert_span.call_args_list
        assert calls[0][1]["status_override"] == 0  # UNSET
        assert calls[1][1]["status_override"] == 0  # UNSET
        assert calls[2][1]["status_override"] == 1  # OK

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

        # 5 upserts: agent start + process suspended + tool suspended + agent suspended + agent end
        assert mock_exporter.upsert_span.call_count == 5


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
