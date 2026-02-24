"""Tests for enhanced InstrumentedRuntime functionality including telemetry callback and property enrichment."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from uipath.agent.models.agent import AgentDefinition, AgentMetadata, AgentSettings
from uipath.runtime import UiPathRuntimeContext, UiPathRuntimeStatus

from uipath_agents._observability.event_emitter import (
    AgentRunEvent,
    TelemetryEventEmitter,
)
from uipath_agents._observability.instrumented_runtime import InstrumentedRuntime
from uipath_agents._observability.llmops.callback import LlmOpsInstrumentationCallback
from uipath_agents._observability.llmops.spans.span_factory import LlmOpsSpanFactory
from uipath_agents._observability.llmops.trace_context_storage import TraceContextData


@pytest.fixture
def mock_delegate():
    """Create a mock delegate runtime."""
    delegate = AsyncMock()
    delegate.runtime_id = "test-runtime-id"
    delegate._get_trace_prompts = MagicMock(return_value=("system", "user"))
    return delegate


@pytest.fixture
def tracer():
    """Create a tracer."""
    return LlmOpsSpanFactory()


@pytest.fixture
def tracing_callback(tracer):
    """Create a tracing callback."""
    return LlmOpsInstrumentationCallback(tracer)


@pytest.fixture
def event_emitter():
    """Create a telemetry callback."""
    return TelemetryEventEmitter()


@pytest.fixture
def mock_runtime_context():
    """Create a mock runtime context."""
    from unittest.mock import MagicMock

    context = MagicMock(spec=UiPathRuntimeContext)
    context.command = "debug"
    context.org_id = "test-org-id"
    context.tenant_id = "test-tenant-id"
    context.job_id = "test-job-id"
    context.resume = False
    return context


@pytest.fixture
def agent_info():
    """Create agent span info with telemetry properties."""
    return AgentDefinition(
        id="test-agent-id",
        name="test-agent",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        messages=[],
        settings=AgentSettings(
            model="gpt-4",
            max_tokens=1000,
            temperature=0.7,
            engine="openai",
            max_iterations=5,
        ),
        metadata=AgentMetadata(
            is_conversational=True,
            storage_version="v1",
        ),
    )


class TestTelemetryCallbackIntegration:
    """Test integration of telemetry callback with runtime wrapper."""

    def test_init_with_event_emitter(
        self,
        mock_delegate,
        tracer,
        tracing_callback,
        event_emitter,
        agent_info,
        mock_runtime_context,
    ):
        wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
            event_emitter=event_emitter,
            agent_definition=agent_info,
        )

        assert wrapper._event_emitter is event_emitter
        assert wrapper._agent_definition is agent_info

    @pytest.mark.asyncio
    async def test_execute_calls_event_emitter(
        self,
        mock_delegate,
        tracer,
        tracing_callback,
        event_emitter,
        agent_info,
        mock_runtime_context,
    ):
        """Test that execute calls telemetry callback for started and completed events."""
        from uipath.runtime import UiPathRuntimeStatus

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUCCESSFUL
        mock_result.output = {"result": "test"}
        mock_delegate.execute.return_value = mock_result

        event_emitter.track_event = MagicMock()
        event_emitter.set_agent_info = MagicMock()

        wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
            event_emitter=event_emitter,
            agent_definition=agent_info,
        )

        await wrapper.execute({"input": "test"}, None)

        # Verify telemetry callback was called
        event_emitter.set_agent_info.assert_called_with("test-agent", "test-agent-id")

        # Should be called twice: once for STARTED, once for COMPLETED
        assert event_emitter.track_event.call_count == 2

        # Verify event names
        calls = event_emitter.track_event.call_args_list
        assert calls[0][0][0] == AgentRunEvent.STARTED
        assert calls[1][0][0] == AgentRunEvent.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_calls_event_emitter_on_failure(
        self,
        mock_delegate,
        tracer,
        tracing_callback,
        event_emitter,
        agent_info,
        mock_runtime_context,
    ):
        """Test that execute calls telemetry callback for failed events."""
        event_emitter.track_event = MagicMock()
        event_emitter.set_agent_info = MagicMock()

        mock_delegate.execute.side_effect = RuntimeError("Test error")

        wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
            event_emitter=event_emitter,
            agent_definition=agent_info,
        )

        with pytest.raises(RuntimeError):
            await wrapper.execute({"input": "test"}, None)

        # Should be called twice: once for STARTED, once for FAILED
        assert event_emitter.track_event.call_count == 2

        # Verify event names
        calls = event_emitter.track_event.call_args_list
        assert calls[0][0][0] == AgentRunEvent.STARTED
        assert calls[1][0][0] == AgentRunEvent.FAILED

    def test_without_event_emitter(
        self, mock_delegate, tracer, tracing_callback, agent_info, mock_runtime_context
    ):
        wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
            agent_definition=agent_info,
        )

        assert wrapper._event_emitter is None


class TestPropertyEnrichment:
    """Test telemetry property enrichment functionality."""

    def test_get_enriched_properties_basic(
        self, mock_delegate, tracer, tracing_callback, agent_info, mock_runtime_context
    ):
        wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
            agent_definition=agent_info,
        )

        base_properties = {"AgentName": "test-agent"}
        enriched = wrapper._get_enriched_properties(base_properties)

        # Verify basic properties are preserved
        assert enriched["AgentName"] == "test-agent"

        # Verify AgentDefinition properties are added
        assert enriched["Model"] == "gpt-4"
        assert enriched["MaxTokens"] == "1000"
        assert enriched["Temperature"] == "0.7"
        assert enriched["Engine"] == "openai"
        assert enriched["MaxIterations"] == "5"
        assert enriched["IsConversational"] == "True"

        # Verify standard properties
        assert enriched["AgentRunSource"] == "playground"
        assert enriched["ApplicationName"] == "UiPath.AgentService"
        assert "AgentRunId" in enriched

    def test_get_enriched_properties_with_trace_id(
        self, mock_delegate, tracer, tracing_callback, agent_info, mock_runtime_context
    ):
        wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
            agent_definition=agent_info,
        )

        # Create mock span with trace context
        span_context = SpanContext(
            trace_id=0x12345678901234567890123456789012,
            span_id=0x1234567890123456,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        )
        mock_span = NonRecordingSpan(span_context)

        base_properties = {"AgentName": "test-agent"}
        enriched = wrapper._get_enriched_properties(base_properties, mock_span)

        # Verify trace ID is added
        assert enriched["TraceId"] == "12345678901234567890123456789012"

    def test_get_enriched_properties_without_agent_info(
        self, mock_delegate, tracer, tracing_callback, mock_runtime_context
    ):
        wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
        )

        base_properties = {"AgentName": "test-agent"}
        enriched = wrapper._get_enriched_properties(base_properties)

        # Verify basic properties are preserved
        assert enriched["AgentName"] == "test-agent"

        # Verify AgentDefinition properties are not present
        assert "Model" not in enriched
        assert "MaxTokens" not in enriched

        # Verify standard properties are still added
        assert enriched["AgentRunSource"] == "playground"
        assert enriched["ApplicationName"] == "UiPath.AgentService"
        assert enriched["Runtime"] == "URT"
        assert enriched["AgentType"] == "LowCode"
        assert "AgentRunId" in enriched

    def test_get_enriched_properties_cloud_context_from_runtime_context(
        self, mock_delegate, tracer, tracing_callback, agent_info, mock_runtime_context
    ):
        """Test cloud context extraction from UiPathConfig and runtime context."""
        # Mock UiPathConfig to provide organization_id
        with patch(
            "uipath_agents._observability.instrumented_runtime.UiPathConfig"
        ) as mock_config:
            mock_config.organization_id = "test-org-id"
            mock_config.folder_key = None
            mock_config.job_key = None
            mock_config.project_id = None
            mock_config.process_uuid = None
            mock_config.process_version = None
            mock_config.is_studio_project = False

            # Mock get_claim_from_token to return None so CloudUserId uses empty string
            with patch(
                "uipath_agents._observability.instrumented_runtime.get_claim_from_token",
                return_value=None,
            ):
                wrapper = InstrumentedRuntime(
                    mock_delegate,
                    tracer,
                    tracing_callback,
                    mock_runtime_context,
                    agent_definition=agent_info,
                )

                base_properties = {"AgentName": "test-agent"}
                enriched = wrapper._get_enriched_properties(base_properties)

                # CloudOrganizationId comes from UiPathConfig
                assert enriched["CloudOrganizationId"] == "test-org-id"
                # CloudUserId comes from JWT token
                assert enriched["CloudUserId"] == ""  # Empty because no JWT token
                # CloudTenantId and JobId come from runtime_context
                assert enriched["CloudTenantId"] == "test-tenant-id"
                assert enriched["JobId"] == "test-job-id"


class TestAgentRunIdConsistency:
    """Test AgentRunId consistency across telemetry events."""

    @pytest.mark.asyncio
    async def test_agent_run_id_consistent_across_events(
        self,
        mock_delegate,
        tracer,
        tracing_callback,
        event_emitter,
        agent_info,
        mock_runtime_context,
    ):
        """Test that AgentRunId is consistent across all telemetry events."""
        from uipath.runtime import UiPathRuntimeStatus

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUCCESSFUL
        mock_result.output = {"result": "test"}
        mock_delegate.execute.return_value = mock_result

        event_emitter.track_event = MagicMock()

        wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
            event_emitter=event_emitter,
            agent_definition=agent_info,
        )

        await wrapper.execute({"input": "test"}, None)

        # Get the calls to track_event
        calls = event_emitter.track_event.call_args_list

        # Extract AgentRunId from both calls
        start_properties = calls[0][0][1]  # First call (STARTED) properties
        complete_properties = calls[1][0][1]  # Second call (COMPLETED) properties

        # Verify AgentRunId is the same in both events
        assert start_properties["AgentRunId"] == complete_properties["AgentRunId"]

        # Verify it's a valid UUID format (36 characters with hyphens)
        agent_run_id = start_properties["AgentRunId"]
        assert len(agent_run_id) == 36
        assert agent_run_id.count("-") == 4

    @pytest.mark.asyncio
    async def test_different_wrapper_instances_have_different_run_ids(
        self,
        mock_delegate,
        tracer,
        tracing_callback,
        event_emitter,
        agent_info,
        mock_runtime_context,
    ):
        """Test that different wrapper instances have different AgentRunIds."""
        wrapper1 = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
            event_emitter=event_emitter,
            agent_definition=agent_info,
        )

        wrapper2 = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
            event_emitter=event_emitter,
            agent_definition=agent_info,
        )

        # Check that different instances have different run IDs
        assert wrapper1._agent_run_id != wrapper2._agent_run_id


class TestEnrichedPropertiesOnResume:
    """Test that enriched properties are available after an interruption and resume."""

    @pytest.fixture
    def mock_trace_context_storage(self):
        """Create a mock trace context storage with async methods."""
        storage = MagicMock()
        storage.load_trace_context = AsyncMock(return_value=None)
        storage.save_trace_context = AsyncMock()
        storage.clear_trace_context = AsyncMock()
        return storage

    @pytest.fixture
    def saved_context(self) -> TraceContextData:
        """Create a saved trace context representing a previously suspended execution."""
        return TraceContextData(
            trace_id="abc123def456789012345678901234567890",
            span_id="1234567890123456",
            parent_span_id=None,
            name="Agent run - test-agent",
            start_time="2024-01-15T10:30:00Z",
            start_time_ns=0,
            attributes={"agentId": "test-agent-id"},
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

    @pytest.mark.asyncio
    async def test_resume_sets_enriched_properties_on_callback(
        self,
        mock_delegate,
        tracer,
        tracing_callback,
        event_emitter,
        agent_info,
        mock_runtime_context,
        mock_trace_context_storage,
        saved_context,
    ):
        """Test that enriched properties are set on the callback during resume.

        This ensures that events tracked after an interruption/resume
        (e.g. guardrail escalation approved/rejected) still carry agent
        and execution metadata.
        """
        mock_trace_context_storage.load_trace_context = AsyncMock(
            return_value=saved_context
        )

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUCCESSFUL
        mock_result.output = {"result": "done"}
        mock_delegate.execute.return_value = mock_result

        event_emitter.track_event = MagicMock()
        event_emitter.set_agent_info = MagicMock()

        wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
            event_emitter=event_emitter,
            agent_definition=agent_info,
            trace_context_storage=mock_trace_context_storage,
        )

        await wrapper.execute({"input": "resume"}, None)

        # After resume, the callback's enriched_properties must be populated
        enriched = tracing_callback._state.enriched_properties
        assert enriched, "enriched_properties should not be empty after resume"
        assert enriched["AgentName"] == "test-agent"
        assert enriched["AgentId"] == "test-agent-id"
        assert enriched["Model"] == "gpt-4"
        assert "AgentRunId" in enriched
        assert "TraceId" in enriched

    @pytest.mark.asyncio
    async def test_resume_enriched_properties_match_normal_flow(
        self,
        mock_delegate,
        tracer,
        agent_info,
        mock_runtime_context,
        mock_trace_context_storage,
        saved_context,
    ):
        """Test that enriched properties after resume match those from a normal execution."""
        event_emitter = MagicMock(spec=TelemetryEventEmitter)

        # --- Normal execution ---
        normal_callback = LlmOpsInstrumentationCallback(tracer)

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUCCESSFUL
        mock_result.output = {"result": "done"}
        mock_delegate.execute.return_value = mock_result

        normal_wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            normal_callback,
            mock_runtime_context,
            event_emitter=event_emitter,
            agent_definition=agent_info,
        )
        await normal_wrapper.execute({"input": "test"}, None)
        normal_enriched = normal_callback._state.enriched_properties.copy()

        # --- Resume execution ---
        resume_callback = LlmOpsInstrumentationCallback(tracer)

        mock_trace_context_storage.load_trace_context = AsyncMock(
            return_value=saved_context
        )

        resume_wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            resume_callback,
            mock_runtime_context,
            event_emitter=event_emitter,
            agent_definition=agent_info,
            trace_context_storage=mock_trace_context_storage,
        )
        await resume_wrapper.execute({"input": "resume"}, None)
        resume_enriched = resume_callback._state.enriched_properties.copy()

        # The same set of keys should be present in both
        assert set(normal_enriched.keys()) == set(resume_enriched.keys()), (
            f"Key mismatch: normal has {set(normal_enriched.keys()) - set(resume_enriched.keys())} extra, "
            f"resume has {set(resume_enriched.keys()) - set(normal_enriched.keys())} extra"
        )

        # Verify shared static properties match
        static_keys = [
            "AgentName",
            "AgentId",
            "Model",
            "MaxTokens",
            "Temperature",
            "Engine",
            "MaxIterations",
            "IsConversational",
            "AgentRunSource",
            "ApplicationName",
            "Runtime",
            "AgentType",
        ]
        for key in static_keys:
            assert normal_enriched.get(key) == resume_enriched.get(key), (
                f"Mismatch for '{key}': normal={normal_enriched.get(key)}, resume={resume_enriched.get(key)}"
            )

    @pytest.mark.asyncio
    async def test_resume_without_event_emitter_does_not_set_enriched_properties(
        self,
        mock_delegate,
        tracer,
        tracing_callback,
        agent_info,
        mock_runtime_context,
        mock_trace_context_storage,
        saved_context,
    ):
        """Test that without an event_emitter, enriched properties remain empty on resume
        (matching the behavior of the normal flow).
        """
        mock_trace_context_storage.load_trace_context = AsyncMock(
            return_value=saved_context
        )

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUCCESSFUL
        mock_result.output = {"result": "done"}
        mock_delegate.execute.return_value = mock_result

        wrapper = InstrumentedRuntime(
            mock_delegate,
            tracer,
            tracing_callback,
            mock_runtime_context,
            agent_definition=agent_info,
            trace_context_storage=mock_trace_context_storage,
            # No event_emitter
        )

        await wrapper.execute({"input": "resume"}, None)

        # Without event_emitter, enriched properties should remain empty
        assert tracing_callback._state.enriched_properties == {}
