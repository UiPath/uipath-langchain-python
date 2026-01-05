"""Tests for enhanced TelemetryRuntimeWrapper functionality including telemetry callback and property enrichment."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from uipath_agents._observability.callback import UiPathTracingCallback
from uipath_agents._observability.runtime_wrapper import TelemetryRuntimeWrapper
from uipath_agents._observability.span_attributes import AgentSpanInfo
from uipath_agents._observability.telemetry_callback import (
    AGENTRUN_COMPLETED,
    AGENTRUN_FAILED,
    AGENTRUN_STARTED,
    AppInsightsTelemetryCallback,
)
from uipath_agents._observability.tracer import UiPathTracer


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
    return UiPathTracer()


@pytest.fixture
def tracing_callback(tracer):
    """Create a tracing callback."""
    return UiPathTracingCallback(tracer)


@pytest.fixture
def telemetry_callback():
    """Create a telemetry callback."""
    return AppInsightsTelemetryCallback()


@pytest.fixture
def agent_info():
    """Create agent span info with telemetry properties."""
    return AgentSpanInfo(
        name="test-agent",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        model="gpt-4",
        max_tokens=1000,
        temperature=0.7,
        engine="openai",
        max_iterations=5,
        is_conversational=True,
    )


class TestTelemetryCallbackIntegration:
    """Test integration of telemetry callback with runtime wrapper."""

    def test_init_with_telemetry_callback(
        self, mock_delegate, tracer, tracing_callback, telemetry_callback, agent_info
    ):
        """Test initialization with telemetry callback."""
        wrapper = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
            telemetry_callback=telemetry_callback,
            agent_info=agent_info,
        )

        assert wrapper._telemetry_callback is telemetry_callback
        assert wrapper._agent_info is agent_info

    @pytest.mark.asyncio
    async def test_execute_calls_telemetry_callback(
        self, mock_delegate, tracer, tracing_callback, telemetry_callback, agent_info
    ):
        """Test that execute calls telemetry callback for started and completed events."""
        from uipath.runtime import UiPathRuntimeStatus

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUCCESSFUL
        mock_result.output = {"result": "test"}
        mock_delegate.execute.return_value = mock_result

        telemetry_callback.track_event = MagicMock()
        telemetry_callback.set_agent_info = MagicMock()

        wrapper = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
            telemetry_callback=telemetry_callback,
            agent_info=agent_info,
        )

        await wrapper.execute({"input": "test"}, None)

        # Verify telemetry callback was called
        telemetry_callback.set_agent_info.assert_called_with(
            "test-agent", "test-runtime-id"
        )

        # Should be called twice: once for STARTED, once for COMPLETED
        assert telemetry_callback.track_event.call_count == 2

        # Verify event names
        calls = telemetry_callback.track_event.call_args_list
        assert calls[0][0][0] == AGENTRUN_STARTED
        assert calls[1][0][0] == AGENTRUN_COMPLETED

    @pytest.mark.asyncio
    async def test_execute_calls_telemetry_callback_on_failure(
        self, mock_delegate, tracer, tracing_callback, telemetry_callback, agent_info
    ):
        """Test that execute calls telemetry callback for failed events."""
        telemetry_callback.track_event = MagicMock()
        telemetry_callback.set_agent_info = MagicMock()

        mock_delegate.execute.side_effect = RuntimeError("Test error")

        wrapper = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
            telemetry_callback=telemetry_callback,
            agent_info=agent_info,
        )

        with pytest.raises(RuntimeError):
            await wrapper.execute({"input": "test"}, None)

        # Should be called twice: once for STARTED, once for FAILED
        assert telemetry_callback.track_event.call_count == 2

        # Verify event names
        calls = telemetry_callback.track_event.call_args_list
        assert calls[0][0][0] == AGENTRUN_STARTED
        assert calls[1][0][0] == AGENTRUN_FAILED

    def test_without_telemetry_callback(
        self, mock_delegate, tracer, tracing_callback, agent_info
    ):
        """Test that wrapper works without telemetry callback."""
        wrapper = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
            agent_info=agent_info,
        )

        assert wrapper._telemetry_callback is None


class TestPropertyEnrichment:
    """Test telemetry property enrichment functionality."""

    def test_get_enriched_properties_basic(
        self, mock_delegate, tracer, tracing_callback, agent_info
    ):
        """Test basic property enrichment from AgentSpanInfo."""
        wrapper = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
            agent_info=agent_info,
        )

        base_properties = {"AgentName": "test-agent"}
        enriched = wrapper._get_enriched_properties(base_properties)

        # Verify basic properties are preserved
        assert enriched["AgentName"] == "test-agent"

        # Verify AgentSpanInfo properties are added
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
        self, mock_delegate, tracer, tracing_callback, agent_info
    ):
        """Test property enrichment with trace ID from span."""
        wrapper = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
            agent_info=agent_info,
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
        self, mock_delegate, tracer, tracing_callback
    ):
        """Test property enrichment without AgentSpanInfo."""
        wrapper = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
        )

        base_properties = {"AgentName": "test-agent"}
        enriched = wrapper._get_enriched_properties(base_properties)

        # Verify basic properties are preserved
        assert enriched["AgentName"] == "test-agent"

        # Verify AgentSpanInfo properties are not present
        assert "Model" not in enriched
        assert "MaxTokens" not in enriched

        # Verify standard properties are still added
        assert enriched["AgentRunSource"] == "playground"
        assert enriched["ApplicationName"] == "UiPath.AgentService"
        assert "AgentRunId" in enriched

    def test_get_enriched_properties_filters_none_values(
        self, mock_delegate, tracer, tracing_callback
    ):
        """Test that None values from AgentSpanInfo are not included."""
        # Create AgentSpanInfo with some None values
        partial_agent_info = AgentSpanInfo(
            name="test-agent",
            model="gpt-4",
            max_tokens=None,  # This should not be included
            temperature=0.7,
            engine=None,  # This should not be included
            is_conversational=True,
        )

        wrapper = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
            agent_info=partial_agent_info,
        )

        base_properties = {"AgentName": "test-agent"}
        enriched = wrapper._get_enriched_properties(base_properties)

        # Verify properties with values are included
        assert enriched["Model"] == "gpt-4"
        assert enriched["Temperature"] == "0.7"
        assert enriched["IsConversational"] == "True"

        # Verify None properties are not included
        assert "MaxTokens" not in enriched
        assert "Engine" not in enriched

    @patch.dict(
        os.environ,
        {
            "UIPATH_CLOUD_ORGANIZATION_ID": "test-org-id",
            "UIPATH_CLOUD_USER_ID": "test-user-id",
        },
    )
    def test_get_enriched_properties_cloud_context_from_env(
        self, mock_delegate, tracer, tracing_callback, agent_info
    ):
        """Test cloud context extraction from environment variables."""
        wrapper = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
            agent_info=agent_info,
        )

        base_properties = {"AgentName": "test-agent"}
        enriched = wrapper._get_enriched_properties(base_properties)

        assert enriched["CloudOrganizationId"] == "test-org-id"
        assert enriched["CloudUserId"] == "test-user-id"


class TestAgentRunIdConsistency:
    """Test AgentRunId consistency across telemetry events."""

    @pytest.mark.asyncio
    async def test_agent_run_id_consistent_across_events(
        self, mock_delegate, tracer, tracing_callback, telemetry_callback, agent_info
    ):
        """Test that AgentRunId is consistent across all telemetry events."""
        from uipath.runtime import UiPathRuntimeStatus

        mock_result = MagicMock()
        mock_result.status = UiPathRuntimeStatus.SUCCESSFUL
        mock_result.output = {"result": "test"}
        mock_delegate.execute.return_value = mock_result

        telemetry_callback.track_event = MagicMock()

        wrapper = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
            telemetry_callback=telemetry_callback,
            agent_info=agent_info,
        )

        await wrapper.execute({"input": "test"}, None)

        # Get the calls to track_event
        calls = telemetry_callback.track_event.call_args_list

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
        self, mock_delegate, tracer, tracing_callback, telemetry_callback, agent_info
    ):
        """Test that different wrapper instances have different AgentRunIds."""
        wrapper1 = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
            telemetry_callback=telemetry_callback,
            agent_info=agent_info,
        )

        wrapper2 = TelemetryRuntimeWrapper(
            mock_delegate,
            tracer,
            tracing_callback,
            telemetry_callback=telemetry_callback,
            agent_info=agent_info,
        )

        # Check that different instances have different run IDs
        assert wrapper1._agent_run_id != wrapper2._agent_run_id
