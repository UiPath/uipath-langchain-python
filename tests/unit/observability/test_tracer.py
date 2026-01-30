"""Tests for LlmOpsSpanFactory manual span instrumentation."""

from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult
from uipath.tracing import SpanStatus

from uipath_agents._observability.llmops.spans.span_attributes import SpanType
from uipath_agents._observability.llmops.spans.span_factory import LlmOpsSpanFactory

# span_exporter fixture comes from conftest.py


@pytest.fixture(autouse=True)
def clear_env_caches():
    """Clear cached environment variable functions before each test."""
    from uipath_agents._observability.llmops.spans.span_attributes import (
        get_agent_version,
        get_execution_type,
    )

    get_execution_type.cache_clear()
    get_agent_version.cache_clear()
    yield
    get_execution_type.cache_clear()
    get_agent_version.cache_clear()


@pytest.fixture
def tracer(span_exporter):
    """Create a fresh tracer for testing."""
    return LlmOpsSpanFactory()


@pytest.fixture
def mock_exporter():
    """Create a mock exporter for upsert tests."""
    exporter = MagicMock()
    exporter.upsert_span = MagicMock(return_value=SpanExportResult.SUCCESS)
    return exporter


@pytest.fixture
def tracer_with_exporter(span_exporter, mock_exporter):
    """Create tracer with mock exporter."""
    return LlmOpsSpanFactory(exporter=mock_exporter)


class TestExecutionContextOnAllSpans:
    """Tests for execution_type and agent_version on all span types."""

    def test_llm_call_includes_execution_type(self, tracer, span_exporter, monkeypatch):
        """Test LLM call span includes executionType from environment."""
        monkeypatch.setenv("UIPATH_IS_DEBUG", "True")

        span = tracer.start_llm_call()
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["executionType"] == 0  # DEBUG

    def test_llm_call_includes_agent_version(self, tracer, span_exporter, monkeypatch):
        """Test LLM call span includes agentVersion from environment."""
        monkeypatch.setenv("UIPATH_PROCESS_VERSION", "3.1.4")

        span = tracer.start_llm_call()
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["agentVersion"] == "3.1.4"

    def test_tool_call_includes_execution_type(
        self, tracer, span_exporter, monkeypatch
    ):
        """Test tool call span includes executionType from environment."""
        monkeypatch.setenv("UIPATH_IS_DEBUG", "False")

        span = tracer.start_tool_call(tool_name="calculator")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["executionType"] == 1  # RUNTIME

    def test_tool_call_includes_agent_version(self, tracer, span_exporter, monkeypatch):
        """Test tool call span includes agentVersion from environment."""
        monkeypatch.setenv("UIPATH_PROCESS_VERSION", "1.0.0")

        span = tracer.start_tool_call(tool_name="calculator")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["agentVersion"] == "1.0.0"

    def test_model_run_includes_execution_type(
        self, tracer, span_exporter, monkeypatch
    ):
        """Test model run span includes executionType from environment."""
        monkeypatch.setenv("UIPATH_IS_DEBUG", "True")

        span = tracer.start_model_run(model_name="gpt-4")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["executionType"] == 0  # DEBUG

    def test_execution_type_defaults_to_runtime_on_all_spans(
        self, tracer, span_exporter, monkeypatch
    ):
        """Test executionType defaults to RUNTIME on all span types when env not set."""
        monkeypatch.delenv("UIPATH_IS_DEBUG", raising=False)

        span = tracer.start_tool_call(tool_name="test_tool")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["executionType"] == 1  # RUNTIME


class TestReferenceIdOnAllSpans:
    """Tests for reference_id propagation to all span types."""

    def test_child_spans_inherit_reference_id_from_agent_run(
        self, tracer, span_exporter
    ):
        """Test child spans inherit reference_id from agent run."""
        with tracer.start_agent_run(agent_name="TestAgent", agent_id="test-ref-123"):
            llm_span = tracer.start_llm_call()
            tool_span = tracer.start_tool_call(tool_name="calculator")
            tracer.end_span_ok(tool_span)
            tracer.end_span_ok(llm_span)

        spans = span_exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name.startswith("Agent run"))
        llm_span_result = next(s for s in spans if s.name == "LLM call")
        tool_span_result = next(s for s in spans if s.name == "Tool call - calculator")

        # All spans should have the same reference_id
        assert agent_span.attributes["referenceId"] == "test-ref-123"
        assert llm_span_result.attributes["referenceId"] == "test-ref-123"
        assert tool_span_result.attributes["referenceId"] == "test-ref-123"

    def test_reference_id_matches_agent_id(self, tracer, span_exporter):
        """Test reference_id is set to agent_id for agent run."""
        with tracer.start_agent_run(agent_name="TestAgent", agent_id="my-agent-id"):
            pass

        spans = span_exporter.get_finished_spans()
        assert spans[0].attributes["referenceId"] == "my-agent-id"
        assert spans[0].attributes["agentId"] == "my-agent-id"

    def test_reference_id_reset_after_agent_run(self, tracer, span_exporter):
        """Test reference_id context is reset after agent run completes."""
        with tracer.start_agent_run(agent_name="TestAgent1", agent_id="agent-1"):
            pass

        # Create span outside of agent run context
        span = tracer.start_tool_call(tool_name="outside_tool")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        agent_span = next(s for s in spans if s.name.startswith("Agent run"))
        outside_span = next(s for s in spans if s.name == "Tool call - outside_tool")

        # Agent span should have reference_id
        assert agent_span.attributes["referenceId"] == "agent-1"
        # Outside span should not have reference_id
        assert "referenceId" not in outside_span.attributes

    def test_model_run_inherits_reference_id(self, tracer, span_exporter):
        """Test model run span inherits reference_id from agent run."""
        with tracer.start_agent_run(agent_name="TestAgent", agent_id="ref-456"):
            model_span = tracer.start_model_run(model_name="gpt-4")
            tracer.end_span_ok(model_span)

        spans = span_exporter.get_finished_spans()
        model_span_result = next(s for s in spans if s.name == "Model run")

        assert model_span_result.attributes["referenceId"] == "ref-456"

    def test_nested_agent_runs_use_separate_reference_ids(self, tracer, span_exporter):
        """Test nested agent runs maintain separate reference_id contexts."""
        with tracer.start_agent_run(agent_name="OuterAgent", agent_id="outer-id"):
            outer_tool = tracer.start_tool_call(tool_name="outer_tool")
            tracer.end_span_ok(outer_tool)

            # Start nested agent run with different reference_id
            with tracer.start_agent_run(agent_name="InnerAgent", agent_id="inner-id"):
                inner_tool = tracer.start_tool_call(tool_name="inner_tool")
                tracer.end_span_ok(inner_tool)

            # Back to outer context
            outer_tool2 = tracer.start_tool_call(tool_name="outer_tool2")
            tracer.end_span_ok(outer_tool2)

        spans = span_exporter.get_finished_spans()
        outer_agent = next(s for s in spans if "OuterAgent" in s.name)
        outer_tool_span = next(
            s for s in spans if "outer_tool" == s.name.split(" - ")[1]
        )
        inner_agent = next(s for s in spans if "InnerAgent" in s.name)
        inner_tool_span = next(s for s in spans if "inner_tool" in s.name)
        outer_tool2_span = next(s for s in spans if "outer_tool2" in s.name)

        # Outer context spans use outer-id
        assert outer_agent.attributes["referenceId"] == "outer-id"
        assert outer_tool_span.attributes["referenceId"] == "outer-id"
        assert outer_tool2_span.attributes["referenceId"] == "outer-id"

        # Inner context spans use inner-id
        assert inner_agent.attributes["referenceId"] == "inner-id"
        assert inner_tool_span.attributes["referenceId"] == "inner-id"


class TestAgentRunSpan:
    """Tests for agent run span creation."""

    def test_creates_span_with_correct_name(self, tracer, span_exporter):
        """Test agent run span has correct name."""
        with tracer.start_agent_run(agent_name="TestAgent"):
            pass

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "Agent run - TestAgent"

    def test_creates_span_with_correct_attributes(self, tracer, span_exporter):
        """Test agent run span has correct attributes."""
        with tracer.start_agent_run(
            agent_name="TestAgent",
            agent_id="test-id-123",
        ):
            pass

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)

        assert attrs["type"] == SpanType.AGENT_RUN
        assert attrs["agentName"] == "TestAgent"
        assert attrs["agentId"] == "test-id-123"
        assert attrs["source"] == 1  # TraceSource.Agents

    def test_includes_execution_type_from_env(self, tracer, span_exporter, monkeypatch):
        """Test agent run span includes executionType from UIPATH_IS_DEBUG env."""
        monkeypatch.setenv("UIPATH_IS_DEBUG", "True")

        with tracer.start_agent_run(agent_name="TestAgent"):
            pass

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["executionType"] == 0  # DEBUG

    def test_includes_agent_version_from_env(self, tracer, span_exporter, monkeypatch):
        """Test agent run span includes agentVersion from UIPATH_PROCESS_VERSION env."""
        monkeypatch.setenv("UIPATH_PROCESS_VERSION", "2.0.5")

        with tracer.start_agent_run(agent_name="TestAgent"):
            pass

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["agentVersion"] == "2.0.5"

    def test_execution_type_defaults_to_runtime(
        self, tracer, span_exporter, monkeypatch
    ):
        """Test executionType defaults to RUNTIME when env not set (per OR)."""
        monkeypatch.delenv("UIPATH_IS_DEBUG", raising=False)

        with tracer.start_agent_run(agent_name="TestAgent"):
            pass

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["executionType"] == 1  # RUNTIME

    def test_conversational_agent_span_name(self, tracer, span_exporter):
        """Test conversational agent has correct span name."""
        with tracer.start_agent_run(
            agent_name="ChatAgent",
            is_conversational=True,
        ):
            pass

        spans = span_exporter.get_finished_spans()
        assert spans[0].name == "Conversational agent run - ChatAgent"
        assert spans[0].attributes["isConversational"] is True

    def test_captures_error_on_exception(self, tracer, span_exporter):
        """Test that exceptions are captured in span attributes."""
        with pytest.raises(ValueError):
            with tracer.start_agent_run(agent_name="FailingAgent"):
                raise ValueError("Test error")

        spans = span_exporter.get_finished_spans()
        assert "error" in spans[0].attributes
        assert "Test error" in spans[0].attributes["error"]


class TestLlmCallSpan:
    """Tests for LLM call span creation."""

    def test_creates_span_with_correct_type(self, tracer, span_exporter):
        """Test LLM call span has correct type attribute (completion)."""
        span = tracer.start_llm_call()
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["type"] == SpanType.COMPLETION

    def test_span_name(self, tracer, span_exporter):
        """Test LLM call span has correct name."""
        span = tracer.start_llm_call()
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert spans[0].name == "LLM call"


class TestModelRunSpan:
    """Tests for model run span creation."""

    def test_creates_span_with_model_attribute(self, tracer, span_exporter):
        """Test model run span has model attribute and completion type."""
        span = tracer.start_model_run(model_name="gpt-4")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)

        # Model run uses completion type
        assert attrs["type"] == SpanType.COMPLETION
        assert attrs["model"] == "gpt-4"


class TestToolCallSpan:
    """Tests for tool call span creation."""

    def test_creates_span_with_tool_name(self, tracer, span_exporter):
        """Test tool call span has tool name attribute."""
        span = tracer.start_tool_call(tool_name="calculator")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)

        assert attrs["type"] == SpanType.TOOL_CALL
        assert attrs["toolName"] == "calculator"
        assert spans[0].name == "Tool call - calculator"

    def test_custom_tool_type(self, tracer, span_exporter):
        """Test tool call span with custom tool type."""
        span = tracer.start_tool_call(
            tool_name="my_process",
            tool_type=SpanType.PROCESS_TOOL,
        )
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert spans[0].attributes["type"] == SpanType.PROCESS_TOOL


class TestIntegrationToolSpan:
    """Tests for integration tool span creation."""

    def test_creates_span_with_correct_type(
        self, tracer: LlmOpsSpanFactory, span_exporter
    ) -> None:
        """Test integration tool span has correct type attribute."""
        span = tracer.start_integration_tool(tool_name="Web_Search")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["type"] == SpanType.INTEGRATION_TOOL

    def test_uses_tool_name_as_span_name(
        self, tracer: LlmOpsSpanFactory, span_exporter
    ) -> None:
        """Test integration tool span uses tool_name as span name."""
        span = tracer.start_integration_tool(tool_name="My_Custom_Tool")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert spans[0].name == "My_Custom_Tool"
        assert spans[0].attributes["toolName"] == "My_Custom_Tool"

    def test_respects_parent_span(
        self, tracer: LlmOpsSpanFactory, span_exporter
    ) -> None:
        """Test integration tool span correctly parents to provided span."""
        parent = tracer.start_tool_call(tool_name="wrapper_tool")
        child = tracer.start_integration_tool(
            tool_name="inner_tool", parent_span=parent
        )
        tracer.end_span_ok(child)
        tracer.end_span_ok(parent)

        spans = span_exporter.get_finished_spans()
        parent_span = next(s for s in spans if s.name == "Tool call - wrapper_tool")
        child_span = next(s for s in spans if s.name == "inner_tool")

        assert child_span.parent.span_id == parent_span.context.span_id


class TestAgentToolSpan:
    """Tests for agent tool span creation."""

    def test_creates_span_with_correct_type(
        self, tracer: LlmOpsSpanFactory, span_exporter
    ) -> None:
        """Test agent tool span has correct type attribute."""
        span = tracer.start_agent_tool(agent_name="A_plus_B")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["type"] == SpanType.AGENT_TOOL

    def test_uses_agent_name_as_span_name(
        self, tracer: LlmOpsSpanFactory, span_exporter
    ) -> None:
        """Test agent tool span uses agent_name as span name."""
        span = tracer.start_agent_tool(agent_name="Calculator_Agent")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert spans[0].name == "Calculator_Agent"
        assert spans[0].attributes["toolName"] == "Calculator_Agent"

    def test_includes_arguments(self, tracer: LlmOpsSpanFactory, span_exporter) -> None:
        """Test agent tool span includes arguments."""
        span = tracer.start_agent_tool(
            agent_name="A_plus_B", arguments={"a": 1, "b": 2}
        )
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert '"a": 1' in spans[0].attributes["arguments"]
        assert '"b": 2' in spans[0].attributes["arguments"]


class TestToolCallWithArguments:
    """Tests for tool call span with arguments and call_id."""

    def test_tool_call_includes_arguments(
        self, tracer: LlmOpsSpanFactory, span_exporter
    ) -> None:
        """Test tool call span includes arguments when provided."""
        span = tracer.start_tool_call(
            tool_name="calculator",
            arguments={"x": 10, "y": 20},
        )
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert '"x": 10' in spans[0].attributes["arguments"]

    def test_tool_call_includes_call_id(
        self, tracer: LlmOpsSpanFactory, span_exporter
    ) -> None:
        """Test tool call span includes call_id when provided."""
        span = tracer.start_tool_call(
            tool_name="calculator",
            call_id="call_abc123",
        )
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert spans[0].attributes["callId"] == "call_abc123"

    def test_tool_call_includes_tool_type_value(
        self, tracer: LlmOpsSpanFactory, span_exporter
    ) -> None:
        """Test tool call span includes toolType value."""
        span = tracer.start_tool_call(
            tool_name="my_agent",
            tool_type_value="Agent",
        )
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert spans[0].attributes["toolType"] == "Agent"


class TestAgentOutputSpan:
    """Tests for agent output span creation."""

    def test_creates_span_with_output(self, tracer, span_exporter):
        """Test agent output span has output attribute."""
        tracer.emit_agent_output({"result": "success"})

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "Agent output"
        assert spans[0].attributes["type"] == SpanType.AGENT_OUTPUT
        assert '"result": "success"' in spans[0].attributes["output"]

    def test_handles_string_output(self, tracer, span_exporter):
        """Test agent output span handles string output."""
        tracer.emit_agent_output("simple string")

        spans = span_exporter.get_finished_spans()
        assert spans[0].attributes["output"] == "simple string"


class TestSpanHierarchy:
    """Tests for span parent-child relationships."""

    def test_nested_spans_have_correct_parent(self, tracer, span_exporter):
        """Test that nested spans maintain correct hierarchy.

        Note: When using start_llm_call/start_model_run (vs context managers),
        we must explicitly pass parent_span for correct hierarchy.
        """
        with tracer.start_agent_run(agent_name="TestAgent"):
            llm_span = tracer.start_llm_call()
            # Must explicitly pass llm_span as parent since start_llm_call
            # doesn't update the current span context
            model_span = tracer.start_model_run("gpt-4", parent_span=llm_span)
            tracer.end_span_ok(model_span)
            tracer.end_span_ok(llm_span)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 3

        # Find spans by name (both LLM call and Model run have type "completion")
        agent_span = next(s for s in spans if s.name.startswith("Agent run"))
        llm_span_result = next(s for s in spans if s.name == "LLM call")
        model_span_result = next(s for s in spans if s.name == "Model run")

        # Agent span is root (no parent)
        assert agent_span.parent is None
        # LLM span's parent is agent span
        assert llm_span_result.parent.span_id == agent_span.context.span_id
        # Model span's parent is LLM span
        assert model_span_result.parent.span_id == llm_span_result.context.span_id


class TestUpsertSpanMethods:
    """Tests for upsert span methods used in interruptible tools."""

    def test_upsert_span_suspended_without_exporter_returns_false(
        self, tracer, span_exporter
    ):
        """Test upsert_span_suspended returns False when no exporter configured."""
        span = tracer.start_tool_call("test_tool")
        result = tracer.upsert_span_suspended(span)
        assert result is False
        tracer.end_span_ok(span)

    def test_upsert_span_suspended_with_exporter_calls_upsert(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test upsert_span_suspended calls exporter with UNSET status."""
        span = tracer_with_exporter.start_tool_call("test_tool")
        mock_exporter.reset_mock()  # Reset after start_tool_call's upsert
        result = tracer_with_exporter.upsert_span_suspended(span)

        assert result is True
        mock_exporter.upsert_span.assert_called_once()
        call_args = mock_exporter.upsert_span.call_args
        assert call_args[1]["status_override"] == SpanStatus.UNSET
        tracer_with_exporter.end_span_ok(span)

    def test_upsert_span_complete_without_exporter_returns_false(
        self, tracer, span_exporter
    ):
        """Test upsert_span_complete returns False when no exporter configured."""
        span = tracer.start_tool_call("test_tool")
        result = tracer.upsert_span_complete(span)
        assert result is False
        tracer.end_span_ok(span)

    def test_upsert_span_complete_with_exporter_calls_upsert(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test upsert_span_complete calls exporter with OK status."""
        span = tracer_with_exporter.start_tool_call("test_tool")
        mock_exporter.reset_mock()  # Reset after start_tool_call's upsert
        result = tracer_with_exporter.upsert_span_complete(span, SpanStatus.OK)

        assert result is True
        mock_exporter.upsert_span.assert_called_once()
        call_args = mock_exporter.upsert_span.call_args
        assert call_args[1]["status_override"] == SpanStatus.OK
        tracer_with_exporter.end_span_ok(span)

    def test_upsert_span_suspended_handles_exporter_failure(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test upsert_span_suspended handles exporter failure gracefully."""
        mock_exporter.upsert_span.return_value = SpanExportResult.FAILURE

        span = tracer_with_exporter.start_tool_call("test_tool")
        result = tracer_with_exporter.upsert_span_suspended(span)

        assert result is False
        tracer_with_exporter.end_span_ok(span)

    def test_upsert_span_suspended_handles_exception(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test upsert_span_suspended handles exceptions gracefully."""
        mock_exporter.upsert_span.side_effect = RuntimeError("Network error")

        span = tracer_with_exporter.start_tool_call("test_tool")
        result = tracer_with_exporter.upsert_span_suspended(span)

        assert result is False
        tracer_with_exporter.end_span_ok(span)


class TestLiveUpdatesUpsert:
    """Tests for live updates - spans upserting on start with UNSET status."""

    def test_upsert_span_started_calls_upsert_with_unset_status(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test upsert_span_started calls exporter with UNSET status."""
        span = tracer_with_exporter.start_tool_call("test_tool")

        # start_tool_call already calls upsert_span_started internally
        # Check that upsert was called with UNSET status
        call_args = mock_exporter.upsert_span.call_args
        assert call_args[1]["status_override"] == SpanStatus.UNSET
        tracer_with_exporter.end_span_ok(span)

    def test_start_llm_call_upserts_immediately(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test start_llm_call upserts span immediately with UNSET status."""
        mock_exporter.reset_mock()

        span = tracer_with_exporter.start_llm_call()

        mock_exporter.upsert_span.assert_called_once()
        call_args = mock_exporter.upsert_span.call_args
        assert call_args[1]["status_override"] == SpanStatus.UNSET
        tracer_with_exporter.end_span_ok(span)

    def test_start_model_run_upserts_immediately(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test start_model_run upserts span immediately with UNSET status."""
        mock_exporter.reset_mock()

        span = tracer_with_exporter.start_model_run(model_name="gpt-4")

        mock_exporter.upsert_span.assert_called_once()
        call_args = mock_exporter.upsert_span.call_args
        assert call_args[1]["status_override"] == SpanStatus.UNSET
        tracer_with_exporter.end_span_ok(span)

    def test_start_tool_call_upserts_immediately(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test start_tool_call upserts span immediately with UNSET status."""
        mock_exporter.reset_mock()

        span = tracer_with_exporter.start_tool_call(tool_name="calculator")

        mock_exporter.upsert_span.assert_called_once()
        call_args = mock_exporter.upsert_span.call_args
        assert call_args[1]["status_override"] == SpanStatus.UNSET
        tracer_with_exporter.end_span_ok(span)

    def test_start_escalation_tool_upserts_immediately(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test start_escalation_tool upserts span immediately with UNSET status."""
        mock_exporter.reset_mock()

        span = tracer_with_exporter.start_escalation_tool(app_name="ApprovalApp")

        mock_exporter.upsert_span.assert_called_once()
        call_args = mock_exporter.upsert_span.call_args
        assert call_args[1]["status_override"] == SpanStatus.UNSET
        tracer_with_exporter.end_span_ok(span)

    def test_start_process_tool_upserts_immediately(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test start_process_tool upserts span immediately with UNSET status."""
        mock_exporter.reset_mock()

        span = tracer_with_exporter.start_process_tool(process_name="InvoiceProcessor")

        mock_exporter.upsert_span.assert_called_once()
        call_args = mock_exporter.upsert_span.call_args
        assert call_args[1]["status_override"] == SpanStatus.UNSET
        tracer_with_exporter.end_span_ok(span)

    def test_start_agent_run_upserts_immediately(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test start_agent_run upserts span immediately with UNSET status."""
        mock_exporter.reset_mock()

        with tracer_with_exporter.start_agent_run(agent_name="TestAgent"):
            # Check upsert was called before yield returns
            mock_exporter.upsert_span.assert_called_once()
            call_args = mock_exporter.upsert_span.call_args
            assert call_args[1]["status_override"] == SpanStatus.UNSET

    def test_no_upsert_when_exporter_not_configured(self, tracer, span_exporter):
        """Test no upsert calls when exporter is not configured."""
        span = tracer.start_tool_call("test_tool")
        # Should not raise, just silently skip upsert
        tracer.end_span_ok(span)

    def test_upsert_failure_does_not_prevent_span_creation(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test that upsert failure doesn't prevent span from being created."""
        mock_exporter.upsert_span.return_value = SpanExportResult.FAILURE

        span = tracer_with_exporter.start_llm_call()

        # Span should still be created even if upsert failed
        assert span is not None
        tracer_with_exporter.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
