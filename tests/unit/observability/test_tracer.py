"""Tests for UiPathTracer manual span instrumentation."""

from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult
from uipath.tracing import SpanStatus

from uipath_agents._observability.schema import SpanType
from uipath_agents._observability.tracer import UiPathTracer

# span_exporter fixture comes from conftest.py


@pytest.fixture
def tracer(span_exporter):
    """Create a fresh tracer for testing."""
    return UiPathTracer()


@pytest.fixture
def mock_exporter():
    """Create a mock exporter for upsert tests."""
    exporter = MagicMock()
    exporter.upsert_span = MagicMock(return_value=SpanExportResult.SUCCESS)
    return exporter


@pytest.fixture
def tracer_with_exporter(span_exporter, mock_exporter):
    """Create tracer with mock exporter."""
    return UiPathTracer(exporter=mock_exporter)


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

        assert attrs["span_type"] == SpanType.AGENT_RUN.value
        assert attrs["agentName"] == "TestAgent"
        assert attrs["agentId"] == "test-id-123"
        assert attrs["source"] == "langchain"

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
        """Test LLM call span has correct type attribute (llmCall, not completion)."""
        span = tracer.start_llm_call()
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["span_type"] == SpanType.LLM_CALL.value

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

        # Model run uses completion type (matching Temporal schema)
        assert attrs["span_type"] == SpanType.COMPLETION.value
        assert attrs["model"] == "gpt-4"


class TestToolCallSpan:
    """Tests for tool call span creation."""

    def test_creates_span_with_tool_name(self, tracer, span_exporter):
        """Test tool call span has tool name attribute."""
        span = tracer.start_tool_call(tool_name="calculator")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)

        assert attrs["span_type"] == SpanType.TOOL_CALL.value
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
        assert spans[0].attributes["span_type"] == SpanType.PROCESS_TOOL.value


class TestAgentOutputSpan:
    """Tests for agent output span creation."""

    def test_creates_span_with_output(self, tracer, span_exporter):
        """Test agent output span has output attribute."""
        tracer.emit_agent_output({"result": "success"})

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "Agent output"
        assert spans[0].attributes["span_type"] == SpanType.AGENT_OUTPUT.value
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

    def test_upsert_span_running_without_exporter_returns_false(
        self, tracer, span_exporter
    ):
        """Test upsert_span_running returns False when no exporter configured."""
        span = tracer.start_tool_call("test_tool")
        result = tracer.upsert_span_running(span)
        assert result is False
        tracer.end_span_ok(span)

    def test_upsert_span_running_with_exporter_calls_upsert(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test upsert_span_running calls exporter with RUNNING status."""
        span = tracer_with_exporter.start_tool_call("test_tool")
        result = tracer_with_exporter.upsert_span_running(span)

        assert result is True
        mock_exporter.upsert_span.assert_called_once()
        call_args = mock_exporter.upsert_span.call_args
        assert call_args[1]["status_override"] == SpanStatus.RUNNING
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
        result = tracer_with_exporter.upsert_span_complete(span, SpanStatus.OK)

        assert result is True
        mock_exporter.upsert_span.assert_called_once()
        call_args = mock_exporter.upsert_span.call_args
        assert call_args[1]["status_override"] == SpanStatus.OK
        tracer_with_exporter.end_span_ok(span)

    def test_upsert_span_running_handles_exporter_failure(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test upsert_span_running handles exporter failure gracefully."""
        mock_exporter.upsert_span.return_value = SpanExportResult.FAILURE

        span = tracer_with_exporter.start_tool_call("test_tool")
        result = tracer_with_exporter.upsert_span_running(span)

        assert result is False
        tracer_with_exporter.end_span_ok(span)

    def test_upsert_span_running_handles_exception(
        self, tracer_with_exporter, mock_exporter, span_exporter
    ):
        """Test upsert_span_running handles exceptions gracefully."""
        mock_exporter.upsert_span.side_effect = RuntimeError("Network error")

        span = tracer_with_exporter.start_tool_call("test_tool")
        result = tracer_with_exporter.upsert_span_running(span)

        assert result is False
        tracer_with_exporter.end_span_ok(span)
