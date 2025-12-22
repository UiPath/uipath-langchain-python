"""Tests for UiPathTracer manual span instrumentation."""

import pytest

from uipath_agents._observability.schema import SpanType
from uipath_agents._observability.tracer import UiPathTracer

# span_exporter fixture comes from conftest.py


@pytest.fixture
def tracer(span_exporter):
    """Create a fresh tracer for testing."""
    return UiPathTracer()


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

        assert attrs["type"] == SpanType.AGENT_RUN.value
        assert attrs["agentName"] == "TestAgent"
        assert attrs["agentId"] == "test-id-123"
        assert attrs["source"] == "langchain"

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
        """Test LLM call span has correct type attribute."""
        span = tracer.start_llm_call()
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["type"] == SpanType.COMPLETION.value

    def test_span_name(self, tracer, span_exporter):
        """Test LLM call span has correct name."""
        span = tracer.start_llm_call()
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        assert spans[0].name == "LLM call"


class TestModelRunSpan:
    """Tests for model run span creation."""

    def test_creates_span_with_model_attribute(self, tracer, span_exporter):
        """Test model run span has model attribute."""
        span = tracer.start_model_run(model_name="gpt-4")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)

        assert attrs["type"] == SpanType.LLM_CALL.value
        assert attrs["model"] == "gpt-4"


class TestToolCallSpan:
    """Tests for tool call span creation."""

    def test_creates_span_with_tool_name(self, tracer, span_exporter):
        """Test tool call span has tool name attribute."""
        span = tracer.start_tool_call(tool_name="calculator")
        tracer.end_span_ok(span)

        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)

        assert attrs["type"] == SpanType.TOOL_CALL.value
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
        assert spans[0].attributes["type"] == SpanType.PROCESS_TOOL.value


class TestAgentOutputSpan:
    """Tests for agent output span creation."""

    def test_creates_span_with_output(self, tracer, span_exporter):
        """Test agent output span has output attribute."""
        tracer.emit_agent_output({"result": "success"})

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "Agent output"
        assert spans[0].attributes["type"] == SpanType.AGENT_OUTPUT.value
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

        # Find spans by type
        agent_span = next(s for s in spans if s.attributes["type"] == "agentRun")
        llm_span_result = next(s for s in spans if s.attributes["type"] == "completion")
        model_span_result = next(s for s in spans if s.attributes["type"] == "llmCall")

        # Agent span is root (no parent)
        assert agent_span.parent is None
        # LLM span's parent is agent span
        assert llm_span_result.parent.span_id == agent_span.context.span_id
        # Model span's parent is LLM span
        assert model_span_result.parent.span_id == llm_span_result.context.span_id
