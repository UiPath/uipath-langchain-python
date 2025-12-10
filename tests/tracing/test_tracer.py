"""Test UiPathTracer produces correct span structure."""
import json

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from uipath_langchain._tracing.schema import SpanType
from uipath_langchain._tracing.tracer import UiPathTracer, get_tracer, reset_tracer


_test_provider: TracerProvider | None = None
_test_exporter: InMemorySpanExporter | None = None


def _setup_test_provider():
    """Setup test provider once per module."""
    global _test_provider, _test_exporter

    if _test_provider is None:
        _test_exporter = InMemorySpanExporter()
        _test_provider = TracerProvider()
        _test_provider.add_span_processor(SimpleSpanProcessor(_test_exporter))
        trace.set_tracer_provider(_test_provider)


@pytest.fixture
def tracer_with_exporter():
    """Setup tracer with in-memory exporter for testing."""
    _setup_test_provider()
    assert _test_exporter is not None
    _test_exporter.clear()
    reset_tracer()
    tracer = UiPathTracer()
    yield tracer, _test_exporter
    _test_exporter.clear()
    reset_tracer()


class TestAgentRunSpan:
    """Test agent run span creation."""

    def test_creates_span_with_attributes(self, tracer_with_exporter):
        """Agent run span has correct name and attributes."""
        tracer, exporter = tracer_with_exporter

        with tracer.start_agent_run("TestAgent", agent_id="agent-123"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "Agent run - TestAgent"
        assert spans[0].attributes["type"] == "agentRun"
        assert spans[0].attributes["agentName"] == "TestAgent"
        assert spans[0].attributes["agentId"] == "agent-123"
        assert spans[0].attributes["source"] == "langchain"

    def test_error_handling(self, tracer_with_exporter):
        """Agent run span captures errors correctly."""
        tracer, exporter = tracer_with_exporter

        with pytest.raises(ValueError):
            with tracer.start_agent_run("TestAgent"):
                raise ValueError("Test error")

        spans = exporter.get_finished_spans()
        error_attr = json.loads(spans[0].attributes["error"])
        assert error_attr["message"] == "Test error"
        assert spans[0].status.is_ok is False


class TestLlmCallSpan:
    """Test LLM call span creation."""

    def test_creates_span_with_attributes(self, tracer_with_exporter):
        """LLM call span has correct name and type."""
        tracer, exporter = tracer_with_exporter

        with tracer.start_llm_call():
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "LLM call"
        assert spans[0].attributes["type"] == "completion"

    def test_error_handling(self, tracer_with_exporter):
        """LLM call span captures errors correctly."""
        tracer, exporter = tracer_with_exporter

        with pytest.raises(RuntimeError):
            with tracer.start_llm_call():
                raise RuntimeError("API error")

        spans = exporter.get_finished_spans()
        error_attr = json.loads(spans[0].attributes["error"])
        assert error_attr["message"] == "API error"


class TestModelRunSpan:
    """Test Model run span creation."""

    def test_creates_span_with_attributes(self, tracer_with_exporter):
        """Model run span has correct name, type, and model."""
        tracer, exporter = tracer_with_exporter

        with tracer.start_model_run("gpt-4o"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "Model run"
        assert spans[0].attributes["type"] == "llmCall"
        assert spans[0].attributes["model"] == "gpt-4o"

    def test_error_handling(self, tracer_with_exporter):
        """Model run span captures errors correctly."""
        tracer, exporter = tracer_with_exporter

        with pytest.raises(RuntimeError):
            with tracer.start_model_run("gpt-4"):
                raise RuntimeError("API error")

        spans = exporter.get_finished_spans()
        error_attr = json.loads(spans[0].attributes["error"])
        assert error_attr["message"] == "API error"


class TestSpanHierarchy:
    """Test parent-child span relationships."""

    def test_full_hierarchy(self, tracer_with_exporter):
        """Full hierarchy: Agent run -> LLM call -> Model run."""
        tracer, exporter = tracer_with_exporter

        with tracer.start_agent_run("TestAgent"):
            with tracer.start_llm_call():
                with tracer.start_model_run("gpt-4"):
                    pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 3

        agent_span = next(s for s in spans if "Agent run" in s.name)
        llm_span = next(s for s in spans if s.name == "LLM call")
        model_span = next(s for s in spans if s.name == "Model run")

        assert llm_span.parent.span_id == agent_span.context.span_id
        assert model_span.parent.span_id == llm_span.context.span_id

    def test_multiple_iterations(self, tracer_with_exporter):
        """Multiple LLM calls are children of agent run."""
        tracer, exporter = tracer_with_exporter

        with tracer.start_agent_run("TestAgent"):
            with tracer.start_llm_call():
                with tracer.start_model_run("gpt-4"):
                    pass
            with tracer.start_llm_call():
                with tracer.start_model_run("gpt-4"):
                    pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 5  # 1 agent + 2 LLM calls + 2 model runs

        agent_span = next(s for s in spans if "Agent run" in s.name)
        llm_spans = [s for s in spans if s.name == "LLM call"]

        assert len(llm_spans) == 2
        for llm_span in llm_spans:
            assert llm_span.parent.span_id == agent_span.context.span_id


class TestToolCallSpan:
    """Test Tool call span creation."""

    def test_creates_span_with_attributes(self, tracer_with_exporter):
        """Tool call span has correct name, type, and toolName."""
        tracer, exporter = tracer_with_exporter

        with tracer.start_tool_call("my_tool"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "Tool call - my_tool"
        assert spans[0].attributes["type"] == "toolCall"
        assert spans[0].attributes["toolName"] == "my_tool"

    def test_custom_tool_type(self, tracer_with_exporter):
        """Tool call span can have custom tool type."""
        tracer, exporter = tracer_with_exporter

        with tracer.start_tool_call("my_process", tool_type=SpanType.PROCESS_TOOL):
            pass

        spans = exporter.get_finished_spans()
        assert spans[0].attributes["type"] == "processTool"


class TestAgentOutputSpan:
    """Test agent output span creation."""

    def test_emits_output_span(self, tracer_with_exporter):
        """Agent output span has correct name, type, and serializes output."""
        tracer, exporter = tracer_with_exporter

        tracer.emit_agent_output({"key": "value"})

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "Agent output"
        assert spans[0].attributes["type"] == "agentOutput"
        output = json.loads(spans[0].attributes["output"])
        assert output == {"key": "value"}


class TestTracerSingleton:
    """Test tracer singleton behavior."""

    def test_singleton_behavior(self):
        """get_tracer returns same instance, reset creates new."""
        reset_tracer()
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2

        reset_tracer()
        tracer3 = get_tracer()
        assert tracer1 is not tracer3
