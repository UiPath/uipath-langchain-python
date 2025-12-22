"""Tests for UiPathTracingCallback LangChain callback handler."""

from typing import Any
from uuid import UUID, uuid4

import pytest

from uipath_agents._observability.callback import UiPathTracingCallback
from uipath_agents._observability.schema import SpanType
from uipath_agents._observability.tracer import UiPathTracer

# span_exporter fixture comes from conftest.py


@pytest.fixture
def tracer(span_exporter):
    """Create a fresh tracer for testing."""
    return UiPathTracer()


@pytest.fixture
def callback(tracer):
    """Create callback with tracer."""
    return UiPathTracingCallback(tracer)


def _model_key(run_id: UUID) -> UUID:
    """Derive model span key from run_id (matches callback implementation)."""
    return UUID(int=run_id.int ^ 1)


class TestLlmCallbackEvents:
    """Tests for LLM callback events."""

    def test_on_llm_start_creates_spans(self, callback, span_exporter):
        """Test that on_llm_start creates LLM call and model run spans."""
        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)

        # Spans are created but not finished yet
        # LLM span stored at run_id, model span at derived key
        assert run_id in callback._spans
        assert _model_key(run_id) in callback._spans

    def test_on_llm_end_closes_spans(self, callback, span_exporter):
        """Test that on_llm_end properly closes spans."""
        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        # Spans should be removed from tracking
        assert run_id not in callback._spans
        assert _model_key(run_id) not in callback._spans

        # Verify spans were created with correct attributes
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2

        span_types = {s.attributes["type"] for s in spans}
        assert SpanType.COMPLETION.value in span_types
        assert SpanType.LLM_CALL.value in span_types

    def test_on_llm_error_closes_spans_with_error(self, callback, span_exporter):
        """Test that on_llm_error closes spans with error status."""
        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_error(ValueError("LLM error"), run_id=run_id)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2

        # Both spans should have error attribute
        for span in spans:
            assert "error" in span.attributes
            assert "LLM error" in span.attributes["error"]

    def test_on_chat_model_start_creates_spans(self, callback, span_exporter):
        """Test that on_chat_model_start creates spans like on_llm_start."""
        run_id = uuid4()
        serialized = {"kwargs": {"model": "gpt-4-turbo"}}

        callback.on_chat_model_start(serialized, [[]], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2

        model_span = next(
            s for s in spans if s.attributes["type"] == SpanType.LLM_CALL.value
        )
        assert model_span.attributes["model"] == "gpt-4-turbo"


class TestToolCallbackEvents:
    """Tests for tool callback events."""

    def test_on_tool_start_creates_span(self, callback, span_exporter):
        """Test that on_tool_start creates a tool call span."""
        run_id = uuid4()
        serialized = {"name": "calculator"}

        callback.on_tool_start(serialized, "input", run_id=run_id)

        assert run_id in callback._spans

    def test_on_tool_end_closes_span(self, callback, span_exporter):
        """Test that on_tool_end properly closes the tool span."""
        run_id = uuid4()
        serialized = {"name": "calculator"}

        callback.on_tool_start(serialized, "input", run_id=run_id)
        callback.on_tool_end("result", run_id=run_id)

        assert run_id not in callback._spans

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["type"] == SpanType.TOOL_CALL.value
        assert spans[0].attributes["toolName"] == "calculator"
        assert spans[0].name == "Tool call - calculator"

    def test_on_tool_error_closes_span_with_error(self, callback, span_exporter):
        """Test that on_tool_error closes span with error status."""
        run_id = uuid4()
        serialized = {"name": "failing_tool"}

        callback.on_tool_start(serialized, "input", run_id=run_id)
        callback.on_tool_error(RuntimeError("Tool failed"), run_id=run_id)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert "error" in spans[0].attributes
        assert "Tool failed" in spans[0].attributes["error"]


class TestModelNameExtraction:
    """Tests for model name extraction from serialized data."""

    def test_extracts_model_name_from_kwargs(self, callback, span_exporter):
        """Test extraction from kwargs.model_name."""
        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        model_span = next(
            s for s in spans if s.attributes["type"] == SpanType.LLM_CALL.value
        )
        assert model_span.attributes["model"] == "gpt-4"

    def test_extracts_model_from_kwargs_model(self, callback, span_exporter):
        """Test extraction from kwargs.model."""
        run_id = uuid4()
        serialized = {"kwargs": {"model": "claude-3"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        model_span = next(
            s for s in spans if s.attributes["type"] == SpanType.LLM_CALL.value
        )
        assert model_span.attributes["model"] == "claude-3"

    def test_falls_back_to_name(self, callback, span_exporter):
        """Test fallback to serialized.name."""
        run_id = uuid4()
        serialized = {"name": "CustomLLM", "kwargs": {}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        model_span = next(
            s for s in spans if s.attributes["type"] == SpanType.LLM_CALL.value
        )
        assert model_span.attributes["model"] == "CustomLLM"

    def test_falls_back_to_unknown(self, callback, span_exporter):
        """Test fallback to 'unknown' when no name found."""
        run_id = uuid4()
        serialized: dict[str, Any] = {}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        model_span = next(
            s for s in spans if s.attributes["type"] == SpanType.LLM_CALL.value
        )
        assert model_span.attributes["model"] == "unknown"


class TestCleanup:
    """Tests for callback cleanup."""

    def test_cleanup_closes_orphaned_spans(self, callback, span_exporter):
        """Test that cleanup closes any remaining open spans."""
        llm_run_id = uuid4()
        tool_run_id = uuid4()

        # Start spans without ending them
        callback.on_llm_start(
            {"kwargs": {"model_name": "gpt-4"}}, ["prompt"], run_id=llm_run_id
        )
        callback.on_tool_start({"name": "tool"}, "input", run_id=tool_run_id)

        # 2 LLM spans (llm + model) + 1 tool span = 3 total
        assert len(callback._spans) == 3

        callback.cleanup()

        assert len(callback._spans) == 0


class TestConcurrentRuns:
    """Tests for handling multiple concurrent runs."""

    def test_tracks_multiple_runs_independently(self, callback, span_exporter):
        """Test that multiple concurrent runs are tracked independently."""
        run_id_1 = uuid4()
        run_id_2 = uuid4()

        # Start two concurrent LLM calls
        callback.on_llm_start(
            {"kwargs": {"model_name": "gpt-4"}}, ["prompt1"], run_id=run_id_1
        )
        callback.on_llm_start(
            {"kwargs": {"model_name": "claude-3"}}, ["prompt2"], run_id=run_id_2
        )

        # 4 spans total (2 llm + 2 model)
        assert len(callback._spans) == 4
        assert run_id_1 in callback._spans
        assert run_id_2 in callback._spans

        # End first run
        callback.on_llm_end(None, run_id=run_id_1)

        assert len(callback._spans) == 2
        assert run_id_2 in callback._spans

        # End second run
        callback.on_llm_end(None, run_id=run_id_2)

        assert len(callback._spans) == 0

        # Should have 4 finished spans (2 LLM + 2 model)
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 4


class TestAgentSpanParenting:
    """Tests for agent_span as root parent."""

    def test_callback_with_agent_span_parents_correctly(self, tracer, span_exporter):
        """Test that set_agent_span makes all spans children of it."""
        callback = UiPathTracingCallback(tracer)

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

            run_id = uuid4()
            callback.on_llm_start(
                {"kwargs": {"model_name": "gpt-4"}}, ["prompt"], run_id=run_id
            )
            callback.on_llm_end(None, run_id=run_id)  # type: ignore[arg-type]

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 3  # agent + llm + model

        agent = next(s for s in spans if s.attributes["type"] == "agentRun")
        llm = next(s for s in spans if s.attributes["type"] == "completion")
        model = next(s for s in spans if s.attributes["type"] == "llmCall")

        # LLM span should be child of agent
        assert llm.parent.span_id == agent.context.span_id
        # Model span should be child of LLM
        assert model.parent.span_id == llm.context.span_id

    def test_tool_span_parents_to_agent(self, tracer, span_exporter):
        """Test tool spans parent to agent_span."""
        callback = UiPathTracingCallback(tracer)

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

            run_id = uuid4()
            callback.on_tool_start({"name": "calc"}, "input", run_id=run_id)
            callback.on_tool_end("result", run_id=run_id)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2

        agent = next(s for s in spans if s.attributes["type"] == "agentRun")
        tool = next(s for s in spans if s.attributes["type"] == "toolCall")

        assert tool.parent.span_id == agent.context.span_id

    def test_set_agent_span_clears_previous_state(self, tracer, span_exporter):
        """Test that set_agent_span clears spans from previous execution."""
        callback = UiPathTracingCallback(tracer)

        # First execution - start LLM but don't end it
        with tracer.start_agent_run("TestAgent1") as agent_span1:
            callback.set_agent_span(agent_span1)
            run_id1 = uuid4()
            callback.on_llm_start(
                {"kwargs": {"model_name": "gpt-4"}}, ["prompt"], run_id=run_id1
            )
            # Don't end - simulate orphaned spans

        # Second execution - set_agent_span should clear previous state
        with tracer.start_agent_run("TestAgent2") as agent_span2:
            callback.set_agent_span(agent_span2)
            # Previous spans should be cleared
            assert len(callback._spans) == 0
            assert callback._prompts_captured is False
