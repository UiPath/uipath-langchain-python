"""Tests for LlmOpsInstrumentationCallback LangChain callback handler."""

from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from uipath.core.guardrails import (
    AllFieldsSelector,
    FieldReference,
    FieldSource,
    GuardrailScope,
    SpecificFieldsSelector,
)
from uipath_langchain.agent.guardrails.types import ExecutionStage

from uipath_agents._observability.llmops.callback import LlmOpsInstrumentationCallback
from uipath_agents._observability.llmops.spans.span_attributes import SpanType
from uipath_agents._observability.llmops.spans.span_factory import LlmOpsSpanFactory
from uipath_agents._observability.llmops.spans.span_name import (
    GUARDRAIL_VALIDATION_DETAILS_KEY,
    GUARDRAIL_VALIDATION_RESULT_KEY,
    INNER_STATE_KEY,
)

# span_exporter fixture comes from conftest.py


@pytest.fixture
def tracer(span_exporter):
    """Create a fresh tracer for testing."""
    return LlmOpsSpanFactory()


@pytest.fixture
def callback(tracer):
    """Create callback with tracer, cleanup after test."""
    from opentelemetry import context

    # Capture initial context token
    initial_context = context.get_current()

    cb = LlmOpsInstrumentationCallback(tracer)
    yield cb
    cb.cleanup()  # Detach any attached OTEL context

    # Force reset to initial context state
    context.attach(initial_context)


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
        assert run_id in callback._state.spans
        assert _model_key(run_id) in callback._state.spans

    def test_on_llm_end_closes_spans(self, callback, span_exporter):
        """Test that on_llm_end properly closes spans."""
        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        # Spans should be removed from tracking
        assert run_id not in callback._state.spans
        assert _model_key(run_id) not in callback._state.spans

        # Verify spans were created
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2

        span_names = {s.name for s in spans}
        assert "LLM call" in span_names
        assert "Model run" in span_names
        # Both LLM call and Model run have type "completion"
        llm_span = next(s for s in spans if s.name == "LLM call")
        model_span = next(s for s in spans if s.name == "Model run")
        assert llm_span.attributes["type"] == SpanType.COMPLETION
        assert model_span.attributes["type"] == SpanType.COMPLETION

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

        # Model run span has the model attribute
        model_span = next(s for s in spans if s.name == "Model run")
        assert model_span.attributes["model"] == "gpt-4-turbo"


class TestToolCallbackEvents:
    """Tests for tool callback events."""

    def test_on_tool_start_creates_span(self, callback, span_exporter):
        """Test that on_tool_start creates a tool call span."""
        run_id = uuid4()
        serialized = {"name": "calculator"}

        callback.on_tool_start(serialized, "input", run_id=run_id)

        assert run_id in callback._state.spans

    def test_on_tool_end_closes_span(self, callback, span_exporter):
        """Test that on_tool_end properly closes the tool span."""
        run_id = uuid4()
        serialized = {"name": "calculator"}

        callback.on_tool_start(serialized, "input", run_id=run_id)
        callback.on_tool_end("result", run_id=run_id)

        assert run_id not in callback._state.spans

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["type"] == SpanType.TOOL_CALL
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
        llm_span = next(s for s in spans if s.name == "LLM call")
        model_span = next(s for s in spans if s.name == "Model run")

        assert llm_span.attributes["model"] == "gpt-4"
        assert model_span.attributes["model"] == "gpt-4"

    def test_extracts_model_from_kwargs_model(self, callback, span_exporter):
        """Test extraction from kwargs.model."""
        run_id = uuid4()
        serialized = {"kwargs": {"model": "claude-3"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        llm_span = next(s for s in spans if s.name == "LLM call")
        model_span = next(s for s in spans if s.name == "Model run")
        assert llm_span.attributes["model"] == "claude-3"
        assert model_span.attributes["model"] == "claude-3"

    def test_falls_back_to_name(self, callback, span_exporter):
        """Test fallback to serialized.name."""
        run_id = uuid4()
        serialized = {"name": "CustomLLM", "kwargs": {}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        model_span = next(s for s in spans if s.name == "Model run")
        assert model_span.attributes["model"] == "CustomLLM"

    def test_falls_back_to_unknown(self, callback, span_exporter):
        """Test fallback to 'unknown' when no name found."""
        run_id = uuid4()
        serialized: dict[str, Any] = {}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        model_span = next(s for s in spans if s.name == "Model run")
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
        assert len(callback._state.spans) == 3

        callback.cleanup()

        assert len(callback._state.spans) == 0


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
        assert len(callback._state.spans) == 4
        assert run_id_1 in callback._state.spans
        assert run_id_2 in callback._state.spans

        # End first run
        callback.on_llm_end(None, run_id=run_id_1)

        assert len(callback._state.spans) == 2
        assert run_id_2 in callback._state.spans

        # End second run
        callback.on_llm_end(None, run_id=run_id_2)

        assert len(callback._state.spans) == 0

        # Should have 4 finished spans (2 LLM + 2 model)
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 4


class TestAgentSpanParenting:
    """Tests for agent_span as root parent."""

    def test_callback_with_agent_span_parents_correctly(self, tracer, span_exporter):
        """Test that set_agent_span makes all spans children of it."""
        callback = LlmOpsInstrumentationCallback(tracer)
        agent_run_id = uuid4()

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            run_id = uuid4()
            callback.on_llm_start(
                {"kwargs": {"model_name": "gpt-4"}}, ["prompt"], run_id=run_id
            )
            callback.on_llm_end(None, run_id=run_id)  # type: ignore[arg-type]

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 3  # agent + llm + model

        # Find spans by name (both LLM call and Model run have type "completion")
        agent = next(s for s in spans if s.name.startswith("Agent run"))
        llm = next(s for s in spans if s.name == "LLM call")
        model = next(s for s in spans if s.name == "Model run")

        # LLM span should be child of agent
        assert llm.parent.span_id == agent.context.span_id
        # Model span should be child of LLM
        assert model.parent.span_id == llm.context.span_id

    def test_tool_span_parents_to_agent(self, tracer, span_exporter):
        """Test tool spans parent to agent_span."""
        callback = LlmOpsInstrumentationCallback(tracer)
        agent_run_id = uuid4()

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

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
        callback = LlmOpsInstrumentationCallback(tracer)
        agent_run_id_1 = uuid4()
        agent_run_id_2 = uuid4()

        # First execution - start LLM but don't end it
        with tracer.start_agent_run("TestAgent1") as agent_span1:
            callback.set_agent_span(agent_span1, agent_run_id_1)
            run_id1 = uuid4()
            callback.on_llm_start(
                {"kwargs": {"model_name": "gpt-4"}}, ["prompt"], run_id=run_id1
            )
            # Don't end - simulate orphaned spans

        # Second execution - set_agent_span should clear previous state
        with tracer.start_agent_run("TestAgent2") as agent_span2:
            callback.set_agent_span(agent_span2, agent_run_id_2)
            # Previous spans should be cleared
            assert len(callback._state.spans) == 0
            assert callback._state.prompts_captured is False


class TestGraphInterruptHandling:
    """Tests for GraphInterrupt detection in tool errors."""

    def test_is_graph_interrupt_by_type_name(self, callback):
        """Test _is_graph_interrupt detects GraphInterrupt by type name."""

        class GraphInterrupt(Exception):
            pass

        error = GraphInterrupt("Suspended for HITL")
        assert callback._tool_instrumentor._is_graph_interrupt(error) is True

    def test_is_graph_interrupt_by_str_prefix(self, callback):
        """Test _is_graph_interrupt detects GraphInterrupt by string prefix."""
        # Some errors may convert to "GraphInterrupt(...)" string
        error = Exception("GraphInterrupt(Waiting for approval)")
        assert callback._tool_instrumentor._is_graph_interrupt(error) is True

    def test_is_not_graph_interrupt_for_regular_error(self, callback):
        """Test _is_graph_interrupt returns False for regular errors."""
        error = ValueError("Some other error")
        assert callback._tool_instrumentor._is_graph_interrupt(error) is False

    def test_on_tool_error_skips_span_close_for_graph_interrupt(
        self, callback, span_exporter
    ):
        """Test on_tool_error does NOT close spans for GraphInterrupt."""

        class GraphInterrupt(Exception):
            pass

        run_id = uuid4()
        serialized = {"name": "interruptible_tool"}

        callback.on_tool_start(serialized, "input", run_id=run_id)

        # Span should be tracked
        assert run_id in callback._state.spans

        # Trigger GraphInterrupt - span should NOT be closed
        callback.on_tool_error(GraphInterrupt("Suspended"), run_id=run_id)

        # Span should still be tracked (not closed)
        assert run_id in callback._state.spans

        # No finished spans
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 0

        # Cleanup for test hygiene
        callback.cleanup()

    def test_on_tool_error_closes_span_for_regular_error(self, callback, span_exporter):
        """Test on_tool_error closes spans for non-GraphInterrupt errors."""
        run_id = uuid4()
        serialized = {"name": "failing_tool"}

        callback.on_tool_start(serialized, "input", run_id=run_id)
        callback.on_tool_error(RuntimeError("Tool failed"), run_id=run_id)

        # Span should be removed
        assert run_id not in callback._state.spans

        # Span should be finished
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert "error" in spans[0].attributes


class TestIntegrationToolCallback:
    """Tests for integration tool child span handling."""

    def _interruptible_key(self, run_id: UUID) -> UUID:
        """Derive interruptible child span key (matches callback impl)."""
        return UUID(int=run_id.int ^ 2)

    def test_on_tool_start_creates_integration_child_span(
        self, callback: LlmOpsInstrumentationCallback, span_exporter
    ) -> None:
        """Test that integration tool_type creates a child span."""
        run_id = uuid4()
        serialized = {"name": "Web_Search"}
        metadata = {"tool_type": "integration", "display_name": "Web Search Tool"}

        callback.on_tool_start(serialized, "input", run_id=run_id, metadata=metadata)

        # Both tool span and integration child span should be tracked
        assert run_id in callback._state.spans
        assert self._interruptible_key(run_id) in callback._state.spans

    def test_on_tool_end_closes_integration_child_span_first(
        self, callback: LlmOpsInstrumentationCallback, span_exporter
    ) -> None:
        """Test that integration child span closes before tool span."""
        run_id = uuid4()
        serialized = {"name": "Web_Search"}
        metadata = {"tool_type": "integration", "display_name": "Web Search Tool"}

        callback.on_tool_start(serialized, "input", run_id=run_id, metadata=metadata)
        callback.on_tool_end("result", run_id=run_id)

        # Both spans should be removed
        assert run_id not in callback._state.spans
        assert self._interruptible_key(run_id) not in callback._state.spans

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2

        tool_span = next(s for s in spans if s.name == "Tool call - Web_Search")
        child_span = next(s for s in spans if s.name == "Web Search Tool")

        # Child should be parented to tool span
        assert child_span.parent.span_id == tool_span.context.span_id
        assert child_span.attributes["type"] == SpanType.INTEGRATION_TOOL


class TestGuardrailActionDetection:
    """Tests for guardrail action detection from action nodes."""

    def test_validation_passed_ends_immediately_with_skip(
        self, callback, tracer, span_exporter
    ):
        """When validation passes, span ends immediately with action=Skip."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            run_id = uuid4()
            # Start guardrail evaluation
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Skip",
                },
            )
            # End with passed validation_result
            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                run_id=run_id,
            )

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Skip"

    def test_validation_failed_defers_until_action_node(
        self, callback, tracer, span_exporter
    ):
        """When validation fails, span ending is deferred until action node fires."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            run_id = uuid4()
            # Start guardrail evaluation
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Log",
                },
            )
            # End with validation_result (failed)
            callback.on_chain_end(
                {
                    INNER_STATE_KEY: {
                        GUARDRAIL_VALIDATION_RESULT_KEY: False,
                        GUARDRAIL_VALIDATION_DETAILS_KEY: "PII detected",
                    }
                },
                run_id=run_id,
            )

            # Action node fires with _log suffix and action_type metadata
            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard_log",
                    "action_type": "Log",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_action",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                },
            )
            callback.on_chain_end({}, run_id=action_run_id)

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Log"
        assert eval_spans[0].attributes.get("validationResult") == "PII detected"

    def test_action_node_log_sets_correct_action(self, callback, tracer, span_exporter):
        """Test action node with _log suffix sets action=log."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "prompt_injection"
            mock_guardrail.description = "Prompt Injection Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.LLM]

            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={
                    "langgraph_node": "llm_pre_execution_prompt_injection",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.LLM,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Log",
                },
            )
            callback.on_chain_end(
                {
                    INNER_STATE_KEY: {
                        GUARDRAIL_VALIDATION_RESULT_KEY: False,
                        GUARDRAIL_VALIDATION_DETAILS_KEY: "Injection detected",
                    }
                },
                run_id=run_id,
            )

            # Action node fires with _log suffix and action_type metadata
            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={
                    "langgraph_node": "llm_pre_execution_prompt_injection_log",
                    "action_type": "Log",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_action",
                    "scope": GuardrailScope.LLM,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                },
            )
            callback.on_chain_end({}, run_id=action_run_id)

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Log"

    def test_action_node_hitl_sets_correct_action(
        self, callback, tracer, span_exporter
    ):
        """Test action node with _hitl suffix sets action=Escalate.

        For Escalate actions, the eval span is NOT ended immediately - it remains
        pending until HITL review completes. A "Review task" child span is created.
        """
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "review_guard"
            mock_guardrail.description = "Review Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.LLM]

            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={
                    "langgraph_node": "llm_post_execution_review_guard",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.LLM,
                    "execution_stage": ExecutionStage.POST_EXECUTION,
                    "action_type": "Escalate",
                },
            )
            callback.on_chain_end(
                {
                    INNER_STATE_KEY: {
                        GUARDRAIL_VALIDATION_RESULT_KEY: False,
                        GUARDRAIL_VALIDATION_DETAILS_KEY: "Review needed",
                    }
                },
                run_id=run_id,
            )

            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={
                    "langgraph_node": "llm_post_execution_review_guard_hitl",
                    "action_type": "Escalate",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_action",
                    "scope": GuardrailScope.LLM,
                    "execution_stage": ExecutionStage.POST_EXECUTION,
                },
            )

            # Verify escalation state is set up correctly
            assert callback._state.pending_escalation_span is not None
            assert callback._state.pending_hitl_guardrail_span is not None

        # For Escalate, eval span is NOT finished - it's pending HITL review
        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("type") == "guardrailEvaluation"
        ]
        # Eval span should NOT be finished yet (pending HITL)
        assert len(eval_spans) == 0

        # Review task span should be created but not finished
        review_spans = [s for s in spans if s.name == "Review task"]
        assert len(review_spans) == 0  # Not finished, pending HITL

    def test_guardrail_named_with_action_suffix_creates_span(
        self, callback, tracer, span_exporter
    ):
        """Guardrail named 'test_guardrail_log' should not be misidentified as action node.

        This tests the fix for a bug where guardrails with names ending in _log, _block,
        _hitl, or _filter were incorrectly detected as action nodes due to suffix matching.
        """
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "test_guardrail_log"
            mock_guardrail.description = "Test Guardrail Log"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            # Eval node for guardrail named "test_guardrail_log"
            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_test_guardrail_log",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Skip",
                },
            )

            # Validation passes - span should end immediately
            callback.on_chain_end(
                {
                    INNER_STATE_KEY: {
                        GUARDRAIL_VALIDATION_RESULT_KEY: True,
                        GUARDRAIL_VALIDATION_DETAILS_KEY: "No issues",
                    }
                },
                run_id=run_id,
            )

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("type") == "guardrailEvaluation"
        ]
        # Span should exist and have correct attributes
        assert len(eval_spans) == 1
        assert eval_spans[0].name == "test_guardrail_log"
        assert eval_spans[0].attributes.get("action") == "Skip"

    def test_command_object_extracts_validation_result(
        self, callback, tracer, span_exporter
    ):
        """Test validation result extraction from LangGraph Command objects."""

        class MockCommand:
            """Mock LangGraph Command object with update dict."""

            def __init__(self, update: dict[str, Any]) -> None:
                self.update = update

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Log",
                },
            )
            # End with Command object containing validation failure
            command = MockCommand(
                update={
                    INNER_STATE_KEY: {
                        GUARDRAIL_VALIDATION_RESULT_KEY: False,
                        GUARDRAIL_VALIDATION_DETAILS_KEY: "PII detected in input",
                    }
                }
            )
            callback.on_chain_end(command, run_id=run_id)

            # Action node fires
            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard_log",
                    "action_type": "Log",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_action",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                },
            )
            callback.on_chain_end({}, run_id=action_run_id)

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Log"
        assert (
            eval_spans[0].attributes.get("validationResult") == "PII detected in input"
        )

    def test_command_object_passed_validation(self, callback, tracer, span_exporter):
        """Test Command object with None validation result is treated as passed."""

        class MockCommand:
            def __init__(self, update: dict[str, Any]) -> None:
                self.update = update

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Skip",
                },
            )
            # Command with validation result = passed
            command = MockCommand(
                update={INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}}
            )
            callback.on_chain_end(command, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Skip"


class TestToolGuardrailParenting:
    """Tests for tool guardrail spans parenting to tool span."""

    def test_tool_pre_guardrail_parents_to_tool_span(
        self, callback, tracer, span_exporter
    ):
        """Tool pre guardrails should be children of the current tool span."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "input_guard"
            mock_guardrail.description = "Input Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            tool_run_id = uuid4()
            callback.on_tool_start({"name": "my_tool"}, "input", run_id=tool_run_id)

            # Now a tool_pre guardrail fires
            guard_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=guard_run_id,
                metadata={
                    "langgraph_node": "tool_pre_execution_input_guard",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.TOOL,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Skip",
                },
            )
            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                run_id=guard_run_id,
            )

            # Close tool_pre container manually since tool_end doesn't close it
            callback._guardrail_instrumentor.close_container(
                GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION
            )

            callback.on_tool_end("result", run_id=tool_run_id)

        spans = span_exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.attributes.get("type") == "toolCall")
        container_span = next(
            s for s in spans if s.attributes.get("type") == "toolPreGuardrails"
        )

        # Container should be child of tool span
        assert container_span.parent.span_id == tool_span.context.span_id

    def test_tool_post_guardrail_parents_to_tool_span(
        self, callback, tracer, span_exporter
    ):
        """Tool post guardrails should be children of the current tool span.

        In actual LangGraph execution, tool_post guardrails fire AFTER on_tool_end.
        The callback keeps _current_tool_span until post guardrails complete.
        """
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "output_guard"
            mock_guardrail.description = "Output Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            tool_run_id = uuid4()
            callback.on_tool_start({"name": "my_tool"}, "input", run_id=tool_run_id)

            # In LangGraph: on_tool_end fires BEFORE tool_post guardrails
            callback.on_tool_end("result", run_id=tool_run_id)

            # Tool post guardrail fires AFTER tool ends (real execution order)
            guard_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=guard_run_id,
                metadata={
                    "langgraph_node": "tool_post_execution_output_guard",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.TOOL,
                    "execution_stage": ExecutionStage.POST_EXECUTION,
                    "action_type": "Skip",
                },
            )
            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                run_id=guard_run_id,
            )

            # Cleanup containers (in real execution, next phase or agent end closes them)
            callback.cleanup_containers()

        spans = span_exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.attributes.get("type") == "toolCall")
        container_span = next(
            s for s in spans if s.attributes.get("type") == "toolPostGuardrails"
        )

        assert container_span.parent.span_id == tool_span.context.span_id


class TestPhaseTransitions:
    """Tests for guardrail phase transition logic."""

    def test_tool_pre_closes_llm_post(self, callback, tracer, span_exporter):
        """Transitioning to tool_pre should close llm_post container."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrails
            mock_guardrail_llm = MagicMock()
            mock_guardrail_llm.name = "guard1"
            mock_guardrail_llm.description = "LLM Guard"
            mock_guardrail_llm.enabled_for_evals = True
            mock_guardrail_llm.guardrail_type = "builtInValidator"
            mock_guardrail_llm.selector = MagicMock()
            mock_guardrail_llm.selector.scopes = [GuardrailScope.LLM]

            mock_guardrail_tool = MagicMock()
            mock_guardrail_tool.name = "guard2"
            mock_guardrail_tool.description = "Tool Guard"
            mock_guardrail_tool.enabled_for_evals = True
            mock_guardrail_tool.guardrail_type = "builtInValidator"
            mock_guardrail_tool.selector = MagicMock()
            mock_guardrail_tool.selector.scopes = [GuardrailScope.TOOL]

            # LLM post guardrail creates llm_post container
            run_id1 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id1,
                metadata={
                    "langgraph_node": "llm_post_execution_guard1",
                    "guardrail": mock_guardrail_llm,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.LLM,
                    "execution_stage": ExecutionStage.POST_EXECUTION,
                    "action_type": "Skip",
                },
            )
            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                run_id=run_id1,
            )

            # Should have llm_post container
            assert (
                GuardrailScope.LLM,
                ExecutionStage.POST_EXECUTION,
            ) in callback._state.guardrail_containers

            # Start tool
            tool_run_id = uuid4()
            callback.on_tool_start({"name": "my_tool"}, "input", run_id=tool_run_id)

            # Tool pre guardrail should close llm_post
            run_id2 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id2,
                metadata={
                    "langgraph_node": "tool_pre_execution_guard",
                    "guardrail": mock_guardrail_tool,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.TOOL,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Skip",
                },
            )

            # llm_post should be closed
            assert (
                GuardrailScope.LLM,
                ExecutionStage.POST_EXECUTION,
            ) not in callback._state.guardrail_containers

            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                run_id=run_id2,
            )
            callback.on_tool_end("result", run_id=tool_run_id)

    def test_tool_post_closes_tool_pre(self, callback, tracer, span_exporter):
        """Transitioning to tool_post should close tool_pre container."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrails
            mock_guardrail_pre = MagicMock()
            mock_guardrail_pre.name = "pre_guard"
            mock_guardrail_pre.description = "Tool Pre Guard"
            mock_guardrail_pre.enabled_for_evals = True
            mock_guardrail_pre.guardrail_type = "builtInValidator"
            mock_guardrail_pre.selector = MagicMock()
            mock_guardrail_pre.selector.scopes = [GuardrailScope.TOOL]

            mock_guardrail_post = MagicMock()
            mock_guardrail_post.name = "post_guard"
            mock_guardrail_post.description = "Tool Post Guard"
            mock_guardrail_post.enabled_for_evals = True
            mock_guardrail_post.guardrail_type = "builtInValidator"
            mock_guardrail_post.selector = MagicMock()
            mock_guardrail_post.selector.scopes = [GuardrailScope.TOOL]

            tool_run_id = uuid4()
            callback.on_tool_start({"name": "my_tool"}, "input", run_id=tool_run_id)

            # Tool pre guardrail
            run_id1 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id1,
                metadata={
                    "langgraph_node": "tool_pre_execution_guard",
                    "guardrail": mock_guardrail_pre,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.TOOL,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Skip",
                },
            )
            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                run_id=run_id1,
            )

            assert (
                GuardrailScope.TOOL,
                ExecutionStage.PRE_EXECUTION,
            ) in callback._state.guardrail_containers

            # Tool post guardrail should close tool_pre
            run_id2 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id2,
                metadata={
                    "langgraph_node": "tool_post_execution_guard",
                    "guardrail": mock_guardrail_post,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.TOOL,
                    "execution_stage": ExecutionStage.POST_EXECUTION,
                    "action_type": "Skip",
                },
            )

            assert (
                GuardrailScope.TOOL,
                ExecutionStage.PRE_EXECUTION,
            ) not in callback._state.guardrail_containers

            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                run_id=run_id2,
            )
            callback.on_tool_end("result", run_id=tool_run_id)

    def test_agent_post_closes_tool_post(self, callback, tracer, span_exporter):
        """Transitioning to agent_post should close tool_post container."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrails
            mock_guardrail_tool = MagicMock()
            mock_guardrail_tool.name = "tool_guard"
            mock_guardrail_tool.description = "Tool Post Guard"
            mock_guardrail_tool.enabled_for_evals = True
            mock_guardrail_tool.guardrail_type = "builtInValidator"
            mock_guardrail_tool.selector = MagicMock()
            mock_guardrail_tool.selector.scopes = [GuardrailScope.TOOL]

            mock_guardrail_agent = MagicMock()
            mock_guardrail_agent.name = "agent_guard"
            mock_guardrail_agent.description = "Agent Post Guard"
            mock_guardrail_agent.enabled_for_evals = True
            mock_guardrail_agent.guardrail_type = "builtInValidator"
            mock_guardrail_agent.selector = MagicMock()
            mock_guardrail_agent.selector.scopes = [GuardrailScope.AGENT]

            tool_run_id = uuid4()
            callback.on_tool_start({"name": "my_tool"}, "input", run_id=tool_run_id)

            # Tool post guardrail
            run_id1 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id1,
                metadata={
                    "langgraph_node": "tool_post_execution_guard",
                    "guardrail": mock_guardrail_tool,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.TOOL,
                    "execution_stage": ExecutionStage.POST_EXECUTION,
                    "action_type": "Skip",
                },
            )
            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                run_id=run_id1,
            )

            # Don't end tool yet - agent_post fires before tool ends in some flows

            assert (
                GuardrailScope.TOOL,
                ExecutionStage.POST_EXECUTION,
            ) in callback._state.guardrail_containers

            # Agent post guardrail should close tool_post
            run_id2 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id2,
                metadata={
                    "langgraph_node": "agent_post_execution_guard",
                    "guardrail": mock_guardrail_agent,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.POST_EXECUTION,
                    "action_type": "Skip",
                },
            )

            assert (
                GuardrailScope.TOOL,
                ExecutionStage.POST_EXECUTION,
            ) not in callback._state.guardrail_containers

            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                run_id=run_id2,
            )
            callback.on_tool_end("result", run_id=tool_run_id)


class TestToolGuardrailRealExecutionOrder:
    """Tests with real LangGraph execution order (guardrails fire before tool start)."""

    def test_tool_pre_guardrail_before_tool_start(
        self, callback, tracer, span_exporter
    ):
        """Tool pre guardrail fires BEFORE on_tool_start - should still parent correctly."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "deterministic"
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            # Real order: guardrail FIRST (before tool starts)
            guard_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=guard_run_id,
                metadata={
                    "langgraph_node": "tool_pre_execution_pii_guard",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.TOOL,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Skip",
                    "tool_name": "my_tool",
                },
            )
            callback.on_chain_end({}, run_id=guard_run_id)

            # THEN tool starts
            tool_run_id = uuid4()
            callback.on_tool_start({"name": "my_tool"}, "input", run_id=tool_run_id)
            callback.on_tool_end("result", run_id=tool_run_id)

        spans = span_exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.attributes.get("type") == "toolCall")
        container = next(
            s for s in spans if s.attributes.get("type") == "toolPreGuardrails"
        )

        # Container should be child of tool span
        assert container.parent.span_id == tool_span.context.span_id
        # Tool span should have correct name after enrichment
        assert tool_span.name == "Tool call - my_tool"

    def test_tool_blocked_by_guardrail_no_orphan(self, callback, tracer, span_exporter):
        """When guardrail blocks, placeholder tool span should end cleanly."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "deterministic"
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            # Guardrail fires (creates placeholder tool span)
            guard_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=guard_run_id,
                metadata={
                    "langgraph_node": "tool_pre_execution_pii_guard",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.TOOL,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Skip",
                    "tool_name": "my_tool",
                },
            )
            # Placeholder should exist
            assert callback._state.current_tool_span is not None

            # Guardrail fails
            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: "PII detected"}},
                run_id=guard_run_id,
            )

            # Block action fires - tool will NOT execute
            block_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=block_run_id,
                metadata={
                    "langgraph_node": "tool_pre_execution_pii_guard_block",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_action",
                    "scope": GuardrailScope.TOOL,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Block",
                    "tool_name": "my_tool",
                    "reason": "PII detected",
                },
            )
            # Block action ends with error
            callback.on_chain_error(
                Exception("PII detected"),
                run_id=block_run_id,
            )

            # Placeholder should be cleaned up
            assert callback._state.current_tool_span is None

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("type") == "toolCall"]
        # Placeholder was created and properly ended
        assert len(tool_spans) == 1

    def test_multiple_tool_pre_guardrails_before_tool(
        self, callback, tracer, span_exporter
    ):
        """Multiple tool_pre guardrails fire before tool - all parent to same tool span."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrails
            mock_guardrail1 = MagicMock()
            mock_guardrail1.name = "guard1"
            mock_guardrail1.description = "Guard 1"
            mock_guardrail1.enabled_for_evals = True
            mock_guardrail1.guardrail_type = "deterministic"
            mock_guardrail1.selector.scopes = [GuardrailScope.TOOL]

            mock_guardrail2 = MagicMock()
            mock_guardrail2.name = "guard2"
            mock_guardrail2.description = "Guard 2"
            mock_guardrail2.enabled_for_evals = True
            mock_guardrail2.guardrail_type = "deterministic"
            mock_guardrail2.selector.scopes = [GuardrailScope.TOOL]

            # First guardrail fires (creates placeholder)
            guard1_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=guard1_id,
                metadata={
                    "langgraph_node": "tool_pre_execution_guard1",
                    "guardrail": mock_guardrail1,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.TOOL,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Skip",
                    "tool_name": "my_tool",
                },
            )
            callback.on_chain_end({}, run_id=guard1_id)

            # Verify placeholder was created
            assert callback._state.current_tool_span is not None
            placeholder_span = callback._state.current_tool_span

            # Second guardrail fires (should reuse same placeholder)
            guard2_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=guard2_id,
                metadata={
                    "langgraph_node": "tool_pre_execution_guard2",
                    "guardrail": mock_guardrail2,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.TOOL,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Skip",
                    "tool_name": "my_tool",
                },
            )
            callback.on_chain_end({}, run_id=guard2_id)

            # Verify same placeholder is still used
            assert callback._state.current_tool_span is placeholder_span

            # Tool starts
            tool_run_id = uuid4()
            callback.on_tool_start({"name": "my_tool"}, "input", run_id=tool_run_id)
            callback.on_tool_end("result", run_id=tool_run_id)

        spans = span_exporter.get_finished_spans()

        # Only one tool span should exist (the placeholder was reused)
        tool_spans = [s for s in spans if s.attributes.get("type") == "toolCall"]
        assert len(tool_spans) == 1

        tool_span = tool_spans[0]
        container = next(
            s for s in spans if s.attributes.get("type") == "toolPreGuardrails"
        )

        # Container should be child of tool span
        assert container.parent.span_id == tool_span.context.span_id
        # Tool should have correct name after enrichment
        assert tool_span.name == "Tool call - my_tool"


class TestSpanStackUnit:
    """Unit tests for span stack functionality."""

    def test_push_span_creates_stack(
        self, callback: LlmOpsInstrumentationCallback
    ) -> None:
        """Pushing a span creates a stack entry for the run_id."""
        from unittest.mock import MagicMock

        from uipath_agents._observability.llmops.span_hierarchy import (
            SpanHierarchyManager,
        )

        run_id = uuid4()
        span = MagicMock()

        SpanHierarchyManager.clear_all()
        SpanHierarchyManager.push(run_id, span)

        assert SpanHierarchyManager.has_stack(run_id)
        assert SpanHierarchyManager.current(run_id) is span

    def test_push_multiple_spans_stacks(
        self, callback: LlmOpsInstrumentationCallback
    ) -> None:
        """Multiple pushes create a proper stack (LIFO)."""
        from unittest.mock import MagicMock

        from uipath_agents._observability.llmops.span_hierarchy import (
            SpanHierarchyManager,
        )

        run_id = uuid4()
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        SpanHierarchyManager.clear_all()
        SpanHierarchyManager.push(run_id, span1)
        SpanHierarchyManager.push(run_id, span2)

        ancestors = SpanHierarchyManager.ancestors(run_id)
        assert len(ancestors) == 2
        assert ancestors[0] is span1
        assert ancestors[1] is span2

    def test_pop_span_returns_lifo(
        self, callback: LlmOpsInstrumentationCallback
    ) -> None:
        """Pop returns spans in LIFO order."""
        from unittest.mock import MagicMock

        from uipath_agents._observability.llmops.span_hierarchy import (
            SpanHierarchyManager,
        )

        run_id = uuid4()
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        SpanHierarchyManager.clear_all()
        SpanHierarchyManager.push(run_id, span1)
        SpanHierarchyManager.push(run_id, span2)

        popped = SpanHierarchyManager.pop(run_id)
        assert popped is span2

        popped = SpanHierarchyManager.pop(run_id)
        assert popped is span1

    def test_pop_empty_returns_none(
        self, callback: LlmOpsInstrumentationCallback
    ) -> None:
        """Pop on empty stack returns None."""
        from uipath_agents._observability.llmops.span_hierarchy import (
            SpanHierarchyManager,
        )

        run_id = uuid4()
        SpanHierarchyManager.clear_all()

        result = SpanHierarchyManager.pop(run_id)
        assert result is None

    def test_get_current_span_returns_top(
        self, callback: LlmOpsInstrumentationCallback
    ) -> None:
        """_get_current_span returns top of stack when run_id is available."""
        from unittest.mock import MagicMock, patch

        from uipath_agents._observability.llmops.callback import _get_current_span
        from uipath_agents._observability.llmops.span_hierarchy import (
            SpanHierarchyManager,
        )

        run_id = uuid4()
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        SpanHierarchyManager.clear_all()
        SpanHierarchyManager.push(run_id, span1)
        SpanHierarchyManager.push(run_id, span2)

        # Mock get_current_run_id to return our run_id
        with patch(
            "uipath_agents._observability.llmops.callback.get_current_run_id",
            return_value=run_id,
        ):
            result = _get_current_span()
            assert result is span2

    def test_get_ancestor_spans_returns_copy(
        self, callback: LlmOpsInstrumentationCallback
    ) -> None:
        """_get_ancestor_spans returns a copy of the stack."""
        from unittest.mock import MagicMock, patch

        from uipath_agents._observability.llmops.callback import _get_ancestor_spans
        from uipath_agents._observability.llmops.span_hierarchy import (
            SpanHierarchyManager,
        )

        run_id = uuid4()
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        SpanHierarchyManager.clear_all()
        SpanHierarchyManager.push(run_id, span1)
        SpanHierarchyManager.push(run_id, span2)

        # Mock get_current_run_id to return our run_id
        with patch(
            "uipath_agents._observability.llmops.callback.get_current_run_id",
            return_value=run_id,
        ):
            ancestors = _get_ancestor_spans()
            assert len(ancestors) == 2
            assert ancestors[0] is span1
            assert ancestors[1] is span2

            # Should be a copy, not the original
            ancestors.pop()
            # Original stack should still have 2 spans
            assert len(SpanHierarchyManager.ancestors(run_id)) == 2

    def test_set_agent_span_clears_only_specified_run_id_stack(
        self, callback: LlmOpsInstrumentationCallback, tracer
    ) -> None:
        """set_agent_span clears only the specified run_id's stack, not others."""
        from unittest.mock import MagicMock

        from uipath_agents._observability.llmops.span_hierarchy import (
            SpanHierarchyManager,
        )

        run_id_1 = uuid4()
        run_id_2 = uuid4()
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        SpanHierarchyManager.clear_all()
        SpanHierarchyManager.push(run_id_1, span1)
        SpanHierarchyManager.push(run_id_2, span2)
        assert SpanHierarchyManager.has_stack(run_id_1)
        assert SpanHierarchyManager.has_stack(run_id_2)

        # set_agent_span for run_id_1 should only clear run_id_1's stack
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, run_id_1)

        # run_id_1 stack should have agent_span as first element
        assert SpanHierarchyManager.has_stack(run_id_1)
        ancestors_1 = SpanHierarchyManager.ancestors(run_id_1)
        assert len(ancestors_1) == 1
        assert ancestors_1[0] is agent_span

        # run_id_2 stack should remain untouched
        assert SpanHierarchyManager.has_stack(run_id_2)
        ancestors_2 = SpanHierarchyManager.ancestors(run_id_2)
        assert len(ancestors_2) == 1
        assert ancestors_2[0] is span2

    def test_set_agent_span_pushes_agent_as_first_span(
        self, callback: LlmOpsInstrumentationCallback, tracer
    ) -> None:
        """set_agent_span pushes agent_span as the first span in the run_id's stack."""
        from uipath_agents._observability.llmops.span_hierarchy import (
            SpanHierarchyManager,
        )

        SpanHierarchyManager.clear_all()
        run_id = uuid4()

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, run_id)

        # Verify agent_span is in stack
        assert SpanHierarchyManager.has_stack(run_id)
        ancestors = SpanHierarchyManager.ancestors(run_id)
        assert len(ancestors) == 1
        assert ancestors[0] is agent_span


class TestSpanStackIntegration:
    """Integration tests for span stack across event types."""

    def test_llm_span_in_stack_during_execution(
        self, callback: LlmOpsInstrumentationCallback, span_exporter
    ) -> None:
        """Model span should be in stack during LLM execution."""
        from unittest.mock import patch

        from uipath_agents._observability.llmops.callback import _get_current_span

        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)

        # Mock get_current_run_id to simulate being in langchain context
        with patch(
            "uipath_agents._observability.llmops.callback.get_current_run_id",
            return_value=run_id,
        ):
            current = _get_current_span()
            assert current is not None
            model_span = callback._state.spans.get(_model_key(run_id))
            assert current is model_span

        callback.on_llm_end(None, run_id=run_id)  # type: ignore[arg-type]

    def test_llm_span_popped_on_end(
        self, callback: LlmOpsInstrumentationCallback, span_exporter
    ) -> None:
        """Stack should be empty after LLM completes."""
        from uipath_agents._observability.llmops.span_hierarchy import (
            SpanHierarchyManager,
        )

        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)  # type: ignore[arg-type]

        # Stack for this run_id should be empty
        assert not SpanHierarchyManager.has_stack(
            run_id
        ) or not SpanHierarchyManager.current(run_id)

    def test_tool_span_in_stack_during_execution(
        self, callback: LlmOpsInstrumentationCallback, span_exporter
    ) -> None:
        """Tool span should be in stack during tool execution."""
        from unittest.mock import patch

        from uipath_agents._observability.llmops.callback import _get_current_span

        run_id = uuid4()
        serialized = {"name": "my_tool"}

        callback.on_tool_start(serialized, "input", run_id=run_id)

        with patch(
            "uipath_agents._observability.llmops.callback.get_current_run_id",
            return_value=run_id,
        ):
            current = _get_current_span()
            assert current is not None
            tool_span = callback._state.spans.get(run_id)
            assert current is tool_span

        callback.on_tool_end("result", run_id=run_id)

    def test_nested_tool_child_on_top(
        self, callback: LlmOpsInstrumentationCallback, span_exporter
    ) -> None:
        """Tool with child span should have child on top of stack."""
        from unittest.mock import patch

        from uipath_agents._observability.llmops.callback import (
            _get_ancestor_spans,
            _get_current_span,
        )

        run_id = uuid4()
        serialized = {"name": "escalate_tool"}
        metadata = {"tool_type": "escalation", "display_name": "Escalate App"}

        callback.on_tool_start(serialized, "input", run_id=run_id, metadata=metadata)

        with patch(
            "uipath_agents._observability.llmops.callback.get_current_run_id",
            return_value=run_id,
        ):
            # Child span should be on top
            current = _get_current_span()
            child_key = UUID(int=run_id.int ^ 2)
            child_span = callback._state.spans.get(child_key)
            assert current is child_span

            # Ancestors should include both
            ancestors = _get_ancestor_spans()
            assert len(ancestors) == 2

        callback.on_tool_end("result", run_id=run_id)

    def test_guardrail_span_in_stack(
        self, callback: LlmOpsInstrumentationCallback, tracer, span_exporter
    ) -> None:
        """Guardrail span should be in stack during evaluation."""
        from uipath_agents._observability.llmops.callback import _get_current_span

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Skip",
                },
            )

            with patch(
                "uipath_agents._observability.llmops.callback.get_current_run_id",
                return_value=run_id,
            ):
                current = _get_current_span()
                assert current is not None

            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                run_id=run_id,
            )

    def test_guardrail_span_popped_on_end(
        self, callback: LlmOpsInstrumentationCallback, tracer, span_exporter
    ) -> None:
        """Guardrail span should be popped after chain end."""
        from uipath_agents._observability.llmops.span_hierarchy import (
            SpanHierarchyManager,
        )

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={"langgraph_node": "agent_pre_execution_pii_guard"},
            )
            callback.on_chain_end({}, run_id=run_id)

            # Stack should be empty for this run_id
            assert not SpanHierarchyManager.has_stack(
                run_id
            ) or not SpanHierarchyManager.current(run_id)


class TestSpanStackParallelExecution:
    """Tests for parallel execution with stack isolation."""

    @pytest.mark.asyncio
    async def test_parallel_tools_isolated_stacks(
        self, callback: LlmOpsInstrumentationCallback, span_exporter
    ) -> None:
        """Two concurrent tools should have independent stacks."""
        import asyncio
        from unittest.mock import patch

        from uipath_agents._observability.llmops.callback import _get_current_span

        run_id_1 = uuid4()
        run_id_2 = uuid4()
        results: dict[str, Any] = {}

        async def tool_1() -> None:
            callback.on_tool_start({"name": "tool_1"}, "input", run_id=run_id_1)
            await asyncio.sleep(0.01)  # Simulate async work
            # Mock get_current_run_id to return this tool's run_id
            with patch(
                "uipath_agents._observability.llmops.callback.get_current_run_id",
                return_value=run_id_1,
            ):
                current = _get_current_span()
                results["tool_1_span"] = current
                results["tool_1_expected"] = callback._state.spans.get(run_id_1)
            callback.on_tool_end("result", run_id=run_id_1)

        async def tool_2() -> None:
            callback.on_tool_start({"name": "tool_2"}, "input", run_id=run_id_2)
            await asyncio.sleep(0.01)
            with patch(
                "uipath_agents._observability.llmops.callback.get_current_run_id",
                return_value=run_id_2,
            ):
                current = _get_current_span()
                results["tool_2_span"] = current
                results["tool_2_expected"] = callback._state.spans.get(run_id_2)
            callback.on_tool_end("result", run_id=run_id_2)

        await asyncio.gather(tool_1(), tool_2())

        # Each tool should see its own span as current
        assert results["tool_1_span"] is results["tool_1_expected"]
        assert results["tool_2_span"] is results["tool_2_expected"]
        assert results["tool_1_span"] is not results["tool_2_span"]


class TestGuardrailTelemetryEvents:
    """Tests for guardrail telemetry event tracking."""

    def test_skip_action_tracks_guardrail_skipped_event(
        self, callback, tracer, span_exporter
    ):
        """When validation passes (Skip), Guardrail.Skipped event should be tracked."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Skip",
                    },
                )
                callback.on_chain_end(
                    {
                        INNER_STATE_KEY: {
                            GUARDRAIL_VALIDATION_RESULT_KEY: True,
                            GUARDRAIL_VALIDATION_DETAILS_KEY: "No PII found",
                        }
                    },
                    run_id=run_id,
                )

                # Verify Guardrail.Skipped event was tracked
                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args[0][0] == "Guardrail.Skipped"
                props = call_args[0][1]
                assert props["ActionType"] == "Skip"
                assert props["AgentName"] == "TestAgent"
                assert props.get("Reason") == "No PII found"

    def test_block_action_tracks_guardrail_blocked_event(
        self, callback, tracer, span_exporter
    ):
        """When validation fails with Block, Guardrail.Blocked event should be tracked."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Block",
                    },
                )
                callback.on_chain_end(
                    {
                        INNER_STATE_KEY: {
                            GUARDRAIL_VALIDATION_RESULT_KEY: False,
                            GUARDRAIL_VALIDATION_DETAILS_KEY: "PII detected",
                        }
                    },
                    run_id=run_id,
                )

                # Action node fires with _block suffix
                action_run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=action_run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard_block",
                        "action_type": "Block",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_action",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                    },
                )
                # Event is tracked on chain end
                callback.on_chain_end({}, run_id=action_run_id)

                # Verify Guardrail.Blocked event was tracked
                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args[0][0] == "Guardrail.Blocked"
                props = call_args[0][1]
                assert props["ActionType"] == "Block"
                assert props["AgentName"] == "TestAgent"

    def test_log_action_tracks_guardrail_logged_event(
        self, callback, tracer, span_exporter
    ):
        """When validation fails with Log, Guardrail.Logged event should be tracked."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "prompt_injection"
            mock_guardrail.description = "Prompt Injection Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.LLM]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "llm_pre_execution_prompt_injection",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.LLM,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Log",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: False}},
                    run_id=run_id,
                )

                # Action node fires with _log suffix
                action_run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=action_run_id,
                    metadata={
                        "langgraph_node": "llm_pre_execution_prompt_injection_log",
                        "severity_level": "Info",
                        "action_type": "Log",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_action",
                        "scope": GuardrailScope.LLM,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                    },
                )
                # Event is tracked on chain end
                callback.on_chain_end({}, run_id=action_run_id)

                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args[0][0] == "Guardrail.Logged"
                props = call_args[0][1]
                assert props["ActionType"] == "Log"
                assert props["SeverityLevel"] == "Info"

    def test_filter_action_tracks_guardrail_filtered_event(
        self, callback, tracer, span_exporter
    ):
        """When validation fails with Filter, Guardrail.Filtered event should be tracked."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii"
            mock_guardrail.description = "PII Filter"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "tool_post_execution_pii",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.TOOL,
                        "execution_stage": ExecutionStage.POST_EXECUTION,
                        "action_type": "Filter",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: False}},
                    run_id=run_id,
                )

                # Action node fires with _filter suffix
                action_run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=action_run_id,
                    metadata={
                        "langgraph_node": "tool_post_execution_pii_filter",
                        "action_type": "Filter",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_action",
                        "scope": GuardrailScope.TOOL,
                        "execution_stage": ExecutionStage.POST_EXECUTION,
                    },
                )
                # Event is tracked on chain end
                callback.on_chain_end({}, run_id=action_run_id)

                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args[0][0] == "Guardrail.Filtered"
                props = call_args[0][1]
                assert props["ActionType"] == "Filter"
                assert props["AgentName"] == "TestAgent"

    def test_enriched_properties_included_in_event(
        self, callback, tracer, span_exporter
    ):
        """Enriched properties from runtime should be included in guardrail events."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties(
                {
                    "AgentName": "MyAgent",
                    "AgentId": "agent-123",
                    "Model": "gpt-4o",
                    "CloudOrganizationId": "org-456",
                }
            )

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "guard"
            mock_guardrail.description = "Test Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Skip",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                    run_id=run_id,
                )

                props = mock_track.call_args[0][1]
                assert props["AgentName"] == "MyAgent"
                assert props["AgentId"] == "agent-123"
                assert props["Model"] == "gpt-4o"
                assert props["CloudOrganizationId"] == "org-456"

    def test_builtin_validator_guardrail_metadata_enrichment(
        self, callback, tracer, span_exporter
    ):
        """BuiltInValidatorGuardrail metadata should enrich telemetry properties."""
        from uipath.platform.guardrails import BuiltInValidatorGuardrail

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock BuiltInValidatorGuardrail
            mock_guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtIn"
            mock_guardrail.validator_type = "pii_detection"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT, GuardrailScope.LLM]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Skip",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                    run_id=run_id,
                )

                props = mock_track.call_args[0][1]
                assert props["EnabledForEvals"] == "true"
                assert props["GuardrailType"] == "BuiltIn"
                assert props["ValidatorType"] == "PiiDetection"
                assert props["CurrentScope"] == "Agent"
                assert "Agent" in props["GuardrailScopes"]
                assert "Llm" in props["GuardrailScopes"]


class TestRuleDetailsExtraction:
    """Tests for rule details extraction for guardrail telemetry events."""

    def test_translate_validation_reason_didnt_match(self, callback):
        """Test that 'didn't match' is translated to 'RuleDidNotMeet'."""
        result = callback._guardrail_instrumentor._translate_validation_reason(
            "Field 'name' didn't match the expected pattern"
        )
        assert result == "RuleDidNotMeet"

    def test_translate_validation_reason_no_translation(self, callback):
        """Test that reasons without 'didn't match' are returned as-is."""
        result = callback._guardrail_instrumentor._translate_validation_reason(
            "No PII found"
        )
        assert result == "No PII found"

    def test_translate_validation_reason_none(self, callback):
        """Test that None is returned as-is."""
        result = callback._guardrail_instrumentor._translate_validation_reason(None)
        assert result is None

    def test_extract_operator_from_description_contains(self, callback):
        """Test extraction of 'contains' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "message.content contains 'forbidden'"
        )
        assert result == "Contains"

    def test_extract_operator_from_description_does_not_contain(self, callback):
        """Test extraction of 'doesNotContain' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "message.content doesNotContain 'allowed'"
        )
        assert result == "DoesNotContain"

    def test_extract_operator_from_description_greater_than(self, callback):
        """Test extraction of 'greaterThan' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.count greaterThan 10.0"
        )
        assert result == "GreaterThan"

    def test_extract_operator_from_description_greater_than_or_equal(self, callback):
        """Test extraction of 'greaterThanOrEqual' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.value greaterThanOrEqual 5.0"
        )
        assert result == "GreaterThanOrEqual"

    def test_extract_operator_from_description_less_than(self, callback):
        """Test extraction of 'lessThan' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.count lessThan 100"
        )
        assert result == "LessThan"

    def test_extract_operator_from_description_equals(self, callback):
        """Test extraction of 'equals' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "All fields equals 'test'"
        )
        assert result == "Equals"

    def test_extract_operator_from_description_is_empty(self, callback):
        """Test extraction of 'isEmpty' operator (no value) from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "message.content isEmpty"
        )
        assert result == "IsEmpty"

    def test_extract_operator_from_description_is_not_empty(self, callback):
        """Test extraction of 'isNotEmpty' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "message.content isNotEmpty"
        )
        assert result == "IsNotEmpty"

    def test_extract_operator_from_description_starts_with(self, callback):
        """Test extraction of 'startsWith' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.prefix startsWith 'hello'"
        )
        assert result == "StartsWith"

    def test_extract_operator_from_description_ends_with(self, callback):
        """Test extraction of 'endsWith' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.suffix endsWith 'world'"
        )
        assert result == "EndsWith"

    def test_extract_operator_from_description_matches_regex(self, callback):
        """Test extraction of 'matchesRegex' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.pattern matchesRegex '^[0-9]+$'"
        )
        assert result == "MatchesRegex"

    def test_extract_operator_from_description_none(self, callback):
        """Test extraction returns 'Unknown' for None description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            None
        )
        assert result == "Unknown"

    def test_extract_operator_from_description_empty(self, callback):
        """Test extraction returns 'Unknown' for empty description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description("")
        assert result == "Unknown"

    def test_extract_operator_from_description_unknown_operator(self, callback):
        """Test extraction returns 'Unknown' for unrecognized operator."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "field.path unknownOp 'value'"
        )
        assert result == "Unknown"

    def test_extract_rule_details_word_rule(self, callback):
        """Test rule details extraction for word rule."""
        from unittest.mock import MagicMock

        mock_rule = MagicMock()
        mock_rule.rule_type = "word"
        mock_rule.field_selector = SpecificFieldsSelector(
            selector_type="specific",
            fields=[FieldReference(path="age", source=FieldSource.INPUT)],
        )
        mock_rule.rule_description = "message.content contains 'forbidden'"

        result = callback._guardrail_instrumentor._extract_rule_details([mock_rule])

        assert len(result) == 1
        assert result[0]["Type"] == "Word"
        assert result[0]["FieldSelectorType"] == "Specific"
        assert result[0]["Operator"] == "Contains"

    def test_extract_rule_details_number_rule(self, callback):
        """Test rule details extraction for number rule."""
        from unittest.mock import MagicMock

        mock_rule = MagicMock()
        mock_rule.rule_type = "number"
        mock_rule.field_selector = AllFieldsSelector(
            selector_type="all", sources=[FieldSource.OUTPUT]
        )
        mock_rule.field_selector_type = "all"
        mock_rule.rule_description = "All fields greaterThan 10.0"

        result = callback._guardrail_instrumentor._extract_rule_details([mock_rule])

        assert len(result) == 1
        assert result[0]["Type"] == "Number"
        assert result[0]["FieldSelectorType"] == "All"
        assert result[0]["Operator"] == "GreaterThan"

    def test_extract_rule_details_boolean_rule(self, callback):
        """Test rule details extraction for boolean rule."""
        from unittest.mock import MagicMock

        mock_rule = MagicMock()
        mock_rule.rule_type = "boolean"
        mock_rule.field_selector = SpecificFieldsSelector(
            selector_type="specific",
            fields=[FieldReference(path="is_active", source=FieldSource.INPUT)],
        )
        mock_rule.rule_description = "data.is_active equals True"

        result = callback._guardrail_instrumentor._extract_rule_details([mock_rule])

        assert len(result) == 1
        assert result[0]["Type"] == "Boolean"
        assert result[0]["FieldSelectorType"] == "Specific"
        assert result[0]["Operator"] == "Equals"

    def test_extract_rule_details_universal_rule(self, callback):
        """Test rule details extraction for universal rule (always enforce)."""
        from uipath.core.guardrails import ApplyTo, UniversalRule

        mock_rule = UniversalRule(
            rule_type="always",
            apply_to=ApplyTo.INPUT_AND_OUTPUT,
        )

        result = callback._guardrail_instrumentor._extract_rule_details([mock_rule])

        assert len(result) == 1
        assert result[0]["Type"] == "Always enforce the guardrail"
        assert result[0]["ApplyTo"] == "InputAndOutput"

    def test_extract_rule_details_multiple_rules(self, callback):
        """Test rule details extraction for multiple rules."""
        from unittest.mock import MagicMock

        word_rule = MagicMock()
        word_rule.rule_type = "word"
        word_rule.field_selector_type = "specific"
        word_rule.rule_description = "field contains 'test'"

        number_rule = MagicMock()
        number_rule.rule_type = "number"
        number_rule.field_selector_type = "all"
        number_rule.rule_description = "All fields lessThan 100"

        result = callback._guardrail_instrumentor._extract_rule_details(
            [word_rule, number_rule]
        )

        assert len(result) == 2
        assert result[0]["Type"] == "Word"
        assert result[0]["Operator"] == "Contains"
        assert result[1]["Type"] == "Number"
        assert result[1]["Operator"] == "LessThan"


class TestDeterministicGuardrailTelemetry:
    """Tests for DeterministicGuardrail telemetry properties."""

    def test_deterministic_guardrail_includes_rule_details(
        self, callback, tracer, span_exporter
    ):
        """DeterministicGuardrail metadata should include NumberOfRules property."""
        import json

        from uipath.core.guardrails import DeterministicGuardrail

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock DeterministicGuardrail with rules
            mock_guardrail = MagicMock(spec=DeterministicGuardrail)
            mock_guardrail.name = "custom_guard"
            mock_guardrail.description = "Custom Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "custom"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            # Create mock rules
            mock_rule1 = MagicMock()
            mock_rule1.rule_type = "word"
            mock_rule1.field_selector = SpecificFieldsSelector(
                selector_type="specific",
                fields=[FieldReference(path="status", source=FieldSource.INPUT)],
            )
            mock_rule1.rule_description = "field contains 'test'"

            mock_rule2 = MagicMock()
            mock_rule2.rule_type = "number"
            mock_rule2.field_selector = AllFieldsSelector(
                selector_type="all", sources=[FieldSource.OUTPUT]
            )
            mock_rule2.rule_description = "All fields greaterThan 5"

            mock_guardrail.rules = [mock_rule1, mock_rule2]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "tool_pre_execution_custom_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.TOOL,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Skip",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                    run_id=run_id,
                )

                props = mock_track.call_args[0][1]
                assert props["NumberOfRules"] == "2"
                assert props["GuardrailType"] == "Custom"

                assert "RuleDetails" in props

                rule_details = json.loads(props["RuleDetails"])
                assert len(rule_details) == 2
                assert rule_details[0]["Type"] == "Word"
                assert rule_details[0]["FieldSelectorType"] == "Specific"
                assert rule_details[0]["Operator"] == "Contains"
                assert rule_details[1]["Type"] == "Number"
                assert rule_details[1]["FieldSelectorType"] == "All"
                assert rule_details[1]["Operator"] == "GreaterThan"

    def test_deterministic_guardrail_with_universal_rule(
        self, callback, tracer, span_exporter
    ):
        """DeterministicGuardrail with UniversalRule should have correct RuleDetails."""
        import json

        from uipath.core.guardrails import (
            ApplyTo,
            DeterministicGuardrail,
            UniversalRule,
        )

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            mock_guardrail = MagicMock(spec=DeterministicGuardrail)
            mock_guardrail.name = "always_guard"
            mock_guardrail.description = "Always Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "custom"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            universal_rule = UniversalRule(
                rule_type="always",
                apply_to=ApplyTo.INPUT_AND_OUTPUT,
            )
            mock_guardrail.rules = [universal_rule]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "tool_pre_execution_always_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.TOOL,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Skip",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                    run_id=run_id,
                )

                props = mock_track.call_args[0][1]
                rule_details = json.loads(props["RuleDetails"])

                assert len(rule_details) == 1
                assert rule_details[0]["Type"] == "Always enforce the guardrail"
                assert rule_details[0]["ApplyTo"] == "InputAndOutput"


class TestLangchainConfigStructure:
    """Integration tests verifying langchain internal config structure.

    These tests detect if langchain changes their internal API that we depend on
    for get_current_run_id(). If these fail after a langchain upgrade, the
    get_current_run_id implementation needs updating.
    """

    def test_var_child_runnable_config_exists(self) -> None:
        """Verify langchain's var_child_runnable_config ContextVar exists."""
        import langchain_core.runnables.config

        assert hasattr(langchain_core.runnables.config, "var_child_runnable_config")

    def test_base_callback_manager_has_parent_run_id(self) -> None:
        """Verify BaseCallbackManager has parent_run_id attribute."""
        from langchain_core.callbacks import BaseCallbackManager

        # Check the class has the attribute (may be None on instance)
        manager = BaseCallbackManager(handlers=[])
        assert hasattr(manager, "parent_run_id")

    def test_get_current_run_id_returns_none_outside_context(self) -> None:
        """get_current_run_id returns None when not in a langchain runnable context."""
        from uipath_agents._observability.llmops.callback import get_current_run_id

        # Outside of a langchain runnable, should return None
        result = get_current_run_id()
        assert result is None

    def test_get_current_run_id_in_runnable_context(self) -> None:
        """get_current_run_id returns run_id when inside a langchain runnable."""
        from langchain_core.runnables import RunnableLambda

        from uipath_agents._observability.llmops.callback import get_current_run_id

        captured_run_id = None

        def capture_run_id(x: str) -> str:
            nonlocal captured_run_id
            captured_run_id = get_current_run_id()
            return x

        runnable = RunnableLambda(capture_run_id)
        runnable.invoke("test")

        # Should have captured a run_id
        assert captured_run_id is not None
        assert isinstance(captured_run_id, UUID)


class TestEscalationReviewedData:
    """Tests for escalation span completion with reviewed data on resume."""

    def test_escalate_approved_on_resume_completes_via_on_chain_end(
        self,
        callback: LlmOpsInstrumentationCallback,
        tracer: LlmOpsSpanFactory,
        span_exporter,
    ) -> None:
        """When HITL is approved on resume, span completes via on_chain_end."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "deterministic"
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            # Simulate resumed escalation context (set by runtime on resume)
            callback._state.resumed_escalation_trace_id = "0123456789abcdef"
            callback._state.resumed_escalation_span_data = {
                "name": "Review task",
                "span_id": "abcd1234",
                "attributes": {
                    "type": "guardrailEscalation",
                    "reviewStatus": "Pending",
                },
            }
            callback._state.resumed_hitl_guardrail_span_data = {
                "name": "pii_guard",
                "span_id": "eval5678",
                "attributes": {},
            }

            # Resume: Escalate action node fires on resume
            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard_hitl",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_action",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Escalate",
                    "escalation_data": {
                        "reviewed_by": "user@example.com",
                        "reviewed_inputs": {"input": "sanitized"},
                    },
                },
            )

            # Mock upsert to verify it's called correctly
            with patch.object(
                callback._state.span_factory, "upsert_span_complete_by_data"
            ) as mock_upsert:
                mock_upsert.return_value = None

                # on_chain_end - HITL approved (no error)
                callback.on_chain_end(
                    {},
                    run_id=action_run_id,
                )

                # Verify upsert was called for the Review task span
                assert mock_upsert.call_count >= 1
                # Find the call for Review task span
                review_call = None
                for call in mock_upsert.call_args_list:
                    span_data = call[1].get("span_data", {})
                    if span_data.get("name") == "Review task":
                        review_call = call
                        break

                assert review_call is not None, "Review task span should be upserted"
                span_data = review_call[1]["span_data"]
                assert span_data["attributes"]["reviewStatus"] == "Completed"
                assert span_data["attributes"]["reviewOutcome"] == "Approved"

    def test_escalate_rejected_on_resume_completes_via_on_chain_error(
        self,
        callback: LlmOpsInstrumentationCallback,
        tracer: LlmOpsSpanFactory,
        span_exporter,
    ) -> None:
        """When HITL is rejected on resume, span completes via on_chain_error."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock guardrail
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "deterministic"
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            # Simulate resumed escalation context (set by runtime on resume)
            callback._state.resumed_escalation_trace_id = "0123456789abcdef"
            callback._state.resumed_escalation_span_data = {
                "name": "Review task",
                "span_id": "abcd1234",
                "attributes": {
                    "type": "guardrailEscalation",
                    "reviewStatus": "Pending",
                },
            }
            callback._state.resumed_hitl_guardrail_span_data = {
                "name": "pii_guard",
                "span_id": "eval5678",
                "attributes": {},
            }

            # Resume: Escalate action node fires on resume
            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard_hitl",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_action",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Escalate",
                    "escalation_data": {
                        "reviewed_by": "user@example.com",
                    },
                },
            )

            # Mock upsert to verify it's called correctly
            with patch.object(
                callback._state.span_factory, "upsert_span_complete_by_data"
            ) as mock_upsert:
                mock_upsert.return_value = None

                # on_chain_error - HITL rejected
                callback.on_chain_error(
                    Exception("User rejected: Invalid data"),
                    run_id=action_run_id,
                )

                # Verify upsert was called for the Review task span
                assert mock_upsert.call_count >= 1
                # Find the call for Review task span
                review_call = None
                for call in mock_upsert.call_args_list:
                    span_data = call[1].get("span_data", {})
                    if span_data.get("name") == "Review task":
                        review_call = call
                        break

                assert review_call is not None, "Review task span should be upserted"
                span_data = review_call[1]["span_data"]
                assert span_data["attributes"]["reviewStatus"] == "Completed"
                assert span_data["attributes"]["reviewOutcome"] == "Rejected"


class TestNestedLlmCallsInTools:
    """Tests for LLM calls made from within tools (e.g., analyze_files tool)."""

    def test_llm_call_inside_tool_parents_to_tool_span(
        self, tracer, callback, span_exporter
    ) -> None:
        """LLM call from within a tool should be parented to the tool span.

        Scenario: analyze_files tool calls llm.ainvoke() internally.
        The LLM span should be a child of the tool span, not the agent span.
        """
        from unittest.mock import patch

        agent_run_id = uuid4()

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Tool starts
            tool_run_id = uuid4()
            callback.on_tool_start(
                {"name": "analyze_files"}, '{"files": ["a.py"]}', run_id=tool_run_id
            )

            # Inside the tool, an LLM call happens.
            # LangChain gives it a different run_id, and parent_run_id may not
            # match tool_run_id (this is the bug we're fixing).
            # But get_current_run_id() should return tool_run_id via context.
            llm_run_id = uuid4()
            unrelated_parent_id = uuid4()  # Simulates LangChain's parent mismatch

            # Mock get_current_run_id to return tool's run_id (simulating context)
            with patch(
                "uipath_agents._observability.llmops.callback.get_current_run_id",
                return_value=tool_run_id,
            ):
                callback.on_chat_model_start(
                    {"kwargs": {"model": "gpt-4"}},
                    [[]],
                    run_id=llm_run_id,
                    parent_run_id=unrelated_parent_id,  # Doesn't match tool_run_id
                )
                callback.on_llm_end(None, run_id=llm_run_id)

            callback.on_tool_end("analysis result", run_id=tool_run_id)

        spans = span_exporter.get_finished_spans()

        agent = next(s for s in spans if s.attributes.get("type") == "agentRun")
        tool = next(s for s in spans if s.attributes.get("type") == "toolCall")
        llm = next(s for s in spans if s.name == "LLM call")
        model = next(s for s in spans if s.name == "Model run")

        # LLM span should be child of tool span (not agent span)
        assert llm.parent.span_id == tool.context.span_id
        # Model span should be child of LLM span
        assert model.parent.span_id == llm.context.span_id
        # Tool span should be child of agent span
        assert tool.parent.span_id == agent.context.span_id

    def test_independent_llm_call_parents_to_agent_span(
        self, tracer, callback, span_exporter
    ) -> None:
        """Independent LLM call (not inside a tool) should parent to agent span.

        When get_current_run_id() returns None (no tool context), LLM span
        should fall back to agent span as parent.
        """
        from unittest.mock import patch

        agent_run_id = uuid4()

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # LLM call without any tool context
            llm_run_id = uuid4()

            # Mock get_current_run_id to return None (no context)
            with patch(
                "uipath_agents._observability.llmops.callback.get_current_run_id",
                return_value=None,
            ):
                callback.on_chat_model_start(
                    {"kwargs": {"model": "gpt-4"}},
                    [[]],
                    run_id=llm_run_id,
                    parent_run_id=None,
                )
                callback.on_llm_end(None, run_id=llm_run_id)

        spans = span_exporter.get_finished_spans()

        agent = next(s for s in spans if s.attributes.get("type") == "agentRun")
        llm = next(s for s in spans if s.name == "LLM call")

        # LLM span should be child of agent span
        assert llm.parent.span_id == agent.context.span_id

    def test_parallel_tool_and_llm_isolated(
        self, tracer, callback, span_exporter
    ) -> None:
        """Parallel tool and LLM call should have correct parents.

        Tool's LLM call should parent to tool, while independent LLM call
        should parent to agent.
        """
        from unittest.mock import patch

        agent_run_id = uuid4()

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Tool starts
            tool_run_id = uuid4()
            callback.on_tool_start({"name": "tool1"}, "{}", run_id=tool_run_id)

            # LLM call inside tool (context returns tool_run_id)
            llm_in_tool_id = uuid4()
            with patch(
                "uipath_agents._observability.llmops.callback.get_current_run_id",
                return_value=tool_run_id,
            ):
                callback.on_chat_model_start(
                    {"kwargs": {"model": "gpt-4"}},
                    [[]],
                    run_id=llm_in_tool_id,
                    parent_run_id=None,
                )
                callback.on_llm_end(None, run_id=llm_in_tool_id)

            callback.on_tool_end("result", run_id=tool_run_id)

            # Independent LLM call (context returns None - no tool)
            llm_independent_id = uuid4()
            with patch(
                "uipath_agents._observability.llmops.callback.get_current_run_id",
                return_value=None,
            ):
                callback.on_chat_model_start(
                    {"kwargs": {"model": "gpt-4"}},
                    [[]],
                    run_id=llm_independent_id,
                    parent_run_id=None,
                )
                callback.on_llm_end(None, run_id=llm_independent_id)

        spans = span_exporter.get_finished_spans()

        agent = next(s for s in spans if s.attributes.get("type") == "agentRun")
        tool = next(s for s in spans if s.attributes.get("type") == "toolCall")
        llm_spans = [s for s in spans if s.name == "LLM call"]

        # Find which LLM is which by checking parent
        llm_in_tool = next(
            s for s in llm_spans if s.parent.span_id == tool.context.span_id
        )
        llm_independent = next(
            s for s in llm_spans if s.parent.span_id == agent.context.span_id
        )

        assert llm_in_tool is not None
        assert llm_independent is not None
        assert llm_in_tool is not llm_independent
