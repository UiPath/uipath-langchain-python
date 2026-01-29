"""Tests for UiPathTracingCallback LangChain callback handler."""

from typing import Any
from uuid import UUID, uuid4

import pytest
from uipath.core.guardrails import (
    AllFieldsSelector,
    FieldReference,
    FieldSource,
    SpecificFieldsSelector,
)

from uipath_agents._observability.callback import UiPathTracingCallback
from uipath_agents._observability.schema import (
    GUARDRAIL_VALIDATION_DETAILS_KEY,
    GUARDRAIL_VALIDATION_RESULT_KEY,
    INNER_STATE_KEY,
    SpanType,
)
from uipath_agents._observability.tracer import UiPathTracer

# span_exporter fixture comes from conftest.py


@pytest.fixture
def tracer(span_exporter):
    """Create a fresh tracer for testing."""
    return UiPathTracer()


@pytest.fixture
def callback(tracer):
    """Create callback with tracer, cleanup after test."""
    from opentelemetry import context

    # Capture initial context token
    initial_context = context.get_current()

    cb = UiPathTracingCallback(tracer)
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

        # Verify spans were created
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2

        span_names = {s.name for s in spans}
        assert "LLM call" in span_names
        assert "Model run" in span_names
        # Both LLM call and Model run have type "completion" (matches C# Temporal)
        llm_span = next(s for s in spans if s.name == "LLM call")
        model_span = next(s for s in spans if s.name == "Model run")
        assert llm_span.attributes["type"] == SpanType.COMPLETION.value
        assert model_span.attributes["type"] == SpanType.COMPLETION.value

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
        callback = UiPathTracingCallback(tracer)
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
        callback = UiPathTracingCallback(tracer)
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
            assert len(callback._spans) == 0
            assert callback._prompts_captured is False


class TestGraphInterruptHandling:
    """Tests for GraphInterrupt detection in tool errors."""

    def test_is_graph_interrupt_by_type_name(self, callback):
        """Test _is_graph_interrupt detects GraphInterrupt by type name."""

        class GraphInterrupt(Exception):
            pass

        error = GraphInterrupt("Suspended for HITL")
        assert callback._is_graph_interrupt(error) is True

    def test_is_graph_interrupt_by_str_prefix(self, callback):
        """Test _is_graph_interrupt detects GraphInterrupt by string prefix."""
        # Some errors may convert to "GraphInterrupt(...)" string
        error = Exception("GraphInterrupt(Waiting for approval)")
        assert callback._is_graph_interrupt(error) is True

    def test_is_not_graph_interrupt_for_regular_error(self, callback):
        """Test _is_graph_interrupt returns False for regular errors."""
        error = ValueError("Some other error")
        assert callback._is_graph_interrupt(error) is False

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
        assert run_id in callback._spans

        # Trigger GraphInterrupt - span should NOT be closed
        callback.on_tool_error(GraphInterrupt("Suspended"), run_id=run_id)

        # Span should still be tracked (not closed)
        assert run_id in callback._spans

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
        assert run_id not in callback._spans

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
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Test that integration tool_type creates a child span."""
        run_id = uuid4()
        serialized = {"name": "Web_Search"}
        metadata = {"tool_type": "integration", "display_name": "Web Search Tool"}

        callback.on_tool_start(serialized, "input", run_id=run_id, metadata=metadata)

        # Both tool span and integration child span should be tracked
        assert run_id in callback._spans
        assert self._interruptible_key(run_id) in callback._spans

    def test_on_tool_end_closes_integration_child_span_first(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Test that integration child span closes before tool span."""
        run_id = uuid4()
        serialized = {"name": "Web_Search"}
        metadata = {"tool_type": "integration", "display_name": "Web Search Tool"}

        callback.on_tool_start(serialized, "input", run_id=run_id, metadata=metadata)
        callback.on_tool_end("result", run_id=run_id)

        # Both spans should be removed
        assert run_id not in callback._spans
        assert self._interruptible_key(run_id) not in callback._spans

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2

        tool_span = next(s for s in spans if s.name == "Tool call - Web_Search")
        child_span = next(s for s in spans if s.name == "Web Search Tool")

        # Child should be parented to tool span
        assert child_span.parent.span_id == tool_span.context.span_id
        assert child_span.attributes["type"] == SpanType.INTEGRATION_TOOL.value


class TestGuardrailActionDetection:
    """Tests for guardrail action detection from action nodes."""

    def test_validation_passed_ends_immediately_with_skip(
        self, callback, tracer, span_exporter
    ):
        """When validation passes, span ends immediately with action=Skip."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            run_id = uuid4()
            # Start guardrail evaluation
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={"langgraph_node": "agent_pre_execution_pii_guard"},
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

            run_id = uuid4()
            # Start guardrail evaluation
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={"langgraph_node": "agent_pre_execution_pii_guard"},
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

            # Span should be pending action
            assert (
                "agent_pre_execution_pii_guard" in callback._pending_guardrail_actions
            )

            # Action node fires with _block suffix and action_type metadata
            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard_block",
                    "action_type": "Block",
                },
            )

            # Pending action should be cleared
            assert (
                "agent_pre_execution_pii_guard"
                not in callback._pending_guardrail_actions
            )

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Block"
        assert eval_spans[0].attributes.get("validationResult") == "PII detected"

    def test_action_node_log_sets_correct_action(self, callback, tracer, span_exporter):
        """Test action node with _log suffix sets action=log."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={"langgraph_node": "llm_pre_execution_prompt_injection"},
            )
            callback.on_chain_end(
                {
                    INNER_STATE_KEY: {
                        GUARDRAIL_VALIDATION_RESULT_KEY: "Injection detected"
                    }
                },
                run_id=run_id,
            )

            # Action node fires with _log suffix and action_type metadata
            callback.on_chain_start(
                {},
                {},
                run_id=uuid4(),
                metadata={
                    "langgraph_node": "llm_pre_execution_prompt_injection_log",
                    "action_type": "Log",
                },
            )

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Log"

    def test_action_node_hitl_sets_correct_action(
        self, callback, tracer, span_exporter
    ):
        """Test action node with _hitl suffix sets action=Escalate."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={"langgraph_node": "llm_post_execution_review_guard"},
            )
            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: "Review needed"}},
                run_id=run_id,
            )

            callback.on_chain_start(
                {},
                {},
                run_id=uuid4(),
                metadata={
                    "langgraph_node": "llm_post_execution_review_guard_hitl",
                    "action_type": "Escalate",
                },
            )

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Escalate"

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

            # Eval node for guardrail named "test_guardrail_log"
            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={"langgraph_node": "agent_pre_execution_test_guardrail_log"},
            )

            # Span should be created (not skipped)
            assert run_id in callback._guardrail_spans

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
        assert eval_spans[0].name == "Guardrail - test_guardrail_log"
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

            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={"langgraph_node": "agent_pre_execution_pii_guard"},
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

            # Span should be pending action
            assert (
                "agent_pre_execution_pii_guard" in callback._pending_guardrail_actions
            )

            # Action node fires
            callback.on_chain_start(
                {},
                {},
                run_id=uuid4(),
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard_block",
                    "action_type": "Block",
                },
            )

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Block"
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

            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={"langgraph_node": "agent_pre_execution_pii_guard"},
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

            tool_run_id = uuid4()
            callback.on_tool_start({"name": "my_tool"}, "input", run_id=tool_run_id)

            # Now a tool_pre guardrail fires
            guard_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=guard_run_id,
                metadata={"langgraph_node": "tool_pre_execution_input_guard"},
            )
            callback.on_chain_end({}, run_id=guard_run_id)

            # Close tool_pre container manually since tool_end doesn't close it
            callback._close_container("tool", "pre")

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
                metadata={"langgraph_node": "tool_post_execution_output_guard"},
            )
            callback.on_chain_end({}, run_id=guard_run_id)

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

            # LLM post guardrail creates llm_post container
            run_id1 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id1,
                metadata={"langgraph_node": "llm_post_execution_guard1"},
            )
            callback.on_chain_end({}, run_id=run_id1)

            # Should have llm_post container
            assert ("llm", "post") in callback._guardrail_containers

            # Start tool
            tool_run_id = uuid4()
            callback.on_tool_start({"name": "my_tool"}, "input", run_id=tool_run_id)

            # Tool pre guardrail should close llm_post
            run_id2 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id2,
                metadata={"langgraph_node": "tool_pre_execution_guard"},
            )

            # llm_post should be closed
            assert ("llm", "post") not in callback._guardrail_containers

            callback.on_chain_end({}, run_id=run_id2)
            callback.on_tool_end("result", run_id=tool_run_id)

    def test_tool_post_closes_tool_pre(self, callback, tracer, span_exporter):
        """Transitioning to tool_post should close tool_pre container."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            tool_run_id = uuid4()
            callback.on_tool_start({"name": "my_tool"}, "input", run_id=tool_run_id)

            # Tool pre guardrail
            run_id1 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id1,
                metadata={"langgraph_node": "tool_pre_execution_guard"},
            )
            callback.on_chain_end({}, run_id=run_id1)

            assert ("tool", "pre") in callback._guardrail_containers

            # Tool post guardrail should close tool_pre
            run_id2 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id2,
                metadata={"langgraph_node": "tool_post_execution_guard"},
            )

            assert ("tool", "pre") not in callback._guardrail_containers

            callback.on_chain_end({}, run_id=run_id2)
            callback.on_tool_end("result", run_id=tool_run_id)

    def test_agent_post_closes_tool_post(self, callback, tracer, span_exporter):
        """Transitioning to agent_post should close tool_post container."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            tool_run_id = uuid4()
            callback.on_tool_start({"name": "my_tool"}, "input", run_id=tool_run_id)

            # Tool post guardrail
            run_id1 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id1,
                metadata={"langgraph_node": "tool_post_execution_guard"},
            )
            callback.on_chain_end({}, run_id=run_id1)

            # Don't end tool yet - agent_post fires before tool ends in some flows

            assert ("tool", "post") in callback._guardrail_containers

            # Agent post guardrail should close tool_post
            run_id2 = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id2,
                metadata={"langgraph_node": "agent_post_execution_guard"},
            )

            assert ("tool", "post") not in callback._guardrail_containers

            callback.on_chain_end({}, run_id=run_id2)
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

            # Real order: guardrail FIRST (before tool starts)
            guard_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=guard_run_id,
                metadata={"langgraph_node": "tool_pre_execution_pii_guard"},
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

            # Guardrail fires (creates placeholder tool span)
            guard_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=guard_run_id,
                metadata={"langgraph_node": "tool_pre_execution_pii_guard"},
            )
            # Placeholder should exist
            assert callback._current_tool_span is not None
            assert callback._tool_span_from_guardrail is True

            # Guardrail fails
            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: "PII detected"}},
                run_id=guard_run_id,
            )

            # Block action fires - tool will NOT execute
            callback.on_chain_start(
                {},
                {},
                run_id=uuid4(),
                metadata={
                    "langgraph_node": "tool_pre_execution_pii_guard_block",
                    "action_type": "Block",
                },
            )

            # Placeholder should be cleaned up
            assert callback._current_tool_span is None
            assert callback._tool_span_from_guardrail is False

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

            # First guardrail fires (creates placeholder)
            guard1_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=guard1_id,
                metadata={"langgraph_node": "tool_pre_execution_guard1"},
            )
            callback.on_chain_end({}, run_id=guard1_id)

            # Verify placeholder was created
            assert callback._current_tool_span is not None
            placeholder_span = callback._current_tool_span

            # Second guardrail fires (should reuse same placeholder)
            guard2_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=guard2_id,
                metadata={"langgraph_node": "tool_pre_execution_guard2"},
            )
            callback.on_chain_end({}, run_id=guard2_id)

            # Verify same placeholder is still used
            assert callback._current_tool_span is placeholder_span

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

    def test_push_span_creates_stack(self, callback: UiPathTracingCallback) -> None:
        """Pushing a span creates a stack entry for the run_id."""
        from uipath_agents._observability.callback import _span_stacks

        run_id = uuid4()
        # Create a mock span
        from unittest.mock import MagicMock

        span = MagicMock()

        _span_stacks.clear()
        callback._push_span(run_id, span)

        assert run_id in _span_stacks
        assert len(_span_stacks[run_id]) == 1
        assert _span_stacks[run_id][0] is span

    def test_push_multiple_spans_stacks(self, callback: UiPathTracingCallback) -> None:
        """Multiple pushes create a proper stack (LIFO)."""
        from unittest.mock import MagicMock

        from uipath_agents._observability.callback import _span_stacks

        run_id = uuid4()
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        _span_stacks.clear()
        callback._push_span(run_id, span1)
        callback._push_span(run_id, span2)

        assert len(_span_stacks[run_id]) == 2
        assert _span_stacks[run_id][0] is span1
        assert _span_stacks[run_id][1] is span2

    def test_pop_span_returns_lifo(self, callback: UiPathTracingCallback) -> None:
        """Pop returns spans in LIFO order."""
        from unittest.mock import MagicMock

        from uipath_agents._observability.callback import _span_stacks

        run_id = uuid4()
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        _span_stacks.clear()
        callback._push_span(run_id, span1)
        callback._push_span(run_id, span2)

        popped = callback._pop_span(run_id)
        assert popped is span2

        popped = callback._pop_span(run_id)
        assert popped is span1

    def test_pop_empty_returns_none(self, callback: UiPathTracingCallback) -> None:
        """Pop on empty stack returns None."""
        from uipath_agents._observability.callback import _span_stacks

        run_id = uuid4()
        _span_stacks.clear()

        result = callback._pop_span(run_id)
        assert result is None

    def test_get_current_span_returns_top(
        self, callback: UiPathTracingCallback
    ) -> None:
        """_get_current_span returns top of stack when run_id is available."""
        from unittest.mock import MagicMock, patch

        from uipath_agents._observability.callback import (
            _get_current_span,
            _span_stacks,
        )

        run_id = uuid4()
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        _span_stacks.clear()
        callback._push_span(run_id, span1)
        callback._push_span(run_id, span2)

        # Mock get_current_run_id to return our run_id
        with patch(
            "uipath_agents._observability.callback.get_current_run_id",
            return_value=run_id,
        ):
            result = _get_current_span()
            assert result is span2

    def test_get_ancestor_spans_returns_copy(
        self, callback: UiPathTracingCallback
    ) -> None:
        """_get_ancestor_spans returns a copy of the stack."""
        from unittest.mock import MagicMock, patch

        from uipath_agents._observability.callback import (
            _get_ancestor_spans,
            _span_stacks,
        )

        run_id = uuid4()
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        _span_stacks.clear()
        callback._push_span(run_id, span1)
        callback._push_span(run_id, span2)

        # Mock get_current_run_id to return our run_id
        with patch(
            "uipath_agents._observability.callback.get_current_run_id",
            return_value=run_id,
        ):
            ancestors = _get_ancestor_spans()
            assert len(ancestors) == 2
            assert ancestors[0] is span1
            assert ancestors[1] is span2

            # Should be a copy, not the original
            ancestors.pop()
            assert len(_span_stacks[run_id]) == 2

    def test_set_agent_span_clears_only_specified_run_id_stack(
        self, callback: UiPathTracingCallback, tracer
    ) -> None:
        """set_agent_span clears only the specified run_id's stack, not others."""
        from unittest.mock import MagicMock

        from uipath_agents._observability.callback import _span_stacks

        run_id_1 = uuid4()
        run_id_2 = uuid4()
        span1 = MagicMock(name="span1")
        span2 = MagicMock(name="span2")

        _span_stacks.clear()
        callback._push_span(run_id_1, span1)
        callback._push_span(run_id_2, span2)
        assert run_id_1 in _span_stacks
        assert run_id_2 in _span_stacks

        # set_agent_span for run_id_1 should only clear run_id_1's stack
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, run_id_1)

        # run_id_1 stack should have agent_span as first element
        assert run_id_1 in _span_stacks
        assert len(_span_stacks[run_id_1]) == 1
        assert _span_stacks[run_id_1][0] is agent_span

        # run_id_2 stack should remain untouched
        assert run_id_2 in _span_stacks
        assert len(_span_stacks[run_id_2]) == 1
        assert _span_stacks[run_id_2][0] is span2

    def test_set_agent_span_pushes_agent_as_first_span(
        self, callback: UiPathTracingCallback, tracer
    ) -> None:
        """set_agent_span pushes agent_span as the first span in the run_id's stack."""
        from uipath_agents._observability.callback import _span_stacks

        _span_stacks.clear()
        run_id = uuid4()

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, run_id)

        # Verify agent_span is in stack
        assert run_id in _span_stacks
        assert len(_span_stacks[run_id]) == 1
        assert _span_stacks[run_id][0] is agent_span


class TestSpanStackIntegration:
    """Integration tests for span stack across event types."""

    def test_llm_span_in_stack_during_execution(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Model span should be in stack during LLM execution."""
        from unittest.mock import patch

        from uipath_agents._observability.callback import _get_current_span

        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)

        # Mock get_current_run_id to simulate being in langchain context
        with patch(
            "uipath_agents._observability.callback.get_current_run_id",
            return_value=run_id,
        ):
            current = _get_current_span()
            assert current is not None
            model_span = callback._spans.get(_model_key(run_id))
            assert current is model_span

        callback.on_llm_end(None, run_id=run_id)  # type: ignore[arg-type]

    def test_llm_span_popped_on_end(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Stack should be empty after LLM completes."""
        from uipath_agents._observability.callback import _span_stacks

        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)  # type: ignore[arg-type]

        # Stack for this run_id should be empty
        assert run_id not in _span_stacks or len(_span_stacks.get(run_id, [])) == 0

    def test_tool_span_in_stack_during_execution(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Tool span should be in stack during tool execution."""
        from unittest.mock import patch

        from uipath_agents._observability.callback import _get_current_span

        run_id = uuid4()
        serialized = {"name": "my_tool"}

        callback.on_tool_start(serialized, "input", run_id=run_id)

        with patch(
            "uipath_agents._observability.callback.get_current_run_id",
            return_value=run_id,
        ):
            current = _get_current_span()
            assert current is not None
            tool_span = callback._spans.get(run_id)
            assert current is tool_span

        callback.on_tool_end("result", run_id=run_id)

    def test_nested_tool_child_on_top(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Tool with child span should have child on top of stack."""
        from unittest.mock import patch

        from uipath_agents._observability.callback import (
            _get_ancestor_spans,
            _get_current_span,
        )

        run_id = uuid4()
        serialized = {"name": "escalate_tool"}
        metadata = {"tool_type": "escalation", "display_name": "Escalate App"}

        callback.on_tool_start(serialized, "input", run_id=run_id, metadata=metadata)

        with patch(
            "uipath_agents._observability.callback.get_current_run_id",
            return_value=run_id,
        ):
            # Child span should be on top
            current = _get_current_span()
            child_key = UUID(int=run_id.int ^ 2)
            child_span = callback._spans.get(child_key)
            assert current is child_span

            # Ancestors should include both
            ancestors = _get_ancestor_spans()
            assert len(ancestors) == 2

        callback.on_tool_end("result", run_id=run_id)

    def test_guardrail_span_in_stack(
        self, callback: UiPathTracingCallback, tracer, span_exporter
    ) -> None:
        """Guardrail span should be in stack during evaluation."""
        from unittest.mock import patch

        from uipath_agents._observability.callback import _get_current_span

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

            with patch(
                "uipath_agents._observability.callback.get_current_run_id",
                return_value=run_id,
            ):
                current = _get_current_span()
                assert current is not None
                eval_span = callback._guardrail_spans.get(run_id)
                assert current is eval_span

            callback.on_chain_end({}, run_id=run_id)

    def test_guardrail_span_popped_on_end(
        self, callback: UiPathTracingCallback, tracer, span_exporter
    ) -> None:
        """Guardrail span should be popped after chain end."""
        from uipath_agents._observability.callback import _span_stacks

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
            assert run_id not in _span_stacks or len(_span_stacks.get(run_id, [])) == 0


class TestSpanStackParallelExecution:
    """Tests for parallel execution with stack isolation."""

    @pytest.mark.asyncio
    async def test_parallel_tools_isolated_stacks(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Two concurrent tools should have independent stacks."""
        import asyncio
        from unittest.mock import patch

        from uipath_agents._observability.callback import _get_current_span

        run_id_1 = uuid4()
        run_id_2 = uuid4()
        results: dict[str, Any] = {}

        async def tool_1() -> None:
            callback.on_tool_start({"name": "tool_1"}, "input", run_id=run_id_1)
            await asyncio.sleep(0.01)  # Simulate async work
            # Mock get_current_run_id to return this tool's run_id
            with patch(
                "uipath_agents._observability.callback.get_current_run_id",
                return_value=run_id_1,
            ):
                current = _get_current_span()
                results["tool_1_span"] = current
                results["tool_1_expected"] = callback._spans.get(run_id_1)
            callback.on_tool_end("result", run_id=run_id_1)

        async def tool_2() -> None:
            callback.on_tool_start({"name": "tool_2"}, "input", run_id=run_id_2)
            await asyncio.sleep(0.01)
            with patch(
                "uipath_agents._observability.callback.get_current_run_id",
                return_value=run_id_2,
            ):
                current = _get_current_span()
                results["tool_2_span"] = current
                results["tool_2_expected"] = callback._spans.get(run_id_2)
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
        from unittest.mock import patch

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            with patch(
                "uipath_agents._observability.callback.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={"langgraph_node": "agent_pre_execution_pii_guard"},
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
        from unittest.mock import MagicMock, patch

        from uipath.platform.guardrails import GuardrailScope

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            with patch(
                "uipath_agents._observability.callback.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={"langgraph_node": "agent_pre_execution_pii_guard"},
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
                        "scope": GuardrailScope.AGENT,
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
        from unittest.mock import MagicMock, patch

        from uipath.platform.guardrails import GuardrailScope

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            with patch(
                "uipath_agents._observability.callback.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={"langgraph_node": "llm_pre_execution_prompt_injection"},
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
                        "scope": GuardrailScope.TOOL,
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
        from unittest.mock import MagicMock, patch

        from uipath.platform.guardrails import GuardrailScope

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            with patch(
                "uipath_agents._observability.callback.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={"langgraph_node": "tool_post_execution_pii"},
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
                        "scope": GuardrailScope.TOOL,
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
        from unittest.mock import patch

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

            with patch(
                "uipath_agents._observability.callback.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={"langgraph_node": "agent_pre_execution_guard"},
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
        from unittest.mock import MagicMock, patch

        from uipath.platform.guardrails import BuiltInValidatorGuardrail, GuardrailScope

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock BuiltInValidatorGuardrail
            mock_guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtIn"
            mock_guardrail.validator_type = "pii_detection"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT, GuardrailScope.LLM]

            with patch(
                "uipath_agents._observability.callback.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard",
                        "guardrail": mock_guardrail,
                        "scope": GuardrailScope.AGENT,
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
        result = callback._translate_validation_reason(
            "Field 'name' didn't match the expected pattern"
        )
        assert result == "RuleDidNotMeet"

    def test_translate_validation_reason_no_translation(self, callback):
        """Test that reasons without 'didn't match' are returned as-is."""
        result = callback._translate_validation_reason("No PII found")
        assert result == "No PII found"

    def test_translate_validation_reason_none(self, callback):
        """Test that None is returned as-is."""
        result = callback._translate_validation_reason(None)
        assert result is None

    def test_extract_operator_from_description_contains(self, callback):
        """Test extraction of 'contains' operator from rule description."""
        result = callback._extract_operator_from_description(
            "message.content contains 'forbidden'"
        )
        assert result == "Contains"

    def test_extract_operator_from_description_does_not_contain(self, callback):
        """Test extraction of 'doesNotContain' operator from rule description."""
        result = callback._extract_operator_from_description(
            "message.content doesNotContain 'allowed'"
        )
        assert result == "DoesNotContain"

    def test_extract_operator_from_description_greater_than(self, callback):
        """Test extraction of 'greaterThan' operator from rule description."""
        result = callback._extract_operator_from_description(
            "data.count greaterThan 10.0"
        )
        assert result == "GreaterThan"

    def test_extract_operator_from_description_greater_than_or_equal(self, callback):
        """Test extraction of 'greaterThanOrEqual' operator from rule description."""
        result = callback._extract_operator_from_description(
            "data.value greaterThanOrEqual 5.0"
        )
        assert result == "GreaterThanOrEqual"

    def test_extract_operator_from_description_less_than(self, callback):
        """Test extraction of 'lessThan' operator from rule description."""
        result = callback._extract_operator_from_description("data.count lessThan 100")
        assert result == "LessThan"

    def test_extract_operator_from_description_equals(self, callback):
        """Test extraction of 'equals' operator from rule description."""
        result = callback._extract_operator_from_description("All fields equals 'test'")
        assert result == "Equals"

    def test_extract_operator_from_description_is_empty(self, callback):
        """Test extraction of 'isEmpty' operator (no value) from rule description."""
        result = callback._extract_operator_from_description("message.content isEmpty")
        assert result == "IsEmpty"

    def test_extract_operator_from_description_is_not_empty(self, callback):
        """Test extraction of 'isNotEmpty' operator from rule description."""
        result = callback._extract_operator_from_description(
            "message.content isNotEmpty"
        )
        assert result == "IsNotEmpty"

    def test_extract_operator_from_description_starts_with(self, callback):
        """Test extraction of 'startsWith' operator from rule description."""
        result = callback._extract_operator_from_description(
            "data.prefix startsWith 'hello'"
        )
        assert result == "StartsWith"

    def test_extract_operator_from_description_ends_with(self, callback):
        """Test extraction of 'endsWith' operator from rule description."""
        result = callback._extract_operator_from_description(
            "data.suffix endsWith 'world'"
        )
        assert result == "EndsWith"

    def test_extract_operator_from_description_matches_regex(self, callback):
        """Test extraction of 'matchesRegex' operator from rule description."""
        result = callback._extract_operator_from_description(
            "data.pattern matchesRegex '^[0-9]+$'"
        )
        assert result == "MatchesRegex"

    def test_extract_operator_from_description_none(self, callback):
        """Test extraction returns 'Unknown' for None description."""
        result = callback._extract_operator_from_description(None)
        assert result == "Unknown"

    def test_extract_operator_from_description_empty(self, callback):
        """Test extraction returns 'Unknown' for empty description."""
        result = callback._extract_operator_from_description("")
        assert result == "Unknown"

    def test_extract_operator_from_description_unknown_operator(self, callback):
        """Test extraction returns 'Unknown' for unrecognized operator."""
        result = callback._extract_operator_from_description(
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

        result = callback._extract_rule_details([mock_rule])

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

        result = callback._extract_rule_details([mock_rule])

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

        result = callback._extract_rule_details([mock_rule])

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

        result = callback._extract_rule_details([mock_rule])

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

        result = callback._extract_rule_details([word_rule, number_rule])

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
        from unittest.mock import MagicMock, patch

        from uipath.core.guardrails import DeterministicGuardrail
        from uipath.platform.guardrails import GuardrailScope

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock DeterministicGuardrail with rules
            mock_guardrail = MagicMock(spec=DeterministicGuardrail)
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
                "uipath_agents._observability.callback.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "tool_pre_execution_custom_guard",
                        "guardrail": mock_guardrail,
                        "scope": GuardrailScope.TOOL,
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
        from unittest.mock import MagicMock, patch

        from uipath.core.guardrails import (
            ApplyTo,
            DeterministicGuardrail,
            UniversalRule,
        )
        from uipath.platform.guardrails import GuardrailScope

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            mock_guardrail = MagicMock(spec=DeterministicGuardrail)
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
                "uipath_agents._observability.callback.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "tool_pre_execution_always_guard",
                        "guardrail": mock_guardrail,
                        "scope": GuardrailScope.TOOL,
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


class TestNestedLlmCallsInTools:
    """Tests for LLM calls made from within tools (e.g., analyze_files tool)."""

    def test_llm_call_inside_tool_parents_to_tool_span(
        self, tracer, span_exporter
    ) -> None:
        """LLM call from within a tool should be parented to the tool span.

        Scenario: analyze_files tool calls llm.ainvoke() internally.
        The LLM span should be a child of the tool span, not the agent span.
        """
        from unittest.mock import patch

        callback = UiPathTracingCallback(tracer)
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
                "uipath_agents._observability.callback.get_current_run_id",
                return_value=tool_run_id,
            ):
                callback.on_chat_model_start(
                    {"kwargs": {"model": "gpt-4"}},
                    [[]],
                    run_id=llm_run_id,
                    parent_run_id=unrelated_parent_id,  # Doesn't match tool_run_id
                )
                callback.on_llm_end(None, run_id=llm_run_id)  # type: ignore[arg-type]

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
        self, tracer, span_exporter
    ) -> None:
        """Independent LLM call (not inside a tool) should parent to agent span.

        When get_current_run_id() returns None (no tool context), LLM span
        should fall back to agent span as parent.
        """
        from unittest.mock import patch

        callback = UiPathTracingCallback(tracer)
        agent_run_id = uuid4()

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # LLM call without any tool context
            llm_run_id = uuid4()

            # Mock get_current_run_id to return None (no context)
            with patch(
                "uipath_agents._observability.callback.get_current_run_id",
                return_value=None,
            ):
                callback.on_chat_model_start(
                    {"kwargs": {"model": "gpt-4"}},
                    [[]],
                    run_id=llm_run_id,
                    parent_run_id=None,
                )
                callback.on_llm_end(None, run_id=llm_run_id)  # type: ignore[arg-type]

        spans = span_exporter.get_finished_spans()

        agent = next(s for s in spans if s.attributes.get("type") == "agentRun")
        llm = next(s for s in spans if s.name == "LLM call")

        # LLM span should be child of agent span
        assert llm.parent.span_id == agent.context.span_id

    def test_parallel_tool_and_llm_isolated(self, tracer, span_exporter) -> None:
        """Parallel tool and LLM call should have correct parents.

        Tool's LLM call should parent to tool, while independent LLM call
        should parent to agent.
        """
        from unittest.mock import patch

        callback = UiPathTracingCallback(tracer)
        agent_run_id = uuid4()

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Tool starts
            tool_run_id = uuid4()
            callback.on_tool_start({"name": "tool1"}, "{}", run_id=tool_run_id)

            # LLM call inside tool (context returns tool_run_id)
            llm_in_tool_id = uuid4()
            with patch(
                "uipath_agents._observability.callback.get_current_run_id",
                return_value=tool_run_id,
            ):
                callback.on_chat_model_start(
                    {"kwargs": {"model": "gpt-4"}},
                    [[]],
                    run_id=llm_in_tool_id,
                    parent_run_id=None,
                )
                callback.on_llm_end(None, run_id=llm_in_tool_id)  # type: ignore[arg-type]

            callback.on_tool_end("result", run_id=tool_run_id)

            # Independent LLM call (context returns None - no tool)
            llm_independent_id = uuid4()
            with patch(
                "uipath_agents._observability.callback.get_current_run_id",
                return_value=None,
            ):
                callback.on_chat_model_start(
                    {"kwargs": {"model": "gpt-4"}},
                    [[]],
                    run_id=llm_independent_id,
                    parent_run_id=None,
                )
                callback.on_llm_end(None, run_id=llm_independent_id)  # type: ignore[arg-type]

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
        from uipath_agents._observability.callback import get_current_run_id

        # Outside of a langchain runnable, should return None
        result = get_current_run_id()
        assert result is None

    def test_get_current_run_id_in_runnable_context(self) -> None:
        """get_current_run_id returns run_id when inside a langchain runnable."""
        from langchain_core.runnables import RunnableLambda

        from uipath_agents._observability.callback import get_current_run_id

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
    """Tests for escalation span completion with reviewed data from outputs."""

    def test_escalate_action_completes_span_in_on_chain_end(
        self, callback: UiPathTracingCallback, tracer: UiPathTracer, span_exporter
    ) -> None:
        """Escalate action span completes in on_chain_end with reviewed data."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Guardrail evaluation fails
            eval_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=eval_run_id,
                metadata={"langgraph_node": "agent_pre_execution_pii_guard"},
            )
            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: False}},
                run_id=eval_run_id,
            )

            # Escalate action node fires (with action_type metadata)
            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard_hitl",
                    "action_type": "Escalate",
                },
            )

            # Verify run_id is tracked for on_chain_end
            assert action_run_id in callback._escalate_action_run_ids

            # on_chain_end with reviewed data in inner_state
            callback.on_chain_end(
                {
                    # INNER_STATE_KEY: {
                    #     ESCALATION_REVIEWED_DATA_KEY: {
                    #         "reviewed_inputs": {"input": "sanitized"},
                    #         "reviewed_outputs": None,
                    #         "reviewed_by": "user@example.com",
                    #     }
                    # }
                },
                run_id=action_run_id,
            )

            # run_id should be removed after completion
            assert action_run_id not in callback._escalate_action_run_ids

        spans = span_exporter.get_finished_spans()
        review_spans = [s for s in spans if s.name == "Review task"]
        assert len(review_spans) == 1

        review_span = review_spans[0]
        assert review_span.attributes.get("reviewOutcome") == "Approved"
        # assert review_span.attributes.get("reviewedBy") == "user@example.com"
        # assert "sanitized" in review_span.attributes.get("reviewedInputs", "")

    def test_resume_escalate_action_completes_span_via_upsert(
        self, callback: UiPathTracingCallback, tracer: UiPathTracer, span_exporter
    ) -> None:
        """Resume escalate action completes saved span via upsert with reviewed data."""
        from unittest.mock import patch

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Simulate resumed escalation context (set by runtime on resume)
            callback._resumed_escalation_trace_id = "trace-123"
            callback._resumed_escalation_span_data = {
                "name": "Review task",
                "attributes": {
                    "type": "reviewTask",
                    "reviewStatus": "waiting",
                },
            }

            # Resume escalate action node fires (with action_type metadata)
            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard_hitl_hitl",
                    "action_type": "Escalate",
                },
            )

            # Verify run_id is tracked for resume completion
            assert action_run_id in callback._escalate_action_resume_data
            assert (
                callback._escalate_action_resume_data[action_run_id]["trace_id"]
                == "trace-123"
            )

            # Mock the upsert call
            with patch.object(
                callback._tracer, "upsert_span_complete_by_data"
            ) as mock_upsert:
                mock_upsert.return_value = None

                # on_chain_end with reviewed data
                callback.on_chain_end(
                    {
                        # INNER_STATE_KEY: {
                        #     ESCALATION_REVIEWED_DATA_KEY: {
                        #         "reviewed_inputs": {"data": "reviewed"},
                        #         "reviewed_outputs": {"result": "approved"},
                        #         "reviewed_by": "reviewer@example.com",
                        #     }
                        # }
                    },
                    run_id=action_run_id,
                )

                # Verify upsert was called with correct data
                mock_upsert.assert_called_once()
                call_args = mock_upsert.call_args
                assert call_args[1]["trace_id"] == "trace-123"
                span_data = call_args[1]["span_data"]
                assert span_data["attributes"]["reviewStatus"] == "completed"
                assert span_data["attributes"]["reviewOutcome"] == "Approved"
                # assert span_data["attributes"]["reviewedBy"] == "reviewer@example.com"
                # assert "reviewed" in span_data["attributes"]["reviewedInputs"]
                # assert "approved" in span_data["attributes"]["reviewedOutputs"]

            # run_id should be removed after completion
            assert action_run_id not in callback._escalate_action_resume_data


class TestToolResultSerialization:
    """Tests for _set_tool_result with Pydantic models (MCP TextContent)."""

    def test_tool_result_with_single_textcontent(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Test tool result with single MCP TextContent object."""
        from mcp.types import TextContent

        run_id = uuid4()
        callback.on_tool_start({"name": "mcp_tool"}, "input", run_id=run_id)

        # Single TextContent object (Pydantic model)
        output = TextContent(type="text", text="Tool execution result")
        callback.on_tool_end(output, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1

        tool_span = spans[0]
        assert tool_span.attributes["type"] == SpanType.TOOL_CALL.value

        # Verify result attribute contains properly serialized JSON
        result_json = tool_span.attributes.get("result")
        assert result_json is not None

        import json

        parsed = json.loads(result_json)
        assert parsed["type"] == "text"
        assert parsed["text"] == "Tool execution result"

    def test_tool_result_with_list_of_textcontent(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Test tool result with list of MCP TextContent objects.

        This is the exact bug scenario we fixed:
        TypeError: Object of type TextContent is not JSON serializable
        """
        from mcp.types import TextContent

        run_id = uuid4()
        callback.on_tool_start({"name": "mcp_tool"}, "input", run_id=run_id)

        # List of TextContent objects - what MCP tools actually return
        output = [
            TextContent(type="text", text="First result"),
            TextContent(type="text", text="Second result"),
        ]

        # Should NOT raise TypeError
        callback.on_tool_end(output, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1

        tool_span = spans[0]
        result_json = tool_span.attributes.get("result")
        assert result_json is not None

        import json

        parsed = json.loads(result_json)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["type"] == "text"
        assert parsed[0]["text"] == "First result"
        assert parsed[1]["type"] == "text"
        assert parsed[1]["text"] == "Second result"

    def test_tool_result_with_dict_containing_pydantic(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Test tool result with dict containing Pydantic models."""
        from mcp.types import TextContent

        run_id = uuid4()
        callback.on_tool_start({"name": "complex_tool"}, "input", run_id=run_id)

        # Dict with Pydantic model values
        output = {
            "status": "success",
            "content": TextContent(type="text", text="Result data"),
            "metadata": {"count": 42},
        }

        callback.on_tool_end(output, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        tool_span = spans[0]
        result_json = tool_span.attributes.get("result")

        import json

        parsed = json.loads(result_json)
        assert parsed["status"] == "success"
        assert parsed["content"]["type"] == "text"
        assert parsed["content"]["text"] == "Result data"
        assert parsed["metadata"]["count"] == 42

    def test_tool_result_with_mixed_list(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Test tool result with mixed list of primitives and Pydantic models."""
        from mcp.types import TextContent

        run_id = uuid4()
        callback.on_tool_start({"name": "mixed_tool"}, "input", run_id=run_id)

        # Mixed list
        output = [
            "string value",
            42,
            TextContent(type="text", text="Pydantic model"),
            {"dict": "value"},
        ]

        callback.on_tool_end(output, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        tool_span = spans[0]
        result_json = tool_span.attributes.get("result")

        import json

        parsed = json.loads(result_json)
        assert parsed[0] == "string value"
        assert parsed[1] == 42
        assert parsed[2]["type"] == "text"
        assert parsed[2]["text"] == "Pydantic model"
        assert parsed[3] == {"dict": "value"}

    def test_tool_result_with_none(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Test tool result with None value."""
        run_id = uuid4()
        callback.on_tool_start({"name": "null_tool"}, "input", run_id=run_id)

        callback.on_tool_end(None, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        tool_span = spans[0]

        # Result should not be set when output is None
        result_json = tool_span.attributes.get("result")
        assert result_json is None

    def test_tool_result_with_simple_dict(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Test tool result with simple dict (backwards compatibility)."""
        run_id = uuid4()
        callback.on_tool_start({"name": "dict_tool"}, "input", run_id=run_id)

        output = {"status": "success", "value": 123}
        callback.on_tool_end(output, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        tool_span = spans[0]
        result_json = tool_span.attributes.get("result")

        import json

        parsed = json.loads(result_json)
        assert parsed == {"status": "success", "value": 123}

    def test_tool_result_with_simple_string(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Test tool result with simple string (backwards compatibility)."""
        run_id = uuid4()
        callback.on_tool_start({"name": "string_tool"}, "input", run_id=run_id)

        callback.on_tool_end("simple result", run_id=run_id)

        spans = span_exporter.get_finished_spans()
        tool_span = spans[0]
        result_json = tool_span.attributes.get("result")

        # Strings are returned as-is, not JSON-encoded
        assert result_json == "simple result"

    def test_tool_result_serialization_error_fallback(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Test fallback to string representation for non-serializable objects."""

        class NonSerializable:
            def __str__(self):
                return "custom_string_repr"

        run_id = uuid4()
        callback.on_tool_start({"name": "custom_tool"}, "input", run_id=run_id)

        output = NonSerializable()
        callback.on_tool_end(output, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        tool_span = spans[0]
        result_json = tool_span.attributes.get("result")

        # serialize_json() converts unknown objects to strings, then JSON-encodes them
        assert result_json == '"custom_string_repr"'


class TestResumedSpanSerialization:
    """Tests for resumed span result serialization with Pydantic models."""

    def test_resumed_tool_span_with_textcontent(
        self, callback: UiPathTracingCallback, tracer, span_exporter
    ) -> None:
        """Test resumed tool span with TextContent result."""
        from mcp.types import TextContent

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            tool_run_id = uuid4()
            callback.on_tool_start(
                {"name": "resumed_tool"}, "input", run_id=tool_run_id
            )

            # Simulate GraphInterrupt (tool is suspended)
            class GraphInterrupt(Exception):
                pass

            callback.on_tool_error(GraphInterrupt("Suspended"), run_id=tool_run_id)

            # Verify span is still tracked (not closed)
            assert tool_run_id in callback._spans

            # Now simulate resume with TextContent result
            output = [
                TextContent(type="text", text="Resumed result 1"),
                TextContent(type="text", text="Resumed result 2"),
            ]

            # This should use the resumed span data path (line 664 in callback.py)
            callback.on_tool_end(output, run_id=tool_run_id)

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("type") == "toolCall"]
        assert len(tool_spans) == 1

        tool_span = tool_spans[0]
        result_json = tool_span.attributes.get("result")
        assert result_json is not None

        import json

        parsed = json.loads(result_json)
        assert len(parsed) == 2
        assert parsed[0]["text"] == "Resumed result 1"

    def test_resumed_process_span_with_pydantic_result(
        self, callback: UiPathTracingCallback, tracer, span_exporter
    ) -> None:
        """Test resumed process span with Pydantic model result."""
        from pydantic import BaseModel

        class ProcessOutput(BaseModel):
            status: str
            metrics: dict[str, int]

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Simulate a process chain with metadata
            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={
                    "process_key": "test-process-123",
                    "langgraph_node": "__start__",
                },
            )

            # Resume with Pydantic output - tests line 644
            output = ProcessOutput(status="success", metrics={"count": 42, "time": 100})
            callback.on_chain_end(output, run_id=run_id)  # type: ignore[arg-type]

        # Should not raise TypeError during serialization
        spans = span_exporter.get_finished_spans()
        assert len(spans) >= 1


class TestToolArgumentsSerialization:
    """Tests for _set_tool_arguments with Pydantic models."""

    def test_tool_arguments_with_pydantic_input(
        self, callback: UiPathTracingCallback, span_exporter
    ) -> None:
        """Test tool arguments containing Pydantic model serialization."""
        from mcp.types import TextContent

        run_id = uuid4()

        # Tool input could contain Pydantic models
        message = TextContent(type="text", text="Input message")

        import json

        serialized_input = json.dumps({"message": message.model_dump(), "count": 5})

        callback.on_tool_start(
            {"name": "pydantic_arg_tool"}, serialized_input, run_id=run_id
        )
        callback.on_tool_end("result", run_id=run_id)

        spans = span_exporter.get_finished_spans()
        tool_span = spans[0]

        # Verify arguments were properly serialized
        arguments_json = tool_span.attributes.get("arguments")
        assert arguments_json is not None

        parsed = json.loads(arguments_json)
        assert parsed["message"]["type"] == "text"
        assert parsed["message"]["text"] == "Input message"
        assert parsed["count"] == 5
