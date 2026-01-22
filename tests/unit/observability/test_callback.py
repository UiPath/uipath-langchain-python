"""Tests for UiPathTracingCallback LangChain callback handler."""

from typing import Any
from uuid import UUID, uuid4

import pytest

from uipath_agents._observability.callback import UiPathTracingCallback
from uipath_agents._observability.schema import (
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
        # LLM call has type "llmCall", Model run has type "completion"
        llm_span = next(s for s in spans if s.name == "LLM call")
        model_span = next(s for s in spans if s.name == "Model run")
        assert llm_span.attributes["span_type"] == SpanType.LLM_CALL.value
        assert model_span.attributes["span_type"] == SpanType.COMPLETION.value

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
        assert spans[0].attributes["span_type"] == SpanType.TOOL_CALL.value
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
        model_span = next(s for s in spans if s.name == "Model run")
        assert model_span.attributes["model"] == "gpt-4"

    def test_extracts_model_from_kwargs_model(self, callback, span_exporter):
        """Test extraction from kwargs.model."""
        run_id = uuid4()
        serialized = {"kwargs": {"model": "claude-3"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        model_span = next(s for s in spans if s.name == "Model run")
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

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

            run_id = uuid4()
            callback.on_tool_start({"name": "calc"}, "input", run_id=run_id)
            callback.on_tool_end("result", run_id=run_id)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2

        agent = next(s for s in spans if s.attributes["span_type"] == "agentRun")
        tool = next(s for s in spans if s.attributes["span_type"] == "toolCall")

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
        assert child_span.attributes["span_type"] == SpanType.INTEGRATION_TOOL.value


class TestGuardrailActionDetection:
    """Tests for guardrail action detection from action nodes."""

    def test_validation_passed_ends_immediately_with_skip(
        self, callback, tracer, span_exporter
    ):
        """When validation passes, span ends immediately with action=Skip."""
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

            run_id = uuid4()
            # Start guardrail evaluation
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={"langgraph_node": "agent_pre_execution_pii_guard"},
            )
            # End with no validation_result (passed)
            callback.on_chain_end({}, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("span_type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Skip"

    def test_validation_failed_defers_until_action_node(
        self, callback, tracer, span_exporter
    ):
        """When validation fails, span ending is deferred until action node fires."""
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: "PII detected"}},
                run_id=run_id,
            )

            # Span should be pending action
            assert (
                "agent_pre_execution_pii_guard" in callback._pending_guardrail_actions
            )

            # Action node fires with _block suffix
            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={"langgraph_node": "agent_pre_execution_pii_guard_block"},
            )

            # Pending action should be cleared
            assert (
                "agent_pre_execution_pii_guard"
                not in callback._pending_guardrail_actions
            )

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("span_type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Block"
        assert eval_spans[0].attributes.get("validationResult") == "PII detected"

    def test_action_node_log_sets_correct_action(self, callback, tracer, span_exporter):
        """Test action node with _log suffix sets action=log."""
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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

            # Action node fires with _log suffix
            callback.on_chain_start(
                {},
                {},
                run_id=uuid4(),
                metadata={"langgraph_node": "llm_pre_execution_prompt_injection_log"},
            )

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("span_type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Log"

    def test_action_node_hitl_sets_correct_action(
        self, callback, tracer, span_exporter
    ):
        """Test action node with _hitl suffix sets action=Escalate."""
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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
                metadata={"langgraph_node": "llm_post_execution_review_guard_hitl"},
            )

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("span_type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Escalate"

    def test_command_object_extracts_validation_result(
        self, callback, tracer, span_exporter
    ):
        """Test validation result extraction from LangGraph Command objects."""

        class MockCommand:
            """Mock LangGraph Command object with update dict."""

            def __init__(self, update: dict[str, Any]) -> None:
                self.update = update

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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
                        GUARDRAIL_VALIDATION_RESULT_KEY: "PII detected in input"
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
                metadata={"langgraph_node": "agent_pre_execution_pii_guard_block"},
            )

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("span_type") == "guardrailEvaluation"
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

        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

            run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=run_id,
                metadata={"langgraph_node": "agent_pre_execution_pii_guard"},
            )
            # Command with None validation result = passed
            command = MockCommand(
                update={INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: None}}
            )
            callback.on_chain_end(command, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        eval_spans = [
            s for s in spans if s.attributes.get("span_type") == "guardrailEvaluation"
        ]
        assert len(eval_spans) == 1
        assert eval_spans[0].attributes.get("action") == "Skip"


class TestToolGuardrailParenting:
    """Tests for tool guardrail spans parenting to tool span."""

    def test_tool_pre_guardrail_parents_to_tool_span(
        self, callback, tracer, span_exporter
    ):
        """Tool pre guardrails should be children of the current tool span."""
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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
        tool_span = next(
            s for s in spans if s.attributes.get("span_type") == "toolCall"
        )
        container_span = next(
            s for s in spans if s.attributes.get("span_type") == "toolPreGuardrails"
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
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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
        tool_span = next(
            s for s in spans if s.attributes.get("span_type") == "toolCall"
        )
        container_span = next(
            s for s in spans if s.attributes.get("span_type") == "toolPostGuardrails"
        )

        assert container_span.parent.span_id == tool_span.context.span_id


class TestPhaseTransitions:
    """Tests for guardrail phase transition logic."""

    def test_tool_pre_closes_llm_post(self, callback, tracer, span_exporter):
        """Transitioning to tool_pre should close llm_post container."""
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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
        tool_span = next(
            s for s in spans if s.attributes.get("span_type") == "toolCall"
        )
        container = next(
            s for s in spans if s.attributes.get("span_type") == "toolPreGuardrails"
        )

        # Container should be child of tool span
        assert container.parent.span_id == tool_span.context.span_id
        # Tool span should have correct name after enrichment
        assert tool_span.name == "Tool call - my_tool"

    def test_tool_blocked_by_guardrail_no_orphan(self, callback, tracer, span_exporter):
        """When guardrail blocks, placeholder tool span should end cleanly."""
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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
                metadata={"langgraph_node": "tool_pre_execution_pii_guard_block"},
            )

            # Placeholder should be cleaned up
            assert callback._current_tool_span is None
            assert callback._tool_span_from_guardrail is False

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.attributes.get("span_type") == "toolCall"]
        # Placeholder was created and properly ended
        assert len(tool_spans) == 1

    def test_multiple_tool_pre_guardrails_before_tool(
        self, callback, tracer, span_exporter
    ):
        """Multiple tool_pre guardrails fire before tool - all parent to same tool span."""
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span)

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
        tool_spans = [s for s in spans if s.attributes.get("span_type") == "toolCall"]
        assert len(tool_spans) == 1

        tool_span = tool_spans[0]
        container = next(
            s for s in spans if s.attributes.get("span_type") == "toolPreGuardrails"
        )

        # Container should be child of tool span
        assert container.parent.span_id == tool_span.context.span_id
        # Tool should have correct name after enrichment
        assert tool_span.name == "Tool call - my_tool"
