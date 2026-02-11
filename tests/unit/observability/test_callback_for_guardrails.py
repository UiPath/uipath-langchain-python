"""Tests for LlmOpsInstrumentationCallback LangChain callback handler."""

from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

from uipath.core.guardrails import (
    GuardrailScope,
)
from uipath_langchain.agent.guardrails.types import ExecutionStage

from uipath_agents._observability.llmops.callback import LlmOpsInstrumentationCallback
from uipath_agents._observability.llmops.spans.span_factory import LlmOpsSpanFactory
from uipath_agents._observability.llmops.spans.span_name import (
    GUARDRAIL_VALIDATION_DETAILS_KEY,
    GUARDRAIL_VALIDATION_RESULT_KEY,
    INNER_STATE_KEY,
)

# span_exporter and callback fixture comes from conftest.py


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
                    "tool_type": "integration",
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
                    "tool_type": "integration",
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
                    "tool_type": "process",
                },
            )
            callback.on_chain_end({}, run_id=guard_run_id)

            # THEN tool starts
            tool_run_id = uuid4()
            callback.on_tool_start(
                {"name": "my_tool"},
                "input",
                run_id=tool_run_id,
                metadata={
                    "tool_type": "process",
                },
            )
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
        assert tool_span.attributes.get("toolType") == "Process"

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
                    "tool_type": "integration",
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
                    "tool_type": "integration",
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
        assert tool_spans[0].attributes.get("toolType") == "Integration"

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
                    "tool_type": "escalation",
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
                    "tool_type": "escalation",
                },
            )
            callback.on_chain_end({}, run_id=guard2_id)

            # Verify same placeholder is still used
            assert callback._state.current_tool_span is placeholder_span

            # Tool starts
            tool_run_id = uuid4()
            callback.on_tool_start(
                {"name": "my_tool"},
                "input",
                run_id=tool_run_id,
                metadata={
                    "tool_type": "escalation",
                },
            )
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
        assert tool_span.attributes.get("toolType") == "Escalation"


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


class TestGuardrailInstrumentorEdgeCases:
    """Tests for guardrail instrumentor guard clauses and error paths."""

    def test_handle_action_end_returns_early_when_metadata_is_none(
        self, callback, tracer, span_exporter
    ):
        """handle_action_end should return immediately when metadata is None (line 201)."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            # Should not raise — just returns
            callback._guardrail_instrumentor.handle_action_end(
                run_id=uuid4(), metadata=None
            )

    def test_handle_action_error_returns_early_when_metadata_is_none(
        self, callback, tracer, span_exporter
    ):
        """handle_action_error should return immediately when metadata is None (line 273)."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            # Should not raise — just returns
            callback._guardrail_instrumentor.handle_action_error(
                run_id=uuid4(), metadata=None, error_str="some error"
            )

    def test_handle_action_error_returns_early_when_node_name_empty(
        self, callback, tracer, span_exporter
    ):
        """handle_action_error should return when node_name is empty (line 278)."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback._guardrail_instrumentor.handle_action_error(
                run_id=uuid4(),
                metadata={"action_type": "Block", "langgraph_node": ""},
                error_str="some error",
            )

    def test_handle_action_end_returns_early_when_eval_node_not_in_upcoming(
        self, callback, tracer, span_exporter
    ):
        """handle_action_end returns early when eval_node_name not in upcoming_guardrail_actions_info (line 242)."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            # upcoming_guardrail_actions_info is empty, so the node won't be found
            callback._guardrail_instrumentor.handle_action_end(
                run_id=uuid4(),
                metadata={
                    "action_type": "Log",
                    "langgraph_node": "agent_pre_execution_some_guard_log",
                },
            )
            # No exception, no spans created — just early return


class TestGuardrailActionErrorWithNonRecordingSpans:
    """Tests for handle_action_error when container/LLM spans are NonRecordingSpans (resume scenario)."""

    def test_action_error_upserts_non_recording_container_span_with_error(
        self, callback, tracer, span_exporter
    ):
        """When container is a NonRecordingSpan, handle_action_error should upsert it
        with error status using saved data"""
        from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
        from uipath.tracing import SpanStatus

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            # Set up resumed escalation context
            callback._state.resumed_escalation_trace_id = "0123456789abcdef"
            callback._state.resumed_escalation_span_data = {
                "name": "Review task",
                "span_id": "abcd1234",
                "attributes": {"type": "guardrailEscalation"},
            }
            callback._state.resumed_hitl_guardrail_span_data = {
                "name": "pii_guard",
                "span_id": "eval5678",
                "attributes": {},
            }
            container_data = {
                "name": "Guardrails - Agent PreExecution",
                "span_id": "cont9999",
                "attributes": {"containerAttr": "value"},
            }
            callback._state.resumed_hitl_guardrail_container_span_data = container_data

            # Put a NonRecordingSpan in the container map
            non_recording_ctx = SpanContext(
                trace_id=0x0123456789ABCDEF,
                span_id=0xC09F9999,
                is_remote=True,
                trace_flags=TraceFlags(0x01),
            )
            container_key = (GuardrailScope.AGENT, ExecutionStage.PRE_EXECUTION)
            callback._state.guardrail_containers[container_key] = NonRecordingSpan(
                non_recording_ctx
            )

            with patch.object(
                callback._state.span_factory, "upsert_span_complete_by_data"
            ) as mock_upsert:
                callback._guardrail_instrumentor.handle_action_error(
                    run_id=uuid4(),
                    metadata={
                        "action_type": "Escalate",
                        "langgraph_node": "agent_pre_execution_pii_guard_hitl",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "escalation_data": {"reviewed_by": "user@test.com"},
                    },
                    error_str="User rejected",
                )

                # Verify upsert was called for the container span with ERROR
                container_calls = [
                    c
                    for c in mock_upsert.call_args_list
                    if c[1].get("span_data", {}).get("name")
                    == "Guardrails - Agent PreExecution"
                ]
                assert len(container_calls) >= 1
                container_call = container_calls[0]
                assert container_call[1]["status"] == SpanStatus.ERROR
                assert (
                    container_call[1]["span_data"]["attributes"]["error"]
                    == "User rejected"
                )

    def test_action_error_upserts_non_recording_llm_span_with_error(
        self, callback, tracer, span_exporter
    ):
        """When LLM span is a NonRecordingSpan, handle_action_error should upsert it
        with error status using saved data."""
        from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
        from uipath.tracing import SpanStatus

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.LLM]

            # Set up resumed escalation context
            callback._state.resumed_escalation_trace_id = "0123456789abcdef"
            callback._state.resumed_escalation_span_data = {
                "name": "Review task",
                "span_id": "abcd1234",
                "attributes": {"type": "guardrailEscalation"},
            }
            callback._state.resumed_hitl_guardrail_span_data = {
                "name": "pii_guard",
                "span_id": "eval5678",
                "attributes": {},
            }
            llm_span_data = {
                "name": "LLM call",
                "span_id": "llm7777",
                "attributes": {"llm.model": "gpt-4o"},
            }
            callback._state.resumed_llm_span_data = llm_span_data

            # Set current_llm_span to a NonRecordingSpan
            non_recording_ctx = SpanContext(
                trace_id=0x0123456789ABCDEF,
                span_id=0x11A7777,
                is_remote=True,
                trace_flags=TraceFlags(0x01),
            )
            callback._state.current_llm_span = NonRecordingSpan(non_recording_ctx)

            with patch.object(
                callback._state.span_factory, "upsert_span_complete_by_data"
            ) as mock_upsert:
                callback._guardrail_instrumentor.handle_action_error(
                    run_id=uuid4(),
                    metadata={
                        "action_type": "Escalate",
                        "langgraph_node": "llm_pre_execution_pii_guard_hitl",
                        "scope": GuardrailScope.LLM,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "escalation_data": {"reviewed_by": "user@test.com"},
                    },
                    error_str="User rejected the LLM output",
                )

                # Verify upsert was called for the LLM span with ERROR
                llm_calls = [
                    c
                    for c in mock_upsert.call_args_list
                    if c[1].get("span_data", {}).get("name") == "LLM call"
                ]
                assert len(llm_calls) >= 1
                llm_call = llm_calls[0]
                assert llm_call[1]["status"] == SpanStatus.ERROR
                assert (
                    llm_call[1]["span_data"]["attributes"]["error"]
                    == "User rejected the LLM output"
                )

            # LLM span should be cleared after error
            assert callback._state.current_llm_span is None


class TestUpsertSuspendedEscalation:
    """Tests for _upsert_suspended_escalation span attribute setting."""

    def test_upsert_suspended_escalation_returns_early_without_pending_span(
        self, callback, tracer, span_exporter
    ):
        """_upsert_suspended_escalation returns early when no pending_escalation_span (line 832)."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            callback._state.pending_escalation_span = None
            # Should not raise
            callback._guardrail_instrumentor._upsert_suspended_escalation(
                metadata={"escalation_data": {"assigned_to": "user@test.com"}}
            )

    def test_upsert_suspended_escalation_sets_assigned_to_and_task_url(
        self, callback, tracer, span_exporter
    ):
        """_upsert_suspended_escalation sets assignedTo and taskUrl attributes"""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            # Start evaluation + action to create a pending escalation span
            eval_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=eval_run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_evaluation",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Escalate",
                },
            )
            callback.on_chain_end(
                {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: False}},
                run_id=eval_run_id,
            )

            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard_hitl",
                    "action_type": "Escalate",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_action",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                },
            )

            # Verify pending escalation span was created
            assert callback._state.pending_escalation_span is not None

            review_span = MagicMock()
            callback._state.pending_escalation_span = review_span

            with patch.object(
                callback._state.span_factory, "upsert_span_suspended"
            ) as mock_upsert:
                callback._guardrail_instrumentor._upsert_suspended_escalation(
                    metadata={
                        "escalation_data": {
                            "assigned_to": "reviewer@example.com",
                            "task_url": "https://tasks.example.com/123",
                        }
                    }
                )

                # Verify attributes were set on the span
                review_span.set_attribute.assert_any_call(
                    "assignedTo", "reviewer@example.com"
                )
                review_span.set_attribute.assert_any_call(
                    "taskUrl", "https://tasks.example.com/123"
                )
                mock_upsert.assert_called_once_with(review_span)


class TestCompleteResumedEscalationOutputs:
    """Tests for _complete_resumed_escalation_from_outputs with reviewed outputs."""

    def test_complete_escalation_with_reviewed_outputs(
        self, callback, tracer, span_exporter
    ):
        """_complete_resumed_escalation_from_outputs includes reviewedOutputs (line 875)."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            callback._state.resumed_escalation_trace_id = "0123456789abcdef"
            callback._state.resumed_escalation_span_data = {
                "name": "Review task",
                "span_id": "abcd1234",
                "attributes": {
                    "type": "guardrailEscalation",
                    "reviewStatus": "Pending",
                },
            }

            with patch.object(
                callback._state.span_factory, "upsert_span_complete_by_data"
            ) as mock_upsert:
                callback._guardrail_instrumentor._complete_resumed_escalation_from_outputs(
                    metadata={
                        "escalation_data": {
                            "reviewed_by": "reviewer@example.com",
                            "reviewed_outputs": {"response": "cleaned output"},
                        }
                    }
                )

                mock_upsert.assert_called_once()
                call_kwargs = mock_upsert.call_args[1]
                span_data = call_kwargs["span_data"]
                assert span_data["attributes"]["reviewedBy"] == "reviewer@example.com"
                assert (
                    '"response": "cleaned output"'
                    in span_data["attributes"]["reviewedOutputs"]
                )
                assert span_data["attributes"]["reviewOutcome"] == "Approved"
                assert span_data["attributes"]["reviewStatus"] == "Completed"
