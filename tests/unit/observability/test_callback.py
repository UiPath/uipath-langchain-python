"""Tests for LlmOpsInstrumentationCallback LangChain callback handler."""

from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from uipath.core.guardrails import (
    GuardrailScope,
)
from uipath.platform.action_center.tasks import TaskRecipient, TaskRecipientType
from uipath_langchain.agent.guardrails.types import ExecutionStage

from uipath_agents._observability.llmops.callback import LlmOpsInstrumentationCallback
from uipath_agents._observability.llmops.spans.span_attributes import SpanType
from uipath_agents._observability.llmops.spans.span_name import (
    GUARDRAIL_VALIDATION_RESULT_KEY,
    INNER_STATE_KEY,
)

# span_exporter and callback fixture comes from conftest.py


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

        assert "model" not in llm_span.attributes
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
        assert "model" not in llm_span.attributes
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

    def test_graph_interrupt_with_wait_escalation_extracts_task_id(
        self, callback, span_exporter
    ):
        """Test on_tool_error extracts taskId from GraphInterrupt with WaitEscalation payload."""

        class MockInterrupt:
            def __init__(self, value: Any) -> None:
                self.value = value

        class MockAction:
            def __init__(self) -> None:
                self.id = 12345
                self.key = "task-key-abc"

        class MockWaitEscalation:
            def __init__(self) -> None:
                self.action = MockAction()
                self.recipient = TaskRecipient(
                    type=TaskRecipientType.EMAIL, value="user@example.com"
                )

        class GraphInterrupt(Exception):
            def __init__(self) -> None:
                interrupts = (MockInterrupt(MockWaitEscalation()),)
                super().__init__(interrupts)

        run_id = uuid4()
        serialized = {"name": "escalate_approval"}
        metadata = {
            "tool_type": "escalation",
            "display_name": "ApprovalApp",
            "channel_type": "actionCenter",
        }

        callback.on_tool_start(serialized, "input", run_id=run_id, metadata=metadata)

        # Child escalation span should exist
        child_key = UUID(int=run_id.int ^ 2)
        assert child_key in callback._state.spans

        child_span = callback._state.spans[child_key]

        callback.on_tool_error(GraphInterrupt(), run_id=run_id)

        # Span should still be tracked (not closed)
        assert run_id in callback._state.spans

        # Child span should have taskId attribute set
        assert child_span.attributes.get("taskId") == "12345"
        assert child_span.attributes.get("assignedTo") == "user@example.com"

        callback.cleanup()

    def test_graph_interrupt_without_interrupts_attr_still_upserts(
        self, callback, span_exporter
    ):
        """Test on_tool_error handles GraphInterrupt without .interrupts gracefully."""

        class GraphInterrupt(Exception):
            pass

        run_id = uuid4()
        serialized = {"name": "escalate_tool"}
        metadata = {
            "tool_type": "escalation",
            "display_name": "TestApp",
            "channel_type": "actionCenter",
        }

        callback.on_tool_start(serialized, "input", run_id=run_id, metadata=metadata)
        callback.on_tool_error(GraphInterrupt("Suspended"), run_id=run_id)

        # Spans should still be tracked (not closed)
        assert run_id in callback._state.spans

        callback.cleanup()


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
