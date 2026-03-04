"""Tests for tool instrumentor callId and arguments capture."""

from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

from uipath_agents._observability.llmops.instrumentors.attribute_helpers import (
    parse_tool_arguments,
)
from uipath_agents._observability.llmops.instrumentors.base import InstrumentationState
from uipath_agents._observability.llmops.instrumentors.tool_instrumentor import (
    ToolSpanInstrumentor,
)


class TestParseToolArguments:
    """Tests for the parse_tool_arguments helper function."""

    def test_parses_valid_json_dict(self) -> None:
        """Valid JSON object should be parsed to dict."""
        result = parse_tool_arguments('{"key": "value", "number": 42}')
        assert result == {"key": "value", "number": 42}

    def test_parses_nested_json(self) -> None:
        """Nested JSON should be parsed correctly."""
        result = parse_tool_arguments('{"log": "Current date is: 2026-02-05"}')
        assert result == {"log": "Current date is: 2026-02-05"}

    def test_returns_none_for_empty_string(self) -> None:
        """Empty string should return None."""
        result = parse_tool_arguments("")
        assert result is None

    def test_returns_none_for_invalid_json(self) -> None:
        """Invalid JSON should return None, not raise exception."""
        result = parse_tool_arguments("not valid json")
        assert result is None

    def test_returns_none_for_json_array(self) -> None:
        """JSON array (not object) should return None."""
        result = parse_tool_arguments("[1, 2, 3]")
        assert result is None

    def test_returns_none_for_json_string(self) -> None:
        """JSON string (not object) should return None."""
        result = parse_tool_arguments('"just a string"')
        assert result is None


class TestToolSpanInstrumentorCallId:
    """Tests for ToolSpanInstrumentor capturing callId from kwargs."""

    def test_on_tool_start_captures_call_id(self) -> None:
        """Tool span should have callId from kwargs.tool_call_id."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "RPA_Workflow"}
        input_str = '{"log": "Current date is: 2026-02-05"}'

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata={"tool_type": "process", "display_name": "RPA Workflow"},
                tool_call_id="toolu_bdrk_01Lj6v2gwof4MUyRAULCJ7Sw",
            )

        mock_span_factory.start_tool_call.assert_called_once()
        call_kwargs = mock_span_factory.start_tool_call.call_args
        assert (
            call_kwargs.kwargs.get("call_id") == "toolu_bdrk_01Lj6v2gwof4MUyRAULCJ7Sw"
        )

    def test_on_tool_start_captures_arguments(self) -> None:
        """Tool span should have arguments from parsed input_str JSON."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "RPA_Workflow"}
        input_str = '{"log": "Current date is: 2026-02-05"}'

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata={"tool_type": "process", "display_name": "RPA Workflow"},
            )

        mock_span_factory.start_tool_call.assert_called_once()
        call_kwargs = mock_span_factory.start_tool_call.call_args
        assert call_kwargs.kwargs.get("arguments") == {
            "log": "Current date is: 2026-02-05"
        }

    def test_on_tool_start_passes_both_call_id_and_arguments(self) -> None:
        """Tool span should receive both callId and arguments together."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "get_weather"}
        input_str = '{"city": "Seattle", "units": "celsius"}'

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata=None,
                tool_call_id="call_abc123xyz",
            )

        mock_span_factory.start_tool_call.assert_called_once()
        call_kwargs = mock_span_factory.start_tool_call.call_args.kwargs

        assert call_kwargs.get("call_id") == "call_abc123xyz"
        assert call_kwargs.get("arguments") == {"city": "Seattle", "units": "celsius"}

    def test_on_tool_start_handles_missing_call_id(self) -> None:
        """Tool span should handle missing tool_call_id gracefully."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "simple_tool"}
        input_str = '{"param": "value"}'

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata=None,
                # No tool_call_id provided
            )

        mock_span_factory.start_tool_call.assert_called_once()
        call_kwargs = mock_span_factory.start_tool_call.call_args.kwargs
        assert call_kwargs.get("call_id") is None
        assert call_kwargs.get("arguments") == {"param": "value"}

    def test_on_tool_start_handles_invalid_json_arguments(self) -> None:
        """Tool span should handle invalid JSON input gracefully."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "broken_tool"}
        input_str = "not valid json at all"

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata=None,
                tool_call_id="call_123",
            )

        mock_span_factory.start_tool_call.assert_called_once()
        call_kwargs = mock_span_factory.start_tool_call.call_args.kwargs
        assert call_kwargs.get("call_id") == "call_123"
        assert call_kwargs.get("arguments") is None


class TestToolSpanInstrumentorMcpTools:
    """Tests for MCP tool name construction with server slug prefix."""

    def test_mcp_tool_name_includes_slug_prefix(self) -> None:
        """MCP tool with slug should get name 'mcp-{sanitized_slug}-tool-{tool_name}'."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "add"}
        input_str = '{"a": 1, "b": 2}'

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata={
                    "tool_type": "mcp",
                    "display_name": "add",
                    "slug": "my_mcp_coded-tool",
                },
            )

        call_args = mock_span_factory.start_tool_call.call_args
        # tool_name is first positional arg
        assert call_args[0][0] == "mcp-my_mcp_coded_tool-tool-add"

    def test_mcp_tool_without_slug_uses_original_name(self) -> None:
        """MCP tool without slug should keep the original tool name."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "add"}
        input_str = '{"a": 1}'

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata={
                    "tool_type": "mcp",
                    "display_name": "add",
                },
            )

        call_args = mock_span_factory.start_tool_call.call_args
        assert call_args[0][0] == "add"

    def test_mcp_tool_type_value_is_mcp(self) -> None:
        """MCP tool should get toolType='Mcp' via get_tool_type_value."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "add"}
        input_str = '{"a": 1}'

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata={
                    "tool_type": "mcp",
                    "display_name": "add",
                    "slug": "my_server",
                },
            )

        call_kwargs = mock_span_factory.start_tool_call.call_args.kwargs
        assert call_kwargs["tool_type_value"] == "Mcp"

    def test_mcp_tool_creates_child_span(self) -> None:
        """MCP tool should create a mcpTool child span via start_mcp_tool."""
        mock_span_factory = MagicMock()
        mock_tool_call_span = MagicMock()
        mock_mcp_child_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_tool_call_span
        mock_span_factory.start_mcp_tool.return_value = mock_mcp_child_span

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "search"}
        input_str = '{"query": "latest iPhone", "max_results": 1}'

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata={
                    "tool_type": "mcp",
                    "display_name": "search",
                    "slug": "duck-duck-go-search",
                },
            )

        mock_span_factory.start_mcp_tool.assert_called_once()
        call_kwargs = mock_span_factory.start_mcp_tool.call_args.kwargs
        assert call_kwargs["tool_name"] == "mcp-duck_duck_go_search-tool-search"
        assert call_kwargs["arguments"] == {
            "query": "latest iPhone",
            "max_results": 1,
        }
        assert call_kwargs["parent_span"] is mock_tool_call_span

    def test_mcp_tool_without_slug_creates_child_span(self) -> None:
        """MCP tool without slug should still create mcpTool child span."""
        mock_span_factory = MagicMock()
        mock_tool_call_span = MagicMock()
        mock_mcp_child_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_tool_call_span
        mock_span_factory.start_mcp_tool.return_value = mock_mcp_child_span

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "search"}
        input_str = '{"query": "test"}'

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata={
                    "tool_type": "mcp",
                    "display_name": "search",
                },
            )

        mock_span_factory.start_mcp_tool.assert_called_once()
        assert (
            mock_span_factory.start_mcp_tool.call_args.kwargs["tool_name"] == "search"
        )

    def test_mcp_tool_end_sets_result_on_child_span(self) -> None:
        """on_tool_end should set 'result' attribute on the mcpTool child span."""
        mock_span_factory = MagicMock()
        mock_tool_call_span = MagicMock()
        mock_mcp_child_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_tool_call_span
        mock_span_factory.start_mcp_tool.return_value = mock_mcp_child_span

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        output = [{"type": "text", "text": "hello"}]

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized={"name": "search"},
                input_str='{"query": "test"}',
                run_id=run_id,
                parent_run_id=None,
                metadata={"tool_type": "mcp", "display_name": "search"},
            )
            instrumentor.on_tool_end(output=output, run_id=run_id)

        mock_mcp_child_span.set_attribute.assert_any_call(
            "result", '[{"type": "text", "text": "hello"}]'
        )


class TestToolSpanInstrumentorGuardrailPath:
    """Tests for tool span with pre-existing guardrail span."""

    def test_guardrail_span_sets_call_id_attribute(self) -> None:
        """When span exists from guardrail, callId should be set as attribute."""
        mock_span_factory = MagicMock()
        mock_existing_span = MagicMock()

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()
        state.current_tool_span = mock_existing_span
        state.tool_span_from_guardrail = True

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "validated_tool"}
        input_str = '{"validated": true}'

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata=None,
                tool_call_id="toolu_guardrail_123",
            )

        # Should NOT create new span
        mock_span_factory.start_tool_call.assert_not_called()

        # Should set attributes on existing span
        mock_existing_span.set_attribute.assert_any_call(
            "call_id", "toolu_guardrail_123"
        )

    def test_guardrail_span_sets_arguments_attribute(self) -> None:
        """When span exists from guardrail, arguments should be set as JSON attribute."""
        mock_span_factory = MagicMock()
        mock_existing_span = MagicMock()

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()
        state.current_tool_span = mock_existing_span
        state.tool_span_from_guardrail = True

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        serialized = {"name": "validated_tool"}
        input_str = '{"validated": true, "score": 0.95}'

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=run_id,
                parent_run_id=None,
                metadata=None,
            )

        # Should set arguments as JSON string
        mock_existing_span.set_attribute.assert_any_call(
            "input", '{"validated": true, "score": 0.95}'
        )


class TestUpsertResumedSpansNoContent:
    """Tests for NO_CONTENT filtering in _upsert_resumed_spans_on_completion."""

    NO_CONTENT_MARKER: dict[str, Any] = {
        "status": "completed",
        "__internal": "NO_CONTENT",
    }

    def _create_instrumentor(
        self,
    ) -> tuple[ToolSpanInstrumentor, InstrumentationState, MagicMock]:
        mock_span_factory = MagicMock()
        state = InstrumentationState(span_factory=mock_span_factory)
        state.resumed_trace_id = "trace-123"
        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )
        return instrumentor, state, mock_span_factory

    def test_no_content_skips_process_span_result(self) -> None:
        """Process span should not get a result attribute for NO_CONTENT output."""
        instrumentor, state, mock_factory = self._create_instrumentor()
        state.resumed_process_span_data = {"attributes": {"type": "processTool"}}
        state.resumed_tool_span_data = None

        instrumentor._upsert_resumed_spans_on_completion(self.NO_CONTENT_MARKER, None)

        mock_factory.upsert_span_complete_by_data.assert_called_once()
        span_data = mock_factory.upsert_span_complete_by_data.call_args.kwargs[
            "span_data"
        ]
        assert "result" not in span_data["attributes"]

    def test_no_content_skips_tool_span_output(self) -> None:
        """Tool span should not get an output attribute for NO_CONTENT output."""
        instrumentor, state, mock_factory = self._create_instrumentor()
        state.resumed_process_span_data = None
        state.resumed_tool_span_data = {"attributes": {}}

        instrumentor._upsert_resumed_spans_on_completion(self.NO_CONTENT_MARKER, None)

        mock_factory.upsert_span_complete_by_data.assert_called_once()
        span_data = mock_factory.upsert_span_complete_by_data.call_args.kwargs[
            "span_data"
        ]
        assert "output" not in span_data["attributes"]

    def test_no_content_skips_both_spans(self) -> None:
        """Both process and tool spans should skip output for NO_CONTENT."""
        instrumentor, state, mock_factory = self._create_instrumentor()
        state.resumed_process_span_data = {"attributes": {"type": "processTool"}}
        state.resumed_tool_span_data = {"attributes": {}}

        instrumentor._upsert_resumed_spans_on_completion(self.NO_CONTENT_MARKER, None)

        assert mock_factory.upsert_span_complete_by_data.call_count == 2
        for call in mock_factory.upsert_span_complete_by_data.call_args_list:
            span_data = call.kwargs["span_data"]
            assert "result" not in span_data["attributes"]
            assert "output" not in span_data["attributes"]

    def test_normal_output_sets_process_span_result(self) -> None:
        """Normal output should be set as result on process span."""
        instrumentor, state, mock_factory = self._create_instrumentor()
        state.resumed_process_span_data = {"attributes": {"type": "processTool"}}
        state.resumed_tool_span_data = None

        instrumentor._upsert_resumed_spans_on_completion({"answer": 42}, None)

        span_data = mock_factory.upsert_span_complete_by_data.call_args.kwargs[
            "span_data"
        ]
        assert span_data["attributes"]["result"] == '{"answer": 42}'

    def test_normal_output_sets_tool_span_output(self) -> None:
        """Normal output should be set as output on tool span."""
        instrumentor, state, mock_factory = self._create_instrumentor()
        state.resumed_process_span_data = None
        state.resumed_tool_span_data = {"attributes": {}}

        instrumentor._upsert_resumed_spans_on_completion({"answer": 42}, None)

        span_data = mock_factory.upsert_span_complete_by_data.call_args.kwargs[
            "span_data"
        ]
        assert span_data["attributes"]["output"] == '{"answer": 42}'


class TestToolSpanParentingHierarchy:
    """Tests for tool span parent selection (C# parity)."""

    def test_regular_tool_parents_to_agent_span_not_llm_span(self) -> None:
        """Standard tools should parent to agent span even when current_llm_span is set."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        mock_agent_span = MagicMock(name="agent_span")
        mock_llm_span = MagicMock(name="llm_span")

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = mock_agent_span
        state.current_llm_span = mock_llm_span

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()
        parent_run_id = uuid4()

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            with patch.object(
                state, "get_span_or_root", return_value=mock_agent_span
            ) as mock_get:
                instrumentor.on_tool_start(
                    serialized={"name": "Test_VB"},
                    input_str='{"param": "value"}',
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    metadata={"tool_type": "process", "display_name": "Test VB"},
                )

        call_kwargs = mock_span_factory.start_tool_call.call_args.kwargs
        assert call_kwargs["parent_span"] is mock_agent_span
        mock_get.assert_called_once_with(parent_run_id)

    def test_context_grounding_tool_parents_to_llm_span(self) -> None:
        """Context grounding tools should parent to current LLM span."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        mock_agent_span = MagicMock(name="agent_span")
        mock_llm_span = MagicMock(name="llm_span")

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = mock_agent_span
        state.current_llm_span = mock_llm_span

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        run_id = uuid4()

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized={"name": "context_grounding_search"},
                input_str='{"query": "test"}',
                run_id=run_id,
                parent_run_id=None,
                metadata={
                    "tool_type": "context_grounding",
                    "display_name": "Context Grounding",
                },
            )

        call_kwargs = mock_span_factory.start_tool_call.call_args.kwargs
        assert call_kwargs["parent_span"] is mock_llm_span

    def test_context_type_tool_parents_to_llm_span(self) -> None:
        """Tools with tool_type='context' should also parent to LLM span."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        mock_llm_span = MagicMock(name="llm_span")

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = MagicMock()
        state.current_llm_span = mock_llm_span

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            instrumentor.on_tool_start(
                serialized={"name": "context_search"},
                input_str="{}",
                run_id=uuid4(),
                parent_run_id=None,
                metadata={"tool_type": "context", "display_name": "Context"},
            )

        call_kwargs = mock_span_factory.start_tool_call.call_args.kwargs
        assert call_kwargs["parent_span"] is mock_llm_span

    def test_context_grounding_falls_back_when_no_llm_span(self) -> None:
        """Context grounding tool should fall back to agent span when no LLM span."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        mock_agent_span = MagicMock(name="agent_span")

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = mock_agent_span
        state.current_llm_span = None

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        parent_run_id = uuid4()

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            with patch.object(
                state, "get_span_or_root", return_value=mock_agent_span
            ) as mock_get:
                instrumentor.on_tool_start(
                    serialized={"name": "context_grounding_search"},
                    input_str="{}",
                    run_id=uuid4(),
                    parent_run_id=parent_run_id,
                    metadata={
                        "tool_type": "context_grounding",
                        "display_name": "Context Grounding",
                    },
                )

        call_kwargs = mock_span_factory.start_tool_call.call_args.kwargs
        assert call_kwargs["parent_span"] is mock_agent_span
        mock_get.assert_called_once_with(parent_run_id)

    def test_escalation_tool_parents_to_agent_span(self) -> None:
        """Escalation tools should parent to agent span (C# ToolWorkflowV4)."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        mock_agent_span = MagicMock(name="agent_span")

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = mock_agent_span
        state.current_llm_span = MagicMock(name="llm_span")

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            with patch.object(state, "get_span_or_root", return_value=mock_agent_span):
                instrumentor.on_tool_start(
                    serialized={"name": "Escalation_1"},
                    input_str='{"message": "need help"}',
                    run_id=uuid4(),
                    parent_run_id=None,
                    metadata={
                        "tool_type": "escalation",
                        "display_name": "Escalation_1",
                    },
                )

        call_kwargs = mock_span_factory.start_tool_call.call_args.kwargs
        assert call_kwargs["parent_span"] is mock_agent_span

    def test_ixp_extraction_tool_parents_to_agent_span(self) -> None:
        """IXP extraction tools should parent to agent span (C# ToolWorkflowV4)."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        mock_agent_span = MagicMock(name="agent_span")

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = mock_agent_span
        state.current_llm_span = MagicMock(name="llm_span")

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            with patch.object(state, "get_span_or_root", return_value=mock_agent_span):
                instrumentor.on_tool_start(
                    serialized={"name": "Extract_Invoice"},
                    input_str='{"document": "invoice.pdf"}',
                    run_id=uuid4(),
                    parent_run_id=None,
                    metadata={
                        "tool_type": "ixp_extraction",
                        "display_name": "Extract Invoice",
                    },
                )

        call_kwargs = mock_span_factory.start_tool_call.call_args.kwargs
        assert call_kwargs["parent_span"] is mock_agent_span

    def test_tool_without_type_parents_to_agent_span(self) -> None:
        """Tool with no tool_type should parent to agent span, not LLM span."""
        mock_span_factory = MagicMock()
        mock_span = MagicMock()
        mock_span_factory.start_tool_call.return_value = mock_span

        mock_agent_span = MagicMock(name="agent_span")

        state = InstrumentationState(span_factory=mock_span_factory)
        state.agent_span = mock_agent_span
        state.current_llm_span = MagicMock(name="llm_span")

        instrumentor = ToolSpanInstrumentor(
            state=state,
            close_container=MagicMock(),
        )

        with patch(
            "uipath_agents._observability.llmops.instrumentors.tool_instrumentor.SpanHierarchyManager"
        ):
            with patch.object(state, "get_span_or_root", return_value=mock_agent_span):
                instrumentor.on_tool_start(
                    serialized={"name": "custom_tool"},
                    input_str="{}",
                    run_id=uuid4(),
                    parent_run_id=None,
                    metadata=None,
                )

        call_kwargs = mock_span_factory.start_tool_call.call_args.kwargs
        assert call_kwargs["parent_span"] is mock_agent_span
