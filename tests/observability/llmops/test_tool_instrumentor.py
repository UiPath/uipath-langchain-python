"""Tests for tool instrumentor callId and arguments capture."""

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
        """MCP tool with slug should get name 'mcp-{slug}-{tool_name}'."""
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
        assert call_args[0][0] == "mcp-my_mcp_coded-tool-add"

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
