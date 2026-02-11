"""Tests for tool span schema naming patterns."""

from unittest.mock import MagicMock

from uipath_agents._observability.llmops.spans.spans_schema.tool import ToolSpanSchema


class TestToolSpanSchemaProcessTool:
    def test_process_tool_span_name_uses_process_name(self) -> None:
        """Process tool span name should use the process display name."""
        tracer = MagicMock()
        mock_span = MagicMock()
        tracer.start_span.return_value = mock_span
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        schema = ToolSpanSchema(tracer=tracer)

        schema.start_process_tool(
            process_name="RPA Workflow",
            arguments={"log": "test"},
        )

        tracer.start_span.assert_called()
        call_args = tracer.start_span.call_args
        span_name = call_args[0][0]
        assert span_name == "RPA Workflow"
