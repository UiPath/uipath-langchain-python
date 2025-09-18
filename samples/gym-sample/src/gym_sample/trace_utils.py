"""OpenTelemetry trace collection utilities for agent evaluation."""

import json
from typing import Any, Dict, List

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from openinference.instrumentation.langchain import LangChainInstrumentor


class SpanCollector(SpanExporter):
    """Span exporter that collects actual ReadableSpan objects."""

    def __init__(self) -> None:
        """Initialize the span collector."""
        self.spans: List[ReadableSpan] = []

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        """Export spans by collecting them."""
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush spans."""
        return True

    def get_spans(self) -> List[ReadableSpan]:
        """Get all collected spans."""
        return self.spans.copy()

    def clear_spans(self) -> None:
        """Clear all collected spans."""
        self.spans.clear()


def setup_tracing() -> SpanCollector:
    """Set up OpenTelemetry tracing with LangChain instrumentation.

    Returns:
        SpanCollector: The configured span collector for capturing traces.
    """
    # Create collector
    collector = SpanCollector()

    # Set up OpenTelemetry trace collection
    tracer_provider = TracerProvider()
    span_processor = SimpleSpanProcessor(collector)
    tracer_provider.add_span_processor(span_processor)

    # Set the tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Initialize LangChain instrumentation (this creates the spans!)
    LangChainInstrumentor().instrument()

    return collector


def extract_tool_calls_names(spans: List[ReadableSpan]) -> List[str]:
    """Extract the tool call names from execution spans IN ORDER.

    Args:
        spans: List of ReadableSpan objects from agent execution.

    Returns:
        List of tool names in the order they were called.
    """
    tool_calls_names = []

    for span in spans:
        # Check for tool.name attribute first
        if span.attributes and (tool_name := span.attributes.get('tool.name')):
            tool_calls_names.append(tool_name)

    return tool_calls_names


def extract_tool_calls(spans: List[ReadableSpan]) -> Dict[str, Any]:
    """Extract the tool calls from execution spans with their arguments.

    Args:
        spans: List of ReadableSpan objects from agent execution.

    Returns:
        Dict of tool calls with their arguments.
    """
    tool_calls = {}

    for span in spans:
        if span.attributes and (tool_name := span.attributes.get('tool.name')):
            try:
                input_value = span.attributes.get('input.value', '{}')
                # Ensure input_value is a string before parsing
                if isinstance(input_value, str):
                    arguments = json.loads(input_value.replace("'", '"'))
                else:
                    arguments = {}
                tool_calls[tool_name] = arguments
            except json.JSONDecodeError:
                # Handle case where input.value is not valid JSON
                tool_calls[tool_name] = {}

    return tool_calls


def lcs_score(
    actual_tool_calls: list[str], expected_tool_calls: list[str], strict: bool = False
) -> float:
    """
    The function calculates the longest common subsequence between the actual tool calls
    and the expected tool calls and returns the ratio of the LCS length to the number of
    expected calls.

    Args:
        actual_tool_calls: List of tool names in the actual order
        expected_tool_calls: List of tool names in the expected order
        strict: If True, the function will return 0 if the actual calls do not match the expected calls

    Returns:
        float: Ratio of the LCS length to the number of expected
    """
    if (
        not expected_tool_calls
        and not actual_tool_calls
        or expected_tool_calls == actual_tool_calls
    ):
        return 1.0
    elif (
        not expected_tool_calls
        or not actual_tool_calls
        or strict
        and actual_tool_calls != expected_tool_calls
    ):
        return 0.0

    # Calculate LCS with DP + memory efficient
    m, n = len(actual_tool_calls), len(expected_tool_calls)
    min_length, max_length = min(m, n), max(m, n)
    dp = [[0] * (min_length + 1) for _ in range(2)]

    aux_actual, aux_expected = (
        (actual_tool_calls, expected_tool_calls)
        if m >= n
        else (expected_tool_calls, actual_tool_calls)
    )

    for i in range(1, max_length + 1):
        for j in range(1, min_length + 1):
            if aux_actual[i - 1] == aux_expected[j - 1]:
                dp[1][j] = dp[0][j - 1] + 1
            else:
                dp[1][j] = max(dp[0][j], dp[1][j - 1])
        dp[0] = dp[1]

    lcs_length = dp[-1][-1]
    return lcs_length / n
