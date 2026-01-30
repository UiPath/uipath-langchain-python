"""Span filter functions for LLMOps telemetry.

These filters determine which spans should be exported to various destinations
based on their instrumentation scope and attributes.
"""

from opentelemetry.sdk.trace import ReadableSpan


def is_openinference_span(span: ReadableSpan) -> bool:
    """Check if span is from OpenInference instrumentation.

    OpenInference instrumentors use scope names like:
    - openinference.instrumentation.langchain
    - openinference.instrumentation.openai
    """
    scope = span.instrumentation_scope
    return scope is not None and scope.name.startswith("openinference.")


def is_http_instrumentation_span(span: ReadableSpan) -> bool:
    """Check if span is from HTTP client instrumentation (httpx, aiohttp)."""
    scope = span.instrumentation_scope
    if scope is None:
        return False
    return scope.name in (
        "opentelemetry.instrumentation.httpx",
        "opentelemetry.instrumentation.aiohttp_client",
    )


def is_azure_monitor_span(span: ReadableSpan) -> bool:
    """Check if span should be exported to Azure Monitor.

    Includes:
    - OpenInference spans (LangGraph/LangChain telemetry)
    - HTTP client spans (httpx, aiohttp for debugging)
    """
    return is_openinference_span(span) or is_http_instrumentation_span(span)


def is_custom_instrumentation_span(span: ReadableSpan) -> bool:
    """Check if span has uipath.custom_instrumentation=True marker.

    This marker identifies LLMOps spans that were manually created
    by the UiPath instrumentation callback.
    """
    return (span.attributes or {}).get("uipath.custom_instrumentation") is True
