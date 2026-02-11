"""Shared fixtures for observability tests."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from uipath_agents._observability.llmops import (
    LlmOpsInstrumentationCallback,
    LlmOpsSpanFactory,
)

# Global exporter to avoid TracerProvider conflicts
_exporter: InMemorySpanExporter | None = None
_provider_set: bool = False


def _get_exporter() -> InMemorySpanExporter:
    """Get or create the global span exporter."""
    global _exporter, _provider_set

    if _exporter is None:
        _exporter = InMemorySpanExporter()

    if not _provider_set:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_exporter))
        trace.set_tracer_provider(provider)
        _provider_set = True

    return _exporter


@pytest.fixture
def span_exporter():
    """Get the span exporter and clear it before each test."""
    exporter = _get_exporter()
    exporter.clear()
    yield exporter
    exporter.clear()


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
