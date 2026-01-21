import logging
import os
from typing import Any, Callable, ClassVar, Optional, Sequence

from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from uipath.core import UiPathTraceManager
from uipath.platform.common import UiPathConfig

from uipath_agents._observability.utils import setup_otel_env

logger = logging.getLogger(__name__)


class FilteringSpanExporter(SpanExporter):
    """Wraps a SpanExporter to filter spans before export."""

    def __init__(
        self,
        delegate: SpanExporter,
        filter_fn: Callable[[ReadableSpan], bool],
    ):
        """Initialize the filtering exporter.

        Args:
            delegate: The underlying exporter to send filtered spans to.
            filter_fn: Function that returns True for spans to export.
        """
        self._delegate = delegate
        self._filter_fn = filter_fn

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export only spans that pass the filter."""
        filtered = [s for s in spans if self._filter_fn(s)]
        if not filtered:
            return SpanExportResult.SUCCESS
        return self._delegate.export(filtered)

    def upsert_span(
        self,
        span: ReadableSpan,
        status_override: Optional[int] = None,
    ) -> SpanExportResult:
        """Upsert a single span, applying filter first."""
        if not self._filter_fn(span):
            return SpanExportResult.SUCCESS
        return self._delegate.upsert_span(span, status_override)  # type: ignore[attr-defined]

    def shutdown(self) -> None:
        """Shutdown the delegate exporter."""
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the delegate exporter."""
        return self._delegate.force_flush(timeout_millis)


def is_openinference_span(span: ReadableSpan) -> bool:
    """Check if span is from OpenInference instrumentation.

    OpenInference instrumentors use scope names like:
    - openinference.instrumentation.langchain
    - openinference.instrumentation.openai
    """
    scope = span.instrumentation_scope
    return scope is not None and scope.name.startswith("openinference.")


def is_custom_instrumentation_span(span: ReadableSpan) -> bool:
    """Check if span has uipath.custom_instrumentation=True marker."""
    return (span.attributes or {}).get("uipath.custom_instrumentation") is True


def patch_trace_manager_with_filter(trace_manager: UiPathTraceManager) -> None:
    """Intercept add_span_exporter to wrap exporters with whitelist filter.

    Needed because cli_run.py (uipath-python) adds exporters we can't modify.
    """
    original = trace_manager.add_span_exporter

    def filtered_add(exporter: SpanExporter) -> None:
        wrapped = FilteringSpanExporter(
            exporter, filter_fn=is_custom_instrumentation_span
        )
        original(wrapped)

    trace_manager.add_span_exporter = filtered_add  # type: ignore[assignment]


class _TelemetryState:
    """Module-level telemetry state for idempotency and cleanup."""

    configured: ClassVar[bool] = False
    instrumentors: ClassVar[list[Any]] = []


def configure_telemetry(trace_manager: UiPathTraceManager | None = None) -> None:
    """Configure telemetry for agents. Idempotent - only runs once.

    Sets up exporters based on environment:
    - OpenInference spans → Azure Monitor (for debugging, if configured)
    - Manual spans → LLMOps HTTP (production)
    - Manual spans → LLMOps file (local only, if LLMOPS_TRACE_FILE is set)

    Applies PII redaction to all spans before export to protect sensitive data.

    Args:
        trace_manager: Optional UiPathTraceManager to add exporters to.
                       If not provided, exporters will be skipped.
    """
    if _TelemetryState.configured:
        logger.debug("Telemetry already configured, skipping")
        return

    setup_otel_env()

    if trace_manager:
        patch_trace_manager_with_filter(trace_manager)

        # Azure Monitor exporter (OpenInference spans only)
        azure_exporter = _get_azure_exporter()
        if azure_exporter:
            # Wrap Azure exporter to:
            # 1. Filter to only OpenInference spans (LangGraph telemetry)
            # 2. Redact PII from attributes before export
            disable_otel_masking = os.getenv("DISABLE_OTEL_MASKING", "false").lower()

            if disable_otel_masking == "true":
                # No PII redaction - just filter spans
                filtered_exporter = FilteringSpanExporter(
                    azure_exporter, filter_fn=is_openinference_span
                )
                logger.warning(
                    "PII redaction is DISABLED - sensitive data may be exported"
                )
            else:
                from uipath_agents._observability.pii_filtering_exporter import (
                    PIIFilteringExporter,
                )

                # Wrap: Azure <- PII Filter <- Span Filter <- Trace Manager
                pii_filtered_exporter = PIIFilteringExporter(azure_exporter)
                filtered_exporter = FilteringSpanExporter(
                    pii_filtered_exporter, filter_fn=is_openinference_span
                )
                logger.debug("Added PII redaction to Azure Monitor exporter")

            trace_manager.add_span_exporter(filtered_exporter)

        # LlmOps file exporter (local execution only)
        llmops_file_exporter = _get_llmops_file_exporter()
        if llmops_file_exporter:
            trace_manager.add_span_exporter(llmops_file_exporter)

    _TelemetryState.instrumentors = [
        HTTPXClientInstrumentor(),
        AioHttpClientInstrumentor(),
    ]
    for instrumentor in _TelemetryState.instrumentors:
        instrumentor.instrument()

    _TelemetryState.configured = True
    logger.debug("Telemetry configured successfully")


def _get_azure_exporter() -> AzureMonitorTraceExporter | None:
    """Get Azure Monitor trace exporter if connection string is configured."""
    connection_string = os.getenv("TELEMETRY_CONNECTION_STRING")
    if not connection_string:
        logger.debug("Azure Monitor exporter not configured - no connection string")
        return None
    return AzureMonitorTraceExporter(connection_string=connection_string)


def _get_llmops_file_exporter() -> SpanExporter | None:
    """Get LlmOps file exporter if configured for local execution.

    Only returns an exporter when:
    - LLMOPS_TRACE_FILE environment variable is set
    - Running locally (no job_key = not in production)

    Returns:
        LlmOpsFileExporter if conditions are met, None otherwise
    """
    llmops_trace_file = os.getenv("LLMOPS_TRACE_FILE")
    if not llmops_trace_file:
        return None

    # Only allow file export for local execution (no job_key = not in production)
    if UiPathConfig.job_key:
        logger.debug(
            "LLMOPS_TRACE_FILE is set but ignored in production environment "
            "(job_key detected)"
        )
        return None

    # Import here to avoid circular dependency
    from uipath_agents._observability.llmops_file_exporter import LlmOpsFileExporter

    return LlmOpsFileExporter(file_path=llmops_trace_file)


def shutdown_telemetry() -> None:
    """Cleanup telemetry resources owned by this module.

    Only uninstruments libraries that we instrumented. Does NOT flush or shutdown
    the TracerProvider - that's the trace_manager's responsibility.
    """
    if not _TelemetryState.configured:
        return

    for instrumentor in _TelemetryState.instrumentors:
        try:
            instrumentor.uninstrument()
        except Exception:
            logger.exception("Failed to uninstrument %s", type(instrumentor).__name__)

    _TelemetryState.configured = False
    _TelemetryState.instrumentors = []
    logger.debug("Telemetry shutdown complete")
