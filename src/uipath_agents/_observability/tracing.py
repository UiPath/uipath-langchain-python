import logging
import os
from typing import Any, ClassVar

from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace.export import SpanExporter
from uipath.core import UiPathTraceManager
from uipath.platform.common import UiPathConfig

from uipath_agents._observability.exporters import FilteringSpanExporter
from uipath_agents._observability.exporters.environment_attributes_exporter import (
    EnvironmentAttributesExporter,
)
from uipath_agents._observability.llmops import is_azure_monitor_span
from uipath_agents._observability.utils import setup_otel_env

logger = logging.getLogger(__name__)


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
        azure_exporter = _get_azure_exporter()
        if azure_exporter:
            pii_masking_disabled = (
                os.getenv("DISABLE_OTEL_MASKING", "false").lower() == "true"
            )

            env_exporter: Any = EnvironmentAttributesExporter(azure_exporter)

            if pii_masking_disabled:
                logger.warning(
                    "PII redaction is DISABLED - sensitive data may be exported"
                )
                base_exporter: Any = env_exporter
            else:
                from uipath_agents._observability.exporters.pii_filtering_exporter import (
                    PIIFilteringExporter,
                )

                base_exporter = PIIFilteringExporter(env_exporter)
                logger.debug("Added PII redaction to Azure Monitor exporter")

            filtered_exporter = FilteringSpanExporter(
                base_exporter, filter_fn=is_azure_monitor_span
            )
            trace_manager.add_span_exporter(filtered_exporter)

        if llmops_trace_file := _enable_llmops_dev_traces():
            trace_manager.add_span_exporter(
                _get_llmops_file_exporter(llmops_trace_file)
            )

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


def _enable_llmops_dev_traces() -> str | None:
    """Check if LLMOps dev traces should be enabled.

    Returns file path if enabled (local execution only), None otherwise.
    """
    llmops_trace_file = os.getenv("LLMOPS_TRACE_FILE")
    if not llmops_trace_file:
        return None

    if UiPathConfig.job_key:
        return None

    return llmops_trace_file


def _get_llmops_file_exporter(file_path: str) -> SpanExporter:
    """Create LlmOps file exporter for the given file path.

    Args:
        file_path: Path to write traces to

    Returns:
        LlmOpsFileExporter instance
    """
    from uipath_agents._observability.exporters.llmops_file_exporter import (
        LlmOpsFileExporter,
    )

    return LlmOpsFileExporter(file_path=file_path)


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
            logger.exception("Failed to un-instrument %s", type(instrumentor).__name__)

    _TelemetryState.configured = False
    _TelemetryState.instrumentors = []
    logger.debug("Telemetry shutdown complete")
