import logging
import os
from typing import Any, ClassVar

from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
from uipath.core import UiPathTraceManager

from uipath_agents._observability.utils import setup_otel_env

logger = logging.getLogger(__name__)


class _TelemetryState:
    """Module-level telemetry state for idempotency and cleanup."""

    configured: ClassVar[bool] = False
    instrumentors: ClassVar[list[Any]] = []


def configure_telemetry(trace_manager: UiPathTraceManager | None = None) -> None:
    """Configure telemetry for agents. Idempotent - only runs once.

    Args:
        trace_manager: Optional UiPathTraceManager to add Azure exporter to.
                       If not provided, Azure Monitor exporter will be skipped.
    """
    if _TelemetryState.configured:
        logger.debug("Telemetry already configured, skipping")
        return

    setup_otel_env()

    if trace_manager:
        azure_exporter = _get_azure_exporter()
        if azure_exporter:
            trace_manager.add_span_exporter(azure_exporter)

    _TelemetryState.instrumentors = [
        AsyncioInstrumentor(),
        HTTPXClientInstrumentor(),
        AioHttpClientInstrumentor(),
        SQLite3Instrumentor(),
    ]
    for instrumentor in _TelemetryState.instrumentors:
        instrumentor.instrument()

    _TelemetryState.configured = True
    logger.debug("Telemetry configured successfully")


def _get_azure_exporter() -> AzureMonitorTraceExporter | None:
    """Get Azure Monitor trace exporter if connection string is configured."""
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        logger.debug("Azure Monitor exporter not configured - no connection string")
        return None
    return AzureMonitorTraceExporter(connection_string=connection_string)


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
