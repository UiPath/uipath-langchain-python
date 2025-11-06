import logging
import os
from typing import Optional

from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry import trace

logger = logging.getLogger(__name__)


def get_azure_exporter() -> Optional[AzureMonitorTraceExporter]:
    """Get Azure Monitor trace exporter if connection string is configured.

    Returns:
        AzureMonitorTraceExporter if APPLICATIONINSIGHTS_CONNECTION_STRING is set, None otherwise.
    """
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

    if not connection_string:
        logger.debug(
            "No Azure Application Insights connection string provided, "
            "Azure Monitor exporter will not be configured"
        )
        return None

    logger.info("Configuring Azure Monitor trace exporter")
    return AzureMonitorTraceExporter(connection_string=connection_string)


def shutdown_telemetry() -> None:
    """Shutdown telemetry and flush pending spans.

    Flushes any pending spans to exporters (including Azure Monitor if configured).
    Waits up to 5 seconds for spans to be exported before forcing shutdown.
    """
    try:
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            logger.debug("Flushing pending telemetry spans...")
            tracer_provider.force_flush(timeout_millis=5000)
            logger.debug("Telemetry spans flushed successfully")
    except Exception as e:
        logger.error(f"Error flushing telemetry: {e}")
