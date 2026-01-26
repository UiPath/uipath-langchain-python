import logging
import os
import socket
from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "uipath-agents"

logger = logging.getLogger(__name__)


def _get_package_version() -> str:
    """Get the package version at runtime."""
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.0.0-dev"


def _build_otel_resource_attributes(service_name: str) -> str:
    """Build OTEL resource attributes string.

    Format: "key1=value1,key2=value2"
    See: https://opentelemetry.io/docs/specs/semconv/resource/
    """
    attributes = {
        "service.name": service_name,
        "service.version": _get_package_version(),
    }

    return ",".join(f"{k}={v}" for k, v in attributes.items())


def setup_otel_env() -> None:
    """Setup OpenTelemetry environment variables for service identification.

    Sets OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES which are used by
    TracerProvider to configure the service.name resource attribute.
    This maps to cloud_RoleName in Azure Application Insights.

    The service name is read from CLOUD_ROLE_NAME environment variable,
    defaulting to PACKAGE_NAME if not set.
    """
    service_name = os.getenv("CLOUD_ROLE_NAME", PACKAGE_NAME)

    # Always set OTEL_SERVICE_NAME to ensure cloud_RoleName is correct
    os.environ["OTEL_SERVICE_NAME"] = service_name

    # Build resource attributes with service name explicitly included
    resource_attrs = _build_otel_resource_attributes(service_name)

    # Merge with existing OTEL_RESOURCE_ATTRIBUTES if present
    existing_attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    if existing_attrs:
        # Merge attributes, with our service.name taking precedence
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"{resource_attrs},{existing_attrs}"
    else:
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = resource_attrs


def configure_appinsights_cloud_role() -> None:
    """Configure cloud role name and instance for Application Insights custom events.

    This configures the Application Insights SDK's TelemetryClient (used for custom events)
    with cloud role name and instance. Must be called after the telemetry client is initialized.
    """
    try:
        # Access the internal Application Insights client from uipath.telemetry
        from uipath.telemetry._track import _AppInsightsEventClient

        # Initialize the client if not already done
        _AppInsightsEventClient._initialize()

        # Get the client instance
        client = _AppInsightsEventClient._client
        if not client:
            return

        # Configure cloud role name and instance
        service_name = os.getenv("CLOUD_ROLE_NAME", PACKAGE_NAME)
        role_instance = socket.gethostname()

        client.context.cloud.role = service_name
        client.context.cloud.role_instance = role_instance

    except Exception as e:
        # Silently fail - telemetry configuration should never break the application
        logger.warning(
            f"Failed to configure AppInsights cloud role: {type(e).__name__}: {e}"
        )
