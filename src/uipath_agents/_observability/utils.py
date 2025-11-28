import os
from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "uipath-agents"


def _get_package_version() -> str:
    """Get the package version at runtime."""
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.0.0-dev"


def _build_otel_resource_attributes() -> str:
    """Build OTEL resource attributes string.

    Format: "key1=value1,key2=value2"
    See: https://opentelemetry.io/docs/specs/semconv/resource/
    """
    attributes = {
        "service.version": _get_package_version(),
    }

    return ",".join(f"{k}={v}" for k, v in attributes.items())


def setup_otel_env() -> None:
    os.environ.setdefault("OTEL_SERVICE_NAME", PACKAGE_NAME)
    os.environ.setdefault(
        "OTEL_RESOURCE_ATTRIBUTES",
        _build_otel_resource_attributes(),
    )
