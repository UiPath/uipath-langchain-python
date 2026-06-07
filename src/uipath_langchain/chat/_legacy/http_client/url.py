"""Gateway URL resolution with service override support.

Integrates chat models into the platform's per-service URL override system
(UIPATH_SERVICE_URL_<SERVICE> env vars) so that local development redirects
work consistently across all UiPath HTTP clients.
"""

import os
from collections.abc import Callable

import uipath.platform.common as _platform_common

_resolve_service_url: Callable[[str], str | None] | None = getattr(
    _platform_common, "resolve_service_url", None
)


def resolve_gateway_url(endpoint_path: str) -> tuple[str, bool]:
    """Resolve the full gateway URL for a given endpoint path.

    Checks for a per-service URL override first (e.g.
    ``UIPATH_SERVICE_URL_AGENTHUB``). Falls back to building the URL
    from ``UIPATH_URL``.

    Args:
        endpoint_path: Endpoint path with service prefix, e.g.
            ``"agenthub_/llm/raw/vendor/openai/model/gpt-4/completions"``.

    Returns:
        Tuple of (resolved_url, is_override). When *is_override* is True
        the caller should inject routing headers since the platform
        routing layer is bypassed.

    Raises:
        ValueError: If neither a service override nor UIPATH_URL is set.
    """
    if _resolve_service_url is not None:
        override_url = _resolve_service_url(endpoint_path)
        if override_url:
            return override_url, True

    env_uipath_url = os.getenv("UIPATH_URL")
    if not env_uipath_url:
        raise ValueError("UIPATH_URL environment variable is required")

    return f"{env_uipath_url.rstrip('/')}/{endpoint_path}", False
