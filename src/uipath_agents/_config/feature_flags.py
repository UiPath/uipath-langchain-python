"""Feature flags client for UiPath Agents."""

from functools import lru_cache
from typing import Any

from uipath.core.feature_flags import FeatureFlags
from uipath.platform import UiPath

from .._services.flags_service import FlagsService


@lru_cache(maxsize=32)
def _fetch_flags(names: tuple[str, ...]) -> dict[str, Any]:
    """Fetch flags from platform and push them into the core registry."""
    uipath = UiPath()
    flags_service = FlagsService(
        config=uipath._config,
        execution_context=uipath._execution_context,
    )
    response = flags_service.get_feature_flags(list(names))
    FeatureFlags.configure_flags(response.flags)
    return response.flags


def get_flags(names: list[str]) -> dict[str, Any]:
    """Get feature flags."""
    return _fetch_flags(tuple(sorted(names)))
