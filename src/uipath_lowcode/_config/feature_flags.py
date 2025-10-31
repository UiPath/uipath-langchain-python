"""Feature flags configuration for UiPath Agents."""

import logging
from typing import Any, Dict, List

from uipath import UiPath

from .._services.flags_service import FlagsService

logger = logging.getLogger(__name__)


class FeatureFlagsConfig:
    """Simple wrapper around feature flags dictionary."""

    def __init__(self, flags: Dict[str, Any]):
        """Initialize with flags dictionary."""
        self._flags = flags

    def get(self, flag_name: str, default: Any = None) -> Any:
        """Get a feature flag value with optional default."""
        return self._flags.get(flag_name, default)

    def to_dict(self) -> Dict[str, Any]:
        """Get all flags as a dictionary."""
        return self._flags.copy()

    def __getitem__(self, flag_name: str) -> Any:
        """Get flag value using dict-like access."""
        return self._flags[flag_name]

    def __contains__(self, flag_name: str) -> bool:
        """Check if flag exists."""
        return flag_name in self._flags


def get_feature_flags(flags: List[str]) -> FeatureFlagsConfig:
    """Retrieve feature flags from UiPath Agent Runtime API."""
    try:
        uipath = UiPath()
        flags_service = FlagsService(
            config=uipath._config,
            execution_context=uipath._execution_context,
        )
        response = flags_service.get_feature_flags(flags)
        return FeatureFlagsConfig(flags=response.flags)
    except Exception as e:
        logger.error(f"Failed to retrieve feature flags: {e}")
        raise
