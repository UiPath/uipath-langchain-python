import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class UiPathConfig:
    """Configuration from uipath.json"""

    def __init__(self, config_path: str = "uipath.json"):
        self.config_path = config_path
        self._config: Optional[Dict[str, Any]] = None

    @property
    def exists(self) -> bool:
        """Check if uipath.json exists"""
        return os.path.exists(self.config_path)

    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        if not self.exists:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            self._config = config
            return config
        except Exception as e:
            logger.error(f"Failed to load uipath.json: {str(e)}")
            raise

    @property
    def is_conversational(self) -> bool:
        """Check if the agent is conversational"""
        if not self._config:
            self.load_config()

        # Check isConversational at root level (testing purposes only)
        return self._config.get("isConversational", False) if self._config else False
