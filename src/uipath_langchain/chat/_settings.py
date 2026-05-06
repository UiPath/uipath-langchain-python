import os

from pydantic import model_validator


class _AgentHubConfigDefaultMixin:
    @model_validator(mode="after")
    def _clear_agenthub_config_default(self):
        if os.getenv("UIPATH_AGENTHUB_CONFIG") is None:
            client_settings = getattr(self, "client_settings", None)
            if client_settings is not None and hasattr(
                client_settings, "agenthub_config"
            ):
                client_settings.agenthub_config = None
        return self
