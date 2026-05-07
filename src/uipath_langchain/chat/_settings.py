import os

from pydantic import field_validator


class _AgentHubConfigDefaultMixin:
    @field_validator("client_settings", mode="after")
    @classmethod
    def _clear_agenthub_config_default(cls, client_settings):
        if (
            client_settings is not None
            and os.getenv("UIPATH_AGENTHUB_CONFIG") is None
            and hasattr(client_settings, "agenthub_config")
        ):
            client_settings.agenthub_config = None
        return client_settings
