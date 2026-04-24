import os
from typing import Any

from pydantic import model_validator
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
)
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatBedrockConverse as _UpstreamUiPathChatBedrockConverse,
)

DEFAULT_MODEL_NAME = "anthropic.claude-haiku-4-5-20251001-v1:0"


def _default_factory() -> str:
    return os.getenv("UIPATH_MODEL_NAME", DEFAULT_MODEL_NAME)


for _cls in (UiPathChatBedrock, UiPathChatAnthropicBedrock):
    _cls.model_fields["model_name"].default_factory = _default_factory
    _cls.model_rebuild(force=True)


class UiPathChatBedrockConverse(_UpstreamUiPathChatBedrockConverse):
    @model_validator(mode="before")
    @classmethod
    def _inject_default_model(cls, values: Any) -> Any:
        if isinstance(values, dict) and not any(
            k in values for k in ("model", "model_id", "model_name")
        ):
            values = {**values, "model": _default_factory()}
        return values


__all__ = [
    "UiPathChatBedrock",
    "UiPathChatBedrockConverse",
    "UiPathChatAnthropicBedrock",
]
