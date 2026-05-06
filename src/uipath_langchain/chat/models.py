import os

from uipath_langchain_client.clients.normalized.chat_models import (
    UiPathChat as _UpstreamUiPathChat,
)

from ._settings import _AgentHubConfigDefaultMixin
from .openai import UiPathAzureChatOpenAI, UiPathChatOpenAI

DEFAULT_MODEL_NAME = "gpt-4.1-mini-2025-04-14"


def _default_factory() -> str:
    return os.getenv("UIPATH_MODEL_NAME", DEFAULT_MODEL_NAME)


class UiPathChat(_AgentHubConfigDefaultMixin, _UpstreamUiPathChat):
    pass


UiPathChat.model_fields["model_name"].default_factory = _default_factory
UiPathChat.model_rebuild(force=True)


__all__ = [
    "UiPathChat",
    "UiPathAzureChatOpenAI",
    "UiPathChatOpenAI",
]
