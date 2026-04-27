import os

from uipath_langchain_client.clients.openai.chat_models import (
    UiPathAzureChatOpenAI,
    UiPathChatOpenAI,
)

DEFAULT_MODEL_NAME = "gpt-4.1-mini-2025-04-14"


def _default_factory() -> str:
    return os.getenv("UIPATH_MODEL_NAME", DEFAULT_MODEL_NAME)


for _cls in (UiPathAzureChatOpenAI, UiPathChatOpenAI):
    _cls.model_fields["model_name"].default_factory = _default_factory
    _cls.model_rebuild(force=True)


__all__ = [
    "UiPathAzureChatOpenAI",
    "UiPathChatOpenAI",
]
