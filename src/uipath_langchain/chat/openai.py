"""OpenAI / Azure-backed chat classes.

Ports the legacy ``UiPathRequestMixin`` ``model_name`` default to the upstream
``uipath_langchain_client`` classes so ``UiPathChat()`` (no args) works:
reads ``UIPATH_MODEL_NAME`` env var with a hardcoded fallback. The mutation
runs at module import and is scoped per-class (pydantic stores ``FieldInfo``
per class, so it does not leak to siblings or parents).
"""

import os

from uipath_langchain_client.clients.normalized.chat_models import UiPathChat
from uipath_langchain_client.clients.openai.chat_models import (
    UiPathAzureChatOpenAI,
    UiPathChatOpenAI,
)

DEFAULT_MODEL_NAME = "gpt-4.1-mini-2025-04-14"


def _default_factory() -> str:
    return os.getenv("UIPATH_MODEL_NAME", DEFAULT_MODEL_NAME)


for _cls in (UiPathChat, UiPathAzureChatOpenAI, UiPathChatOpenAI):
    _cls.model_fields["model_name"].default_factory = _default_factory
    _cls.model_rebuild(force=True)


__all__ = [
    "UiPathChat",
    "UiPathAzureChatOpenAI",
    "UiPathChatOpenAI",
]
