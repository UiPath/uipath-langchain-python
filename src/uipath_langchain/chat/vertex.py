import os

from uipath_langchain_client.clients.google.chat_models import (
    UiPathChatGoogleGenerativeAI,
)

DEFAULT_MODEL_NAME = "gemini-2.5-flash"


def _default_factory() -> str:
    return os.getenv("UIPATH_MODEL_NAME", DEFAULT_MODEL_NAME)


UiPathChatGoogleGenerativeAI.model_fields[
    "model_name"
].default_factory = _default_factory
UiPathChatGoogleGenerativeAI.model_rebuild(force=True)

UiPathChatVertex = UiPathChatGoogleGenerativeAI


__all__ = [
    "UiPathChatGoogleGenerativeAI",
    "UiPathChatVertex",
]
