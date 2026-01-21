"""Provider detection utilities for LLM models."""

from langchain_core.language_models import BaseChatModel

from uipath_langchain.chat.types import (
    APIFlavor,
    LLMProvider,
    UiPathPassthroughChatModel,
)


def get_llm_provider(model: BaseChatModel) -> LLMProvider:
    """Get the LLM provider from a UiPath chat model."""
    if isinstance(model, UiPathPassthroughChatModel):
        return model.llm_provider
    raise ValueError(f"Can't determine llm_provider for model={model}")


def get_api_flavor(model: BaseChatModel) -> APIFlavor:
    """Get the API flavor from a UiPath chat model."""
    if isinstance(model, UiPathPassthroughChatModel):
        return model.api_flavor
    raise ValueError(f"Can't determine api_flavor for model={model}")
