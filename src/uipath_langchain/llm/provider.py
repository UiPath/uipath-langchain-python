"""Provider detection utilities for LLM models."""

from langchain_core.language_models import BaseChatModel

from uipath_langchain.chat.types import APIFlavor, LLMProvider


def get_llm_provider(model: BaseChatModel) -> LLMProvider:
    """Get the LLM provider from a model's llm_provider attribute."""
    if hasattr(model, "llm_provider") and isinstance(model.llm_provider, LLMProvider):
        return model.llm_provider

    raise ValueError(f"Can't determine llm_provider for model={model}")


def get_api_flavor(model: BaseChatModel) -> APIFlavor:
    """Get the API flavor from a model's api_flavor attribute."""
    if hasattr(model, "api_flavor") and isinstance(model.api_flavor, APIFlavor):
        return model.api_flavor

    raise ValueError(f"Can't determine api_flavor for model={model}")
