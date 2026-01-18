"""Factory for creating message content builders."""

from langchain_core.language_models import BaseChatModel

from uipath_langchain.chat.types import APIFlavor, LLMProvider

from .builders import (
    BedrockConverseBuilder,
    BedrockInvokeBuilder,
    MessageContentBuilder,
    OpenAICompletionsBuilder,
    OpenAIResponsesBuilder,
    VertexGeminiBuilder,
)
from .provider import get_api_flavor, get_llm_provider

_BUILDER_REGISTRY: dict[tuple[LLMProvider, APIFlavor], type[MessageContentBuilder]] = {
    (LLMProvider.BEDROCK, APIFlavor.AWS_BEDROCK_CONVERSE): BedrockConverseBuilder,
    (LLMProvider.BEDROCK, APIFlavor.AWS_BEDROCK_INVOKE): BedrockInvokeBuilder,
    (LLMProvider.OPENAI, APIFlavor.OPENAI_RESPONSES): OpenAIResponsesBuilder,
    (LLMProvider.OPENAI, APIFlavor.OPENAI_COMPLETIONS): OpenAICompletionsBuilder,
    (LLMProvider.VERTEX, APIFlavor.VERTEX_GEMINI_GENERATE_CONTENT): VertexGeminiBuilder,
}


def get_content_builder(model: BaseChatModel) -> MessageContentBuilder:
    """Get the appropriate content builder for a model.

    Args:
        model: The LLM model instance.

    Returns:
        A MessageContentBuilder instance for the model's provider and API flavor.

    Raises:
        ValueError: If no builder is registered for the model's provider/API flavor.
    """
    provider = get_llm_provider(model)
    api_flavor = get_api_flavor(model)

    key = (provider, api_flavor)
    builder_class = _BUILDER_REGISTRY.get(key)

    if builder_class is None:
        raise ValueError(
            f"No content builder registered for provider={provider}, "
            f"api_flavor={api_flavor}"
        )

    return builder_class()
