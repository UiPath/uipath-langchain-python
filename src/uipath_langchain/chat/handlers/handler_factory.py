"""Factory for creating model payload handlers."""

from langchain_core.language_models import BaseChatModel

from uipath_langchain.chat.types import (
    APIFlavor,
    LLMProvider,
)

from .base import ModelPayloadHandler
from .bedrock_converse import BedrockConversePayloadHandler
from .bedrock_invoke import BedrockInvokePayloadHandler
from .openai_completions import OpenAICompletionsPayloadHandler
from .openai_responses import OpenAIResponsesPayloadHandler
from .vertex_gemini import VertexGeminiPayloadHandler

_HANDLER_REGISTRY: dict[tuple[LLMProvider, APIFlavor], type[ModelPayloadHandler]] = {
    (LLMProvider.OPENAI, APIFlavor.OPENAI_COMPLETIONS): OpenAICompletionsPayloadHandler,
    (LLMProvider.OPENAI, APIFlavor.OPENAI_RESPONSES): OpenAIResponsesPayloadHandler,
    (
        LLMProvider.BEDROCK,
        APIFlavor.AWS_BEDROCK_CONVERSE,
    ): BedrockConversePayloadHandler,
    (LLMProvider.BEDROCK, APIFlavor.AWS_BEDROCK_INVOKE): BedrockInvokePayloadHandler,
    (
        LLMProvider.VERTEX,
        APIFlavor.VERTEX_GEMINI_GENERATE_CONTENT,
    ): VertexGeminiPayloadHandler,
}


def get_payload_handler(model: BaseChatModel) -> ModelPayloadHandler | None:
    """Get the appropriate payload handler for a model.

    Args:
        model: A UiPath chat model instance with llm_provider and api_flavor.

    Returns:
        A ModelPayloadHandler instance for the model or None if could not be determined.
    """
    try:
        key = (model.api_config.vendor_type, model.api_config.api_flavor)
        handler_class = _HANDLER_REGISTRY[key]
    except (AttributeError, KeyError) as _:
        return None

    return handler_class()
