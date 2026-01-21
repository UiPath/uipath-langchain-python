"""Factory for creating model payload handlers."""

from langchain_core.language_models import BaseChatModel

from uipath_langchain.chat.types import (
    APIFlavor,
    LLMProvider,
    UiPathPassthroughChatModel,
)

from .handlers import (
    BedrockConversePayloadHandler,
    BedrockInvokePayloadHandler,
    ModelPayloadHandler,
    OpenAICompletionsPayloadHandler,
    OpenAIResponsesPayloadHandler,
    VertexGeminiPayloadHandler,
)

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


def get_payload_handler(model: BaseChatModel) -> ModelPayloadHandler:
    """Get the appropriate payload handler for a model.

    Args:
         model: A UiPath chat model instance with llm_provider and api_flavor.

    Returns:
        A ModelPayloadHandler instance for the model.

    Raises:
        TypeError: If the model doesn't implement UiPathPassthroughChatModel.
        ValueError: If no handler is registered for the model's provider/API flavor.
    """
    if not isinstance(model, UiPathPassthroughChatModel):
        raise TypeError(
            f"Model {type(model).__name__} does not implement UiPathPassthroughChatModel"
        )
    key = (model.llm_provider, model.api_flavor)
    handler_class = _HANDLER_REGISTRY.get(key)

    if handler_class is None:
        raise ValueError(
            f"No payload handler registered for provider={model.llm_provider}, "
            f"api_flavor={model.api_flavor}"
        )

    return handler_class()
