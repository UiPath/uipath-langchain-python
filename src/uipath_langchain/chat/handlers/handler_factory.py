"""Factory for creating model payload handlers."""

from langchain_core.language_models import BaseChatModel

from .anthropic import AnthropicPayloadHandler
from .base import DefaultModelPayloadHandler, ModelPayloadHandler
from .bedrock import BedrockConversePayloadHandler, BedrockInvokePayloadHandler
from .fireworks import FireworksPayloadHandler
from .gemini import GeminiPayloadHandler
from .openai import OpenAIPayloadHandler


def get_payload_handler(model: BaseChatModel) -> ModelPayloadHandler:
    """Get the appropriate payload handler for a model based on its MRO.

    Resolves the handler by checking the model's class hierarchy (MRO) for
    known LLM provider base classes (e.g. AzureChatOpenAI, ChatBedrock).
    Falls back to DefaultModelPayloadHandler if no match is found.

    Args:
        model: A LangChain chat model instance.

    Returns:
        A ModelPayloadHandler instance for the model.
    """

    model_mro = set([m.__name__ for m in type(model).mro()])

    if "BaseChatOpenAI" in model_mro:
        return OpenAIPayloadHandler(model)
    if "ChatAnthropic" in model_mro:
        return AnthropicPayloadHandler(model)
    if "ChatGoogleGenerativeAI" in model_mro:
        return GeminiPayloadHandler(model)
    if "ChatBedrockConverse" in model_mro:
        return BedrockConversePayloadHandler(model)
    if "ChatBedrock" in model_mro:
        return BedrockInvokePayloadHandler(model)
    if "ChatFireworks" in model_mro:
        return FireworksPayloadHandler(model)
    return DefaultModelPayloadHandler(model)
