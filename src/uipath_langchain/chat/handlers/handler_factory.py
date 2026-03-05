"""Factory for creating model payload handlers."""

from langchain_core.language_models import BaseChatModel

from .anthropic import AnthropicPayloadHandler
from .base import DefaultModelPayloadHandler, ModelPayloadHandler
from .bedrock import BedrockPayloadHandler
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

    model_mro = [m.__name__ for m in type(model).mro()]

    if "AzureChatOpenAI" in model_mro or "ChatOpenAI" in model_mro:
        return OpenAIPayloadHandler()
    if "ChatAnthropic" in model_mro:
        return AnthropicPayloadHandler()
    if "ChatGoogleGenerativeAI" in model_mro:
        return GeminiPayloadHandler()
    if "ChatBedrock" in model_mro or "ChatBedrockConverse" in model_mro:
        return BedrockPayloadHandler()
    return DefaultModelPayloadHandler()
