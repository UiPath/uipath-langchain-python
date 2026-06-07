"""
UiPath LangChain Chat module.

NOTE: This module uses lazy imports via __getattr__ to avoid loading heavy
dependencies (langchain_openai, openai SDK) at import time. This significantly
improves CLI startup performance.

Do NOT add eager imports like:
    from .models import UiPathChat  # BAD - loads langchain_openai immediately

Instead, all exports are loaded on-demand when first accessed.
"""

from .hitl import (
    request_approval,
    request_conversational_tool_confirmation,
    requires_approval,
)


def __getattr__(name):
    if name == "get_chat_model":
        from .chat_model_factory import get_chat_model

        return get_chat_model
    if name == "UiPathChat":
        from .models import UiPathChat

        return UiPathChat
    if name == "UiPathAzureChatOpenAI":
        from .openai import UiPathAzureChatOpenAI

        return UiPathAzureChatOpenAI
    if name == "UiPathChatOpenAI":
        from .openai import UiPathChatOpenAI

        return UiPathChatOpenAI
    if name == "UiPathChatGoogleGenerativeAI":
        from .vertex import UiPathChatGoogleGenerativeAI

        return UiPathChatGoogleGenerativeAI
    if name == "UiPathChatVertex":
        from .vertex import UiPathChatVertex

        return UiPathChatVertex
    if name == "UiPathChatBedrock":
        from .bedrock import UiPathChatBedrock

        return UiPathChatBedrock
    if name == "UiPathChatBedrockConverse":
        from .bedrock import UiPathChatBedrockConverse

        return UiPathChatBedrockConverse
    if name == "UiPathChatAnthropicBedrock":
        from .bedrock import UiPathChatAnthropicBedrock

        return UiPathChatAnthropicBedrock
    if name == "UiPathChatAnthropic":
        from uipath_langchain_client.clients.anthropic.chat_models import (
            UiPathChatAnthropic,
        )

        return UiPathChatAnthropic
    if name == "UiPathChatAnthropicVertex":
        from uipath_langchain_client.clients.vertexai.chat_models import (
            UiPathChatAnthropicVertex,
        )

        return UiPathChatAnthropicVertex
    if name == "UiPathChatFireworks":
        from uipath_langchain_client.clients.fireworks.chat_models import (
            UiPathChatFireworks,
        )

        return UiPathChatFireworks
    if name in ("OpenAIModels", "BedrockModels", "GeminiModels"):
        from uipath_langchain.chat._legacy import supported_models

        return getattr(supported_models, name)
    if name in ("LLMProvider", "APIFlavor"):
        from uipath_langchain.chat._legacy import types

        return getattr(types, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_chat_model",
    "UiPathChat",
    "UiPathAzureChatOpenAI",
    "UiPathChatOpenAI",
    "UiPathChatGoogleGenerativeAI",
    "UiPathChatBedrock",
    "UiPathChatBedrockConverse",
    "UiPathChatAnthropicBedrock",
    "UiPathChatAnthropic",
    "UiPathChatAnthropicVertex",
    "UiPathChatFireworks",
    "UiPathChatVertex",
    "OpenAIModels",
    "BedrockModels",
    "GeminiModels",
    "LLMProvider",
    "APIFlavor",
    "request_approval",
    "request_conversational_tool_confirmation",
    "requires_approval",
]
