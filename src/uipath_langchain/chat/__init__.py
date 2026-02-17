"""
UiPath LangChain Chat module.

NOTE: This module uses lazy imports via __getattr__ to avoid loading heavy
dependencies (langchain_openai, openai SDK) at import time. This significantly
improves CLI startup performance.

Do NOT add eager imports like:
    from .models import UiPathChat  # BAD - loads langchain_openai immediately

Instead, all exports are loaded on-demand when first accessed.
"""


def __getattr__(name):
    if name == "UiPathBaseLLMClient":
        from uipath_langchain_client.base_client import UiPathBaseLLMClient

        return UiPathBaseLLMClient
    if name == "UiPathChat":
        from uipath_langchain_client.clients.normalized import UiPathNormalizedChatModel

        return UiPathNormalizedChatModel
    if name == "UiPathAzureChatOpenAI":
        from uipath_langchain_client.clients.openai import UiPathAzureChatOpenAI

        return UiPathAzureChatOpenAI
    if name == "UiPathChatOpenAI":
        from uipath_langchain_client.clients.openai import UiPathChatOpenAI

        return UiPathChatOpenAI
    if name == "requires_approval":
        from .hitl import requires_approval

        return requires_approval
    if name in ("OpenAIModels", "BedrockModels", "GeminiModels"):
        from . import supported_models

        return getattr(supported_models, name)
    if name in ("LLMProvider", "APIFlavor"):
        from . import types

        return getattr(types, name)
    if name == "UiPathChatBedrock":
        from uipath_langchain_client.clients.bedrock import UiPathChatBedrock

        return UiPathChatBedrock
    if name == "UiPathChatBedrockConverse":
        from uipath_langchain_client.clients.bedrock import UiPathChatBedrockConverse

        return UiPathChatBedrockConverse
    if name == "UiPathChatGoogleGenerativeAI":
        from uipath_langchain_client.clients.google import UiPathChatGoogleGenerativeAI

        return UiPathChatGoogleGenerativeAI
    if name == "UiPathChatAnthropic":
        from uipath_langchain_client.clients.anthropic import UiPathChatAnthropic

        return UiPathChatAnthropic
    if name == "UiPathChatAnthropicVertex":
        from uipath_langchain_client.clients.vertexai import UiPathChatAnthropicVertex

        return UiPathChatAnthropicVertex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UiPathBaseLLMClient",
    "UiPathChat",
    "UiPathAzureChatOpenAI",
    "UiPathChatOpenAI",
    "UiPathChatBedrock",
    "UiPathChatBedrockConverse",
    "UiPathChatGoogleGenerativeAI",
    "UiPathChatAnthropic",
    "UiPathChatAnthropicVertex",
    "OpenAIModels",
    "BedrockModels",
    "GeminiModels",
    "LLMProvider",
    "APIFlavor",
    "requires_approval",
]
