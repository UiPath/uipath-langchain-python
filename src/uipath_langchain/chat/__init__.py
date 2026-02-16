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
    if name == "UiPathAzureChatOpenAI":
        from .models import UiPathAzureChatOpenAI

        return UiPathAzureChatOpenAI
    if name == "UiPathChat":
        from .models import UiPathChat

        return UiPathChat
    if name == "UiPathChatOpenAI":
        from .openai import UiPathChatOpenAI

        return UiPathChatOpenAI
    if name == "requires_approval":
        from .hitl import requires_approval

        return requires_approval
    if name in ("OpenAIModels", "BedrockModels", "GeminiModels"):
        from . import supported_models

        return getattr(supported_models, name)
    if name in ("LLMProvider", "APIFlavor", "UiPathPassthroughChatModel"):
        from . import types

        return getattr(types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UiPathChat",
    "UiPathAzureChatOpenAI",
    "UiPathChatOpenAI",
    "UiPathPassthroughChatModel",
    "OpenAIModels",
    "BedrockModels",
    "GeminiModels",
    "LLMProvider",
    "APIFlavor",
    "requires_approval",
]
