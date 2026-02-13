"""
UiPath LangChain Embeddings module.

NOTE: This module uses lazy imports via __getattr__ to avoid loading heavy
dependencies (langchain_openai, openai SDK) at import time. This significantly
improves CLI startup performance.

Do NOT add eager imports like:
    from .models import UiPathOpenAIEmbeddings  # BAD - loads langchain_openai immediately

Instead, all exports are loaded on-demand when first accessed.
"""


def __getattr__(name: str):
    if name == "UiPathEmbeddings":
        from uipath_langchain_client.clients.normalized import (
            UiPathNormalizedEmbeddings,
        )

        return UiPathNormalizedEmbeddings
    if name == "UiPathAzureOpenAIEmbeddings":
        from uipath_langchain_client.clients.openai.embeddings import (
            UiPathAzureOpenAIEmbeddings,
        )

        return UiPathAzureOpenAIEmbeddings
    if name == "UiPathOpenAIEmbeddings":
        from uipath_langchain_client.clients.openai.embeddings import (
            UiPathOpenAIEmbeddings,
        )

        return UiPathOpenAIEmbeddings
    if name == "UiPathGoogleGenerativeAIEmbeddings":
        from uipath_langchain_client.clients.google.embeddings import (
            UiPathGoogleGenerativeAIEmbeddings,
        )

        return UiPathGoogleGenerativeAIEmbeddings
    if name == "UiPathBedrockEmbeddings":
        from uipath_langchain_client.clients.bedrock.embeddings import (
            UiPathBedrockEmbeddings,
        )

        return UiPathBedrockEmbeddings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UiPathAzureOpenAIEmbeddings",
    "UiPathOpenAIEmbeddings",
    "UiPathGoogleGenerativeAIEmbeddings",
    "UiPathBedrockEmbeddings",
]
