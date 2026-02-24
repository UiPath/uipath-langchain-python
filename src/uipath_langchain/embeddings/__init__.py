from .embeddings_factory import get_embeddings


def __getattr__(name):
    if name == "UiPathEmbeddings":
        from uipath_langchain_client.clients.normalized.embeddings import (
            UiPathEmbeddings,
        )

        return UiPathEmbeddings
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
    if name == "UiPathBedrockEmbeddings":
        from uipath_langchain_client.clients.bedrock.embeddings import (
            UiPathBedrockEmbeddings,
        )

        return UiPathBedrockEmbeddings
    if name == "UiPathGoogleGenerativeAIEmbeddings":
        from uipath_langchain_client.clients.google.embeddings import (
            UiPathGoogleGenerativeAIEmbeddings,
        )

        return UiPathGoogleGenerativeAIEmbeddings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_embeddings",
    "UiPathEmbeddings",
    "UiPathAzureOpenAIEmbeddings",
    "UiPathOpenAIEmbeddings",
    "UiPathBedrockEmbeddings",
    "UiPathGoogleGenerativeAIEmbeddings",
]
