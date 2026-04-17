"""Embeddings factory with legacy/new implementation switching.

The ``EnabledNewLlmClients`` feature flag controls which embedding client is
instantiated. When False, the pre-commit-3f7da07d
:class:`~uipath_langchain.embeddings._legacy.embeddings.UiPathAzureOpenAIEmbeddings`
is returned so behavior matches the prior implementation. When True, the new
``uipath_langchain_client`` factory is used.
"""

from typing import Any

from langchain_core.embeddings import Embeddings


def get_embeddings(
    model: str,
    *,
    byo_connection_id: str | None = None,
    use_new_llm_clients: bool = False,
    **kwargs: Any,
) -> Embeddings:
    """Create and configure an embeddings client, dispatching legacy vs new.

    Args:
        model: Embeddings model name (e.g., ``"text-embedding-3-large"``).
        byo_connection_id: Optional Integration Service connection ID. Only
            used by the new factory.
        use_new_llm_clients: Routes to the new ``uipath_langchain_client``
            factory when True. Defaults to False (legacy).
        **kwargs: Forwarded to the underlying factory or to the legacy
            embeddings class constructor.

    Returns:
        An embeddings client.
    """
    if use_new_llm_clients:
        from uipath_langchain_client.base_client import UiPathBaseEmbeddings
        from uipath_langchain_client.factory import (
            get_embedding_model as get_embedding_model_factory,
        )
        from uipath_langchain_client.settings import RoutingMode

        routing_mode = kwargs.pop("routing_mode", RoutingMode.PASSTHROUGH)
        result: UiPathBaseEmbeddings = get_embedding_model_factory(
            model,
            byo_connection_id=byo_connection_id,
            routing_mode=routing_mode,
            **kwargs,
        )
        return result

    from uipath_langchain.embeddings._legacy.embeddings import (
        UiPathAzureOpenAIEmbeddings,
    )

    return UiPathAzureOpenAIEmbeddings(model=model, **kwargs)
