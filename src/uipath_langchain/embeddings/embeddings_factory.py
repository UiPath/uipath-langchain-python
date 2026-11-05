"""Embeddings factory backed by ``uipath_langchain_client``."""

from typing import Any

from langchain_core.embeddings import Embeddings
from uipath_langchain_client.base_client import UiPathBaseEmbeddings
from uipath_langchain_client.factory import (
    get_embedding_model as get_embedding_model_factory,
)
from uipath_langchain_client.settings import RoutingMode


def get_embeddings(
    model: str,
    *,
    byo_connection_id: str | None = None,
    routing_mode: RoutingMode | str = RoutingMode.PASSTHROUGH,
    **kwargs: Any,
) -> Embeddings:
    """Create and configure an embeddings client.

    Args:
        model: Embeddings model name (e.g., ``"text-embedding-3-large"``).
        byo_connection_id: Optional Integration Service connection ID.
        routing_mode: ``PASSTHROUGH`` (vendor-specific) or ``NORMALIZED``.
        **kwargs: Forwarded to the underlying factory.

    Returns:
        An embeddings client.
    """
    result: UiPathBaseEmbeddings = get_embedding_model_factory(
        model,
        byo_connection_id=byo_connection_id,
        routing_mode=routing_mode,
        **kwargs,
    )
    return result
