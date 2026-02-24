from typing import Literal

from uipath_langchain_client.base_client import UiPathBaseEmbeddings
from uipath_langchain_client.factory import (
    get_embedding_model as get_embedding_model_factory,
)


def get_embeddings(
    model: str,
    *,
    client_type: Literal["passthrough", "normalized"] = "passthrough",
) -> UiPathBaseEmbeddings:
    return get_embedding_model_factory(model, client_type=client_type)
