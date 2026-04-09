from uipath.llm_client.settings import UiPathBaseSettings
from uipath_langchain_client.base_client import UiPathBaseEmbeddings
from uipath_langchain_client.factory import (
    get_embedding_model as get_embedding_model_factory,
)
from uipath_langchain_client.settings import ApiFlavor, RoutingMode, VendorType


def get_embeddings(
    model: str,
    *,
    byo_connection_id: str | None = None,
    client_settings: UiPathBaseSettings | None = None,
    routing_mode: RoutingMode | str = RoutingMode.PASSTHROUGH,
    vendor_type: VendorType | str | None = None,
    api_flavor: ApiFlavor | str | None = None,
    timeout: float | None = None,
    max_retries: int = 5,
) -> UiPathBaseEmbeddings:
    return get_embedding_model_factory(
        model,
        byo_connection_id=byo_connection_id,
        client_settings=client_settings,
        routing_mode=routing_mode,
        vendor_type=vendor_type,
        api_flavor=api_flavor,
        timeout=timeout,
        max_retries=max_retries,
    )
