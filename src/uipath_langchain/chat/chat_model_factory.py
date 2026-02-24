from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.factory import get_chat_model as get_chat_model_factory
from uipath_langchain_client.settings import (
    ApiFlavor,
    RoutingMode,
    UiPathBaseSettings,
    VendorType,
    get_default_client_settings,
)

_PREFERRED_FLAVOR_BY_VENDOR: dict[str, ApiFlavor] = {
    VendorType.OPENAI.value: ApiFlavor.RESPONSES,
    VendorType.AWSBEDROCK.value: ApiFlavor.CONVERSE,
}


def get_chat_model(
    model: str,
    *,
    byo_connection_id: str | None = None,
    client_settings: UiPathBaseSettings | None = None,
    routing_mode: RoutingMode | str = RoutingMode.PASSTHROUGH,
    vendor_type: VendorType | str | None = None,
    api_flavor: ApiFlavor | str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int = 5,
) -> UiPathBaseChatModel:
    if api_flavor is None and routing_mode == RoutingMode.PASSTHROUGH:
        client_settings = client_settings or get_default_client_settings()
        api_flavor = _resolve_default_api_flavor(
            client_settings,
            model,
            byo_connection_id=byo_connection_id,
            vendor_type=vendor_type,
        )

    return get_chat_model_factory(
        model,
        byo_connection_id=byo_connection_id,
        client_settings=client_settings,
        routing_mode=routing_mode,
        vendor_type=vendor_type,
        api_flavor=api_flavor,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


def _resolve_default_api_flavor(
    settings: UiPathBaseSettings,
    model_name: str,
    *,
    byo_connection_id: str | None,
    vendor_type: VendorType | str | None,
) -> ApiFlavor | None:
    """Pick a modern default api_flavor when the gateway exposes both flavors.

    ``get_model_info`` reuses the cached discovery list, so calling it here
    does not add an HTTP round-trip on top of the lookup the factory performs
    next. When the discovered entry doesn't pin an ``apiFlavor`` — the
    UiPath-owned case where chat-completions/responses (OpenAI) or
    invoke/converse (Bedrock) are both routable — prefer Responses for OpenAI
    and Converse for Bedrock so the factory instantiates
    ``UiPathChatBedrockConverse``. BYOM entries keep the flavor the factory
    derives from the discovered value.
    """
    try:
        info = settings.get_model_info(
            model_name,
            byo_connection_id=byo_connection_id,
            vendor_type=vendor_type,
        )
    except ValueError:
        return None
    if info.get("apiFlavor") is not None:
        return None
    vendor = str(info.get("vendor") or "").lower()
    return _PREFERRED_FLAVOR_BY_VENDOR.get(vendor)
