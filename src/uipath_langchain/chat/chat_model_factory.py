from typing import Any, Final

from langchain_core.callbacks import Callbacks
from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.factory import get_chat_model as get_chat_model_factory
from uipath_langchain_client.settings import (
    ApiFlavor,
    RoutingMode,
    UiPathBaseSettings,
    VendorType,
)

_UNSET: Final[Any] = object()


def get_chat_model(
    model: str,
    *,
    byo_connection_id: str | None = None,
    client_settings: UiPathBaseSettings | None = None,
    routing_mode: RoutingMode | str = RoutingMode.PASSTHROUGH,
    vendor_type: VendorType | str | None = None,
    api_flavor: ApiFlavor | str | None = None,
    custom_class: type[UiPathBaseChatModel] | None = None,
    temperature: float | None = _UNSET,
    max_tokens: int | None = _UNSET,
    timeout: float | None = _UNSET,
    max_retries: int | None = _UNSET,
    callbacks: Callbacks = _UNSET,
    **kwargs: Any,
) -> UiPathBaseChatModel:
    optional_kwargs = {
        k: v
        for k, v in {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
            "callbacks": callbacks,
        }.items()
        if v is not _UNSET
    }
    return get_chat_model_factory(
        model,
        byo_connection_id=byo_connection_id,
        client_settings=client_settings,
        routing_mode=routing_mode,
        vendor_type=vendor_type,
        api_flavor=api_flavor,
        custom_class=custom_class,
        **optional_kwargs,
        **kwargs,
    )
