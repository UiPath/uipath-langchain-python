"""Chat model factory with legacy/new implementation switching.

The ``EnabledNewLlmClients`` feature flag is sourced from ``uipath_agents`` and
passed through as the ``use_new_llm_clients`` argument of :func:`get_chat_model`.

- ``use_new_llm_clients=False`` (default): routes to the legacy in-repo clients
  under :mod:`uipath_langchain.chat._legacy`, preserving behavior exactly as it
  was before the ``uipath_langchain_client`` migration.
- ``use_new_llm_clients=True``: routes to the new ``uipath_langchain_client``
  factory, preserving behavior exactly as on commit 3f7da07d.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel


def get_chat_model(
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    agenthub_config: str | None = None,
    byo_connection_id: str | None = None,
    *,
    use_new_llm_clients: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Create and configure a chat model, dispatching legacy vs new clients.

    Args:
        model: The model name (e.g., ``"gpt-4o"``, ``"claude-3-sonnet"``).
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        agenthub_config: AgentHub config header value. Required by the legacy
            factory; ignored by the new factory (which resolves config from the
            ``uipath_langchain_client`` settings).
        byo_connection_id: Optional Integration Service connection ID.
        use_new_llm_clients: Routes to the new ``uipath_langchain_client``
            factory when True. Defaults to False (legacy).
        **kwargs: Forwarded to the underlying factory. The legacy factory
            accepts ``disable_streaming``; the new factory accepts
            ``client_settings``, ``routing_mode``, ``vendor_type``,
            ``api_flavor``, ``timeout``, ``max_retries``.

    Returns:
        A configured ``BaseChatModel`` instance.
    """
    if use_new_llm_clients:
        return _get_chat_model_new(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            byo_connection_id=byo_connection_id,
            **kwargs,
        )

    if agenthub_config is None:
        raise ValueError(
            "agenthub_config is required when use_new_llm_clients=False"
        )

    from uipath_langchain.chat._legacy.chat_model_factory import (
        get_chat_model as _legacy_get_chat_model,
    )

    return _legacy_get_chat_model(
        model,
        temperature if temperature is not None else 0.0,
        max_tokens if max_tokens is not None else 0,
        agenthub_config,
        byo_connection_id,
        **kwargs,
    )


def _get_chat_model_new(
    model: str,
    *,
    temperature: float | None,
    max_tokens: int | None,
    byo_connection_id: str | None,
    client_settings: Any = None,
    routing_mode: Any = None,
    vendor_type: Any = None,
    api_flavor: Any = None,
    timeout: float | None = None,
    max_retries: int = 5,
    **_: Any,
) -> BaseChatModel:
    """Dispatch to the new ``uipath_langchain_client`` factory.

    Replicates the ``get_chat_model`` function introduced in commit 3f7da07d,
    including the ``api_flavor`` default resolution when the discovered model
    does not pin a flavor.
    """
    from uipath_langchain_client.factory import (
        get_chat_model as get_chat_model_factory,
    )
    from uipath_langchain_client.settings import (
        ApiFlavor,
        RoutingMode,
        VendorType,
        get_default_client_settings,
    )

    if routing_mode is None:
        routing_mode = RoutingMode.PASSTHROUGH

    if api_flavor is None and routing_mode == RoutingMode.PASSTHROUGH:
        client_settings = client_settings or get_default_client_settings()
        api_flavor = _resolve_default_api_flavor_new(
            client_settings,
            model,
            byo_connection_id=byo_connection_id,
            vendor_type=vendor_type,
            preferred_by_vendor={
                VendorType.OPENAI.value: ApiFlavor.RESPONSES,
                VendorType.AWSBEDROCK.value: ApiFlavor.CONVERSE,
            },
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


def _resolve_default_api_flavor_new(
    settings: Any,
    model_name: str,
    *,
    byo_connection_id: str | None,
    vendor_type: Any,
    preferred_by_vendor: dict[str, Any],
) -> Any:
    """Pick a modern default api_flavor when the gateway exposes both flavors."""
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
    return preferred_by_vendor.get(vendor)
