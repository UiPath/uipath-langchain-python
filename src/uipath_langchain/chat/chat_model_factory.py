"""Chat model factory with legacy/new implementation switching.

The ``EnabledNewLlmClients`` feature flag is sourced from ``uipath_agents`` and
passed through as the ``use_new_llm_clients`` argument of :func:`get_chat_model`.

- ``use_new_llm_clients=True`` (default): routes to the new
  ``uipath_langchain_client`` factory.
- ``use_new_llm_clients=False``: routes to the legacy in-repo clients under
  :mod:`uipath_langchain.chat._legacy`, preserving behavior exactly as it was
  before the ``uipath_langchain_client`` migration.
"""

from typing import Any, Final

from langchain_core.callbacks import BaseCallbackHandler, Callbacks
from langchain_core.language_models import BaseChatModel
from uipath.llm_client.utils.headers import (
    get_dynamic_request_headers,
    set_dynamic_request_headers,
)
from uipath.platform.chat.llm_trace_context import build_trace_context_headers
from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.factory import get_chat_model as get_chat_model_factory
from uipath_langchain_client.settings import (
    ApiFlavor,
    RoutingMode,
    UiPathBaseSettings,
    VendorType,
)


class _TraceContextHeadersCallback(BaseCallbackHandler):
    """Inject W3C-style trace context headers into each LLM gateway request.

    Merges into the existing dynamic-headers ContextVar so that headers
    set by earlier callbacks (e.g. ``LicenseRefIdHeadersCallback``) are
    preserved instead of overwritten.
    """

    run_inline: bool = True

    def __init__(self, trace_context_extra_baggage: list[str] | None = None) -> None:
        super().__init__()
        self._trace_context_extra_baggage = trace_context_extra_baggage

    def _merge_headers(self) -> None:
        existing = get_dynamic_request_headers()
        existing.update(
            build_trace_context_headers(extra_baggage=self._trace_context_extra_baggage)
        )
        set_dynamic_request_headers(existing)

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any
    ) -> None:
        self._merge_headers()

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        self._merge_headers()


_UNSET: Final[Any] = object()
DEFAULT_TIMEOUT_SECONDS: Final[float] = 895.0
DEFAULT_MAX_TOKENS: Final[int] = 1000
DEFAULT_TEMPERATURE: Final[float] = 0.0
DEFAULT_MAX_RETRIES: Final[int] = 3


def get_chat_model(
    model: str,
    *,
    byo_connection_id: str | None = None,
    client_settings: UiPathBaseSettings | None = None,
    routing_mode: RoutingMode | str = RoutingMode.PASSTHROUGH,
    vendor_type: VendorType | str | None = None,
    api_flavor: ApiFlavor | str | None = None,
    custom_class: type[UiPathBaseChatModel] | None = None,
    temperature: float | None = DEFAULT_TEMPERATURE,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    timeout: float | None = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int | None = DEFAULT_MAX_RETRIES,
    callbacks: Callbacks = _UNSET,
    agenthub_config: str | None = None,
    use_new_llm_clients: bool = True,
    trace_context_extra_baggage: list[str] | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create and configure a chat model, dispatching legacy vs new clients.

    Args:
        model: The model name (e.g., ``"gpt-4o"``, ``"claude-3-sonnet"``).
        byo_connection_id: Optional Integration Service connection ID.
        client_settings: Overrides the default ``uipath_langchain_client`` settings.
        routing_mode: ``PASSTHROUGH`` (vendor-specific) or ``NORMALIZED``.
        vendor_type: Filter models by vendor; auto-detected when omitted.
        api_flavor: Vendor-specific API flavor (e.g. OpenAI Responses, Bedrock
            Converse). Auto-detected when omitted.
        custom_class: Custom ``UiPathBaseChatModel`` subclass to instantiate
            instead of the auto-detected one.
        temperature: Sampling temperature. Defaults to 0.0. Pass ``None`` to
            omit the parameter when the underlying client supports it.
        max_tokens: Maximum output tokens. Defaults to 1000 to match the
            historical default from ``UiPathRequestMixin``. Pass ``None`` to
            forward an explicit unset value (lets the underlying client apply
            its own default or use no limit).
        timeout: Request timeout in seconds. Defaults to 895 seconds.
        max_retries: Max retry count. Defaults to 3.
        callbacks: LangChain callbacks (handlers or a manager) attached to the
            returned chat model. Accepts ``list[BaseCallbackHandler]`` or a
            ``BaseCallbackManager``. Forwarded only when explicitly set.
            Ignored by the legacy factory.
        agenthub_config: AgentHub config header value. Required by the legacy
            factory; forwarded to the new factory.
        use_new_llm_clients: Routes to the new ``uipath_langchain_client``
            factory when True (default). When False, routes to the legacy
            in-repo clients.
        trace_context_extra_baggage: Optional caller-owned trace baggage
            additions, for example ``source=<product>``. This shared package does
            not default product source attribution.
        **kwargs: Forwarded to the underlying factory. The legacy factory
            accepts ``disable_streaming``; the new factory forwards extras as
            model kwargs to the LangChain constructor.

    Returns:
        A configured ``BaseChatModel`` instance.
    """
    # Always inject trace context headers per-request via a dynamic-headers
    # callback.  For the new path the UiPathHttpxClient reads the ContextVar
    # set by the callback; for the legacy path the callback is a no-op but
    # keeps the wiring consistent.
    callbacks = _ensure_trace_context_callback(
        callbacks,
        trace_context_extra_baggage=trace_context_extra_baggage,
    )

    if not use_new_llm_clients:
        return _legacy_chat_model(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            agenthub_config=agenthub_config,
            byo_connection_id=byo_connection_id,
            trace_context_extra_baggage=trace_context_extra_baggage,
            **kwargs,
        )

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
        agenthub_config=agenthub_config,
        **optional_kwargs,
        **kwargs,
    )


def _ensure_trace_context_callback(
    callbacks: Callbacks,
    *,
    trace_context_extra_baggage: list[str] | None = None,
) -> list[BaseCallbackHandler]:
    """Append or update a ``_TraceContextHeadersCallback`` for trace headers."""
    if callbacks is _UNSET or callbacks is None:
        cb_list: list[BaseCallbackHandler] = []
    else:
        cb_list = list(callbacks)  # type: ignore[arg-type]

    existing_callback = next(
        (cb for cb in cb_list if isinstance(cb, _TraceContextHeadersCallback)),
        None,
    )
    if existing_callback is not None:
        if trace_context_extra_baggage is not None:
            existing_callback._trace_context_extra_baggage = trace_context_extra_baggage
        return cb_list

    cb_list.append(
        _TraceContextHeadersCallback(
            trace_context_extra_baggage=trace_context_extra_baggage
        )
    )
    return cb_list


def _legacy_chat_model(
    model: str,
    *,
    temperature: float | None,
    max_tokens: int | None,
    agenthub_config: str | None,
    byo_connection_id: str | None,
    trace_context_extra_baggage: list[str] | None,
    **kwargs: Any,
) -> BaseChatModel:
    if agenthub_config is None:
        raise ValueError("agenthub_config is required when use_new_llm_clients=False")

    from uipath_langchain.chat._legacy.chat_model_factory import (
        get_chat_model as _legacy_get_chat_model,
    )

    return _legacy_get_chat_model(
        model,
        temperature,
        max_tokens,
        agenthub_config,
        byo_connection_id,
        trace_context_extra_baggage=trace_context_extra_baggage,
        **kwargs,
    )
