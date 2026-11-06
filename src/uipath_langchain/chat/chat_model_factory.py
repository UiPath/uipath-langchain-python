"""Chat model factory backed by ``uipath_langchain_client``."""

import os
from collections.abc import Mapping
from typing import Any, Final

from langchain_core.callbacks import BaseCallbackHandler, Callbacks
from langchain_core.language_models import BaseChatModel
from uipath.llm_client.utils.exceptions import ModelNotFoundError
from uipath.llm_client.utils.headers import (
    get_dynamic_request_headers,
    set_dynamic_request_headers,
)
from uipath.platform.chat.llm_trace_context import build_trace_context_headers
from uipath.runtime.errors import UiPathErrorCategory
from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.factory import get_chat_model as get_chat_model_factory
from uipath_langchain_client.settings import (
    ApiFlavor,
    RoutingMode,
    UiPathBaseSettings,
    VendorType,
)

from uipath_langchain.agent.exceptions import (
    AgentStartupError,
    AgentStartupErrorCode,
)
from uipath_langchain.chat.attachment_baggage import trace_baggage

USE_CLIENT_LLM_SPANS_ENV = "UIPATH_USE_CLIENT_LLM_SPANS"


def use_client_llm_spans() -> bool:
    """Whether LLM calls are traced by client-side spans instead of the LLM Gateway.

    Set ``UIPATH_USE_CLIENT_LLM_SPANS=true`` to keep the client-side LLM spans
    and leave the LLM Gateway's spans (Model run, governance, guardrails) out
    of the agent's trace. By default the Gateway's spans are used.
    """
    return os.environ.get(USE_CLIENT_LLM_SPANS_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


class _TraceContextHeadersCallback(BaseCallbackHandler):
    """Inject W3C-style trace context headers into each LLM gateway request.

    Merges into the existing dynamic-headers ContextVar so that headers
    set by earlier callbacks (e.g. ``LicenseRefIdHeadersCallback``) are
    preserved instead of overwritten. Sends nothing when
    ``UIPATH_USE_CLIENT_LLM_SPANS`` is true, so the Gateway's spans don't
    join the agent's trace.
    """

    run_inline: bool = True

    def _merge_headers(self) -> None:
        if use_client_llm_spans():
            return
        existing = get_dynamic_request_headers()
        existing.update(
            build_trace_context_headers(extra_baggage=trace_baggage("source=agents"))
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
# 5 total attempts (initial request + up to 4 retries), matching uipath-llm-client defaults.
DEFAULT_MAX_RETRIES: Final[int] = 5


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
    model_settings: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Create and configure a chat model.

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
        max_tokens: Maximum output tokens. Defaults to 1000. Pass ``None`` to
            forward an explicit unset value (lets the underlying client apply
            its own default or use no limit).
        timeout: Request timeout in seconds. Defaults to 895 seconds.
        max_retries: Maximum number of attempts. Defaults to 5.
        callbacks: LangChain callbacks (handlers or a manager) attached to the
            returned chat model. Accepts ``list[BaseCallbackHandler]`` or a
            ``BaseCallbackManager``. Forwarded only when explicitly set.
        agenthub_config: AgentHub config header value.
        **kwargs: Forwarded to the underlying factory as model kwargs to the
            LangChain constructor.

    Returns:
        A configured ``BaseChatModel`` instance.
    """
    # Inject trace context headers per-request via a dynamic-headers callback;
    # the UiPathHttpxClient reads the ContextVar set by the callback.
    callbacks = _ensure_trace_context_callback(callbacks)

    try:
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
            model_settings=model_settings,
            **optional_kwargs,
            **kwargs,
        )
    except ModelNotFoundError as e:
        if byo_connection_id:
            detail = (
                f"The model '{model}' is not available. Check that your custom "
                "Model Configuration is available on this tenant."
            )
        else:
            detail = (
                f"The model '{model}' is not available. Verify the model name in "
                "the agent configuration is correct and that the model is enabled "
                "for this tenant. If the error persists, contact your administrator."
            )

        raise AgentStartupError(
            code=AgentStartupErrorCode.LLM_INVALID_MODEL,
            title="LLM model not available",
            detail=detail,
            category=UiPathErrorCategory.DEPLOYMENT,
        ) from e


def _ensure_trace_context_callback(callbacks: Callbacks) -> list[BaseCallbackHandler]:
    """Append a ``_TraceContextHeadersCallback`` if one is not already present."""
    if callbacks is _UNSET or callbacks is None:
        cb_list: list[BaseCallbackHandler] = []
    else:
        cb_list = list(callbacks)  # type: ignore[arg-type]
    if not any(isinstance(cb, _TraceContextHeadersCallback) for cb in cb_list):
        cb_list.append(_TraceContextHeadersCallback())
    return cb_list
