import logging
import os
from collections.abc import Mapping
from typing import Any, Optional, cast

import httpx
import openai
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import AzureChatOpenAI
from pydantic import PrivateAttr
from uipath.platform.chat.llm_trace_context import build_trace_context_headers
from uipath.platform.common import (
    EndpointManager,
    get_httpx_client_kwargs,
    resource_override,
)

from .http_client import build_uipath_headers, resolve_gateway_url
from .license_ref_id import get_license_ref_id
from .supported_models import OpenAIModels
from .types import APIFlavor, LLMProvider

logger = logging.getLogger(__name__)

_OPENAI_TOOL_CALL_EXTRAS_KEY = "__openai_tool_call_extras__"
_STANDARD_TOOL_CALL_FIELDS = frozenset({"id", "type", "function", "index"})


def _get_tool_call_extras(
    tool_call: Mapping[str, Any], fallback_key: str | None = None
) -> tuple[str | None, dict[str, Any]]:
    """Extract provider extensions without retaining replaceable call fields."""
    extras = {
        key: value
        for key, value in tool_call.items()
        if key not in _STANDARD_TOOL_CALL_FIELDS
    }
    if not extras:
        return None, {}
    return tool_call.get("id") or fallback_key, extras


def _store_tool_call_extras(
    message: AIMessage | AIMessageChunk,
    tool_calls: list[Mapping[str, Any]],
) -> None:
    stored_extras = message.additional_kwargs.get(_OPENAI_TOOL_CALL_EXTRAS_KEY)
    extras_by_id = dict(stored_extras) if isinstance(stored_extras, Mapping) else {}
    for index, tool_call in enumerate(tool_calls):
        fallback_key = (
            f"__index_{tool_call['index']}"
            if "index" in tool_call
            else f"__index_{index}"
        )
        key, extras = _get_tool_call_extras(tool_call, fallback_key)
        if key and extras:
            extras_by_id[key] = extras
    if extras_by_id:
        message.additional_kwargs[_OPENAI_TOOL_CALL_EXTRAS_KEY] = extras_by_id


class _OpenAIToolCallExtrasMixin:
    """Keep provider-specific tool-call fields across LangChain conversion."""

    def _create_chat_result(
        self,
        response: dict[str, Any] | openai.BaseModel,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        result = cast(Any, super())._create_chat_result(response, generation_info)

        for choice, generation in zip(
            response_dict.get("choices") or [], result.generations, strict=False
        ):
            raw_tool_calls = choice.get("message", {}).get("tool_calls") or []
            if raw_tool_calls and isinstance(generation.message, AIMessage):
                _store_tool_call_extras(generation.message, raw_tool_calls)

        return cast(ChatResult, result)

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type[BaseMessageChunk],
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        generation = cast(Any, super())._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None or not isinstance(generation.message, AIMessageChunk):
            return cast(ChatGenerationChunk | None, generation)

        choices = chunk.get("choices", []) or chunk.get("chunk", {}).get("choices", [])
        if choices and choices[0].get("delta"):
            raw_tool_calls = choices[0]["delta"].get("tool_calls") or []
            if raw_tool_calls:
                _store_tool_call_extras(generation.message, raw_tool_calls)

        return cast(ChatGenerationChunk, generation)

    def _get_generation_chunk_from_completion(
        self, completion: openai.BaseModel
    ) -> ChatGenerationChunk:
        generation = cast(Any, super())._get_generation_chunk_from_completion(
            completion
        )
        # This final summary chunk follows the actual tool-call deltas. Repeating
        # string-valued extras here would make LangChain concatenate the signature.
        generation.message.additional_kwargs.pop(_OPENAI_TOOL_CALL_EXTRAS_KEY, None)
        return cast(ChatGenerationChunk, generation)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = cast(Any, self)._convert_input(input_).to_messages()
        payload = cast(Any, super())._get_request_payload(input_, stop=stop, **kwargs)
        payload_messages = payload.get("messages")
        if not isinstance(payload_messages, list):
            return cast(dict[str, Any], payload)

        for message, payload_message in zip(messages, payload_messages, strict=False):
            if not isinstance(message, AIMessage) or not isinstance(
                payload_message, dict
            ):
                continue

            extras_by_id = message.additional_kwargs.get(_OPENAI_TOOL_CALL_EXTRAS_KEY)
            tool_calls = payload_message.get("tool_calls")
            if not isinstance(extras_by_id, dict) or not isinstance(tool_calls, list):
                continue

            for index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                extras = extras_by_id.get(tool_call.get("id")) or extras_by_id.get(
                    f"__index_{index}"
                )
                if isinstance(extras, Mapping):
                    tool_call.update(
                        {
                            key: value
                            for key, value in extras.items()
                            if key not in _STANDARD_TOOL_CALL_FIELDS
                        }
                    )

        return cast(dict[str, Any], payload)


def _rewrite_openai_url(
    original_url: str, params: httpx.QueryParams
) -> httpx.URL | None:
    """Rewrite OpenAI URLs to UiPath gateway completions endpoint.

    Handles three URL patterns:
    - responses: false -> .../openai/deployments/.../chat/completions?api-version=...
    - responses: true  -> .../openai/responses?api-version=...
    - responses API base -> .../{model}?api-version=... (no /openai/ path)

    All are rewritten to .../completions
    """
    if "/openai/deployments/" in original_url:
        base_url = original_url.split("/openai/deployments/")[0]
    elif "/openai/responses" in original_url:
        base_url = original_url.split("/openai/responses")[0]
    else:
        # Handle base URL case (no /openai/ path appended yet)
        # Strip query string to get base URL
        base_url = original_url.split("?")[0]

    new_url_str = f"{base_url}/completions"
    if params:
        return httpx.URL(new_url_str, params=params)
    return httpx.URL(new_url_str)


def _inject_license_ref_id(request: httpx.Request) -> None:
    """Inject X-UiPath-License-RefId header if a model_run span is active."""
    license_ref_id = get_license_ref_id()
    if license_ref_id:
        request.headers["X-UiPath-License-RefId"] = license_ref_id


def _inject_trace_context_headers(request: httpx.Request) -> None:
    """Inject trace context headers per-request from the active OTEL span."""
    for key, value in build_trace_context_headers(
        extra_baggage=["source=agents"]
    ).items():
        request.headers[key] = value


class UiPathURLRewriteTransport(httpx.AsyncHTTPTransport):
    def __init__(self, verify: bool = True, **kwargs):
        super().__init__(verify=verify, **kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        new_url = _rewrite_openai_url(str(request.url), request.url.params)
        if new_url:
            request.url = new_url
        _inject_license_ref_id(request)
        _inject_trace_context_headers(request)

        return await super().handle_async_request(request)


class UiPathSyncURLRewriteTransport(httpx.HTTPTransport):
    def __init__(self, verify: bool = True, **kwargs):
        super().__init__(verify=verify, **kwargs)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        new_url = _rewrite_openai_url(str(request.url), request.url.params)
        if new_url:
            request.url = new_url
        _inject_license_ref_id(request)
        _inject_trace_context_headers(request)

        return super().handle_request(request)


class UiPathChatOpenAI(_OpenAIToolCallExtrasMixin, AzureChatOpenAI):
    llm_provider: LLMProvider = LLMProvider.OPENAI
    _api_flavor: APIFlavor = PrivateAttr()

    @property
    def api_flavor(self) -> APIFlavor:
        return self._api_flavor

    @resource_override(
        resource_identifier="byo_connection_id", resource_type="connection"
    )
    def __init__(
        self,
        use_responses_api: bool = True,
        token: Optional[str] = None,
        model_name: str = OpenAIModels.gpt_4_1_mini_2025_04_14,
        api_version: str = "2024-12-01-preview",
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        agenthub_config: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
        byo_connection_id: Optional[str] = None,
        **kwargs,
    ):
        org_id = org_id or os.getenv("UIPATH_ORGANIZATION_ID")
        tenant_id = tenant_id or os.getenv("UIPATH_TENANT_ID")
        token = token or os.getenv("UIPATH_ACCESS_TOKEN")

        if not org_id:
            raise ValueError(
                "UIPATH_ORGANIZATION_ID environment variable or org_id parameter is required"
            )
        if not tenant_id:
            raise ValueError(
                "UIPATH_TENANT_ID environment variable or tenant_id parameter is required"
            )
        if not token:
            raise ValueError(
                "UIPATH_ACCESS_TOKEN environment variable or token parameter is required"
            )

        self._openai_api_version = api_version
        self._vendor = "openai"
        self._model_name = model_name
        self._agenthub_config = agenthub_config
        self._byo_connection_id = byo_connection_id
        self._extra_headers = extra_headers or {}

        url, is_override = self._resolve_url()

        client_kwargs = get_httpx_client_kwargs()
        client_kwargs["timeout"] = 300.0
        verify = client_kwargs.get("verify", True)

        api_flavor = (
            APIFlavor.OPENAI_RESPONSES
            if use_responses_api
            else APIFlavor.OPENAI_COMPLETIONS
        )

        super().__init__(
            azure_endpoint=url,
            model_name=model_name,
            default_headers=self._build_headers(token, inject_routing=is_override),
            http_async_client=httpx.AsyncClient(
                transport=UiPathURLRewriteTransport(verify=verify),
                **client_kwargs,
            ),
            http_client=httpx.Client(
                transport=UiPathSyncURLRewriteTransport(verify=verify),
                **client_kwargs,
            ),
            api_key=token,
            api_version=api_version,
            validate_base_url=False,
            use_responses_api=use_responses_api,
            include_response_headers=True,
            **kwargs,
        )

        self._api_flavor = api_flavor

    def _build_headers(
        self, token: str, *, inject_routing: bool = False
    ) -> dict[str, str]:
        headers: dict[str, str] = {"Authorization": f"Bearer {token}"}
        headers.update(
            build_uipath_headers(
                agenthub_config=self._agenthub_config,
                byo_connection_id=self._byo_connection_id,
                inject_routing=inject_routing,
            )
        )
        headers["X-UiPath-LlmGateway-ApiFlavor"] = "auto"
        headers.update(self._extra_headers)
        return headers

    @property
    def endpoint(self) -> str:
        vendor_endpoint = EndpointManager.get_vendor_endpoint()
        formatted_endpoint = vendor_endpoint.format(
            vendor=self._vendor,
            model=self._model_name,
        )
        base_endpoint = formatted_endpoint.replace("/completions", "")
        return f"{base_endpoint}?api-version={self._openai_api_version}"

    def _resolve_url(self) -> tuple[str, bool]:
        return resolve_gateway_url(self.endpoint)
