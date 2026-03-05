import logging
import os
from typing import Optional

import httpx
from langchain_openai import AzureChatOpenAI
from pydantic import PrivateAttr
from uipath.platform.common import (
    EndpointManager,
    get_httpx_client_kwargs,
    resource_override,
)

from .http_client import build_uipath_headers, resolve_gateway_url
from .supported_models import OpenAIModels
from .types import APIFlavor, LLMProvider

logger = logging.getLogger(__name__)


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


class UiPathURLRewriteTransport(httpx.AsyncHTTPTransport):
    def __init__(self, verify: bool = True, **kwargs):
        super().__init__(verify=verify, **kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        new_url = _rewrite_openai_url(str(request.url), request.url.params)
        if new_url:
            request.url = new_url

        return await super().handle_async_request(request)


class UiPathSyncURLRewriteTransport(httpx.HTTPTransport):
    def __init__(self, verify: bool = True, **kwargs):
        super().__init__(verify=verify, **kwargs)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        new_url = _rewrite_openai_url(str(request.url), request.url.params)
        if new_url:
            request.url = new_url

        return super().handle_request(request)


class UiPathChatOpenAI(AzureChatOpenAI):
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
