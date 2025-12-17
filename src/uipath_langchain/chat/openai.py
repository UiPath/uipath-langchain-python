import logging
import os
from typing import Optional

import httpx
from langchain_openai import AzureChatOpenAI
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.utils import EndpointManager

from .supported_models import OpenAIModels

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
    def __init__(
        self,
        token: Optional[str] = None,
        model_name: str = OpenAIModels.gpt_5_mini_2025_08_07,
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
        self._url: Optional[str] = None
        self._agenthub_config = agenthub_config
        self._byo_connection_id = byo_connection_id
        self._extra_headers = extra_headers or {}

        client_kwargs = get_httpx_client_kwargs()
        verify = client_kwargs.get("verify", True)

        super().__init__(
            azure_endpoint=self._build_base_url(),
            model_name=model_name,
            default_headers=self._build_headers(token),
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
            **kwargs,
        )

    def _build_headers(self, token: str) -> dict[str, str]:
        headers = {
            "X-UiPath-LlmGateway-ApiFlavor": "auto",
            "Authorization": f"Bearer {token}",
        }

        if self._agenthub_config:
            headers["X-UiPath-AgentHub-Config"] = self._agenthub_config
        if self._byo_connection_id:
            headers["X-UiPath-LlmGateway-ByoIsConnectionId"] = self._byo_connection_id
        if job_key := os.getenv("UIPATH_JOB_KEY"):
            headers["X-UiPath-JobKey"] = job_key
        if process_key := os.getenv("UIPATH_PROCESS_KEY"):
            headers["X-UiPath-ProcessKey"] = process_key

        # Allow extra_headers to override defaults
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

    def _build_base_url(self) -> str:
        if not self._url:
            env_uipath_url = os.getenv("UIPATH_URL")

            if env_uipath_url:
                self._url = f"{env_uipath_url.rstrip('/')}/{self.endpoint}"
            else:
                raise ValueError("UIPATH_URL environment variable is required")

        return self._url
