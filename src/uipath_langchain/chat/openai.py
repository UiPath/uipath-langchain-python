import logging
import os
from typing import Optional

import httpx
from langchain_openai import AzureChatOpenAI
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.utils import EndpointManager

from .supported_models import OpenAIModels

logger = logging.getLogger(__name__)


class UiPathURLRewriteTransport(httpx.AsyncHTTPTransport):
    def __init__(self, verify: bool = True, **kwargs):
        super().__init__(verify=verify, **kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        original_url = str(request.url)

        if "/openai/deployments/" in original_url:
            base_url = original_url.split("/openai/deployments/")[0]
            query_string = request.url.params
            new_url_str = f"{base_url}/completions"
            if query_string:
                request.url = httpx.URL(new_url_str, params=query_string)
            else:
                request.url = httpx.URL(new_url_str)

        return await super().handle_async_request(request)


class UiPathSyncURLRewriteTransport(httpx.HTTPTransport):
    def __init__(self, verify: bool = True, **kwargs):
        super().__init__(verify=verify, **kwargs)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        original_url = str(request.url)

        if "/openai/deployments/" in original_url:
            base_url = original_url.split("/openai/deployments/")[0]
            query_string = request.url.params
            new_url_str = f"{base_url}/completions"
            if query_string:
                request.url = httpx.URL(new_url_str, params=query_string)
            else:
                request.url = httpx.URL(new_url_str)

        return super().handle_request(request)


class UiPathChatOpenAI(AzureChatOpenAI):
    def __init__(
        self,
        token: Optional[str] = None,
        model_name: str = OpenAIModels.gpt_5_mini_2025_08_07,
        api_version: str = "2024-12-01-preview",
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
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

        super().__init__(
            azure_endpoint=self._build_base_url(),
            model_name=model_name,
            default_headers=self._build_headers(token),
            http_async_client=httpx.AsyncClient(
                transport=UiPathURLRewriteTransport(verify=True),
                **get_httpx_client_kwargs(),
            ),
            http_client=httpx.Client(
                transport=UiPathSyncURLRewriteTransport(verify=True),
                **get_httpx_client_kwargs(),
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
        if job_key := os.getenv("UIPATH_JOB_KEY"):
            headers["X-UiPath-JobKey"] = job_key
        if process_key := os.getenv("UIPATH_PROCESS_KEY"):
            headers["X-UiPath-ProcessKey"] = process_key
        return headers

    @property
    def endpoint(self) -> str:
        vendor_endpoint = EndpointManager.get_vendor_endpoint()
        formatted_endpoint = vendor_endpoint.format(
            vendor=self._vendor,
            model=self._model_name,
            api_version=self._openai_api_version,
        )
        return formatted_endpoint.replace("/completions", "")

    def _build_base_url(self) -> str:
        if not self._url:
            env_uipath_url = os.getenv("UIPATH_URL")

            if env_uipath_url:
                self._url = f"{env_uipath_url.rstrip('/')}/{self.endpoint}"
            else:
                raise ValueError("UIPATH_URL environment variable is required")

        return self._url
