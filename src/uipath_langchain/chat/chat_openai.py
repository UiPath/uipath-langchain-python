import logging
import os
from typing import Optional

import httpx
from langchain_openai import AzureChatOpenAI
from uipath.utils import EndpointManager

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


class ChatOpenAIUiPath(AzureChatOpenAI):
    """
    LangChain chat model for UiPath LLM Gateway with OpenAI models.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        model_name: str = "gpt-4o-mini-2024-07-18",
        api_version: str = "2024-12-01-preview",
        verify_ssl: bool = True,
        gateway_url: Optional[str] = None,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        vendor: str = "openai",
        **kwargs,
    ):
        self.model_name = model_name
        self.openai_api_version = api_version

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

        gateway_url = gateway_url or os.getenv(
            "UIPATH_LLM_GATEWAY_URL", "https://alpha.uipath.com"
        )

        endpoint = EndpointManager.get_vendor_endpoint().format(
            model=model_name, api_version=api_version
        )
        base_url = f"{gateway_url}/{org_id}/{tenant_id}/{endpoint}/openai/deployments/placeholder"

        logger.debug(
            f"Initializing ChatOpenAIUiPath with base_url={base_url}, model={model_name}"
        )

        headers = {
            "X-UiPath-Streaming-Enabled": "false",
            "Authorization": f"Bearer {token}",
        }

        job_key = os.getenv("UIPATH_JOB_KEY")
        process_key = os.getenv("UIPATH_PROCESS_KEY")
        if job_key:
            headers["X-UiPath-JobKey"] = job_key
        if process_key:
            headers["X-UiPath-ProcessKey"] = process_key

        http_async_client = httpx.AsyncClient(
            transport=UiPathURLRewriteTransport(verify=verify_ssl),
            verify=verify_ssl,
        )
        http_client = httpx.Client(
            transport=UiPathSyncURLRewriteTransport(verify=verify_ssl),
            verify=verify_ssl,
        )

        super().__init__(
            azure_endpoint=base_url,
            model_name=model_name,
            default_headers=headers,
            http_async_client=http_async_client,
            http_client=http_client,
            api_key=token,
            api_version=api_version,
            validate_base_url=False,
            **kwargs,
        )

    @property
    def endpoint(self) -> str:
        endpoint = EndpointManager.get_vendor_endpoint()
        logger.debug("Using endpoint: %s", endpoint)
        return endpoint.format(
            model=self.model_name, api_version=self.openai_api_version
        )
