import os
from typing import Any, Optional

import httpx
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.utils import EndpointManager

from .supported_models import GeminiModels


def _check_genai_dependencies() -> None:
    """Check if required dependencies for UiPathChatVertex are installed."""
    import importlib.util

    missing_packages = []

    if importlib.util.find_spec("langchain_google_genai") is None:
        missing_packages.append("langchain-google-genai")

    if importlib.util.find_spec("google.genai") is None:
        missing_packages.append("google-genai")

    if missing_packages:
        packages_str = ", ".join(missing_packages)
        raise ImportError(
            f"The following packages are required to use UiPathChatVertex: {packages_str}\n"
            "Please install them using one of the following methods:\n\n"
            "  # Using pip:\n"
            f"  pip install uipath-langchain[vertex]\n\n"
            "  # Using uv:\n"
            f"  uv add 'uipath-langchain[vertex]'\n\n"
        )


_check_genai_dependencies()

import google.genai
from google.genai import types as genai_types
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import PrivateAttr


def _rewrite_vertex_url(original_url: str, gateway_url: str) -> httpx.URL | None:
    """Rewrite Google GenAI URLs to UiPath gateway endpoint.

    Handles URL patterns containing generateContent or streamGenerateContent.
    Returns the gateway URL, or None if no rewrite needed.
    """
    if "generateContent" in original_url or "streamGenerateContent" in original_url:
        return httpx.URL(gateway_url)
    return None


class _UrlRewriteTransport(httpx.HTTPTransport):
    """Transport that rewrites URLs to redirect to UiPath gateway."""

    def __init__(self, gateway_url: str, verify: bool = True):
        super().__init__(verify=verify)
        self.gateway_url = gateway_url

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        original_url = str(request.url)
        new_url = _rewrite_vertex_url(original_url, self.gateway_url)
        if new_url:
            # Set streaming header based on original URL before modifying
            is_streaming = "alt=sse" in original_url
            request.headers["X-UiPath-Streaming-Enabled"] = (
                "true" if is_streaming else "false"
            )
            # Update host header to match the new URL
            request.headers["host"] = new_url.host
            request.url = new_url
        return super().handle_request(request)


class _AsyncUrlRewriteTransport(httpx.AsyncHTTPTransport):
    """Async transport that rewrites URLs to redirect to UiPath gateway."""

    def __init__(self, gateway_url: str, verify: bool = True):
        super().__init__(verify=verify)
        self.gateway_url = gateway_url

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        original_url = str(request.url)
        new_url = _rewrite_vertex_url(original_url, self.gateway_url)
        if new_url:
            # Set streaming header based on original URL before modifying
            is_streaming = "alt=sse" in original_url
            request.headers["X-UiPath-Streaming-Enabled"] = (
                "true" if is_streaming else "false"
            )
            # Update host header to match the new URL
            request.headers["host"] = new_url.host
            request.url = new_url
        return await super().handle_async_request(request)


class UiPathChatVertex(ChatGoogleGenerativeAI):
    """UiPath Vertex AI Chat model that routes requests through UiPath's LLM Gateway."""

    _vendor: str = PrivateAttr(default="vertexai")
    _model_name: str = PrivateAttr()
    _uipath_token: str = PrivateAttr()
    _uipath_llmgw_url: Optional[str] = PrivateAttr(default=None)
    _agenthub_config: Optional[str] = PrivateAttr(default=None)
    _byo_connection_id: Optional[str] = PrivateAttr(default=None)

    def __init__(
        self,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        model_name: str = GeminiModels.gemini_2_5_flash,
        temperature: Optional[float] = None,
        agenthub_config: Optional[str] = None,
        byo_connection_id: Optional[str] = None,
        **kwargs: Any,
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

        uipath_url = self._build_base_url(model_name)
        headers = self._build_headers(token, agenthub_config, byo_connection_id)

        client_kwargs = get_httpx_client_kwargs()
        verify = client_kwargs.get("verify", True)

        http_options = genai_types.HttpOptions(
            httpx_client=httpx.Client(
                transport=_UrlRewriteTransport(uipath_url, verify=verify),
                headers=headers,
                **client_kwargs,
            ),
            httpx_async_client=httpx.AsyncClient(
                transport=_AsyncUrlRewriteTransport(uipath_url, verify=verify),
                headers=headers,
                **client_kwargs,
            ),
        )

        if temperature is None and (
            "gemini-3" in model_name or "gemini-2" in model_name
        ):
            temperature = 1.0

        super().__init__(
            model=model_name,
            google_api_key="uipath-gateway",
            temperature=temperature,
            **kwargs,
        )

        custom_client = google.genai.Client(
            api_key="uipath-gateway",
            http_options=http_options,
        )

        object.__setattr__(self, "client", custom_client)

        self._model_name = model_name
        self._uipath_token = token
        self._uipath_llmgw_url = uipath_url
        self._agenthub_config = agenthub_config
        self._byo_connection_id = byo_connection_id

        if self.temperature is not None and not 0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in the range [0.0, 2.0]")

        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")

        additional_headers = self.additional_headers or {}
        self.default_metadata = tuple(additional_headers.items())

    @staticmethod
    def _build_headers(
        token: str,
        agenthub_config: Optional[str] = None,
        byo_connection_id: Optional[str] = None,
    ) -> dict[str, str]:
        """Build HTTP headers for UiPath Gateway requests."""
        headers = {
            "Authorization": f"Bearer {token}",
        }
        if agenthub_config:
            headers["X-UiPath-AgentHub-Config"] = agenthub_config
        if byo_connection_id:
            headers["X-UiPath-LlmGateway-ByoIsConnectionId"] = byo_connection_id
        if job_key := os.getenv("UIPATH_JOB_KEY"):
            headers["X-UiPath-JobKey"] = job_key
        if process_key := os.getenv("UIPATH_PROCESS_KEY"):
            headers["X-UiPath-ProcessKey"] = process_key
        return headers

    @staticmethod
    def _build_base_url(model_name: str) -> str:
        """Build the full URL for the UiPath LLM Gateway."""
        env_uipath_url = os.getenv("UIPATH_URL")

        if not env_uipath_url:
            raise ValueError("UIPATH_URL environment variable is required")

        vendor_endpoint = EndpointManager.get_vendor_endpoint()
        formatted_endpoint = vendor_endpoint.format(
            vendor="vertexai",
            model=model_name,
        )
        return f"{env_uipath_url.rstrip('/')}/{formatted_endpoint}"

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """Streaming fallback - calls _generate and yields single response."""
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        if result.generations:
            message = result.generations[0].message
            chunk = AIMessageChunk(
                content=message.content,
                additional_kwargs=message.additional_kwargs,
                response_metadata=getattr(message, "response_metadata", {}),
                id=message.id,
                tool_calls=getattr(message, "tool_calls", []),
                tool_call_chunks=getattr(message, "tool_call_chunks", []),
            )
            if hasattr(message, "usage_metadata") and message.usage_metadata:
                chunk.usage_metadata = message.usage_metadata

            yield ChatGenerationChunk(message=chunk)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """Async streaming fallback - calls _agenerate and yields single response."""
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        result = await self._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

        if result.generations:
            message = result.generations[0].message
            chunk = AIMessageChunk(
                content=message.content,
                additional_kwargs=message.additional_kwargs,
                response_metadata=getattr(message, "response_metadata", {}),
                id=message.id,
                tool_calls=getattr(message, "tool_calls", []),
                tool_call_chunks=getattr(message, "tool_call_chunks", []),
            )
            if hasattr(message, "usage_metadata") and message.usage_metadata:
                chunk.usage_metadata = message.usage_metadata

            yield ChatGenerationChunk(message=chunk)
