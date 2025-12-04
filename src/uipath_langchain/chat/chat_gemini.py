import logging
import os
from typing import Optional, Union

import aiohttp
from google.auth.credentials import AnonymousCredentials
from google.cloud.aiplatform_v1.services.prediction_service import (
    PredictionServiceAsyncClient as v1PredictionServiceAsyncClient,
)
from google.cloud.aiplatform_v1.services.prediction_service import (
    PredictionServiceClient as v1PredictionServiceClient,
)
from google.cloud.aiplatform_v1beta1.services.prediction_service import (
    PredictionServiceAsyncClient as v1beta1PredictionServiceAsyncClient,
)
from google.cloud.aiplatform_v1beta1.services.prediction_service import (
    PredictionServiceClient as v1beta1PredictionServiceClient,
)
from google.cloud.aiplatform_v1beta1.services.prediction_service.transports.base import (
    PredictionServiceTransport,
)
from google.cloud.aiplatform_v1beta1.services.prediction_service.transports.rest import (
    PredictionServiceRestTransport,
)
from langchain_community.utilities.vertexai import get_client_info
from langchain_google_vertexai import ChatVertexAI
from pydantic import Field
from uipath.utils import EndpointManager

from .chat_models import GeminiModels

logger = logging.getLogger(__name__)


class CustomPredictionServiceRestTransport(PredictionServiceRestTransport):
    def __init__(self, llmgw_url: str, custom_headers: dict[str, str], **kwargs):
        self.llmgw_url = llmgw_url
        self.custom_headers = custom_headers or {}

        kwargs.setdefault("credentials", AnonymousCredentials())
        super().__init__(**kwargs)

        # Disable SSL verification for testing
        self._session.verify = False

        original_request = self._session.request

        def redirected_request(method, url, **kwargs_inner):
            headers = kwargs_inner.pop("headers", {})
            headers.update(self.custom_headers)

            is_streaming = kwargs_inner.get("stream", False)
            headers["X-UiPath-Streaming-Enabled"] = "true" if is_streaming else "false"

            return original_request(
                method, self.llmgw_url, headers=headers, **kwargs_inner
            )

        self._session.request = redirected_request  # type: ignore


class CustomPredictionServiceRestAsyncTransport:
    """
    Custom async transport for calling UiPath LLM Gateway.

    Uses aiohttp for REST/HTTP communication instead of gRPC.
    Handles both regular and streaming responses from the gateway.
    """

    def __init__(self, llmgw_url: str, custom_headers: dict[str, str], **kwargs):
        self.llmgw_url = llmgw_url
        self.custom_headers = custom_headers or {}

    def _serialize_request(self, request) -> str:
        """Convert proto-plus request to JSON string."""
        import json

        from proto import Message as ProtoMessage  # type: ignore[import-untyped]

        if isinstance(request, ProtoMessage):
            request_dict = type(request).to_dict(
                request, preserving_proto_field_name=False
            )
            return json.dumps(request_dict)
        else:
            from google.protobuf.json_format import MessageToJson

            return MessageToJson(request, preserving_proto_field_name=False)

    def _get_response_class(self, request):
        """Get the response class corresponding to the request class."""
        import importlib

        response_class_name = request.__class__.__name__.replace("Request", "Response")
        response_class = getattr(
            request.__class__.__module__, response_class_name, None
        )

        if response_class is None:
            module = importlib.import_module(request.__class__.__module__)
            response_class = getattr(module, response_class_name, None)

        return response_class

    def _deserialize_response(self, response_json: str, request):
        """Convert JSON string to proto-plus response object."""
        import json

        from proto import Message as ProtoMessage

        response_class = self._get_response_class(request)

        if response_class and isinstance(request, ProtoMessage):
            return response_class.from_json(response_json, ignore_unknown_fields=True)
        elif response_class:
            from google.protobuf.json_format import Parse

            return Parse(response_json, response_class(), ignore_unknown_fields=True)
        else:
            return json.loads(response_json)

    async def _make_request(self, request_json: str, streaming: bool = False):
        """Make HTTP POST request to UiPath gateway."""
        headers = self.custom_headers.copy()
        headers["Content-Type"] = "application/json"

        if streaming:
            headers["X-UiPath-Streaming-Enabled"] = "true"

        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                self.llmgw_url, headers=headers, data=request_json
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

                return await response.text()

    async def generate_content(self, request, **kwargs):
        """Handle non-streaming generate_content calls."""
        request_json = self._serialize_request(request)
        response_text = await self._make_request(request_json, streaming=False)
        return self._deserialize_response(response_text, request)

    def stream_generate_content(self, request, **kwargs):
        """
        Handle streaming generate_content calls.

        Returns a coroutine that yields an async iterator.
        """
        return self._create_stream_awaitable(request)

    async def _create_stream_awaitable(self, request):
        """Awaitable wrapper that returns the async generator."""
        return self._stream_implementation(request)

    async def _stream_implementation(self, request):
        """
        Async generator that yields streaming response chunks.

        Parses the array and yields each chunk individually.
        """
        import json

        request_json = self._serialize_request(request)
        response_text = await self._make_request(request_json, streaming=True)

        try:
            chunks_array = json.loads(response_text)
            if isinstance(chunks_array, list):
                logger.info(f"Streaming: yielding {len(chunks_array)} chunks")
                for chunk_data in chunks_array:
                    chunk_json = json.dumps(chunk_data)
                    yield self._deserialize_response(chunk_json, request)
                return
        except Exception as e:
            logger.info(f"Not a JSON array, trying single response: {e}")

        try:
            yield self._deserialize_response(response_text, request)
        except Exception as e:
            logger.error(f"Failed to parse streaming response: {e}")


class ChatVertexUiPath(ChatVertexAI):
    transport: Optional[PredictionServiceTransport] = Field(default=None)
    async_transport: Optional[CustomPredictionServiceRestAsyncTransport] = Field(
        default=None
    )
    async_client: Optional[
        Union[v1beta1PredictionServiceAsyncClient, v1PredictionServiceAsyncClient]
    ] = Field(default=None)

    def __init__(
        self,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        model_name: str = GeminiModels.gemini_2_5_flash,
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

        self._vendor = "vertexai"
        self._model_name = model_name
        self._url: Optional[str] = None

        llmgw_url = self._build_base_url()

        headers = self._build_headers(token)

        super().__init__(
            model=model_name,
            project=os.getenv("VERTEXAI_PROJECT", "none"),
            location=os.getenv("VERTEXAI_LOCATION", "us-central1"),
            **kwargs,
        )

        self.transport = CustomPredictionServiceRestTransport(
            llmgw_url=llmgw_url, custom_headers=headers
        )

        self.async_transport = CustomPredictionServiceRestAsyncTransport(
            llmgw_url=llmgw_url, custom_headers=headers
        )

    @property
    def prediction_client(
        self,
    ) -> Union[v1beta1PredictionServiceClient, v1PredictionServiceClient]:
        if self.client is None:
            if self.endpoint_version == "v1":
                self.client = v1PredictionServiceClient(
                    client_options=self.client_options,
                    client_info=get_client_info(module=self._user_agent),
                    transport=self.transport,  # type: ignore[arg-type]
                )
            else:
                self.client = v1beta1PredictionServiceClient(
                    client_options=self.client_options,
                    client_info=get_client_info(module=self._user_agent),
                    transport=self.transport,
                )
        return self.client

    @property
    def async_prediction_client(
        self,
    ) -> Union[
        v1beta1PredictionServiceAsyncClient,
        v1PredictionServiceAsyncClient,
    ]:
        return self.async_transport  # type: ignore[return-value]

    @property
    def endpoint(self) -> str:
        vendor_endpoint = EndpointManager.get_vendor_endpoint()
        formatted_endpoint = vendor_endpoint.format(
            vendor=self._vendor,
            model=self._model_name,
        )
        return formatted_endpoint

    def _build_headers(self, token: str) -> dict[str, str]:
        headers = {
            # "X-UiPath-LlmGateway-ApiFlavor": "auto",
            "Authorization": f"Bearer {token}",
        }
        if job_key := os.getenv("UIPATH_JOB_KEY"):
            headers["X-UiPath-JobKey"] = job_key
        if process_key := os.getenv("UIPATH_PROCESS_KEY"):
            headers["X-UiPath-ProcessKey"] = process_key
        return headers

    def _build_base_url(self) -> str:
        if not self._url:
            env_uipath_url = os.getenv("UIPATH_URL")

            if env_uipath_url:
                self._url = f"{env_uipath_url.rstrip('/')}/{self.endpoint}"
            else:
                raise ValueError("UIPATH_URL environment variable is required")

        return self._url
