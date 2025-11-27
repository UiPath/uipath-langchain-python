import logging
import os
from typing import Optional, Union

from google.auth.credentials import AnonymousCredentials
from google.cloud.aiplatform_v1.services.prediction_service import (
    PredictionServiceClient as v1PredictionServiceClient,
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

logger = logging.getLogger(__name__)


class CustomPredictionServiceRestTransport(PredictionServiceRestTransport):
    """
    Custom transport that redirects Vertex AI API calls to UiPath LLM Gateway.
    """

    def __init__(self, llmgw_url: str, custom_headers: dict[str, str], **kwargs):
        self.llmgw_url = llmgw_url
        self.custom_headers = custom_headers or {}

        kwargs.setdefault("credentials", AnonymousCredentials())
        super().__init__(**kwargs)

        original_request = self._session.request

        def redirected_request(method, url, **kwargs_inner):
            headers = kwargs_inner.pop("headers", {})
            headers.update(self.custom_headers)
            return original_request(
                method, self.llmgw_url, headers=headers, **kwargs_inner
            )

        self._session.request = redirected_request  # type: ignore


class ChatGeminiUiPath(ChatVertexAI):
    transport: Optional[PredictionServiceTransport] = Field(None)

    def __init__(
        self,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        gateway_url: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-001",
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

        gateway_url = gateway_url or os.getenv(
            "UIPATH_LLM_GATEWAY_URL", "https://alpha.uipath.com"
        )

        llmgw_url = (
            f"{gateway_url}/{org_id}/{tenant_id}/agenthub_/llm/raw/vendor/vertexai/"
            f"model/{model_name}/completions"
        )

        logger.debug(
            f"Initializing ChatGeminiUiPath with llmgw_url={llmgw_url}, model={model_name}"
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

        super().__init__(
            model=model_name,
            project=os.getenv("VERTEXAI_PROJECT", "none"),
            location=os.getenv("VERTEXAI_LOCATION", "us-central1"),
            **kwargs,
        )

        self.transport = CustomPredictionServiceRestTransport(
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
