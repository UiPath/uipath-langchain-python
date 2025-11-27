import logging
import os
import urllib.parse
from typing import Optional

import boto3
from langchain_aws import ChatBedrockConverse

logger = logging.getLogger(__name__)


class AwsBedrockCompletionsPassthroughClient:
    def __init__(
        self,
        model: str,
        org_id: str,
        tenant_id: str,
        token: str,
        gateway_url: str,
        streaming: bool = True,
    ):
        self.model = model
        self.org_id = org_id
        self.tenant_id = tenant_id
        self.token = token
        self.gateway_url = gateway_url
        self.streaming = streaming

    def get_client(self):
        client = boto3.client(
            "bedrock-runtime",
            region_name="none",
            aws_access_key_id="none",
            aws_secret_access_key="none",
            verify=False,
        )
        # method = "ConverseStream" if self.streaming else "Converse"
        # client.meta.events.register(
        #     f"before-send.bedrock-runtime.{method}", self._modify_request
        # )
        client.meta.events.register(
            "before-send.bedrock-runtime.*", self._modify_request
        )
        return client

    def _modify_request(self, request, **kwargs):
        """Intercept boto3 request and redirect to LLM Gateway"""
        encoded_model = urllib.parse.quote(self.model, safe="")

        request.url = (
            f"{self.gateway_url}/{self.org_id}/{self.tenant_id}/agenthub_/llm/raw/vendor/"
            f"awsbedrock/model/{encoded_model}/completions"
        )

        headers = {
            "Authorization": f"Bearer {self.token}",
        }

        job_key = os.getenv("UIPATH_JOB_KEY")
        process_key = os.getenv("UIPATH_PROCESS_KEY")
        if job_key:
            headers["X-UiPath-JobKey"] = job_key
        if process_key:
            headers["X-UiPath-ProcessKey"] = process_key

        request.headers.update(headers)


class ChatBedrockConverseUiPath(ChatBedrockConverse):
    def __init__(
        self,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        gateway_url: Optional[str] = None,
        model_name: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
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

        logger.debug(
            f"Initializing ChatBedrockConverseUiPath with gateway_url={gateway_url}, model={model_name}"
        )

        passthrough_client = AwsBedrockCompletionsPassthroughClient(
            model=model_name,
            org_id=org_id,
            tenant_id=tenant_id,
            token=token,
            gateway_url=gateway_url,  # type: ignore
            streaming=kwargs.get("streaming", True),
        )

        client = passthrough_client.get_client()
        kwargs["client"] = client
        kwargs["model"] = model_name
        super().__init__(**kwargs)
