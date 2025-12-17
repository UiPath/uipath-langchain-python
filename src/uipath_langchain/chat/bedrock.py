import logging
import os
from typing import Optional

from uipath.utils import EndpointManager

from .supported_models import BedrockModels

logger = logging.getLogger(__name__)


def _check_bedrock_dependencies() -> None:
    """Check if required dependencies for UiPathChatBedrock are installed."""
    import importlib.util

    missing_packages = []

    if importlib.util.find_spec("langchain_aws") is None:
        missing_packages.append("langchain-aws")

    if importlib.util.find_spec("boto3") is None:
        missing_packages.append("boto3")

    if missing_packages:
        packages_str = ", ".join(missing_packages)
        raise ImportError(
            f"The following packages are required to use UiPathChatBedrock: {packages_str}\n"
            "Please install them using one of the following methods:\n\n"
            "  # Using pip:\n"
            f"  pip install uipath-langchain[bedrock]\n\n"
            "  # Using uv:\n"
            f"  uv add 'uipath-langchain[bedrock]'\n\n"
        )


_check_bedrock_dependencies()

import boto3
from langchain_aws import (
    ChatBedrock,
    ChatBedrockConverse,
)


class AwsBedrockCompletionsPassthroughClient:
    def __init__(
        self,
        model: str,
        token: str,
        api_flavor: str,
        agenthub_config: Optional[str] = None,
        byo_connection_id: Optional[str] = None,
    ):
        self.model = model
        self.token = token
        self.api_flavor = api_flavor
        self.agenthub_config = agenthub_config
        self.byo_connection_id = byo_connection_id
        self._vendor = "awsbedrock"
        self._url: Optional[str] = None

    @property
    def endpoint(self) -> str:
        vendor_endpoint = EndpointManager.get_vendor_endpoint()
        formatted_endpoint = vendor_endpoint.format(
            vendor=self._vendor,
            model=self.model,
        )
        return formatted_endpoint

    def _build_base_url(self) -> str:
        if not self._url:
            env_uipath_url = os.getenv("UIPATH_URL")

            if env_uipath_url:
                self._url = f"{env_uipath_url.rstrip('/')}/{self.endpoint}"
            else:
                raise ValueError("UIPATH_URL environment variable is required")

        return self._url

    def get_client(self):
        client = boto3.client(
            "bedrock-runtime",
            region_name="none",
            aws_access_key_id="none",
            aws_secret_access_key="none",
        )
        client.meta.events.register(
            "before-send.bedrock-runtime.*", self._modify_request
        )
        return client

    def _modify_request(self, request, **kwargs):
        """Intercept boto3 request and redirect to LLM Gateway"""
        # Detect streaming based on URL suffix:
        # - converse-stream / invoke-with-response-stream -> streaming
        # - converse / invoke -> non-streaming
        streaming = "true" if request.url.endswith("-stream") else "false"
        request.url = self._build_base_url()

        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-UiPath-LlmGateway-ApiFlavor": self.api_flavor,
            "X-UiPath-Streaming-Enabled": streaming,
        }

        if self.agenthub_config:
            headers["X-UiPath-AgentHub-Config"] = self.agenthub_config
        if self.byo_connection_id:
            headers["X-UiPath-LlmGateway-ByoIsConnectionId"] = self.byo_connection_id
        job_key = os.getenv("UIPATH_JOB_KEY")
        process_key = os.getenv("UIPATH_PROCESS_KEY")
        if job_key:
            headers["X-UiPath-JobKey"] = job_key
        if process_key:
            headers["X-UiPath-ProcessKey"] = process_key

        request.headers.update(headers)


class UiPathChatBedrockConverse(ChatBedrockConverse):
    def __init__(
        self,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        model_name: str = BedrockModels.anthropic_claude_haiku_4_5,
        agenthub_config: Optional[str] = None,
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

        passthrough_client = AwsBedrockCompletionsPassthroughClient(
            model=model_name,
            token=token,
            api_flavor="converse",
            agenthub_config=agenthub_config,
            byo_connection_id=byo_connection_id,
        )

        client = passthrough_client.get_client()
        kwargs["client"] = client
        kwargs["model"] = model_name
        super().__init__(**kwargs)


class UiPathChatBedrock(ChatBedrock):
    def __init__(
        self,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        model_name: str = BedrockModels.anthropic_claude_haiku_4_5,
        agenthub_config: Optional[str] = None,
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

        passthrough_client = AwsBedrockCompletionsPassthroughClient(
            model=model_name,
            token=token,
            api_flavor="invoke",
            agenthub_config=agenthub_config,
            byo_connection_id=byo_connection_id,
        )

        client = passthrough_client.get_client()
        kwargs["client"] = client
        kwargs["model"] = model_name
        super().__init__(**kwargs)
