import logging
import os
from collections.abc import Iterator
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from tenacity import AsyncRetrying, Retrying
from uipath._utils import resource_override
from uipath.utils import EndpointManager

from .header_capture import HeaderCapture
from .retryers.bedrock import AsyncBedrockRetryer, BedrockRetryer
from .supported_models import BedrockModels
from .types import APIFlavor, LLMProvider

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
import botocore.config
from langchain_aws import (
    ChatBedrock,
    ChatBedrockConverse,
)


class AwsBedrockCompletionsPassthroughClient:
    @resource_override(
        resource_identifier="byo_connection_id", resource_type="connection"
    )
    def __init__(
        self,
        model: str,
        token: str,
        api_flavor: str,
        agenthub_config: Optional[str] = None,
        byo_connection_id: Optional[str] = None,
        header_capture: HeaderCapture | None = None,
    ):
        self.model = model
        self.token = token
        self.api_flavor = api_flavor
        self.agenthub_config = agenthub_config
        self.byo_connection_id = byo_connection_id
        self._vendor = "awsbedrock"
        self._url: Optional[str] = None
        self.header_capture = header_capture

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

    def _capture_response_headers(self, parsed, model, **kwargs):
        if "ResponseMetadata" in parsed:
            headers = parsed["ResponseMetadata"].get("HTTPHeaders", {})
            if self.header_capture:
                self.header_capture.set(dict(headers))

    def get_client(self):
        read_timeout = int(os.getenv("UIPATH_TIMEOUT_SECONDS", "120"))
        client = boto3.client(
            "bedrock-runtime",
            region_name="none",
            aws_access_key_id="none",
            aws_secret_access_key="none",
            config=botocore.config.Config(
                retries={
                    "total_max_attempts": 1,
                },
                read_timeout=read_timeout,
            ),
        )
        client.meta.events.register(
            "before-send.bedrock-runtime.*", self._modify_request
        )
        client.meta.events.register(
            "after-call.bedrock-runtime.*", self._capture_response_headers
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
    llm_provider: LLMProvider = LLMProvider.BEDROCK
    api_flavor: APIFlavor = APIFlavor.AWS_BEDROCK_CONVERSE
    model: str = ""  # For tracing serialization
    retryer: Optional[Retrying] = None
    aretryer: Optional[AsyncRetrying] = None

    def __init__(
        self,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        model_name: str = BedrockModels.anthropic_claude_haiku_4_5,
        agenthub_config: Optional[str] = None,
        byo_connection_id: Optional[str] = None,
        retryer: Optional[Retrying] = None,
        aretryer: Optional[AsyncRetrying] = None,
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
        self.model = model_name
        self.retryer = retryer
        self.aretryer = aretryer

    def invoke(self, *args, **kwargs):
        retryer = self.retryer or _get_default_retryer()
        return retryer(super().invoke, *args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        retryer = self.aretryer or _get_default_async_retryer()
        return await retryer(super().ainvoke, *args, **kwargs)


class UiPathChatBedrock(ChatBedrock):
    llm_provider: LLMProvider = LLMProvider.BEDROCK
    api_flavor: APIFlavor = APIFlavor.AWS_BEDROCK_INVOKE
    model: str = ""  # For tracing serialization
    retryer: Optional[Retrying] = None
    aretryer: Optional[AsyncRetrying] = None
    header_capture: HeaderCapture

    def __init__(
        self,
        org_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token: Optional[str] = None,
        model_name: str = BedrockModels.anthropic_claude_haiku_4_5,
        agenthub_config: Optional[str] = None,
        byo_connection_id: Optional[str] = None,
        retryer: Optional[Retrying] = None,
        aretryer: Optional[AsyncRetrying] = None,
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

        header_capture = HeaderCapture(name=f"bedrock_headers_{id(self)}")

        passthrough_client = AwsBedrockCompletionsPassthroughClient(
            model=model_name,
            token=token,
            api_flavor="invoke",
            agenthub_config=agenthub_config,
            byo_connection_id=byo_connection_id,
            header_capture=header_capture,
        )

        client = passthrough_client.get_client()
        kwargs["client"] = client
        kwargs["model"] = model_name
        kwargs["header_capture"] = header_capture
        super().__init__(**kwargs)
        self.model = model_name
        self.retryer = retryer
        self.aretryer = aretryer

    def invoke(self, *args, **kwargs):
        retryer = self.retryer or _get_default_retryer()
        return retryer(super().invoke, *args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        retryer = self.aretryer or _get_default_async_retryer()
        return await retryer(super().ainvoke, *args, **kwargs)

    @staticmethod
    def _convert_file_blocks_to_anthropic_documents(
        messages: list[BaseMessage],
    ) -> list[BaseMessage]:
        """Convert FileContentBlock items to Anthropic document format.

        langchain_aws's _format_data_content_block() does not support
        type='file' blocks (only images). This pre-processes messages to
        convert PDF FileContentBlocks into Anthropic's native 'document'
        format so they pass through formatting without error.
        """
        for message in messages:
            if not isinstance(message.content, list):
                continue
            for i, block in enumerate(message.content):
                if (
                    isinstance(block, dict)
                    and block.get("type") == "file"
                    and block.get("mime_type") == "application/pdf"
                    and "base64" in block
                ):
                    anthropic_block: dict[str, Any] = {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": block["mime_type"],
                            "data": block["base64"],
                        },
                    }
                    message.content[i] = anthropic_block
        return messages

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        messages = self._convert_file_blocks_to_anthropic_documents(messages)
        result = super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        self.header_capture.attach_to_chat_result(result)
        self.header_capture.clear()
        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        messages = self._convert_file_blocks_to_anthropic_documents(messages)
        chunks = super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

        for chunk in chunks:
            self.header_capture.attach_to_chat_generation(chunk)
            yield chunk
        self.header_capture.clear()


def _get_default_retryer() -> BedrockRetryer:
    return BedrockRetryer(logger=logger)


def _get_default_async_retryer() -> AsyncBedrockRetryer:
    return AsyncBedrockRetryer(logger=logger)
