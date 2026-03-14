import logging
import os
from unittest.mock import patch

import botocore
from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.content import create_file_block
from langchain_core.outputs import ChatGeneration, ChatResult

from uipath_langchain.chat.bedrock import (
    AwsBedrockCompletionsPassthroughClient,
    UiPathChatBedrock,
)


class TestGetClientSkipsImds:
    def test_client_creation_does_not_trigger_credential_resolution(self, caplog):
        passthrough = AwsBedrockCompletionsPassthroughClient(
            model="anthropic.claude-haiku-4-5-20251001",
            token="test-token",
            api_flavor="converse",
        )

        with caplog.at_level(logging.DEBUG, logger="botocore"):
            client = passthrough.get_client()

        assert caplog.records
        credential_log_records = [
            r for r in caplog.records if r.name.startswith("botocore.credentials")
        ]
        assert not credential_log_records, (
            f"Unexpected credential resolution: {[r.getMessage() for r in credential_log_records]}"
        )
        assert client._request_signer._signature_version == botocore.UNSIGNED


class TestConvertFileBlocksToAnthropicDocuments:
    def test_converts_pdf_file_block_to_document(self):
        messages: list[BaseMessage] = [
            HumanMessage(
                content_blocks=[
                    {"type": "text", "text": "Summarize this PDF"},
                    create_file_block(base64="JVBER==", mime_type="application/pdf"),
                ]
            )
        ]

        result = UiPathChatBedrock._convert_file_blocks_to_anthropic_documents(messages)

        assert result[0].content[0] == {"type": "text", "text": "Summarize this PDF"}
        assert result[0].content[1] == {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBER==",
            },
        }


class TestGenerate:
    @patch.dict(
        os.environ,
        {
            "UIPATH_URL": "https://example.com",
            "UIPATH_ORGANIZATION_ID": "org",
            "UIPATH_TENANT_ID": "tenant",
            "UIPATH_ACCESS_TOKEN": "token",
        },
    )
    @patch("uipath_langchain.chat.bedrock.boto3.Session")
    def test_generate_converts_file_blocks(self, mock_session_cls):
        chat = UiPathChatBedrock()

        messages: list[BaseMessage] = [
            HumanMessage(
                content_blocks=[
                    {"type": "text", "text": "Summarize this PDF"},
                    create_file_block(base64="JVBER==", mime_type="application/pdf"),
                ]
            )
        ]

        fake_result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Summary"))]
        )

        with patch.object(
            ChatBedrock, "_generate", return_value=fake_result
        ) as mock_parent_generate:
            result = chat._generate(messages)

        called_messages = mock_parent_generate.call_args[0][0]
        assert called_messages[0].content[0] == {
            "type": "text",
            "text": "Summarize this PDF",
        }
        assert called_messages[0].content[1] == {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBER==",
            },
        }
        assert result == fake_result
