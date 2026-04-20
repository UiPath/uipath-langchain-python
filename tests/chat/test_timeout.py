"""Tests that the timeout parameter propagates correctly to the underlying HTTP clients."""

import os
from unittest.mock import patch

import pytest

BASE_ENV = {
    "UIPATH_URL": "https://cloud.uipath.com/org/tenant",
    "UIPATH_ORGANIZATION_ID": "org-id",
    "UIPATH_TENANT_ID": "tenant-id",
    "UIPATH_ACCESS_TOKEN": "test-token",
}


class TestUiPathChatOpenAITimeout:
    def _make(self, timeout: float):
        from uipath_langchain.chat.openai import UiPathChatOpenAI

        with patch.dict(os.environ, BASE_ENV, clear=False):
            return UiPathChatOpenAI(timeout=timeout)

    def test_default_timeout(self):
        llm = self._make(300.0)
        assert llm.http_async_client.timeout.read == 300.0
        assert llm.http_client.timeout.read == 300.0

    def test_custom_timeout_propagates_to_async_client(self):
        llm = self._make(600.0)
        assert llm.http_async_client.timeout.read == 600.0

    def test_custom_timeout_propagates_to_sync_client(self):
        llm = self._make(120.0)
        assert llm.http_client.timeout.read == 120.0


@pytest.mark.skipif(
    pytest.importorskip("google.genai", reason="google-genai not installed") is None,
    reason="google-genai not installed",
)
class TestUiPathChatVertexTimeout:
    def _make(self, timeout: float):
        from uipath_langchain.chat.vertex import UiPathChatVertex

        with patch.dict(os.environ, BASE_ENV, clear=False):
            return UiPathChatVertex(timeout=timeout)

    def test_default_timeout(self):
        llm = self._make(300.0)
        assert llm.client._api_client._httpx_client.timeout.read == 300.0

    def test_custom_timeout_propagates(self):
        llm = self._make(600.0)
        assert llm.client._api_client._httpx_client.timeout.read == 600.0


class TestAwsBedrockPassthroughClientTimeout:
    def _make(self, timeout: float):
        from uipath_langchain.chat.bedrock import AwsBedrockCompletionsPassthroughClient

        return AwsBedrockCompletionsPassthroughClient(
            model="anthropic.claude-haiku-4-5",
            token="test-token",
            api_flavor="converse",
            timeout=timeout,
        )

    def test_default_timeout(self):
        client = self._make(300.0)
        boto_client = client.get_client()
        assert boto_client.meta.config.read_timeout == 300.0

    def test_custom_timeout_propagates(self):
        client = self._make(600.0)
        boto_client = client.get_client()
        assert boto_client.meta.config.read_timeout == 600.0
