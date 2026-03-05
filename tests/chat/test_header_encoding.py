import os
from unittest.mock import MagicMock, patch
from urllib.parse import quote

import pytest

from uipath_langchain.chat.openai import UiPathChatOpenAI

NON_ASCII_PROCESS_KEY = "Solution.17.agent.GetCompanyIdAgent-請-test"
ASCII_PROCESS_KEY = "Solution.17.agent.MyAgent-test"

BASE_ENV = {
    "UIPATH_URL": "https://cloud.uipath.com/org/tenant",
    "UIPATH_ORGANIZATION_ID": "org-id",
    "UIPATH_TENANT_ID": "tenant-id",
    "UIPATH_ACCESS_TOKEN": "test-token",
}


class TestOpenAIHeaderEncoding:
    """Verify UiPathChatOpenAI percent-encodes non-ASCII header values."""

    def _build_headers_with_process_key(self, process_key: str) -> dict[str, str]:
        env = {**BASE_ENV, "UIPATH_PROCESS_KEY": process_key}
        with patch.dict(os.environ, env, clear=False):
            obj = object.__new__(UiPathChatOpenAI)
            obj._agenthub_config = None
            obj._byo_connection_id = None
            obj._extra_headers = {}
            return obj._build_headers("fake-token")

    def test_ascii_process_key_unchanged(self) -> None:
        headers = self._build_headers_with_process_key(ASCII_PROCESS_KEY)
        assert headers["X-UiPath-ProcessKey"] == quote(ASCII_PROCESS_KEY, safe="")

    def test_non_ascii_process_key_encoded(self) -> None:
        headers = self._build_headers_with_process_key(NON_ASCII_PROCESS_KEY)
        value = headers["X-UiPath-ProcessKey"]
        assert "請" not in value
        assert value == quote(NON_ASCII_PROCESS_KEY, safe="")
        assert "%E8%AB%8B" in value

    def test_header_value_is_ascii_safe(self) -> None:
        headers = self._build_headers_with_process_key(NON_ASCII_PROCESS_KEY)
        value = headers["X-UiPath-ProcessKey"]
        value.encode("ascii")

    def test_missing_process_key_omitted(self) -> None:
        env = {**BASE_ENV}
        env.pop("UIPATH_PROCESS_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            obj = object.__new__(UiPathChatOpenAI)
            obj._agenthub_config = None
            obj._byo_connection_id = None
            obj._extra_headers = {}
            headers = obj._build_headers("fake-token")
        assert "X-UiPath-ProcessKey" not in headers


class TestVertexHeaderEncoding:
    """Verify UiPathChatVertex._build_headers percent-encodes non-ASCII header values."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_google(self) -> None:
        pytest.importorskip("google.genai", reason="google-genai not installed")

    def test_non_ascii_process_key_encoded(self) -> None:
        from uipath_langchain.chat.vertex import UiPathChatVertex

        env = {**BASE_ENV, "UIPATH_PROCESS_KEY": NON_ASCII_PROCESS_KEY}
        with patch.dict(os.environ, env, clear=False):
            headers = UiPathChatVertex._build_headers("fake-token")
        value = headers["X-UiPath-ProcessKey"]
        assert "請" not in value
        assert "%E8%AB%8B" in value
        value.encode("ascii")

    def test_ascii_process_key_unchanged(self) -> None:
        from uipath_langchain.chat.vertex import UiPathChatVertex

        env = {**BASE_ENV, "UIPATH_PROCESS_KEY": ASCII_PROCESS_KEY}
        with patch.dict(os.environ, env, clear=False):
            headers = UiPathChatVertex._build_headers("fake-token")
        assert headers["X-UiPath-ProcessKey"] == quote(ASCII_PROCESS_KEY, safe="")


class TestBedrockHeaderEncoding:
    """Verify AwsBedrockCompletionsPassthroughClient percent-encodes non-ASCII header values."""

    def test_non_ascii_process_key_encoded(self) -> None:
        pytest.importorskip("botocore", reason="botocore not installed")
        from uipath_langchain.chat.bedrock import AwsBedrockCompletionsPassthroughClient

        env = {**BASE_ENV, "UIPATH_PROCESS_KEY": NON_ASCII_PROCESS_KEY}
        with (
            patch.dict(os.environ, env, clear=False),
            patch(
                "uipath_langchain.chat.bedrock.boto3.client", return_value=MagicMock()
            ),
        ):
            client = AwsBedrockCompletionsPassthroughClient(
                model="test-model",
                token="fake-token",
                api_flavor="converse",
            )
            request = MagicMock()
            request.url = "https://example.com/converse"
            request.headers = {}
            client._modify_request(request)

        value = request.headers["X-UiPath-ProcessKey"]
        assert "請" not in value
        assert "%E8%AB%8B" in value
        value.encode("ascii")


class TestRequestMixinHeaderEncoding:
    """Verify UiPathRequestMixin default_headers percent-encodes non-ASCII values."""

    def test_non_ascii_process_key_encoded_in_defaults(self) -> None:
        env = {**BASE_ENV, "UIPATH_PROCESS_KEY": NON_ASCII_PROCESS_KEY}
        with patch.dict(os.environ, env, clear=False):
            import importlib

            import uipath_langchain._utils._request_mixin as mod

            importlib.reload(mod)
            value = mod.UiPathRequestMixin.model_fields["default_headers"].default[
                "X-UiPath-ProcessKey"
            ]
            assert "請" not in value
            assert "%E8%AB%8B" in value
            value.encode("ascii")
