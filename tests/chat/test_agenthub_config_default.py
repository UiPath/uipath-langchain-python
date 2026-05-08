"""Tests that direct construction of UiPathChat / UiPathChatOpenAI /
UiPathAzureChatOpenAI / UiPathChatBedrock / UiPathChatBedrockConverse /
UiPathChatAnthropicBedrock / UiPathChatGoogleGenerativeAI / UiPathChatVertex
defaults ``client_settings.agenthub_config`` to None and omits the
``x-uipath-agenthub-config`` header on the outgoing httpx client unless
``UIPATH_AGENTHUB_CONFIG`` is set."""

import pytest

from uipath_langchain.chat import (
    UiPathAzureChatOpenAI,
    UiPathChat,
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
    UiPathChatGoogleGenerativeAI,
    UiPathChatOpenAI,
    UiPathChatVertex,
)

_FAKE_JWT = (
    "eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9."
    "eyJzdWIiOiAidGVzdCIsICJpc3MiOiAidGVzdCJ9."
    "signature"
)


@pytest.fixture(autouse=True)
def _platform_env(monkeypatch):
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", _FAKE_JWT)
    monkeypatch.setenv("UIPATH_URL", "https://example.com/org/tenant/orchestrator_/")
    monkeypatch.setenv("UIPATH_TENANT_ID", "tenant")
    monkeypatch.setenv("UIPATH_ORGANIZATION_ID", "org")
    monkeypatch.delenv("UIPATH_AGENTHUB_CONFIG", raising=False)
    monkeypatch.delenv("UIPATH_MODEL_NAME", raising=False)


_DIRECT_CTOR_CASES = [
    UiPathChat,
    UiPathAzureChatOpenAI,
    UiPathChatOpenAI,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
    UiPathChatAnthropicBedrock,
    UiPathChatGoogleGenerativeAI,
    UiPathChatVertex,
]


@pytest.mark.parametrize("cls", _DIRECT_CTOR_CASES)
class TestDirectConstructorAgentHubConfig:
    def test_default_is_none(self, cls):
        llm = cls()
        assert llm.client_settings.agenthub_config is None

    def test_env_var_is_honored(self, cls, monkeypatch):
        monkeypatch.setenv("UIPATH_AGENTHUB_CONFIG", "agentsplayground")
        llm = cls()
        assert llm.client_settings.agenthub_config == "agentsplayground"

    def test_no_agenthub_header_on_inner_http_client(self, cls):
        llm = cls()
        client = getattr(llm, "uipath_sync_client", None)
        if client is None:
            pytest.skip(f"{cls.__name__} has no uipath_sync_client to inspect")
        assert "x-uipath-agenthub-config" not in {key.lower() for key in client.headers}

    def test_env_var_is_honored_on_inner_http_client(self, cls, monkeypatch):
        monkeypatch.setenv("UIPATH_AGENTHUB_CONFIG", "agentsplayground")
        llm = cls()
        client = getattr(llm, "uipath_sync_client", None)
        if client is None:
            pytest.skip(f"{cls.__name__} has no uipath_sync_client to inspect")
        normalized = {key.lower(): value for key, value in client.headers.items()}
        assert normalized.get("x-uipath-agenthub-config") == "agentsplayground"
