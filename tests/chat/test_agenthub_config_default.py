"""Tests for the chat-client agenthub_config default behavior.

Direct construction of UiPathChat / UiPathChatOpenAI / UiPathAzureChatOpenAI /
UiPathChatBedrock / UiPathChatBedrockConverse / UiPathChatAnthropicBedrock /
UiPathChatGoogleGenerativeAI / UiPathChatVertex must default
client_settings.agenthub_config to None.

The upstream classes (used by chat_model_factory for low-code runtime)
must keep the upstream library default of "agentsruntime", proving the
local override does not leak globally onto the upstream class.
"""

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


_UPSTREAM_CASES = [
    "uipath_langchain_client.clients.normalized.chat_models:UiPathChat",
    "uipath_langchain_client.clients.openai.chat_models:UiPathChatOpenAI",
    "uipath_langchain_client.clients.openai.chat_models:UiPathAzureChatOpenAI",
    "uipath_langchain_client.clients.bedrock.chat_models:UiPathChatBedrock",
    "uipath_langchain_client.clients.bedrock.chat_models:UiPathChatBedrockConverse",
    "uipath_langchain_client.clients.bedrock.chat_models:UiPathChatAnthropicBedrock",
    "uipath_langchain_client.clients.google.chat_models:UiPathChatGoogleGenerativeAI",
]


@pytest.mark.parametrize("upstream_path", _UPSTREAM_CASES)
class TestUpstreamAgentHubConfigUntouched:
    """Deployed runtimes go through chat_model_factory, which instantiates the
    upstream classes directly. Those must keep the upstream library default of
    'agentsruntime'."""

    def _resolve(self, upstream_path: str):
        import importlib

        module_name, attr = upstream_path.split(":")
        return getattr(importlib.import_module(module_name), attr)

    def test_upstream_keeps_agentsruntime_default(self, upstream_path):
        # make sure model rebinds are not breaking the agenthub_config
        import uipath_langchain.chat  # noqa: F401

        upstream_cls = self._resolve(upstream_path)
        llm = upstream_cls(model="gpt-4.1-mini-2025-04-14")
        assert llm.client_settings.agenthub_config == "agentsruntime"
