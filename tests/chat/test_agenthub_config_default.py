"""Tests that direct construction of UiPathChat / UiPathChatOpenAI /
UiPathAzureChatOpenAI / UiPathChatBedrock / UiPathChatBedrockConverse /
UiPathChatAnthropicBedrock / UiPathChatGoogleGenerativeAI / UiPathChatVertex
defaults ``client_settings.agenthub_config`` based on the execution context.

Since ``uipath-llm-client`` 1.18.0, a design-time run (local, Studio Web, or a
debug session) defaults ``agenthub_config`` to ``codedagentsplayground`` so coded
agents draw the CodedAgents.Playground licensing pool, while a deployed run (a real
Orchestrator job that is not a Studio Web project and not rooted to a debug session)
keeps it ``None`` and omits the ``x-uipath-agenthub-config`` header. An explicit
``UIPATH_AGENTHUB_CONFIG`` always wins."""

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

_CODED_PLAYGROUND = "codedagentsplayground"


@pytest.fixture(autouse=True)
def _platform_env(monkeypatch):
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", _FAKE_JWT)
    monkeypatch.setenv("UIPATH_URL", "https://example.com/org/tenant/orchestrator_/")
    monkeypatch.setenv("UIPATH_TENANT_ID", "tenant")
    monkeypatch.setenv("UIPATH_ORGANIZATION_ID", "org")
    monkeypatch.delenv("UIPATH_AGENTHUB_CONFIG", raising=False)
    monkeypatch.delenv("UIPATH_MODEL_NAME", raising=False)
    # No job key by default -> design-time context.
    monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
    monkeypatch.delenv("UIPATH_PROJECT_ID", raising=False)


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
    def test_default_is_coded_playground_at_design_time(self, cls):
        """No job key -> design-time -> codedagentsplayground."""
        llm = cls()
        assert llm.client_settings.agenthub_config == _CODED_PLAYGROUND

    def test_default_is_none_when_deployed(self, cls, monkeypatch):
        """Deployed job (not a studio project, not a debug session) -> None."""
        monkeypatch.setenv("UIPATH_JOB_KEY", "deployed-job")
        llm = cls()
        assert llm.client_settings.agenthub_config is None

    def test_env_var_is_honored(self, cls, monkeypatch):
        monkeypatch.setenv("UIPATH_AGENTHUB_CONFIG", "agentsplayground")
        llm = cls()
        assert llm.client_settings.agenthub_config == "agentsplayground"

    def test_coded_playground_header_on_inner_http_client_at_design_time(self, cls):
        """Design-time run emits the coded-playground header on the httpx client."""
        llm = cls()
        client = getattr(llm, "uipath_sync_client", None)
        if client is None:
            pytest.skip(f"{cls.__name__} has no uipath_sync_client to inspect")
        normalized = {key.lower(): value for key, value in client.headers.items()}
        assert normalized.get("x-uipath-agenthub-config") == _CODED_PLAYGROUND

    def test_no_agenthub_header_on_inner_http_client_when_deployed(
        self, cls, monkeypatch
    ):
        """Deployed run omits the agenthub-config header."""
        monkeypatch.setenv("UIPATH_JOB_KEY", "deployed-job")
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
