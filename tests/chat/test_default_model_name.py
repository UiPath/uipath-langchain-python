import pytest
from pydantic import ValidationError
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatBedrockConverse as _UpstreamBedrockConverse,
)
from uipath_langchain_client.clients.normalized.chat_models import (
    UiPathChat as _UpstreamUiPathChat,
)

from uipath_langchain.chat import (
    UiPathAzureChatOpenAI,
    UiPathChat,
    UiPathChatAnthropic,
    UiPathChatAnthropicBedrock,
    UiPathChatAnthropicVertex,
    UiPathChatBedrock,
    UiPathChatBedrockConverse,
    UiPathChatFireworks,
    UiPathChatGoogleGenerativeAI,
    UiPathChatOpenAI,
    UiPathChatVertex,
)

_DEFAULT_OPENAI = "gpt-4.1-mini-2025-04-14"
_DEFAULT_BEDROCK = "anthropic.claude-haiku-4-5-20251001-v1:0"
_DEFAULT_VERTEX = "gemini-2.5-flash"

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
    monkeypatch.delenv("UIPATH_MODEL_NAME", raising=False)


_CASES = [
    (UiPathChat, _DEFAULT_OPENAI),
    (UiPathAzureChatOpenAI, _DEFAULT_OPENAI),
    (UiPathChatOpenAI, _DEFAULT_OPENAI),
    (UiPathChatGoogleGenerativeAI, _DEFAULT_VERTEX),
    (UiPathChatVertex, _DEFAULT_VERTEX),
    (UiPathChatBedrock, _DEFAULT_BEDROCK),
    (UiPathChatBedrockConverse, _DEFAULT_BEDROCK),
    (UiPathChatAnthropicBedrock, _DEFAULT_BEDROCK),
]


@pytest.mark.parametrize("cls, expected", _CASES)
class TestInstantiationWithoutModelKwarg:
    def test_no_args_uses_default(self, cls, expected):
        llm = cls()
        assert llm.model_name == expected

    def test_explicit_model_kwarg_overrides_default(self, cls, expected):
        llm = cls(model="custom-model-id")
        assert llm.model_name == "custom-model-id"


def test_uipath_chat_no_args():
    llm = UiPathChat()
    assert llm.model_name == _DEFAULT_OPENAI


def test_uipath_chat_bedrock_no_args():
    llm = UiPathChatBedrock()
    assert llm.model_name == _DEFAULT_BEDROCK


def test_uipath_chat_bedrock_converse_no_args():
    llm = UiPathChatBedrockConverse()
    assert llm.model_name == _DEFAULT_BEDROCK


def test_uipath_chat_vertex_no_args():
    llm = UiPathChatVertex()
    assert llm.model_name == _DEFAULT_VERTEX


class TestUipathModelNameEnvVarOverride:
    def test_env_var_overrides_openai_default(self, monkeypatch):
        monkeypatch.setenv("UIPATH_MODEL_NAME", "custom-override")
        llm = UiPathChat()
        assert llm.model_name == "custom-override"

    def test_env_var_overrides_bedrock_default(self, monkeypatch):
        monkeypatch.setenv("UIPATH_MODEL_NAME", "custom-override")
        llm = UiPathChatBedrock()
        assert llm.model_name == "custom-override"

    def test_env_var_overrides_bedrock_converse_default(self, monkeypatch):
        monkeypatch.setenv("UIPATH_MODEL_NAME", "custom-override")
        llm = UiPathChatBedrockConverse()
        assert llm.model_name == "custom-override"

    def test_env_var_overrides_vertex_default(self, monkeypatch):
        monkeypatch.setenv("UIPATH_MODEL_NAME", "custom-override")
        llm = UiPathChatGoogleGenerativeAI()
        assert llm.model_name == "custom-override"


class TestExportsWithoutDefaults:
    def test_anthropic_raises_without_model(self):
        with pytest.raises(ValidationError, match="model"):
            UiPathChatAnthropic()

    def test_anthropic_vertex_raises_without_model(self):
        with pytest.raises(ValidationError, match="model"):
            UiPathChatAnthropicVertex()

    def test_fireworks_raises_without_model(self):
        with pytest.raises(ValidationError, match="model"):
            UiPathChatFireworks()


class TestReExportedClassIdentity:
    def test_uipath_chat_is_upstream_class(self):
        assert UiPathChat is _UpstreamUiPathChat

    def test_uipath_chat_vertex_alias_matches_google(self):
        assert UiPathChatVertex is UiPathChatGoogleGenerativeAI

    def test_bedrock_converse_is_subclass_of_upstream(self):
        assert issubclass(UiPathChatBedrockConverse, _UpstreamBedrockConverse)
