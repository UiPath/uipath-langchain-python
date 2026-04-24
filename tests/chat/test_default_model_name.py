"""Tests for per-vendor ``model_name`` defaults on re-exported client classes.

Exercises the public ``uipath_langchain.chat`` API only. Behavior is verified
by instantiating the public classes and checking the resulting ``model_name``
(or ``model_id`` for Bedrock), rather than introspecting pydantic fields.
"""

import pytest

from tests.settings import agent_hub_dummy_settings

# Expected per-vendor defaults — duplicated intentionally so these tests fail
# loudly if someone "silently" changes the defaults without updating the test.
_DEFAULT_OPENAI = "gpt-4.1-mini-2025-04-14"
_DEFAULT_BEDROCK = "anthropic.claude-haiku-4-5-20251001-v1:0"
_DEFAULT_VERTEX = "gemini-2.5-flash"


def _get_exported(class_name: str):
    import uipath_langchain.chat as chat_pkg

    return getattr(chat_pkg, class_name)


# (exported class name, expected default model_name)
# Classes that had a legacy equivalent — ``UiPathChat()`` (no args) must
# produce a usable instance with the vendor-appropriate default model.
_CASES = [
    ("UiPathChat", _DEFAULT_OPENAI),
    ("UiPathAzureChatOpenAI", _DEFAULT_OPENAI),
    ("UiPathChatOpenAI", _DEFAULT_OPENAI),
    ("UiPathChatGoogleGenerativeAI", _DEFAULT_VERTEX),
    ("UiPathChatVertex", _DEFAULT_VERTEX),
    ("UiPathChatBedrock", _DEFAULT_BEDROCK),
    ("UiPathChatBedrockConverse", _DEFAULT_BEDROCK),
    ("UiPathChatAnthropicBedrock", _DEFAULT_BEDROCK),
]


@pytest.mark.parametrize("class_name, expected", _CASES)
class TestInstantiationWithoutModelKwarg:
    """``UiPathChat()`` (no ``model=``) must produce an instance with the
    vendor-appropriate default ``model_name``."""

    def test_no_args_uses_default(self, class_name, expected):
        cls = _get_exported(class_name)
        instance = cls(settings=agent_hub_dummy_settings, model_details={})
        assert instance.model_name == expected

    def test_explicit_model_kwarg_overrides_default(self, class_name, expected):
        cls = _get_exported(class_name)
        instance = cls(
            model="custom-model-id",
            settings=agent_hub_dummy_settings,
            model_details={},
        )
        assert instance.model_name == "custom-model-id"


class TestOpenAIEnvVarOverride:
    """``UIPATH_MODEL_NAME`` overrides only the OpenAI/Azure-backed classes."""

    def test_env_var_overrides_openai_default(self, monkeypatch):
        monkeypatch.setenv("UIPATH_MODEL_NAME", "gpt-5-preview")
        from uipath_langchain.chat import UiPathChat

        instance = UiPathChat(settings=agent_hub_dummy_settings, model_details={})
        assert instance.model_name == "gpt-5-preview"

    def test_env_var_overrides_azure(self, monkeypatch):
        monkeypatch.setenv("UIPATH_MODEL_NAME", "gpt-5-preview")
        from uipath_langchain.chat import UiPathAzureChatOpenAI

        instance = UiPathAzureChatOpenAI(
            settings=agent_hub_dummy_settings, model_details={}
        )
        assert instance.model_name == "gpt-5-preview"

    def test_env_var_does_not_leak_into_bedrock(self, monkeypatch):
        """Bedrock classes stay at their vendor-specific default even when
        ``UIPATH_MODEL_NAME`` is set."""
        monkeypatch.setenv("UIPATH_MODEL_NAME", "gpt-5-preview")
        from uipath_langchain.chat import UiPathChatBedrock

        instance = UiPathChatBedrock(
            settings=agent_hub_dummy_settings, model_details={}
        )
        assert instance.model_name == _DEFAULT_BEDROCK

    def test_env_var_does_not_leak_into_vertex(self, monkeypatch):
        """Vertex classes stay at their vendor-specific default even when
        ``UIPATH_MODEL_NAME`` is set."""
        monkeypatch.setenv("UIPATH_MODEL_NAME", "gpt-5-preview")
        from uipath_langchain.chat import UiPathChatGoogleGenerativeAI

        instance = UiPathChatGoogleGenerativeAI(
            settings=agent_hub_dummy_settings, model_details={}
        )
        assert instance.model_name == _DEFAULT_VERTEX


class TestExportsWithoutDefaults:
    """Classes without a legacy equivalent keep ``model=`` required."""

    def test_anthropic_raises_without_model(self):
        from pydantic import ValidationError

        from uipath_langchain.chat import UiPathChatAnthropic

        with pytest.raises(ValidationError, match="model"):
            UiPathChatAnthropic(settings=agent_hub_dummy_settings, model_details={})

    def test_anthropic_vertex_raises_without_model(self):
        """``UiPathChatAnthropicVertex`` has no legacy equivalent."""
        from pydantic import ValidationError

        from uipath_langchain.chat import UiPathChatAnthropicVertex

        with pytest.raises(ValidationError, match="model"):
            UiPathChatAnthropicVertex(
                settings=agent_hub_dummy_settings, model_details={}
            )

    def test_fireworks_raises_without_model(self):
        from pydantic import ValidationError

        from uipath_langchain.chat import UiPathChatFireworks

        with pytest.raises(ValidationError, match="model"):
            UiPathChatFireworks(settings=agent_hub_dummy_settings, model_details={})


class TestReExportedClassIdentity:
    def test_uipath_chat_is_upstream_class(self):
        """Non-wrapped re-exports share identity with the upstream class."""
        from uipath_langchain_client.clients.normalized.chat_models import (
            UiPathChat as _Upstream,
        )

        from uipath_langchain.chat import UiPathChat

        assert UiPathChat is _Upstream

    def test_uipath_chat_vertex_alias_matches_google(self):
        """``UiPathChatVertex`` is a legacy alias for
        ``UiPathChatGoogleGenerativeAI``."""
        from uipath_langchain.chat import (
            UiPathChatGoogleGenerativeAI,
            UiPathChatVertex,
        )

        assert UiPathChatVertex is UiPathChatGoogleGenerativeAI

    def test_bedrock_converse_is_subclass_of_upstream(self):
        """``UiPathChatBedrockConverse`` is wrapped in a subclass with an
        injecting before-validator — it is NOT identical to upstream, but
        ``issubclass`` still holds for langchain interop."""
        from uipath_langchain_client.clients.bedrock.chat_models import (
            UiPathChatBedrockConverse as _Upstream,
        )

        from uipath_langchain.chat import UiPathChatBedrockConverse

        assert issubclass(UiPathChatBedrockConverse, _Upstream)
