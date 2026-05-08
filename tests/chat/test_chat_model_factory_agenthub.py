"""Tests for chat_model_factory.get_chat_model agenthub_config plumbing.

The new-LLM-clients path must thread agenthub_config onto client_settings
before delegating to the upstream factory. PlatformSettings.agenthub_config
defaults to "agentsruntime", so without explicit threading, the carefully
computed value at the call site never reaches the wire.
"""

from types import SimpleNamespace

import pytest

from uipath_langchain.chat import chat_model_factory

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


class _Sentinel:
    """Lightweight stand-in for the chat model returned by the upstream factory."""


def _stub_factory(monkeypatch):
    """Replace the upstream factory with a capturing stub.

    Returns a dict that will receive the kwargs the factory was called with.
    """
    captured: dict = {}

    def fake_factory(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return _Sentinel()

    monkeypatch.setattr(chat_model_factory, "get_chat_model_factory", fake_factory)
    return captured


def test_new_path_threads_agenthub_config_into_client_settings(monkeypatch):
    captured = _stub_factory(monkeypatch)

    chat_model_factory.get_chat_model(
        "gpt-4.1-mini-2025-04-14",
        agenthub_config="agentsplayground",
        use_new_llm_clients=True,
    )

    settings = captured["client_settings"]
    assert settings is not None, "factory must receive a non-None client_settings"
    assert settings.agenthub_config == "agentsplayground"


def test_new_path_passes_through_when_agenthub_config_is_none(monkeypatch):
    captured = _stub_factory(monkeypatch)

    chat_model_factory.get_chat_model(
        "gpt-4.1-mini-2025-04-14",
        agenthub_config=None,
        use_new_llm_clients=True,
    )

    # Without explicit agenthub_config, the factory must not synthesize a
    # client_settings — leave it for the upstream factory to default and let
    # the existing env-var / mixin paths govern.
    assert captured["client_settings"] is None


def test_new_path_mutates_caller_supplied_client_settings(monkeypatch):
    captured = _stub_factory(monkeypatch)

    caller_settings = SimpleNamespace(
        agenthub_config="agentsruntime",
        other_field="preserved",
    )

    chat_model_factory.get_chat_model(
        "gpt-4.1-mini-2025-04-14",
        client_settings=caller_settings,  # type: ignore[arg-type]
        agenthub_config="agentsplayground",
        use_new_llm_clients=True,
    )

    forwarded = captured["client_settings"]
    assert forwarded is caller_settings, (
        "caller's settings instance must be preserved, not replaced"
    )
    assert forwarded.agenthub_config == "agentsplayground"
    assert forwarded.other_field == "preserved"


def test_legacy_path_forwards_agenthub_config(monkeypatch):
    import uipath_langchain.chat._legacy.chat_model_factory as legacy_module

    captured: dict = {}

    def fake_legacy_factory(
        model, temperature, max_tokens, agenthub_config, byo_connection_id, **kwargs
    ):
        captured["model"] = model
        captured["agenthub_config"] = agenthub_config
        return _Sentinel()

    monkeypatch.setattr(legacy_module, "get_chat_model", fake_legacy_factory)

    chat_model_factory.get_chat_model(
        "gpt-4.1-mini-2025-04-14",
        agenthub_config="agentsplayground",
        use_new_llm_clients=False,
    )

    assert captured["agenthub_config"] == "agentsplayground"
