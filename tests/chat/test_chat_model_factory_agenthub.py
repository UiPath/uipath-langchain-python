"""Tests that ``chat_model_factory.get_chat_model`` forwards ``agenthub_config``
to the upstream ``uipath_langchain_client`` factory on the new-clients path."""

from unittest.mock import MagicMock

import pytest

from uipath_langchain.chat.chat_model_factory import get_chat_model


class TestForwardAgentHubConfig:
    def test_agenthub_config_forwarded_on_new_path(self, mocker):
        """The kwarg the caller passes must reach the upstream factory verbatim."""
        upstream = mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory",
            return_value=MagicMock(),
        )

        get_chat_model(
            "gpt-4o",
            agenthub_config="agentsplayground",
            use_new_llm_clients=True,
        )

        _, kwargs = upstream.call_args
        assert kwargs.get("agenthub_config") == "agentsplayground"

    def test_agenthub_config_none_forwarded_on_new_path(self, mocker):
        """Pass-through preserves None so upstream applies its own default."""
        upstream = mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory",
            return_value=MagicMock(),
        )

        get_chat_model("gpt-4o", use_new_llm_clients=True)

        _, kwargs = upstream.call_args
        assert kwargs.get("agenthub_config") is None

    def test_legacy_path_still_requires_agenthub_config(self, mocker):
        """Legacy factory contract is unchanged: missing agenthub_config raises."""
        with pytest.raises(ValueError, match="agenthub_config is required"):
            get_chat_model("gpt-4o", use_new_llm_clients=False)
