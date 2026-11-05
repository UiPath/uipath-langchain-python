"""Tests that ``chat_model_factory.get_chat_model`` forwards ``agenthub_config``
to the upstream ``uipath_langchain_client`` factory."""

from unittest.mock import MagicMock

from uipath_langchain.chat.chat_model_factory import get_chat_model


class TestForwardAgentHubConfig:
    def test_agenthub_config_forwarded(self, mocker):
        """The kwarg the caller passes must reach the upstream factory verbatim."""
        upstream = mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory",
            return_value=MagicMock(),
        )

        get_chat_model(
            "gpt-4o",
            agenthub_config="agentsplayground",
        )

        _, kwargs = upstream.call_args
        assert kwargs.get("agenthub_config") == "agentsplayground"

    def test_agenthub_config_none_forwarded(self, mocker):
        """Pass-through preserves None so upstream applies its own default."""
        upstream = mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory",
            return_value=MagicMock(),
        )

        get_chat_model("gpt-4o")

        _, kwargs = upstream.call_args
        assert kwargs.get("agenthub_config") is None
