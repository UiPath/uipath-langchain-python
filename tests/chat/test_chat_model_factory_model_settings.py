"""Tests for how ``chat_model_factory.get_chat_model`` forwards ``model_settings``."""

from unittest.mock import MagicMock

from uipath_langchain.chat.chat_model_factory import get_chat_model


class TestModelSettingsDispatch:
    def test_forwards_model_settings(self, mocker):
        upstream = mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory",
            return_value=MagicMock(),
        )

        get_chat_model(
            "gpt-4o",
            model_settings={"reasoning_effort": "high"},
        )

        assert upstream.call_args.kwargs["model_settings"] == {
            "reasoning_effort": "high"
        }
