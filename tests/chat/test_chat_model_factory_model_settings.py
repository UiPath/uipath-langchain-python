"""Tests for how ``chat_model_factory.get_chat_model`` dispatches ``model_settings``."""

import logging
from unittest.mock import MagicMock

from uipath_langchain.chat.chat_model_factory import get_chat_model


class TestModelSettingsDispatch:
    def test_new_path_forwards_model_settings(self, mocker):
        upstream = mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory",
            return_value=MagicMock(),
        )

        get_chat_model(
            "gpt-4o",
            use_new_llm_clients=True,
            model_settings={"reasoning_effort": "high"},
        )

        assert upstream.call_args.kwargs["model_settings"] == {
            "reasoning_effort": "high"
        }

    def test_legacy_path_warns_when_model_settings_dropped(self, mocker, caplog):
        """The legacy clients can't apply model_settings; dropping them must be
        loud so a tenant with EnableModelSpecificSettings on but
        EnabledNewLlmClients off can be diagnosed from logs."""
        legacy = mocker.patch(
            "uipath_langchain.chat.chat_model_factory._legacy_chat_model",
            return_value=MagicMock(),
        )

        with caplog.at_level(logging.WARNING):
            get_chat_model(
                "gpt-4o",
                agenthub_config="cfg",
                use_new_llm_clients=False,
                model_settings={"reasoning_effort": "high"},
            )

        legacy.assert_called_once()
        assert any("model_settings" in record.message for record in caplog.records)

    def test_legacy_path_silent_without_model_settings(self, mocker, caplog):
        mocker.patch(
            "uipath_langchain.chat.chat_model_factory._legacy_chat_model",
            return_value=MagicMock(),
        )

        with caplog.at_level(logging.WARNING):
            get_chat_model(
                "gpt-4o",
                agenthub_config="cfg",
                use_new_llm_clients=False,
            )

        assert not any("model_settings" in record.message for record in caplog.records)
