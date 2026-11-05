"""Tests for chat/chat_model_factory.py module."""

import pytest


class TestDefaultMaxRetriesForwarding:
    """Pins ``DEFAULT_MAX_RETRIES`` to 5 and asserts the default reaches the
    ``uipath_langchain_client`` factory."""

    def test_default_max_retries_is_5(self):
        from uipath_langchain.chat.chat_model_factory import DEFAULT_MAX_RETRIES

        assert DEFAULT_MAX_RETRIES == 5

    def test_default_max_retries_forwarded_to_client_factory(self, mocker):
        from uipath_langchain.chat import chat_model_factory as outer_factory

        mock_factory = mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory"
        )

        outer_factory.get_chat_model(
            "gpt-test",
            agenthub_config="cfg",
        )

        _, kwargs = mock_factory.call_args
        assert kwargs["max_retries"] == 5


class TestGetChatModelModelNotFound:
    """get_chat_model maps a discovery-miss to AgentStartupError."""

    @pytest.mark.parametrize("byo_connection_id", [None, "conn-1"])
    def test_maps_model_not_found_to_agent_startup_error(
        self, mocker, byo_connection_id
    ):
        from uipath.llm_client.utils.exceptions import ModelNotFoundError
        from uipath.runtime.errors import UiPathErrorCategory

        from uipath_langchain.agent.exceptions import (
            AgentStartupError,
            AgentStartupErrorCode,
        )
        from uipath_langchain.chat.chat_model_factory import (
            get_chat_model as dispatch_get_chat_model,
        )

        mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory",
            side_effect=ModelNotFoundError("Model gpt-x not found."),
        )

        with pytest.raises(AgentStartupError) as exc_info:
            dispatch_get_chat_model(
                "gpt-x", byo_connection_id=byo_connection_id, agenthub_config="cfg"
            )

        info = exc_info.value.error_info
        assert info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.LLM_INVALID_MODEL
        )
        assert info.category == UiPathErrorCategory.DEPLOYMENT
        assert isinstance(exc_info.value.__cause__, ModelNotFoundError)
