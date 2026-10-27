from types import SimpleNamespace
from typing import Any

import pytest
from pytest_mock import MockerFixture
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode
from uipath_langchain.chat.chat_model_factory import get_chat_model


def _patch_new_client_factory(
    mocker: MockerFixture, model_details: dict[str, Any] | None
) -> SimpleNamespace:
    chat_model = SimpleNamespace(model_details=model_details)
    mocker.patch(
        "uipath_langchain.chat.chat_model_factory.get_chat_model_factory",
        return_value=chat_model,
    )
    return chat_model


class TestNewClientsMaxTokensCap:
    def test_over_cap_raises_user_startup_error_naming_both_values(
        self, mocker: MockerFixture
    ) -> None:
        _patch_new_client_factory(mocker, {"maxOutputTokens": 64000})

        with pytest.raises(AgentStartupError) as exc_info:
            get_chat_model(
                "anthropic.claude-sonnet-4-5-20250929-v1:0",
                max_tokens=128000,
                agenthub_config="cfg",
            )

        info = exc_info.value.error_info
        assert info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.LLM_MAX_TOKENS_EXCEEDED
        )
        assert info.category == UiPathErrorCategory.USER
        assert "128000" in info.detail
        assert "64000" in info.detail
        assert "anthropic.claude-sonnet-4-5-20250929-v1:0" in info.detail

    def test_exactly_at_cap_returns_model(self, mocker: MockerFixture) -> None:
        chat_model = _patch_new_client_factory(mocker, {"maxOutputTokens": 64000})

        result = get_chat_model("m", max_tokens=64000, agenthub_config="cfg")

        assert result is chat_model

    @pytest.mark.parametrize(
        "model_details",
        [None, {}, {"maxOutputTokens": 0}, {"maxOutputTokens": None}],
        ids=["no-details", "empty-details", "zero-cap", "null-cap"],
    )
    def test_unknown_cap_returns_model(
        self, mocker: MockerFixture, model_details: dict[str, Any] | None
    ) -> None:
        chat_model = _patch_new_client_factory(mocker, model_details)

        result = get_chat_model("m", max_tokens=128000, agenthub_config="cfg")

        assert result is chat_model

    def test_unset_max_tokens_returns_model(self, mocker: MockerFixture) -> None:
        chat_model = _patch_new_client_factory(mocker, {"maxOutputTokens": 64000})

        result = get_chat_model("m", max_tokens=None, agenthub_config="cfg")

        assert result is chat_model


class TestLegacyClientsMaxTokensCap:
    def _patch_discovery(self, mocker: MockerFixture, max_output_tokens: int) -> None:
        mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._get_model_info",
            return_value={
                "modelName": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                "vendor": "AwsBedrock",
                "apiFlavor": "AwsBedrockConverse",
                "modelDetails": {"maxOutputTokens": max_output_tokens},
            },
        )

    def test_over_cap_raises_user_startup_error(self, mocker: MockerFixture) -> None:
        pytest.importorskip("langchain_aws")
        self._patch_discovery(mocker, 64000)
        mocker.patch("uipath_langchain.chat._legacy.bedrock.UiPathChatBedrockConverse")

        with pytest.raises(AgentStartupError) as exc_info:
            get_chat_model(
                "anthropic.claude-sonnet-4-5-20250929-v1:0",
                max_tokens=128000,
                agenthub_config="cfg",
                use_new_llm_clients=False,
            )

        info = exc_info.value.error_info
        assert info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.LLM_MAX_TOKENS_EXCEEDED
        )
        assert info.category == UiPathErrorCategory.USER

    def test_within_cap_builds_model_with_requested_max_tokens(
        self, mocker: MockerFixture
    ) -> None:
        pytest.importorskip("langchain_aws")
        self._patch_discovery(mocker, 64000)
        mock_cls = mocker.patch(
            "uipath_langchain.chat._legacy.bedrock.UiPathChatBedrockConverse"
        )

        get_chat_model(
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            max_tokens=4096,
            agenthub_config="cfg",
            use_new_llm_clients=False,
        )

        _, kwargs = mock_cls.call_args
        assert kwargs["max_tokens"] == 4096
