"""Tests for forgiving enum coercion in chat/chat_model_factory.get_chat_model."""

from unittest.mock import patch

import pytest
from uipath_langchain_client.settings import ApiFlavor, RoutingMode, VendorType

from uipath_langchain.chat.chat_model_factory import _coerce_enum, get_chat_model


class TestCoerceEnum:
    """Test cases for the _coerce_enum helper."""

    @pytest.mark.parametrize(
        "raw",
        ["awsbedrock", "AWSBedrock", "AWS_BEDROCK", "aws-bedrock", " awsbedrock "],
    )
    def test_vendor_spelling_variants(self, raw):
        assert _coerce_enum(raw, VendorType) is VendorType.AWSBEDROCK

    @pytest.mark.parametrize(
        "raw",
        [
            "AnthropicMessages",
            "anthropic_messages",
            "ANTHROPIC-MESSAGES",
            "anthropicmessages",
        ],
    )
    def test_flavor_spelling_variants(self, raw):
        assert _coerce_enum(raw, ApiFlavor) is ApiFlavor.ANTHROPIC_MESSAGES

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("chat_completions", ApiFlavor.CHAT_COMPLETIONS),
            ("chat-completions", ApiFlavor.CHAT_COMPLETIONS),
            ("CONVERSE", ApiFlavor.CONVERSE),
            ("generate_content", ApiFlavor.GENERATE_CONTENT),
            ("responses", ApiFlavor.RESPONSES),
            ("invoke", ApiFlavor.INVOKE),
        ],
    )
    def test_flavor_values(self, raw, expected):
        assert _coerce_enum(raw, ApiFlavor) is expected

    def test_enum_member_passes_through(self):
        assert _coerce_enum(VendorType.OPENAI, VendorType) is VendorType.OPENAI

    def test_none_passes_through(self):
        assert _coerce_enum(None, ApiFlavor) is None

    def test_unknown_string_raises_with_valid_values(self):
        with pytest.raises(ValueError) as exc_info:
            _coerce_enum("nonsense", ApiFlavor)
        message = str(exc_info.value)
        assert "nonsense" in message
        assert "converse" in message  # lists valid values


class TestGetChatModelCoercion:
    """get_chat_model forwards coerced enum members to the client factory."""

    def _capture(self, **call_kwargs):
        captured = {}

        def fake_factory(model, **kwargs):
            captured.update(kwargs)
            return object()

        with patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory",
            side_effect=fake_factory,
        ):
            get_chat_model("some-model", **call_kwargs)
        return captured

    def test_string_vendor_and_flavor_are_coerced(self):
        captured = self._capture(
            vendor_type="AWSBedrock", api_flavor="anthropic_messages"
        )
        assert captured["vendor_type"] is VendorType.AWSBEDROCK
        assert captured["api_flavor"] is ApiFlavor.ANTHROPIC_MESSAGES

    def test_string_routing_mode_is_coerced(self):
        captured = self._capture(routing_mode="NORMALIZED")
        assert captured["routing_mode"] is RoutingMode.NORMALIZED

    def test_enum_arguments_pass_through_unchanged(self):
        captured = self._capture(
            vendor_type=VendorType.VERTEXAI, api_flavor=ApiFlavor.GENERATE_CONTENT
        )
        assert captured["vendor_type"] is VendorType.VERTEXAI
        assert captured["api_flavor"] is ApiFlavor.GENERATE_CONTENT

    def test_omitted_vendor_and_flavor_stay_none(self):
        captured = self._capture()
        assert captured["vendor_type"] is None
        assert captured["api_flavor"] is None

    def test_invalid_flavor_raises_before_factory_call(self):
        with patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory"
        ) as factory:
            with pytest.raises(ValueError, match="Unknown ApiFlavor"):
                get_chat_model("some-model", api_flavor="not-a-flavor")
            factory.assert_not_called()
