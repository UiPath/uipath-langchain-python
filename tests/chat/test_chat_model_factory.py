"""Tests for chat/chat_model_factory.py module."""

import pytest

from uipath_langchain.chat._legacy.chat_model_factory import (
    _API_FLAVOR_TO_PROVIDER,
    _DEFAULT_API_FLAVOR,
    _compute_vendor_and_api_flavor,
    get_chat_model,
)
from uipath_langchain.chat._legacy.types import APIFlavor, LLMProvider


class TestComputeVendorAndApiFlavor:
    """Test cases for _compute_vendor_and_api_flavor function."""

    # ========== Both vendor and api_flavor present ==========

    def test_both_vendor_and_api_flavor_present_openai(self):
        """Test when both vendor and api_flavor are provided for OpenAI."""
        model = {
            "modelName": "gpt-4",
            "vendor": "OpenAi",
            "apiFlavor": "OpenAiResponses",
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.OPENAI
        assert api_flavor == APIFlavor.OPENAI_RESPONSES

    def test_both_vendor_and_api_flavor_present_bedrock(self):
        """Test when both vendor and api_flavor are provided for Bedrock."""
        model = {
            "modelName": "anthropic.claude-3",
            "vendor": "AwsBedrock",
            "apiFlavor": "AwsBedrockConverse",
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.BEDROCK
        assert api_flavor == APIFlavor.AWS_BEDROCK_CONVERSE

    def test_both_vendor_and_api_flavor_present_vertex(self):
        """Test when both vendor and api_flavor are provided for Vertex."""
        model = {
            "modelName": "gemini-pro",
            "vendor": "VertexAi",
            "apiFlavor": "GeminiGenerateContent",
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.VERTEX
        assert api_flavor == APIFlavor.VERTEX_GEMINI_GENERATE_CONTENT

    # ========== Only api_flavor present (vendor is null) ==========

    def test_only_api_flavor_openai_responses(self):
        """Test deriving vendor from OpenAiResponses api_flavor."""
        model = {
            "modelName": "gpt-4",
            "vendor": None,
            "apiFlavor": "OpenAiResponses",
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.OPENAI
        assert api_flavor == APIFlavor.OPENAI_RESPONSES

    def test_only_api_flavor_openai_completions(self):
        """Test deriving vendor from OpenAiChatCompletions api_flavor."""
        model = {
            "modelName": "gpt-3.5-turbo",
            "vendor": None,
            "apiFlavor": "OpenAiChatCompletions",
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.OPENAI
        assert api_flavor == APIFlavor.OPENAI_COMPLETIONS

    def test_only_api_flavor_bedrock_converse(self):
        """Test deriving vendor from AwsBedrockConverse api_flavor."""
        model = {
            "modelName": "anthropic.claude-3",
            "vendor": None,
            "apiFlavor": "AwsBedrockConverse",
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.BEDROCK
        assert api_flavor == APIFlavor.AWS_BEDROCK_CONVERSE

    def test_only_api_flavor_bedrock_invoke(self):
        """Test deriving vendor from AwsBedrockInvoke api_flavor."""
        model = {
            "modelName": "anthropic.claude-2",
            "vendor": None,
            "apiFlavor": "AwsBedrockInvoke",
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.BEDROCK
        assert api_flavor == APIFlavor.AWS_BEDROCK_INVOKE

    def test_only_api_flavor_vertex_gemini(self):
        """Test deriving vendor from GeminiGenerateContent api_flavor."""
        model = {
            "modelName": "gemini-pro",
            "vendor": None,
            "apiFlavor": "GeminiGenerateContent",
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.VERTEX
        assert api_flavor == APIFlavor.VERTEX_GEMINI_GENERATE_CONTENT

    def test_only_api_flavor_vertex_anthropic(self):
        """Test deriving vendor from AnthropicClaude api_flavor."""
        model = {
            "modelName": "claude-3-sonnet",
            "vendor": None,
            "apiFlavor": "AnthropicClaude",
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.VERTEX
        assert api_flavor == APIFlavor.VERTEX_ANTHROPIC_CLAUDE

    def test_only_api_flavor_vendor_missing_key(self):
        """Test when vendor key is missing entirely (not just None)."""
        model = {
            "modelName": "gpt-4",
            "apiFlavor": "OpenAiResponses",
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.OPENAI
        assert api_flavor == APIFlavor.OPENAI_RESPONSES

    # ========== Only vendor present (api_flavor is null) ==========

    def test_only_vendor_openai_default_flavor(self):
        """Test default api_flavor for OpenAI vendor."""
        model = {
            "modelName": "gpt-4",
            "vendor": "OpenAi",
            "apiFlavor": None,
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.OPENAI
        assert api_flavor == APIFlavor.OPENAI_RESPONSES

    def test_only_vendor_bedrock_default_flavor(self):
        """Test default api_flavor for Bedrock vendor."""
        model = {
            "modelName": "anthropic.claude-3",
            "vendor": "AwsBedrock",
            "apiFlavor": None,
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.BEDROCK
        assert api_flavor == APIFlavor.AWS_BEDROCK_CONVERSE

    def test_only_vendor_vertex_default_flavor(self):
        """Test default api_flavor for Vertex vendor (non-Claude model)."""
        model = {
            "modelName": "gemini-pro",
            "vendor": "VertexAi",
            "apiFlavor": None,
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.VERTEX
        assert api_flavor == APIFlavor.VERTEX_GEMINI_GENERATE_CONTENT

    def test_only_vendor_vertex_claude_special_case(self):
        """Test Vertex vendor with Claude in model name uses VERTEX_ANTHROPIC_CLAUDE."""
        model = {
            "modelName": "claude-3-sonnet@20240229",
            "vendor": "VertexAi",
            "apiFlavor": None,
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.VERTEX
        assert api_flavor == APIFlavor.VERTEX_ANTHROPIC_CLAUDE

    def test_only_vendor_vertex_claude_case_sensitive(self):
        """Test that 'claude' detection is case-sensitive (lowercase only)."""
        model = {
            "modelName": "CLAUDE-MODEL",
            "vendor": "VertexAi",
            "apiFlavor": None,
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.VERTEX
        # Should NOT match because 'claude' is lowercase check
        assert api_flavor == APIFlavor.VERTEX_GEMINI_GENERATE_CONTENT

    def test_only_vendor_api_flavor_missing_key(self):
        """Test when apiFlavor key is missing entirely (not just None)."""
        model = {
            "modelName": "gpt-4",
            "vendor": "OpenAi",
        }

        vendor, api_flavor = _compute_vendor_and_api_flavor(model)

        assert vendor == LLMProvider.OPENAI
        assert api_flavor == APIFlavor.OPENAI_RESPONSES

    # ========== Error cases ==========

    def test_error_neither_vendor_nor_api_flavor(self):
        """Test error when neither vendor nor api_flavor is provided."""
        model = {
            "modelName": "unknown-model",
            "vendor": None,
            "apiFlavor": None,
        }

        with pytest.raises(ValueError) as exc_info:
            _compute_vendor_and_api_flavor(model)

        assert "Neither vendor nor apiFlavor provided" in str(exc_info.value)
        assert "unknown-model" in str(exc_info.value)

    def test_error_both_missing_keys(self):
        """Test error when both vendor and apiFlavor keys are missing."""
        model = {
            "modelName": "unknown-model",
        }

        with pytest.raises(ValueError) as exc_info:
            _compute_vendor_and_api_flavor(model)

        assert "Neither vendor nor apiFlavor provided" in str(exc_info.value)

    def test_error_unknown_api_flavor(self):
        """Test error for unknown api_flavor value."""
        model = {
            "modelName": "some-model",
            "vendor": None,
            "apiFlavor": "UnknownFlavor",
        }

        with pytest.raises(ValueError) as exc_info:
            _compute_vendor_and_api_flavor(model)

        assert "Unknown apiFlavor 'UnknownFlavor'" in str(exc_info.value)
        assert "some-model" in str(exc_info.value)

    def test_error_unknown_vendor_without_api_flavor(self):
        """Test error for unknown vendor when api_flavor is not provided."""
        model = {
            "modelName": "some-model",
            "vendor": "UnknownVendor",
            "apiFlavor": None,
        }

        with pytest.raises(ValueError) as exc_info:
            _compute_vendor_and_api_flavor(model)

        assert "Unknown vendor 'UnknownVendor'" in str(exc_info.value)
        assert "some-model" in str(exc_info.value)

    def test_error_unknown_vendor_with_api_flavor(self):
        """Test error for unknown vendor even when api_flavor is provided."""
        model = {
            "modelName": "some-model",
            "vendor": "UnknownVendor",
            "apiFlavor": "OpenAiResponses",
        }

        with pytest.raises(ValueError) as exc_info:
            _compute_vendor_and_api_flavor(model)

        assert "Unknown vendor 'UnknownVendor'" in str(exc_info.value)


class TestMappingConsistency:
    """Test that the mapping dictionaries are consistent and complete."""

    def test_all_api_flavors_have_provider_mapping(self):
        """Test that every APIFlavor has a mapping to LLMProvider."""
        for flavor in APIFlavor:
            assert flavor in _API_FLAVOR_TO_PROVIDER, (
                f"APIFlavor.{flavor.name} is missing from _API_FLAVOR_TO_PROVIDER"
            )

    def test_all_providers_have_default_flavor(self):
        """Test that every LLMProvider has a default APIFlavor."""
        for provider in LLMProvider:
            assert provider in _DEFAULT_API_FLAVOR, (
                f"LLMProvider.{provider.name} is missing from _DEFAULT_API_FLAVOR"
            )

    def test_default_flavors_map_back_to_same_provider(self):
        """Test that default flavors map back to their provider."""
        for provider, default_flavor in _DEFAULT_API_FLAVOR.items():
            mapped_provider = _API_FLAVOR_TO_PROVIDER[default_flavor]
            assert mapped_provider == provider, (
                f"Default flavor {default_flavor} for {provider} "
                f"maps to {mapped_provider} instead"
            )


class TestGetChatModelTemperatureGating:
    """End-to-end tests that call ``get_chat_model`` and assert how
    ``temperature`` is forwarded to the underlying LangChain chat class.

    The gate is driven by discovery's ``modelDetails.shouldSkipTemperature``:
    when True, ``temperature`` must be omitted from the constructor kwargs;
    when False/absent, it must be passed through as-is.
    """

    def test_opus_4_7_bedrock_converse_omits_temperature(self, mocker):
        """flag=True + Bedrock Converse: UiPathChatBedrockConverse must be
        instantiated without a ``temperature`` kwarg."""
        pytest.importorskip("langchain_aws")
        mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._get_model_info",
            return_value={
                "modelName": "anthropic.claude-opus-4-7",
                "vendor": "AwsBedrock",
                "apiFlavor": "AwsBedrockConverse",
                "modelDetails": {"shouldSkipTemperature": True},
            },
        )
        mock_cls = mocker.patch(
            "uipath_langchain.chat._legacy.bedrock.UiPathChatBedrockConverse"
        )

        get_chat_model(
            model="anthropic.claude-opus-4-7",
            temperature=0.0,
            max_tokens=4096,
            agenthub_config="cfg",
        )

        _, kwargs = mock_cls.call_args
        assert "temperature" not in kwargs

    def test_sonnet_4_5_bedrock_converse_forwards_temperature(self, mocker):
        """flag=False: UiPathChatBedrockConverse receives the exact caller
        temperature."""
        pytest.importorskip("langchain_aws")
        mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._get_model_info",
            return_value={
                "modelName": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                "vendor": "AwsBedrock",
                "apiFlavor": "AwsBedrockConverse",
                "modelDetails": {"shouldSkipTemperature": False},
            },
        )
        mock_cls = mocker.patch(
            "uipath_langchain.chat._legacy.bedrock.UiPathChatBedrockConverse"
        )

        get_chat_model(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            temperature=0.7,
            max_tokens=4096,
            agenthub_config="cfg",
        )

        _, kwargs = mock_cls.call_args
        assert kwargs.get("temperature") == 0.7

    def test_gpt_openai_responses_forwards_temperature_when_flag_absent(self, mocker):
        """Older discovery payloads have ``modelDetails: null``; the gate
        must default to not-skipping and UiPathChatOpenAI must receive the
        caller temperature."""
        pytest.importorskip("langchain_openai")
        mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._get_model_info",
            return_value={
                "modelName": "gpt-5-2025-08-07",
                "vendor": "OpenAi",
                "apiFlavor": "OpenAiResponses",
                "modelDetails": None,
            },
        )
        mock_cls = mocker.patch("uipath_langchain.chat._legacy.openai.UiPathChatOpenAI")

        get_chat_model(
            model="gpt-5-2025-08-07",
            temperature=0.3,
            max_tokens=2048,
            agenthub_config="cfg",
        )

        _, kwargs = mock_cls.call_args
        assert kwargs.get("temperature") == 0.3

    def test_byom_custom_name_honors_discovery_flag(self, mocker):
        """BYOM display names don't match any known alias, but the discovery
        flag still identifies the underlying model — the gate must use it
        and the leaf client must be built without a temperature kwarg."""
        pytest.importorskip("langchain_aws")
        mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._get_model_info",
            return_value={
                "modelName": "Custom BYOM Opus 4.7",
                "vendor": "AwsBedrock",
                "apiFlavor": "AwsBedrockConverse",
                "modelDetails": {"shouldSkipTemperature": True},
            },
        )
        mock_cls = mocker.patch(
            "uipath_langchain.chat._legacy.bedrock.UiPathChatBedrockConverse"
        )

        get_chat_model(
            model="Custom BYOM Opus 4.7",
            temperature=0.7,
            max_tokens=4096,
            agenthub_config="cfg",
        )

        _, kwargs = mock_cls.call_args
        assert "temperature" not in kwargs

    def test_gemini_vertex_forwards_temperature(self, mocker):
        """Third vendor path: flag=False on a Vertex Gemini model must
        forward the caller temperature to UiPathChatVertex."""
        pytest.importorskip("langchain_google_genai")
        pytest.importorskip("google.genai")
        mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._get_model_info",
            return_value={
                "modelName": "gemini-2.5-pro",
                "vendor": "VertexAi",
                "apiFlavor": "GeminiGenerateContent",
                "modelDetails": {"shouldSkipTemperature": False},
            },
        )
        mock_cls = mocker.patch("uipath_langchain.chat._legacy.vertex.UiPathChatVertex")

        get_chat_model(
            model="gemini-2.5-pro",
            temperature=0.5,
            max_tokens=2048,
            agenthub_config="cfg",
        )

        _, kwargs = mock_cls.call_args
        assert kwargs.get("temperature") == 0.5
