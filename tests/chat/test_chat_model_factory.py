"""Tests for chat/chat_model_factory.py module."""

import pytest

from uipath_langchain.chat.chat_model_factory import (
    _API_FLAVOR_TO_PROVIDER,
    _DEFAULT_API_FLAVOR,
    _compute_vendor_and_api_flavor,
)
from uipath_langchain.chat.types import APIFlavor, LLMProvider


class TestComputeVendorAndApiFlavor:
    """Test cases for _compute_vendor_and_api_flavor function."""

    # ========== Both vendor and api_flavor present ==========

    def test_both_vendor_and_api_flavor_present_openai(self):
        """Test when both vendor and api_flavor are provided for OpenAI."""
        model = {
            "modelName": "gpt-4",
            "vendor": "OpenAi",
            "apiFlavor": "OpenAIResponses",
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
        """Test deriving vendor from OpenAIResponses api_flavor."""
        model = {
            "modelName": "gpt-4",
            "vendor": None,
            "apiFlavor": "OpenAIResponses",
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
            "apiFlavor": "OpenAIResponses",
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
            "apiFlavor": "OpenAIResponses",
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
