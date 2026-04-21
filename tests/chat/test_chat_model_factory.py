"""Tests for chat/chat_model_factory.py module."""

import pytest

from uipath_langchain.chat._legacy.chat_model_factory import (
    _API_FLAVOR_TO_PROVIDER,
    _DEFAULT_API_FLAVOR,
    _compute_vendor_and_api_flavor,
    _should_skip_temperature,
)
from uipath_langchain.chat._legacy.chat_model_factory import (
    get_chat_model as _legacy_get_chat_model,
)
from uipath_langchain.chat._legacy.types import APIFlavor, LLMProvider
from uipath_langchain.chat.chat_model_factory import get_chat_model


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


class TestShouldSkipTemperature:
    """Tests for the _should_skip_temperature capability helper.

    The gate reads ``modelDetails.shouldSkipTemperature`` from the discovery
    response, which is authoritative even for BYOM deployments where the
    display name does not match UiPath-owned model names.
    """

    def test_flag_true_skips_temperature(self):
        model_info = {
            "modelName": "anthropic.claude-opus-4-7",
            "modelDetails": {"shouldSkipTemperature": True},
        }
        assert _should_skip_temperature(model_info) is True

    def test_flag_false_keeps_temperature(self):
        model_info = {
            "modelName": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "modelDetails": {"shouldSkipTemperature": False},
        }
        assert _should_skip_temperature(model_info) is False

    def test_flag_absent_defaults_to_false(self):
        """When the flag is missing (older discovery responses), preserve
        the caller-supplied temperature."""
        model_info = {
            "modelName": "anthropic.claude-opus-4-6",
            "modelDetails": {},
        }
        assert _should_skip_temperature(model_info) is False

    def test_model_details_null_defaults_to_false(self):
        """Embedding models and some vendors have ``modelDetails: null`` in
        the discovery payload; the helper must treat that as not-skipping."""
        model_info = {
            "modelName": "text-embedding-3-large",
            "modelDetails": None,
        }
        assert _should_skip_temperature(model_info) is False

    def test_byom_flag_true_is_honored(self):
        """Under a custom BYOM display name, the discovery flag is still
        the authoritative source — the gate must not depend on model name."""
        model_info = {
            "modelName": "Denis LLM V3",
            "modelDetails": {"shouldSkipTemperature": True},
        }
        assert _should_skip_temperature(model_info) is True


class TestLegacyGetChatModelTemperatureGating:
    """Tests that the legacy factory's get_chat_model nullifies temperature
    when the discovery flag ``modelDetails.shouldSkipTemperature`` is set."""

    def test_discovery_flag_true_gates_temperature_to_none_converse(self, mocker):
        """Bedrock Converse: flag=True must reach _create_bedrock_llm with
        temperature=None, regardless of the caller-supplied value."""
        mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._get_model_info",
            return_value={
                "modelName": "anthropic.claude-opus-4-7",
                "vendor": "AwsBedrock",
                "apiFlavor": "AwsBedrockConverse",
                "modelDetails": {"shouldSkipTemperature": True},
            },
        )
        mock_create = mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._create_bedrock_llm"
        )

        _legacy_get_chat_model(
            model="anthropic.claude-opus-4-7",
            temperature=0.0,
            max_tokens=4096,
            agenthub_config="cfg",
        )

        # Signature: (model, api_flavor, temperature, max_tokens, ...)
        args, _ = mock_create.call_args
        assert args[2] is None

    def test_discovery_flag_true_gates_temperature_to_none_invoke(self, mocker):
        """Bedrock Invoke: flag=True must also nullify temperature."""
        mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._get_model_info",
            return_value={
                "modelName": "anthropic.claude-opus-4-7",
                "vendor": "AwsBedrock",
                "apiFlavor": "AwsBedrockInvoke",
                "modelDetails": {"shouldSkipTemperature": True},
            },
        )
        mock_create = mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._create_bedrock_llm"
        )

        _legacy_get_chat_model(
            model="anthropic.claude-opus-4-7",
            temperature=0.5,
            max_tokens=4096,
            agenthub_config="cfg",
        )

        args, _ = mock_create.call_args
        assert args[2] is None

    def test_discovery_flag_false_preserves_temperature(self, mocker):
        """Flag=False must leave temperature intact even for Anthropic models."""
        mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._get_model_info",
            return_value={
                "modelName": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                "vendor": "AwsBedrock",
                "apiFlavor": "AwsBedrockConverse",
                "modelDetails": {"shouldSkipTemperature": False},
            },
        )
        mock_create = mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._create_bedrock_llm"
        )

        _legacy_get_chat_model(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            temperature=0.7,
            max_tokens=4096,
            agenthub_config="cfg",
        )

        args, _ = mock_create.call_args
        assert args[2] == 0.7

    def test_discovery_flag_absent_preserves_temperature(self, mocker):
        """When discovery omits the flag, temperature must be forwarded.

        Older discovery responses and embedding models expose ``modelDetails``
        as ``None`` or without the flag; the gate must default to not-skipping.
        """
        mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._get_model_info",
            return_value={
                "modelName": "gpt-5-2025-08-07",
                "vendor": "OpenAi",
                "apiFlavor": "OpenAiResponses",
                "modelDetails": None,
            },
        )
        mock_create = mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._create_openai_llm"
        )

        _legacy_get_chat_model(
            model="gpt-5-2025-08-07",
            temperature=0.3,
            max_tokens=2048,
            agenthub_config="cfg",
        )

        args, _ = mock_create.call_args
        assert args[2] == 0.3

    def test_byom_custom_name_honors_discovery_flag(self, mocker):
        """BYOM display names don't match any known alias, but the discovery
        flag still identifies the underlying model — the gate must use it."""
        mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._get_model_info",
            return_value={
                "modelName": "Custom BYOM Opus 4.7",
                "vendor": "AwsBedrock",
                "apiFlavor": "AwsBedrockConverse",
                "modelDetails": {"shouldSkipTemperature": True},
            },
        )
        mock_create = mocker.patch(
            "uipath_langchain.chat._legacy.chat_model_factory._create_bedrock_llm"
        )

        _legacy_get_chat_model(
            model="Custom BYOM Opus 4.7",
            temperature=0.7,
            max_tokens=4096,
            agenthub_config="cfg",
        )

        args, _ = mock_create.call_args
        assert args[2] is None


class TestCreateBedrockLlmKwargSpread:
    """Tests that _create_bedrock_llm omits the `temperature` kwarg entirely
    when the gated value is None, rather than passing `temperature=None`.

    The distinction matters because Anthropic's migration-guide wording is
    "omit these parameters entirely from request payloads." Passing
    `temperature=None` happens to work today (langchain-aws filters None
    out of inferenceConfig) but is less forward-compatible than omitting.
    """

    def test_converse_omits_temperature_kwarg_when_none(self, mocker):
        """None temperature must not appear in UiPathChatBedrockConverse kwargs."""
        pytest.importorskip("langchain_aws")
        from uipath_langchain.chat._legacy.chat_model_factory import (
            _create_bedrock_llm,
        )

        mock_cls = mocker.patch(
            "uipath_langchain.chat._legacy.bedrock.UiPathChatBedrockConverse"
        )

        _create_bedrock_llm(
            model="anthropic.claude-opus-4-7",
            api_flavor=APIFlavor.AWS_BEDROCK_CONVERSE,
            temperature=None,
            max_tokens=4096,
            agenthub_config="cfg",
        )

        _, kwargs = mock_cls.call_args
        assert "temperature" not in kwargs

    def test_invoke_omits_temperature_kwarg_when_none(self, mocker):
        """None temperature must not appear in UiPathChatBedrock (Invoke) kwargs."""
        pytest.importorskip("langchain_aws")
        from uipath_langchain.chat._legacy.chat_model_factory import (
            _create_bedrock_llm,
        )

        mock_cls = mocker.patch(
            "uipath_langchain.chat._legacy.bedrock.UiPathChatBedrock"
        )

        _create_bedrock_llm(
            model="anthropic.claude-opus-4-7",
            api_flavor=APIFlavor.AWS_BEDROCK_INVOKE,
            temperature=None,
            max_tokens=4096,
            agenthub_config="cfg",
        )

        _, kwargs = mock_cls.call_args
        assert "temperature" not in kwargs

    def test_converse_includes_temperature_kwarg_when_set(self, mocker):
        """A concrete float must be forwarded as a `temperature` kwarg."""
        pytest.importorskip("langchain_aws")
        from uipath_langchain.chat._legacy.chat_model_factory import (
            _create_bedrock_llm,
        )

        mock_cls = mocker.patch(
            "uipath_langchain.chat._legacy.bedrock.UiPathChatBedrockConverse"
        )

        _create_bedrock_llm(
            model="anthropic.claude-sonnet-4-5-20250929-v1:0",
            api_flavor=APIFlavor.AWS_BEDROCK_CONVERSE,
            temperature=0.7,
            max_tokens=4096,
            agenthub_config="cfg",
        )

        _, kwargs = mock_cls.call_args
        assert kwargs.get("temperature") == 0.7


class TestDispatcherTemperatureGating:
    """Tests that the top-level dispatcher strips temperature on the new
    path based on the discovery response's ``shouldSkipTemperature`` flag.

    The legacy path is gated by the inner ``_legacy`` factory, which does its
    own discovery fetch; the dispatcher does not pre-gate the legacy path.
    """

    def test_discovery_flag_true_strips_temperature_on_new_path(self, mocker):
        """When discovery reports ``shouldSkipTemperature: true``, the
        dispatcher must not forward ``temperature`` to the new factory."""
        mock_settings = mocker.MagicMock()
        mock_settings.get_model_info.return_value = {
            "modelName": "anthropic.claude-opus-4-7",
            "modelDetails": {"shouldSkipTemperature": True},
        }
        mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_default_client_settings",
            return_value=mock_settings,
        )
        mock_factory = mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory"
        )

        get_chat_model(
            "anthropic.claude-opus-4-7",
            temperature=0.0,
            max_tokens=4096,
        )

        _, kwargs = mock_factory.call_args
        assert "temperature" not in kwargs

    def test_discovery_flag_false_forwards_temperature_on_new_path(self, mocker):
        """When the flag is false, temperature must pass through unchanged."""
        mock_settings = mocker.MagicMock()
        mock_settings.get_model_info.return_value = {
            "modelName": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "modelDetails": {"shouldSkipTemperature": False},
        }
        mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_default_client_settings",
            return_value=mock_settings,
        )
        mock_factory = mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory"
        )

        get_chat_model(
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            temperature=0.7,
            max_tokens=4096,
        )

        _, kwargs = mock_factory.call_args
        assert kwargs.get("temperature") == 0.7

    def test_byom_display_name_honors_discovery_flag_on_new_path(self, mocker):
        """Under a custom BYOM name the model alias won't match any known
        opus-4.7 string, but the discovery flag must still gate it."""
        mock_settings = mocker.MagicMock()
        mock_settings.get_model_info.return_value = {
            "modelName": "Customer BYOM Opus",
            "modelDetails": {"shouldSkipTemperature": True},
        }
        mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_default_client_settings",
            return_value=mock_settings,
        )
        mock_factory = mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_chat_model_factory"
        )

        get_chat_model(
            "Customer BYOM Opus",
            temperature=0.0,
            max_tokens=4096,
        )

        _, kwargs = mock_factory.call_args
        assert "temperature" not in kwargs

    def test_temperature_unset_skips_discovery_lookup(self, mocker):
        """When the caller does not supply temperature, the dispatcher must
        not pay for an extra discovery lookup."""
        mock_settings = mocker.MagicMock()
        mocker.patch(
            "uipath_langchain.chat.chat_model_factory.get_default_client_settings",
            return_value=mock_settings,
        )
        mocker.patch("uipath_langchain.chat.chat_model_factory.get_chat_model_factory")

        get_chat_model(
            "anthropic.claude-opus-4-7",
            max_tokens=4096,
        )

        mock_settings.get_model_info.assert_not_called()

    def test_legacy_path_does_not_pre_gate_temperature(self, mocker):
        """On the legacy path the dispatcher forwards the caller-supplied
        temperature as-is; the inner legacy factory owns the gate."""
        mock_legacy = mocker.patch(
            "uipath_langchain.chat.chat_model_factory._legacy_chat_model"
        )

        get_chat_model(
            "anthropic.claude-opus-4-7",
            temperature=0.0,
            max_tokens=4096,
            agenthub_config="cfg",
            use_new_llm_clients=False,
        )

        _, kwargs = mock_legacy.call_args
        assert kwargs["temperature"] == 0.0
