"""Tests for LLM span attributes."""

from uipath_agents._observability.llmops.spans.span_attributes.llm import (
    CompletionSpanAttributes,
    LlmCallSpanAttributes,
    ModelSettings,
)
from uipath_agents._observability.llmops.spans.span_attributes.types import SpanType


class TestLlmCallSpanAttributes:
    def test_type_is_llm_call(self) -> None:
        """LlmCallSpanAttributes.type is 'llmCall'."""
        attrs = LlmCallSpanAttributes(input="Hello")
        assert attrs.type == SpanType.LLM_CALL
        assert attrs.type == "llmCall"

    def test_to_otel_attributes_has_correct_type(self) -> None:
        attrs = LlmCallSpanAttributes(input="Hello")
        otel = attrs.to_otel_attributes()
        assert otel["type"] == "llmCall"

    def test_license_ref_id_field(self) -> None:
        """LlmCallSpanAttributes should support licenseRefId from BaseSpanAttributes."""
        attrs = LlmCallSpanAttributes(
            input="Hello",
            license_ref_id="bf6631fd-9eba-4f02-b7fd-cb8ca66c44a7",
        )
        otel = attrs.to_otel_attributes()
        assert otel["licenseRefId"] == "bf6631fd-9eba-4f02-b7fd-cb8ca66c44a7"


class TestCompletionSpanAttributes:
    def test_type_is_completion(self) -> None:
        """CompletionSpanAttributes.type must be 'completion'."""
        attrs = CompletionSpanAttributes(model="gpt-4")
        assert attrs.type == SpanType.COMPLETION
        assert attrs.type == "completion"


class TestModelSettings:
    def test_temperature_serializes_as_int_when_whole(self) -> None:
        """Temperature 0.0 should serialize as 0 (int) for C# parity."""
        settings = ModelSettings(max_tokens=16384, temperature=0.0)
        dumped = settings.model_dump(by_alias=True)
        assert dumped["temperature"] == 0
        assert isinstance(dumped["temperature"], int)

    def test_temperature_serializes_as_float_when_fractional(self) -> None:
        """Temperature 0.7 should serialize as 0.7 (float)."""
        settings = ModelSettings(max_tokens=16384, temperature=0.7)
        dumped = settings.model_dump(by_alias=True)
        assert dumped["temperature"] == 0.7
