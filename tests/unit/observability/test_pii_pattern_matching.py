"""Tests for pattern-based attribute name matching in PII filtering."""

import os
from unittest.mock import patch

import pytest

from uipath_agents._observability.pii_filtering_exporter import (
    _redact_attributes,
    _should_redact_attribute,
)


@pytest.fixture(autouse=True)
def _setup_patterns():
    """Set up pattern matching for all tests."""
    with patch.dict(
        os.environ,
        {
            "OTEL_PII_REDACTION_PATTERNS": "llm.*.message.content,llm.*.message_content,*.messages.*.content"
        },
    ):
        # Force reload of module-level variables
        from uipath_agents._observability import pii_filtering_exporter

        pii_filtering_exporter._PII_PATTERNS = [
            "llm.*.message.content",
            "llm.*.message_content",
            "*.messages.*.content",
        ]
        yield
        pii_filtering_exporter._PII_PATTERNS = []


class TestShouldRedactAttribute:
    """Tests for _should_redact_attribute function."""

    def test_exact_match_input_value(self) -> None:
        """Test exact match for input.value."""
        assert _should_redact_attribute("input.value") is True

    def test_exact_match_output_value(self) -> None:
        """Test exact match for output.value."""
        assert _should_redact_attribute("output.value") is True

    def test_pattern_match_llm_message_content(self) -> None:
        """Test pattern matching for llm.*.message.content."""
        # Should match
        assert _should_redact_attribute("llm.input_messages.0.message.content") is True
        assert _should_redact_attribute("llm.input_messages.1.message.content") is True
        assert _should_redact_attribute("llm.output_messages.0.message.content") is True

    def test_pattern_match_llm_message_content_underscore(self) -> None:
        """Test pattern matching for llm.*.message_content."""
        assert _should_redact_attribute("llm.input_messages.0.message_content") is True
        assert _should_redact_attribute("llm.output_messages.1.message_content") is True

    def test_pattern_match_generic_messages_content(self) -> None:
        """Test pattern matching for *.messages.*.content."""
        assert _should_redact_attribute("foo.messages.0.content") is True
        assert _should_redact_attribute("bar.messages.1.content") is True
        assert (
            _should_redact_attribute("nested.path.messages.2.content") is True
        )  # fnmatch * is greedy

    def test_no_match_for_safe_attributes(self) -> None:
        """Test that safe attributes are not matched."""
        assert _should_redact_attribute("llm.model_name") is False
        assert _should_redact_attribute("llm.token_count.total") is False
        assert _should_redact_attribute("openinference.span.kind") is False

    def test_no_match_for_role_field(self) -> None:
        """Test that role fields are not matched (not PII)."""
        assert _should_redact_attribute("llm.input_messages.0.message.role") is False


class TestRedactAttributesWithPatterns:
    """Tests for _redact_attributes with pattern matching."""

    def test_redact_dynamic_attribute_names(self) -> None:
        """Test redacting attributes with dynamic names."""
        attrs: dict[str, str | int] = {
            "llm.input_messages.0.message.content": "User said: Hello",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.1.message.content": "System prompt here",
            "llm.input_messages.1.message.role": "system",
            "llm.model_name": "gpt-4o",
            "llm.token_count.total": 150,
        }

        redacted = _redact_attributes(attrs)

        # Content should be redacted
        assert "REDACTED" in str(redacted["llm.input_messages.0.message.content"])
        assert "REDACTED" in str(redacted["llm.input_messages.1.message.content"])

        # Role should be preserved (not PII)
        assert redacted["llm.input_messages.0.message.role"] == "user"
        assert redacted["llm.input_messages.1.message.role"] == "system"

        # Metadata should be preserved
        assert redacted["llm.model_name"] == "gpt-4o"
        assert redacted["llm.token_count.total"] == 150

    def test_redact_multiple_indexed_messages(self) -> None:
        """Test redacting multiple indexed message attributes."""
        attrs = {
            "llm.output_messages.0.message_content": "Response 1",
            "llm.output_messages.1.message_content": "Response 2",
            "llm.output_messages.2.message_content": "Response 3",
        }

        redacted = _redact_attributes(attrs)

        # All should be redacted
        for key in attrs.keys():
            assert "REDACTED" in str(redacted[key])

    def test_mixed_exact_and_pattern_matches(self) -> None:
        """Test that both exact matches and pattern matches work together."""
        attrs = {
            "input.value": '{"messages": ["sensitive"]}',  # Exact match
            "output.value": "sensitive response",  # Exact match
            "llm.input_messages.0.message.content": "pattern match",  # Pattern match
            "llm.model_name": "gpt-4o",  # Preserved
        }

        redacted = _redact_attributes(attrs)

        # All PII should be redacted
        assert "REDACTED" in str(redacted["input.value"])
        assert "REDACTED" in str(redacted["output.value"])
        assert "REDACTED" in str(redacted["llm.input_messages.0.message.content"])

        # Metadata preserved
        assert redacted["llm.model_name"] == "gpt-4o"

    def test_no_redaction_for_non_matching_attributes(self) -> None:
        """Test that attributes that don't match any pattern are preserved."""
        attrs = {
            "custom.attribute": "some value",
            "llm.provider": "openai",
            "span.name": "test_span",
        }

        redacted = _redact_attributes(attrs)

        # All should be preserved (no matches)
        assert redacted["custom.attribute"] == "some value"
        assert redacted["llm.provider"] == "openai"
        assert redacted["span.name"] == "test_span"


@pytest.mark.parametrize(
    ("attr_name", "should_match"),
    [
        # Should match patterns
        ("llm.input_messages.0.message.content", True),
        ("llm.input_messages.999.message.content", True),
        ("llm.output_messages.0.message_content", True),
        ("foo.messages.0.content", True),
        ("very.nested.messages.5.content", True),
        # Should not match
        ("llm.input_messages.0.message.role", False),
        ("llm.input_messages.0.message.type", False),
        ("llm.model_name", False),
        ("messages", False),  # Pattern requires more structure
        ("content", False),  # Pattern requires more structure
    ],
)
def test_pattern_matching_scenarios(attr_name: str, should_match: bool) -> None:
    """Test various pattern matching scenarios."""
    result = _should_redact_attribute(attr_name)
    assert result is should_match, (
        f"Expected {attr_name} to {'' if should_match else 'NOT '}match"
    )
