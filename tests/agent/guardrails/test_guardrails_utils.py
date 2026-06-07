"""Tests for guardrails utils module."""

from __future__ import annotations

from uipath.core.guardrails import GuardrailSelector
from uipath.platform.guardrails import GuardrailScope

from uipath_langchain.agent.guardrails.utils import _sanitize_selector_tool_names


class TestSanitizeSelectorToolNames:
    """Tests for _sanitize_selector_tool_names function."""

    def test_sanitizes_tool_names_for_tool_scope_guardrails(self) -> None:
        """Tool names in match_names are sanitized when selector has TOOL scope."""
        # Create a selector with TOOL scope and tool names that need sanitization
        selector = GuardrailSelector(
            scopes=[GuardrailScope.TOOL],
            match_names=["My Tool!", "Another Tool@123", "Tool With Spaces"],
        )

        result = _sanitize_selector_tool_names(selector)

        # Tool names should be sanitized (special chars removed, spaces replaced with _)
        assert result.match_names == ["My_Tool", "Another_Tool123", "Tool_With_Spaces"]

    def test_handles_tool_names_with_valid_characters(self) -> None:
        """Tool names with only valid characters (alphanumeric, underscore, hyphen) remain unchanged."""
        selector = GuardrailSelector(
            scopes=[GuardrailScope.TOOL],
            match_names=["valid_tool", "another-tool", "Tool123"],
        )

        result = _sanitize_selector_tool_names(selector)

        assert result.match_names == ["valid_tool", "another-tool", "Tool123"]

    def test_truncates_long_tool_names(self) -> None:
        """Tool names longer than 64 characters are truncated."""
        long_name = "a" * 100  # 100 characters
        selector = GuardrailSelector(
            scopes=[GuardrailScope.TOOL],
            match_names=[long_name],
        )

        result = _sanitize_selector_tool_names(selector)

        assert len(result.match_names[0]) == 64
        assert result.match_names[0] == "a" * 64

    def test_does_not_sanitize_when_tool_scope_not_present(self) -> None:
        """Tool names are not sanitized when TOOL scope is not in the scopes list."""
        selector = GuardrailSelector(
            scopes=[GuardrailScope.LLM, GuardrailScope.AGENT],
            match_names=["My Tool!", "Another Tool@123"],
        )

        result = _sanitize_selector_tool_names(selector)

        assert result.match_names == ["My Tool!", "Another Tool@123"]

    def test_handles_mixed_scopes_with_tool_scope_present(self) -> None:
        """Tool names are sanitized when TOOL scope is present among other scopes."""
        selector = GuardrailSelector(
            scopes=[GuardrailScope.LLM, GuardrailScope.TOOL, GuardrailScope.AGENT],
            match_names=["My Tool!", "Another Tool@123"],
        )

        result = _sanitize_selector_tool_names(selector)

        assert result.match_names == ["My_Tool", "Another_Tool123"]

    def test_handles_whitespace_in_tool_names(self) -> None:
        """Whitespace in tool names is replaced with underscores."""
        selector = GuardrailSelector(
            scopes=[GuardrailScope.TOOL],
            match_names=[
                "tool with spaces",
                "tool\twith\ttabs",
                "tool\nwith\nnewlines",
            ],
        )

        result = _sanitize_selector_tool_names(selector)

        assert result.match_names == [
            "tool_with_spaces",
            "tool_with_tabs",
            "tool_with_newlines",
        ]

    def test_preserves_hyphens_and_underscores(self) -> None:
        """Hyphens and underscores are preserved in tool names."""
        selector = GuardrailSelector(
            scopes=[GuardrailScope.TOOL],
            match_names=["my-tool_name", "another_tool-name"],
        )

        result = _sanitize_selector_tool_names(selector)

        assert result.match_names == ["my-tool_name", "another_tool-name"]

    def test_sanitizes_with_proper_guardrail_selector(self) -> None:
        """Function works correctly with proper GuardrailSelector type."""
        selector = GuardrailSelector(
            scopes=[GuardrailScope.TOOL],
            match_names=["My Tool!", "Another Tool@123"],
        )

        result = _sanitize_selector_tool_names(selector)

        assert result.match_names == ["My_Tool", "Another_Tool123"]
        assert result is selector  # Same object is returned
