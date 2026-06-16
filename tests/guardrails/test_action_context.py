"""Unit tests for the guardrail action runtime context."""

from uipath.core.guardrails import GuardrailScope

from uipath_langchain.guardrails._action_context import (
    GuardrailActionContext,
    _action_context,
    component_label,
    current_action_context,
)
from uipath_langchain.guardrails.enums import GuardrailExecutionStage


class TestComponentLabel:
    def test_agent(self) -> None:
        assert component_label(GuardrailScope.AGENT) == "Agent"

    def test_llm(self) -> None:
        assert component_label(GuardrailScope.LLM) == "LLM call"

    def test_tool_returns_none(self) -> None:
        # TOOL has no static label — the tool name is supplied separately.
        assert component_label(GuardrailScope.TOOL) is None

    def test_none(self) -> None:
        assert component_label(None) is None


class TestContextVar:
    def test_default_is_none(self) -> None:
        assert current_action_context() is None

    def test_set_get_reset_round_trip(self) -> None:
        ctx = GuardrailActionContext(
            scope=GuardrailScope.AGENT,
            execution_stage=GuardrailExecutionStage.PRE,
            component="Agent",
        )
        token = _action_context.set(ctx)
        try:
            assert current_action_context() is ctx
        finally:
            _action_context.reset(token)
        assert current_action_context() is None
