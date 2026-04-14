"""Tests that each built-in guardrail middleware wires the correct hook types.

The validate endpoint payload has no stage field, so e2e/integration tests cannot
distinguish a PRE-execution call from a POST-execution call for the same validator.
These unit tests verify the wiring contract at the source: by checking the
``name`` of each AgentMiddleware instance, which is set from the hook function
``__name__`` (e.g. ``"Intellectual_Property_Detection_after_model"``).

Contracts under test:
- ``UiPathIntellectualPropertyMiddleware`` — POST-only (only after_* hooks)
- ``UiPathUserPromptAttacksMiddleware``    — PRE-only  (only before_* hooks)
- ``UiPathHarmfulContentMiddleware``       — PRE+POST  (both before_* and after_* hooks)
"""

import pytest
from uipath.platform.guardrails import GuardrailScope
from uipath.platform.guardrails.decorators import (
    BlockAction,
    HarmfulContentEntity,
    LogAction,
)

from uipath_langchain.guardrails.middlewares import (
    UiPathHarmfulContentMiddleware,
    UiPathIntellectualPropertyMiddleware,
    UiPathUserPromptAttacksMiddleware,
)


def _hook_names(middleware: object) -> list[str]:
    """Return the ``name`` attribute of each AgentMiddleware instance."""
    return [inst.name for inst in middleware]  # type: ignore[union-attr]


_LOG = LogAction()
_BLOCK = BlockAction()


class TestIntellectualPropertyHookWiring:
    """UiPathIntellectualPropertyMiddleware registers only POST (after_*) hooks."""

    def test_llm_scope_registers_only_after_model(self) -> None:
        """LLM scope produces a single after_model hook, no before_* hooks."""
        middleware = UiPathIntellectualPropertyMiddleware(
            scopes=[GuardrailScope.LLM],
            action=_LOG,
            entities=["Text"],
        )
        names = _hook_names(middleware)
        assert len(names) == 1
        assert all("after" in n for n in names), (
            f"Expected only after_* hooks, got: {names}"
        )
        assert not any("before" in n for n in names), (
            f"No before_* hooks expected, got: {names}"
        )

    def test_agent_scope_registers_only_after_agent(self) -> None:
        """AGENT scope produces a single after_agent hook, no before_* hooks."""
        middleware = UiPathIntellectualPropertyMiddleware(
            scopes=[GuardrailScope.AGENT],
            action=_LOG,
            entities=["Text"],
        )
        names = _hook_names(middleware)
        assert len(names) == 1
        assert all("after" in n for n in names), (
            f"Expected only after_* hooks, got: {names}"
        )
        assert not any("before" in n for n in names), (
            f"No before_* hooks expected, got: {names}"
        )

    def test_agent_and_llm_scopes_register_only_after_hooks(self) -> None:
        """Both scopes produce after_agent + after_model — no before_* hooks."""
        middleware = UiPathIntellectualPropertyMiddleware(
            scopes=[GuardrailScope.AGENT, GuardrailScope.LLM],
            action=_BLOCK,
            entities=["Text", "Code"],
        )
        names = _hook_names(middleware)
        assert len(names) == 2
        assert all("after" in n for n in names), (
            f"Expected only after_* hooks, got: {names}"
        )
        assert not any("before" in n for n in names), (
            f"No before_* hooks expected, got: {names}"
        )


class TestUserPromptAttacksHookWiring:
    """UiPathUserPromptAttacksMiddleware registers only PRE (before_*) hooks."""

    def test_llm_scope_registers_only_before_model(self) -> None:
        """LLM scope produces a single before_model hook, no after_* hooks."""
        middleware = UiPathUserPromptAttacksMiddleware(
            scopes=[GuardrailScope.LLM],
            action=_BLOCK,
        )
        names = _hook_names(middleware)
        assert len(names) == 1
        assert all("before" in n for n in names), (
            f"Expected only before_* hooks, got: {names}"
        )
        assert not any("after" in n for n in names), (
            f"No after_* hooks expected, got: {names}"
        )


class TestHarmfulContentHookWiring:
    """UiPathHarmfulContentMiddleware registers both PRE (before_*) and POST (after_*) hooks."""

    def test_llm_scope_registers_before_and_after_model(self) -> None:
        """LLM scope produces before_model and after_model hooks."""
        middleware = UiPathHarmfulContentMiddleware(
            scopes=[GuardrailScope.LLM],
            action=_LOG,
            entities=[HarmfulContentEntity("Hate"), HarmfulContentEntity("Violence")],
        )
        names = _hook_names(middleware)
        assert any("before" in n for n in names), (
            f"Expected a before_* hook, got: {names}"
        )
        assert any("after" in n for n in names), (
            f"Expected an after_* hook, got: {names}"
        )

    def test_agent_scope_registers_before_and_after_agent(self) -> None:
        """AGENT scope produces before_agent and after_agent hooks."""
        middleware = UiPathHarmfulContentMiddleware(
            scopes=[GuardrailScope.AGENT],
            action=_LOG,
            entities=[HarmfulContentEntity("Hate")],
        )
        names = _hook_names(middleware)
        assert any("before" in n for n in names), (
            f"Expected a before_* hook, got: {names}"
        )
        assert any("after" in n for n in names), (
            f"Expected an after_* hook, got: {names}"
        )

    @pytest.mark.parametrize("action", [_LOG, _BLOCK])
    def test_all_scopes_register_four_hooks(
        self, action: LogAction | BlockAction
    ) -> None:
        """AGENT+LLM produces before_agent, after_agent, before_model, after_model."""
        middleware = UiPathHarmfulContentMiddleware(
            scopes=[GuardrailScope.AGENT, GuardrailScope.LLM],
            action=action,
            entities=[HarmfulContentEntity("Hate"), HarmfulContentEntity("Violence")],
        )
        names = _hook_names(middleware)
        assert len(names) == 4
        assert sum(1 for n in names if "before" in n) == 2, (
            f"Expected 2 before_* hooks: {names}"
        )
        assert sum(1 for n in names if "after" in n) == 2, (
            f"Expected 2 after_* hooks: {names}"
        )
