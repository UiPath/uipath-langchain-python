"""Tests that each built-in guardrail middleware wires the correct hook types.

The validate endpoint payload has no stage field, so e2e/integration tests cannot
distinguish a PRE-execution call from a POST-execution call for the same validator.
These unit tests verify the wiring contract at the source. AGENT scope is
checked by the ``name`` of each AgentMiddleware instance, which is set from the
hook function ``__name__`` (e.g. ``"Harmful_Content_Detection_after_agent"``).
LLM scope has one ``wrap_model_call`` hook around the model call, so its stage
is checked by running it and recording which side (``PRE`` input, ``POST``
output) it evaluates.

Contracts under test:
- ``UiPathIntellectualPropertyMiddleware`` — POST-only (after_agent; LLM output)
- ``UiPathUserPromptAttacksMiddleware``    — PRE-only  (LLM input)
- ``UiPathHarmfulContentMiddleware``       — PRE+POST  (both agent hooks; LLM input and output)
"""

from collections.abc import Iterable
from typing import Any

import pytest
from uipath.platform.guardrails import GuardrailScope
from uipath.platform.guardrails.decorators import (
    BlockAction,
    HarmfulContentEntity,
    LogAction,
)

from uipath_langchain.guardrails.enums import (
    GuardrailExecutionStage,
    PIIDetectionEntityType,
)
from uipath_langchain.guardrails.middlewares import (
    UiPathHarmfulContentMiddleware,
    UiPathIntellectualPropertyMiddleware,
    UiPathPIIDetectionMiddleware,
    UiPathPromptInjectionMiddleware,
    UiPathUserPromptAttacksMiddleware,
)
from uipath_langchain.guardrails.models import PIIDetectionEntity

from ._llm_hooks import llm_stages


def _hook_names(middleware: Iterable[Any]) -> list[str]:
    """Return the ``name`` attribute of each AgentMiddleware instance."""
    return [inst.name for inst in middleware]


_LOG = LogAction()
_BLOCK = BlockAction()


class TestIntellectualPropertyHookWiring:
    """UiPathIntellectualPropertyMiddleware registers only POST (after_*) hooks."""

    def test_llm_scope_checks_only_the_output(self) -> None:
        """LLM scope produces a single wrap_model_call hook that checks the output."""
        middleware = UiPathIntellectualPropertyMiddleware(
            scopes=[GuardrailScope.LLM],
            action=_LOG,
            entities=["Text"],
        )
        assert _hook_names(middleware) == [
            "Intellectual_Property_Detection_post_wrap_model_call"
        ]
        assert llm_stages(middleware) == ["POST"]

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

    def test_agent_and_llm_scopes_check_only_outputs(self) -> None:
        """Both scopes produce after_agent + an output-only wrap_model_call."""
        middleware = UiPathIntellectualPropertyMiddleware(
            scopes=[GuardrailScope.AGENT, GuardrailScope.LLM],
            action=_BLOCK,
            entities=["Text", "Code"],
        )
        assert _hook_names(middleware) == [
            "Intellectual_Property_Detection_after_agent",
            "Intellectual_Property_Detection_post_wrap_model_call",
        ]
        assert llm_stages(middleware) == ["POST"]


class TestUserPromptAttacksHookWiring:
    """UiPathUserPromptAttacksMiddleware registers only PRE (before_*) hooks."""

    def test_llm_scope_checks_only_the_input(self) -> None:
        """LLM scope produces a single wrap_model_call hook that checks the input."""
        middleware = UiPathUserPromptAttacksMiddleware(
            scopes=[GuardrailScope.LLM],
            action=_BLOCK,
        )
        [name] = _hook_names(middleware)
        assert name.endswith("_wrap_model_call")
        assert llm_stages(middleware) == ["PRE"]


class TestPromptInjectionHookWiring:
    """UiPathPromptInjectionMiddleware registers only PRE (before_*) hooks."""

    def test_llm_scope_checks_only_the_input(self) -> None:
        """LLM scope produces a single wrap_model_call hook that checks the input."""
        middleware = UiPathPromptInjectionMiddleware(
            scopes=[GuardrailScope.LLM],
            action=_BLOCK,
        )
        [name] = _hook_names(middleware)
        assert name.endswith("_wrap_model_call")
        assert llm_stages(middleware) == ["PRE"]


class TestPIIDetectionHookWiringToolScope:
    """UiPathPIIDetectionMiddleware TOOL scope registers exactly one wrap_tool_call hook."""

    def test_tool_scope_registers_one_wrap_tool_call_hook(self) -> None:
        middleware = UiPathPIIDetectionMiddleware(
            scopes=[GuardrailScope.TOOL],
            action=_LOG,
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
            tools=["my_tool"],
        )
        names = _hook_names(middleware)
        assert len(names) == 1, f"Expected 1 hook, got: {names}"
        assert "wrap_tool_call" in names[0], (
            f"Expected wrap_tool_call hook, got: {names}"
        )

    def test_tool_scope_no_before_or_after_hooks(self) -> None:
        middleware = UiPathPIIDetectionMiddleware(
            scopes=[GuardrailScope.TOOL],
            action=_BLOCK,
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
            tools=["my_tool"],
        )
        names = _hook_names(middleware)
        assert not any("before" in n or "after" in n for n in names), (
            f"No before_*/after_* hooks expected for TOOL scope, got: {names}"
        )


class TestPIIDetectionHookWiringStageGating:
    """PII middleware honors ``stage`` for AGENT/LLM scopes (not just TOOL).

    PRE registers only ``before_*``, POST only ``after_*``, PRE_AND_POST both;
    for LLM scope, the one ``wrap_model_call`` hook checks the matching sides.
    """

    def _pii(self, scope: GuardrailScope, stage: GuardrailExecutionStage):
        return UiPathPIIDetectionMiddleware(
            scopes=[scope],
            action=_LOG,
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
            stage=stage,
        )

    def test_agent_pre_registers_only_before_agent(self) -> None:
        names = _hook_names(
            self._pii(GuardrailScope.AGENT, GuardrailExecutionStage.PRE)
        )
        assert names == ["PII_Detection_before_agent"]

    def test_agent_post_registers_only_after_agent(self) -> None:
        names = _hook_names(
            self._pii(GuardrailScope.AGENT, GuardrailExecutionStage.POST)
        )
        assert names == ["PII_Detection_after_agent"]

    def test_agent_pre_and_post_registers_both(self) -> None:
        names = _hook_names(
            self._pii(GuardrailScope.AGENT, GuardrailExecutionStage.PRE_AND_POST)
        )
        assert sorted(names) == [
            "PII_Detection_after_agent",
            "PII_Detection_before_agent",
        ]

    @pytest.mark.parametrize(
        ("stage", "checked"),
        [
            (GuardrailExecutionStage.PRE, ["PRE"]),
            (GuardrailExecutionStage.POST, ["POST"]),
            (GuardrailExecutionStage.PRE_AND_POST, ["PRE", "POST"]),
        ],
    )
    def test_llm_stage_gates_what_the_model_hook_checks(
        self, stage: GuardrailExecutionStage, checked: list[str]
    ) -> None:
        middleware = self._pii(GuardrailScope.LLM, stage)
        assert _hook_names(middleware) == [
            f"PII_Detection_{stage.name.lower()}_wrap_model_call"
        ]
        assert llm_stages(middleware) == checked


class TestHarmfulContentHookWiringToolScope:
    """UiPathHarmfulContentMiddleware TOOL scope registers exactly one wrap_tool_call hook."""

    def test_tool_scope_registers_one_wrap_tool_call_hook(self) -> None:
        middleware = UiPathHarmfulContentMiddleware(
            scopes=[GuardrailScope.TOOL],
            action=_LOG,
            entities=[HarmfulContentEntity("Hate")],
            tools=["my_tool"],
        )
        names = _hook_names(middleware)
        assert len(names) == 1, f"Expected 1 hook, got: {names}"
        assert "wrap_tool_call" in names[0], (
            f"Expected wrap_tool_call hook, got: {names}"
        )

    def test_tool_scope_no_before_or_after_hooks(self) -> None:
        middleware = UiPathHarmfulContentMiddleware(
            scopes=[GuardrailScope.TOOL],
            action=_BLOCK,
            entities=[HarmfulContentEntity("Violence")],
            tools=["my_tool"],
        )
        names = _hook_names(middleware)
        assert not any("before" in n or "after" in n for n in names), (
            f"No before_*/after_* hooks expected for TOOL scope, got: {names}"
        )


class TestHarmfulContentHookWiring:
    """UiPathHarmfulContentMiddleware registers both PRE (before_*) and POST (after_*) hooks."""

    def test_llm_scope_checks_input_and_output(self) -> None:
        """LLM scope produces one wrap_model_call hook checking both sides."""
        middleware = UiPathHarmfulContentMiddleware(
            scopes=[GuardrailScope.LLM],
            action=_LOG,
            entities=[HarmfulContentEntity("Hate"), HarmfulContentEntity("Violence")],
        )
        assert _hook_names(middleware) == [
            "Harmful_Content_Detection_pre_and_post_wrap_model_call"
        ]
        assert llm_stages(middleware) == ["PRE", "POST"]

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
    def test_all_scopes_register_three_hooks(
        self, action: LogAction | BlockAction
    ) -> None:
        """AGENT+LLM produces before_agent, after_agent and wrap_model_call."""
        middleware = UiPathHarmfulContentMiddleware(
            scopes=[GuardrailScope.AGENT, GuardrailScope.LLM],
            action=action,
            entities=[HarmfulContentEntity("Hate"), HarmfulContentEntity("Violence")],
        )
        assert _hook_names(middleware) == [
            "Harmful_Content_Detection_before_agent",
            "Harmful_Content_Detection_after_agent",
            "Harmful_Content_Detection_pre_and_post_wrap_model_call",
        ]
        assert llm_stages(middleware) == ["PRE", "POST"]


class TestHarmfulContentHookWiringStageGating:
    """Harmful content honors ``stage`` for AGENT/LLM scopes (mirrors PII).

    PRE registers only ``before_*``, POST only ``after_*``, PRE_AND_POST both —
    so ``stage=PRE`` no longer also wires ``after_*`` (which would escalate twice).
    For LLM scope, the one ``wrap_model_call`` hook checks the matching sides.
    """

    def _hc(self, scope: GuardrailScope, stage: GuardrailExecutionStage):
        return UiPathHarmfulContentMiddleware(
            scopes=[scope],
            action=_LOG,
            entities=[HarmfulContentEntity("Hate")],
            stage=stage,
        )

    def test_agent_pre_registers_only_before_agent(self) -> None:
        names = _hook_names(self._hc(GuardrailScope.AGENT, GuardrailExecutionStage.PRE))
        assert names == ["Harmful_Content_Detection_before_agent"]

    def test_agent_post_registers_only_after_agent(self) -> None:
        names = _hook_names(
            self._hc(GuardrailScope.AGENT, GuardrailExecutionStage.POST)
        )
        assert names == ["Harmful_Content_Detection_after_agent"]

    def test_agent_pre_and_post_registers_both(self) -> None:
        names = _hook_names(
            self._hc(GuardrailScope.AGENT, GuardrailExecutionStage.PRE_AND_POST)
        )
        assert sorted(names) == [
            "Harmful_Content_Detection_after_agent",
            "Harmful_Content_Detection_before_agent",
        ]

    @pytest.mark.parametrize(
        ("stage", "checked"),
        [
            (GuardrailExecutionStage.PRE, ["PRE"]),
            (GuardrailExecutionStage.POST, ["POST"]),
            (GuardrailExecutionStage.PRE_AND_POST, ["PRE", "POST"]),
        ],
    )
    def test_llm_stage_gates_what_the_model_hook_checks(
        self, stage: GuardrailExecutionStage, checked: list[str]
    ) -> None:
        middleware = self._hc(GuardrailScope.LLM, stage)
        assert _hook_names(middleware) == [
            f"Harmful_Content_Detection_{stage.name.lower()}_wrap_model_call"
        ]
        assert llm_stages(middleware) == checked
