"""Tests for the enforcement context a built-in guardrail check is sent with."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails import (
    BuiltInValidatorGuardrail,
    GuardrailScope,
)

from uipath_langchain.agent.guardrails.guardrail_nodes import (
    create_agent_init_guardrail_node,
)
from uipath_langchain.agent.guardrails.types import (
    ExecutionStage,
    GuardrailEnforcementContext,
)
from uipath_langchain.agent.react.types import AgentGuardrailsGraphState

_ENFORCEMENT_KEYS = {"scope", "execution_stage", "action_type"}


class _StrictGuardrails:
    """Takes the enforcement keywords only when the node has them, like an older SDK would."""

    def __init__(self, accepts_enforcement: bool):
        self.calls: list[dict[str, Any]] = []
        self._accepts_enforcement = accepts_enforcement

    def evaluate_guardrail(self, text, guardrail, **kwargs):
        if _ENFORCEMENT_KEYS & kwargs.keys() and not self._accepts_enforcement:
            raise TypeError("unexpected keyword argument 'scope'")
        self.calls.append(kwargs)
        return GuardrailValidationResult(
            result=GuardrailValidationResultType.PASSED, reason="ok"
        )


def _patch(monkeypatch, guardrails: _StrictGuardrails) -> None:
    class _UiPath:
        def __init__(self):
            self.guardrails = guardrails

    monkeypatch.setattr(
        "uipath_langchain.agent.guardrails.guardrail_nodes.UiPath", _UiPath
    )


def _built_in() -> BuiltInValidatorGuardrail:
    guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
    guardrail.name = "Example"
    return guardrail


class TestEnforcementContextForwarding:
    @pytest.mark.asyncio
    async def test_node_sends_its_enforcement_context(self, monkeypatch):
        guardrails = _StrictGuardrails(accepts_enforcement=True)
        _patch(monkeypatch, guardrails)
        _, node = create_agent_init_guardrail_node(
            _built_in(),
            ExecutionStage.PRE_EXECUTION,
            "ok",
            "nope",
            enforcement_context=GuardrailEnforcementContext(
                scope=GuardrailScope.AGENT,
                execution_stage=ExecutionStage.PRE_EXECUTION,
                action_type="Block",
            ),
        )

        await node(AgentGuardrailsGraphState(messages=[HumanMessage("payload")]))

        call = guardrails.calls[0]
        assert (call["scope"], call["execution_stage"], call["action_type"]) == (
            GuardrailScope.AGENT,
            "preExecution",
            "Block",
        )

    @pytest.mark.asyncio
    async def test_node_without_a_context_sends_no_enforcement_keywords(
        self, monkeypatch
    ):
        guardrails = _StrictGuardrails(accepts_enforcement=False)
        _patch(monkeypatch, guardrails)
        _, node = create_agent_init_guardrail_node(
            _built_in(), ExecutionStage.PRE_EXECUTION, "ok", "nope"
        )

        await node(AgentGuardrailsGraphState(messages=[HumanMessage("payload")]))

        assert _ENFORCEMENT_KEYS.isdisjoint(guardrails.calls[0])

    def test_subgraph_gives_each_node_its_actions_context(self):
        from tests.agent.guardrails.test_guardrail_utils import FakeStateGraph
        from uipath_langchain.agent.guardrails.actions.block_action import BlockAction
        from uipath_langchain.agent.guardrails.actions.filter_action import (
            FilterAction,
        )
        from uipath_langchain.agent.guardrails.actions.log_action import LogAction
        from uipath_langchain.agent.react.guardrails import guardrails_subgraph

        seen: dict[str, Any] = {}

        def factory(
            guardrail,
            execution_stage,
            success_node,
            failure_node,
            *,
            termination_mode=None,
            enforcement_context=None,
        ):
            seen[guardrail.name] = enforcement_context
            return f"eval_{guardrail.name}", (lambda s: s)

        guardrails = []
        for name, action in (
            ("blocking", BlockAction("r")),
            ("logging", LogAction(None)),
            ("filtering", FilterAction()),
        ):
            guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
            guardrail.name = name
            guardrails.append((guardrail, action))

        guardrails_subgraph._build_guardrail_node_chain(
            FakeStateGraph(None),  # type: ignore[arg-type]
            guardrails,
            GuardrailScope.TOOL,
            ExecutionStage.POST_EXECUTION,
            factory,
            "next",
            "inner",
        )

        def context(action_type: str) -> GuardrailEnforcementContext:
            return GuardrailEnforcementContext(
                scope=GuardrailScope.TOOL,
                execution_stage=ExecutionStage.POST_EXECUTION,
                action_type=action_type,
            )

        assert seen == {
            "blocking": context("Block"),
            "logging": context("Log"),
            "filtering": context("Filter"),
        }
