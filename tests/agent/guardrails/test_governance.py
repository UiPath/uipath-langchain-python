"""Tests for the governance decision agent-defined guardrail checks run under."""

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
    GuardrailGovernanceContext,
    GuardrailScope,
)

from uipath_langchain.agent.guardrails.governance import (
    governance_context,
    governance_decision,
    governance_hook,
)
from uipath_langchain.agent.guardrails.guardrail_nodes import (
    create_agent_init_guardrail_node,
)
from uipath_langchain.agent.guardrails.types import ExecutionStage
from uipath_langchain.agent.react.types import AgentGuardrailsGraphState


class TestGovernanceMapping:
    @pytest.mark.parametrize(
        ("scope", "stage", "hook"),
        [
            (GuardrailScope.AGENT, ExecutionStage.PRE_EXECUTION, "BEFORE_AGENT"),
            (GuardrailScope.AGENT, ExecutionStage.POST_EXECUTION, "AFTER_AGENT"),
            (GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION, "BEFORE_MODEL"),
            (GuardrailScope.LLM, ExecutionStage.POST_EXECUTION, "AFTER_MODEL"),
            (GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION, "TOOL_CALL"),
            (GuardrailScope.TOOL, ExecutionStage.POST_EXECUTION, "AFTER_TOOL"),
        ],
    )
    def test_each_scope_and_stage_has_a_hook(self, scope, stage, hook):
        assert governance_hook(scope, stage) == hook
        # Node metadata may carry the raw values instead of the enums.
        assert governance_hook(scope.value, stage.value) == hook

    @pytest.mark.parametrize(
        ("scope", "stage"),
        [(None, ExecutionStage.PRE_EXECUTION), (GuardrailScope.LLM, None), ("", "")],
    )
    def test_unknown_scope_or_stage_has_no_hook(self, scope, stage):
        assert governance_hook(scope, stage) is None

    @pytest.mark.parametrize(
        ("action_type", "decision"),
        [
            ("Block", ("DENY", "ENFORCE")),
            ("Filter", ("DENY", "ENFORCE")),
            ("Log", ("AUDIT", "SIMULATE")),
            ("Escalate", ("HITL", "ENFORCE")),
            ("Custom", None),
            (None, None),
        ],
    )
    def test_each_action_maps_to_its_decision(self, action_type, decision):
        assert governance_decision(action_type) == decision

    def test_context_carries_the_hook_mode_version_and_action(self):
        assert governance_context(
            GuardrailScope.TOOL, ExecutionStage.POST_EXECUTION, "Log"
        ) == GuardrailGovernanceContext(
            hook="AFTER_TOOL", mode="SIMULATE", version="1", action_on_violation="AUDIT"
        )

    def test_no_context_for_an_unknown_action(self):
        assert (
            governance_context(GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION, "X")
            is None
        )


class _StrictGuardrails:
    """Takes ``governance`` only when the node has one, like an older SDK would."""

    def __init__(self, accepts_governance: bool):
        self.calls: list[dict[str, Any]] = []
        self._accepts_governance = accepts_governance

    def evaluate_guardrail(self, text, guardrail, **kwargs):
        if "governance" in kwargs and not self._accepts_governance:
            raise TypeError("unexpected keyword argument 'governance'")
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


class TestGovernanceForwarding:
    @pytest.mark.asyncio
    async def test_node_sends_its_governance_context(self, monkeypatch):
        guardrails = _StrictGuardrails(accepts_governance=True)
        _patch(monkeypatch, guardrails)
        governance = GuardrailGovernanceContext(
            hook="BEFORE_AGENT", mode="ENFORCE", version="1", action_on_violation="DENY"
        )
        _, node = create_agent_init_guardrail_node(
            _built_in(),
            ExecutionStage.PRE_EXECUTION,
            "ok",
            "nope",
            governance=governance,
        )

        await node(AgentGuardrailsGraphState(messages=[HumanMessage("payload")]))

        assert guardrails.calls[0]["governance"] == governance

    @pytest.mark.asyncio
    async def test_node_without_a_context_sends_no_governance_keyword(
        self, monkeypatch
    ):
        guardrails = _StrictGuardrails(accepts_governance=False)
        _patch(monkeypatch, guardrails)
        _, node = create_agent_init_guardrail_node(
            _built_in(), ExecutionStage.PRE_EXECUTION, "ok", "nope"
        )

        await node(AgentGuardrailsGraphState(messages=[HumanMessage("payload")]))

        assert "governance" not in guardrails.calls[0]

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
            governance=None,
        ):
            seen[guardrail.name] = governance
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
            ExecutionStage.PRE_EXECUTION,
            factory,
            "next",
            "inner",
        )

        def context(mode: str, action: str) -> GuardrailGovernanceContext:
            return GuardrailGovernanceContext(
                hook="TOOL_CALL", mode=mode, version="1", action_on_violation=action
            )

        assert seen == {
            "blocking": context("ENFORCE", "DENY"),
            "logging": context("SIMULATE", "AUDIT"),
            "filtering": context("ENFORCE", "DENY"),
        }
