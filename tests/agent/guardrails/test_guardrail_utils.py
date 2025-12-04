"""Shared utilities for guardrail tests."""

import types

from uipath.platform.guardrails import BaseGuardrail

from uipath_langchain.agent.guardrails.actions.base_action import (
    GuardrailAction,
    GuardrailActionNode,
)


class FakeStateGraph:
    def __init__(self, _state_type):
        self.added_nodes = []
        self.added_edges = []

    def add_node(self, name, node):
        self.added_nodes.append((name, node))

    def add_edge(self, src, dst):
        self.added_edges.append((src, dst))

    def compile(self):
        # Return a simple object we can inspect if needed
        return types.SimpleNamespace(nodes=self.added_nodes, edges=self.added_edges)


def fake_action(fail_prefix: str) -> GuardrailAction:
    class _Action(GuardrailAction):
        def action_node(
            self,
            *,
            guardrail: BaseGuardrail,
            scope,
            execution_stage,
        ) -> GuardrailActionNode:
            name = f"{fail_prefix}_{execution_stage.name.lower()}_{guardrail.name}"
            return name, lambda s: s

    return _Action()


def fake_factory(eval_prefix):
    def _factory(guardrail, execution_stage, success_node, failure_node):
        name = f"{eval_prefix}_{execution_stage.name.lower()}_{guardrail.name}"
        return name, (lambda s: s)  # node function not invoked in this test

    return _factory
