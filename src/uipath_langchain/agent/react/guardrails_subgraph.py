from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence, cast

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from uipath.agent.models.agent import AgentGuardrail
from uipath.models.guardrails import GuardrailScope

from .guardrail_nodes import (
    ActionEnforcementNode,
    ActionInlineEnforcement,
    GuardrailAction,
    create_agent_guardrail_node,
    create_llm_guardrail_node,
    create_tool_guardrail_node,
)
from .types import AgentGraphState


def _build_guardrails_chain_by_execution_stage(
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
    *,
    scope: GuardrailScope,
    hook_type: Literal["PreExecution", "PostExecution"],
    node_factory: Callable[
        [
            AgentGuardrail,
            Literal["PreExecution", "PostExecution"],
            Optional[ActionInlineEnforcement],
        ],
        tuple[str, Callable],
    ],
) -> list[tuple[str, Callable]]:
    chain: list[tuple[str, Callable]] = []

    for guardrail, action in guardrails or []:
        action_enforcement_node = None
        inline_action_to_enforce = None

        action_enforcement = action.enforcement_outcome(
            guardrail=guardrail,
            scope=scope,
            hook_type=hook_type,
        )
        if isinstance(action_enforcement, ActionEnforcementNode):
            action_enforcement_node = action_enforcement
        else:
            inline_action_to_enforce = cast(ActionInlineEnforcement, action_enforcement)

        eval_node = node_factory(guardrail, hook_type, inline_action_to_enforce)
        if eval_node is not None:
            chain.append(eval_node)
        if action_enforcement_node is not None:
            chain.append(action_enforcement)
    return chain


def create_guardrails_subgraph(
    main_inner_node: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
    scope: GuardrailScope,
    node_factory: Callable[
        [
            AgentGuardrail,
            Literal["PreExecution", "PostExecution"],
            Optional[ActionInlineEnforcement],
        ],
        tuple[str, Callable],
    ] = create_llm_guardrail_node,
) -> Any:
    """Wrap a named inner runnable in a subgraph with before/after guardrail nodes.

    Accepts per-guardrail actions by passing a sequence of (AgentGuardrail, GuardrailAction)
    tuples. Inline actions execute within the guardrail nodes; node-producing actions are linked
    immediately after their corresponding guardrail nodes.
    """
    inner_name, inner_node = main_inner_node
    before_chain = _build_guardrails_chain_by_execution_stage(
        guardrails,
        scope=scope,
        hook_type="PreExecution",
        node_factory=node_factory,
    )
    after_chain = _build_guardrails_chain_by_execution_stage(
        guardrails,
        scope=scope,
        hook_type="PostExecution",
        node_factory=node_factory,
    )

    subgraph = StateGraph(AgentGraphState)
    for node_name, before_node in before_chain:
        subgraph.add_node(node_name, before_node)
    for node_name, after_node in after_chain:
        subgraph.add_node(node_name, after_node)
    subgraph.add_node(inner_name, inner_node)

    before_names = [name for name, _ in before_chain]
    after_names = [name for name, _ in after_chain]
    if before_names:
        subgraph.add_edge(START, before_names[0])
        for cur, nxt in zip(before_names, before_names[1:], strict=False):
            subgraph.add_edge(cur, nxt)
        subgraph.add_edge(before_names[-1], inner_name)
    else:
        subgraph.add_edge(START, inner_name)

    if after_names:
        subgraph.add_edge(inner_name, after_names[0])
        for cur, nxt in zip(after_names, after_names[1:], strict=False):
            subgraph.add_edge(cur, nxt)
        subgraph.add_edge(after_names[-1], END)
    else:
        subgraph.add_edge(inner_name, END)

    return subgraph.compile()


def create_llm_guardrails_subgraph(
    llm_node: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
) -> Any:
    applicable_guardrails = [
        (guardrail, _)
        for (guardrail, _) in (guardrails or [])
        if GuardrailScope.LLM in guardrail.selector.scopes
    ]
    return create_guardrails_subgraph(
        main_inner_node=llm_node,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.LLM,
        node_factory=create_llm_guardrail_node,
    )


def create_agent_guardrails_subgraph(
    agent_node: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
) -> Any:
    applicable_guardrails = [
        (guardrail, _)
        for (guardrail, _) in (guardrails or [])
        if GuardrailScope.AGENT in guardrail.selector.scopes
    ]
    return create_guardrails_subgraph(
        main_inner_node=agent_node,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.AGENT,
        node_factory=create_agent_guardrail_node,
    )


def create_tool_guardrails_subgraph(
    tool_node: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
) -> Any:
    tool_name, _ = tool_node
    applicable_guardrails = [
        (guardrail, _action)
        for (guardrail, _action) in (guardrails or [])
        if GuardrailScope.TOOL in guardrail.selector.scopes
        and tool_name in guardrail.selector.match_names
    ]
    return create_guardrails_subgraph(
        main_inner_node=tool_node,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.TOOL,
        node_factory=create_tool_guardrail_node,
    )
