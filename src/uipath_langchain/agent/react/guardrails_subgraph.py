from __future__ import annotations

from typing import Any, Callable, Literal, Sequence, cast

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from uipath.agent.models.agent import AgentGuardrail
from uipath.models.guardrails import GuardrailScope

from .guardrail_nodes import (
    GuardrailAction,
    create_agent_guardrail_node,
    create_llm_guardrail_node,
    create_tool_guardrail_node,
)
from .types import AgentGraphState


def create_guardrails_subgraph(
    inner: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
    node_factory: Callable[
        [AgentGuardrail, Literal["before", "after"], GuardrailAction],
        tuple[str, Callable] | None,
    ] = create_llm_guardrail_node,
    guardrail_filter: Callable[[AgentGuardrail], bool]
    | None = lambda gr: GuardrailScope.LLM in gr.selector.scopes,
) -> Any:
    """Wrap a named inner runnable in a subgraph with before/after guardrail nodes.

    Accepts per-guardrail actions by passing a sequence of (AgentGuardrail, GuardrailAction)
    tuples. Inline actions execute within the guardrail nodes; node-producing actions are linked
    immediately after their corresponding guardrail nodes.
    """
    inner_name, inner_node = inner
    before_chain: list[tuple[str, Callable]] = []
    after_chain: list[tuple[str, Callable]] = []
    predicate = guardrail_filter or (lambda gr: True)
    pairs: list[tuple[AgentGuardrail, GuardrailAction]] = (
        [(gr, act) for gr, act in guardrails if predicate(gr)] if guardrails else []
    )

    for gr, resolved_action in pairs:
        created = node_factory(gr, "before", resolved_action)
        if created is not None:
            before_chain.append(created)
        # If action returns a node, link it right after the guardrail node
        act = resolved_action.apply(
            guardrail=gr,
            scope=gr.selector.scopes[0] if gr.selector.scopes else GuardrailScope.LLM,
            hook_type="before",
            payload_generator=lambda state: "",  # not used for linking decision
        )
        if isinstance(act, tuple):
            before_chain.append(cast(tuple[str, Callable], act))
    for gr, resolved_action in pairs:
        created = node_factory(gr, "after", resolved_action)
        if created is not None:
            after_chain.append(created)
        act = resolved_action.apply(
            guardrail=gr,
            scope=gr.selector.scopes[0] if gr.selector.scopes else GuardrailScope.LLM,
            hook_type="after",
            payload_generator=lambda state: "",
        )
        if isinstance(act, tuple):
            after_chain.append(cast(tuple[str, Callable], act))

    subgraph = StateGraph(AgentGraphState)
    for node_name, node_callable in before_chain:
        subgraph.add_node(node_name, node_callable)
    for node_name, node_callable in after_chain:
        subgraph.add_node(node_name, node_callable)
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
    inner: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
) -> Any:
    return create_guardrails_subgraph(
        inner=inner,
        guardrails=guardrails,
        node_factory=create_llm_guardrail_node,  # type: ignore[arg-type]
        guardrail_filter=lambda gr: GuardrailScope.LLM in gr.selector.scopes,
    )


def create_agent_guardrails_subgraph(
    inner: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
) -> Any:
    return create_guardrails_subgraph(
        inner=inner,
        guardrails=guardrails,
        node_factory=create_agent_guardrail_node,  # type: ignore[arg-type]
        guardrail_filter=lambda gr: GuardrailScope.AGENT in gr.selector.scopes,
    )


def create_tool_guardrails_subgraph(
    inner: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
) -> Any:
    return create_guardrails_subgraph(
        inner=inner,
        guardrails=guardrails,
        node_factory=create_tool_guardrail_node,  # type: ignore[arg-type]
        guardrail_filter=lambda gr: GuardrailScope.TOOL in gr.selector.scopes,
    )
