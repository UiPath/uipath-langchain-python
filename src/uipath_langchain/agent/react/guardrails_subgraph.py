from __future__ import annotations

from typing import Any, Callable, Literal, Sequence

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from uipath.agent.models.agent import AgentGuardrail
from uipath.models.guardrails import GuardrailScope

from .guardrail_nodes import (
    create_agent_guardrail_node,
    create_llm_guardrail_node,
    create_tool_guardrail_node,
)
from .types import AgentGraphState


def create_guardrails_subgraph(
    inner: tuple[str, Any],
    guardrails: Sequence[AgentGuardrail] | None,
    node_factory: Callable[
        [AgentGuardrail, Literal["before", "after"]],
        tuple[str, Callable] | None,
    ] = create_llm_guardrail_node,
    guardrail_filter: Callable[[AgentGuardrail], bool]
    | None = lambda gr: GuardrailScope.LLM in gr.selector.scopes,
) -> Any:
    """Wrap a named inner runnable in a subgraph with before/after guardrail nodes."""
    inner_name, inner_node = inner
    before_nodes: dict[str, Callable] = {}
    after_nodes: dict[str, Callable] = {}
    predicate = guardrail_filter or (lambda gr: True)
    applicable_guardrails = [gr for gr in (guardrails or []) if predicate(gr)]

    for gr in applicable_guardrails:
        created = node_factory(gr, "before")
        if created is not None:
            name, fn = created
            before_nodes[name] = fn
    for gr in applicable_guardrails:
        created = node_factory(gr, "after")
        if created is not None:
            name, fn = created
            after_nodes[name] = fn

    subgraph = StateGraph(AgentGraphState)
    for node_name, node_callable in before_nodes.items():
        subgraph.add_node(node_name, node_callable)
    for node_name, node_callable in after_nodes.items():
        subgraph.add_node(node_name, node_callable)
    subgraph.add_node(inner_name, inner_node)

    before_names = list(before_nodes.keys())
    after_names = list(after_nodes.keys())
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
    inner: tuple[str, Any], guardrails: Sequence[AgentGuardrail] | None
) -> Any:
    return create_guardrails_subgraph(
        inner=inner,
        guardrails=guardrails,
        node_factory=create_llm_guardrail_node,  # type: ignore[arg-type]
        guardrail_filter=lambda gr: GuardrailScope.LLM in gr.selector.scopes,
    )


def create_agent_guardrails_subgraph(
    inner: tuple[str, Any], guardrails: Sequence[AgentGuardrail] | None
) -> Any:
    return create_guardrails_subgraph(
        inner=inner,
        guardrails=guardrails,
        node_factory=create_agent_guardrail_node,  # type: ignore[arg-type]
        guardrail_filter=lambda gr: GuardrailScope.AGENT in gr.selector.scopes,
    )


def create_tool_guardrails_subgraph(
    inner: tuple[str, Any], guardrails: Sequence[AgentGuardrail] | None
) -> Any:
    return create_guardrails_subgraph(
        inner=inner,
        guardrails=guardrails,
        node_factory=create_tool_guardrail_node,  # type: ignore[arg-type]
        guardrail_filter=lambda gr: GuardrailScope.TOOL in gr.selector.scopes,
    )
