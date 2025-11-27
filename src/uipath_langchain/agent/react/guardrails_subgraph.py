from __future__ import annotations

from typing import Any, Callable, Literal, Sequence

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from uipath.agent.models.agent import AgentGuardrail
from uipath.models.guardrails import GuardrailScope

from .guardrail_nodes import (
    GraphNode,
    GuardrailAction,
    create_agent_guardrail_node,
    create_llm_guardrail_node,
    create_tool_guardrail_node,
)
from .types import AgentGuardrailsGraphState


def _build_guardrails_chain_by_execution_stage(
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
    *,
    scope: GuardrailScope,
    execution_stage: Literal["PreExecution", "PostExecution"],
) -> list[tuple[AgentGuardrail, GraphNode]]:
    """Produce (guardrail, mandatory failure action node as GraphNode) for a stage."""
    items: list[tuple[AgentGuardrail, GraphNode]] = []
    for guardrail, action in guardrails or []:
        failure_node = action.enforcement_outcome(
            guardrail=guardrail, scope=scope, execution_stage=execution_stage
        )
        items.append((guardrail, failure_node))
    return items


def create_guardrails_subgraph(
    main_inner_node: tuple[str, Any],
    guardrails: Sequence[tuple[AgentGuardrail, GuardrailAction]] | None,
    scope: GuardrailScope,
    node_factory: Callable[
        [
            AgentGuardrail,
            Literal["PreExecution", "PostExecution"],
            GraphNode,  # success node (name, callable)
            GraphNode,  # failure node (name, callable)
        ],
        GraphNode,
    ] = create_llm_guardrail_node,
) -> Any:
    """Build a subgraph that enforces guardrails around an inner node.

    START -> pre-eval nodes (dynamic goto) -> inner -> post-eval nodes (dynamic goto) -> END

    No static edges are added between guardrail nodes; each eval decides via Command.
    Failure nodes are added but not chained; they are expected to route via Command.
    """
    inner_name, inner_node = main_inner_node

    # Build stage descriptors (guardrail + failure action node) for both phases
    before_items = _build_guardrails_chain_by_execution_stage(
        guardrails, scope=scope, execution_stage="PreExecution"
    )
    after_items = _build_guardrails_chain_by_execution_stage(
        guardrails, scope=scope, execution_stage="PostExecution"
    )

    subgraph = StateGraph(AgentGuardrailsGraphState)

    # Construct pre-execution eval nodes from tail to head, so we can pass the success target
    before_eval_nodes_rev: list[GraphNode] = []
    next_success: GraphNode = (inner_name, inner_node)
    for guardrail, failure_node in reversed(before_items):
        eval_node = node_factory(guardrail, "PreExecution", next_success, failure_node)
        before_eval_nodes_rev.append(eval_node)
        next_success = eval_node
    before_eval_nodes = list(reversed(before_eval_nodes_rev))

    # Construct post-execution eval nodes from tail to head. Success target for last is END
    after_eval_nodes_rev: list[GraphNode] = []
    end_node: GraphNode = (str(END), lambda _s: {})
    next_success_post: GraphNode = end_node
    for guardrail, failure_node in reversed(after_items):
        eval_node = node_factory(guardrail, "PostExecution", next_success_post, failure_node)
        after_eval_nodes_rev.append(eval_node)
        next_success_post = eval_node
    after_eval_nodes = list(reversed(after_eval_nodes_rev))

    # Add nodes: all eval nodes, inner node, and all failure action nodes
    for name, node in before_eval_nodes:
        subgraph.add_node(name, node)
    subgraph.add_node(inner_name, inner_node)
    for name, node in after_eval_nodes:
        subgraph.add_node(name, node)

    for _, fail_node in before_items:
        subgraph.add_node(fail_node[0], fail_node[1])
    for _, fail_node in after_items:
        subgraph.add_node(fail_node[0], fail_node[1])

    # Only minimal static edges: entry to first pre, inner to first post (or END)
    if before_eval_nodes:
        subgraph.add_edge(START, before_eval_nodes[0][0])
    else:
        subgraph.add_edge(START, inner_name)

    if after_eval_nodes:
        subgraph.add_edge(inner_name, after_eval_nodes[0][0])
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
        (guardrail, action)
        for (guardrail, action) in (guardrails or [])
        if GuardrailScope.TOOL in guardrail.selector.scopes
        and tool_name in guardrail.selector.match_names
    ]
    return create_guardrails_subgraph(
        main_inner_node=tool_node,
        guardrails=applicable_guardrails,
        scope=GuardrailScope.TOOL,
        node_factory=create_tool_guardrail_node,
    )
